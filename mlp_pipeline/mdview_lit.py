"""Lit 3D molecule view in braille characters (vectorised NumPy): spheres with ambient + diffuse + specular shading and a true z-buffer, shaded cylinder bonds, and a vectorised pixel-to-braille
packing. This is a separate renderer next to mdview.py (which stays as it was): same box / view options, same LAST hook for DiffPrinter, so toy_live.py --lit uses it.
The technique is the standard 'sphere impostor' rasterisation (per pixel normal from the disc, Blinn-Phong), the same idea MolTUI uses; this code is written from scratch for this project.
  from mdview_lit import render_lit;  print(render_lit(atoms, yaw=30, width=140, height=44))
"""
import time
import numpy as np
import mdview

BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.uint16)
LIGHT = np.array([-0.45, 0.55, 0.70]); LIGHT = LIGHT / np.linalg.norm(LIGHT)          # towards the light, view space: x right, y up, z towards the viewer
HALF = LIGHT + np.array([0.0, 0.0, 1.0]); HALF = HALF / np.linalg.norm(HALF)
COL = {8: np.array([255.0, 62.0, 52.0]), 1: np.array([120.0, 225.0, 255.0])}
RAD = {8: 0.30, 1: 0.17}                                    # sphere radii in Angstrom (display radii, much smaller than van der Waals: 162 atoms in a 7 A box and bonds must stay visible)


class Raster:
    def __init__(self, w, h):
        self.w, self.h = w, h; self.z = np.full((h, w), np.inf); self.rgb = np.zeros((h, w, 3)); self.hit = np.zeros((h, w), bool)

    def sphere(self, cx, cy, depth, rpx, rA, color, fade, lit=True):
        if rpx < 0.6: 
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= xi < self.w and 0 <= yi < self.h and depth < self.z[yi, xi]: self.z[yi, xi] = depth; self.rgb[yi, xi] = color * fade; self.hit[yi, xi] = True
            return
        x0, x1 = max(0, int(cx - rpx - 1)), min(self.w - 1, int(cx + rpx + 1)); y0, y1 = max(0, int(cy - rpx - 1)), min(self.h - 1, int(cy + rpx + 1))
        if x0 > x1 or y0 > y1: return
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)); dx = (gx - cx) / rpx; dy = (gy - cy) / rpx; r2 = dx * dx + dy * dy; mask = r2 <= 1.0
        dz = np.sqrt(np.maximum(0.0, 1.0 - r2)); nl = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9; nx, ny, nz = dx / nl, -dy / nl, dz / nl
        diff = np.maximum(0.0, nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]); spec = np.maximum(0.0, nx * HALF[0] + ny * HALF[1] + nz * HALF[2]) ** 28 * 0.45
        inten = (np.minimum(1.0, 0.30 + 0.70 * diff) if lit else 0.85) * fade; spec = spec if lit else spec * 0.0; pz = depth - rA * dz                                      # surface depth: nearer than the centre
        zs = self.z[y0:y1 + 1, x0:x1 + 1]; ok = mask & (pz < zs); zs[ok] = pz[ok]
        shaded = np.minimum(255.0, color[None, None, :] * inten[:, :, None] + 255.0 * spec[:, :, None] * fade)
        sub = self.rgb[y0:y1 + 1, x0:x1 + 1]; sub[ok] = shaded[ok]; self.hit[y0:y1 + 1, x0:x1 + 1] |= ok

    def line(self, p0, p1, d0, d1, color, hw=0.5, fade=1.0, spec=True):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]; L = float(np.hypot(dx, dy)); n = int(L * 2.5) + 2; t = np.linspace(0, 1, n)
        nxp, nyp = (-dy / L, dx / L) if L > 1e-6 else (0.0, 1.0); offs = np.arange(-int(hw + 1), int(hw + 1) + 1, dtype=float); offs = offs[np.abs(offs) <= max(hw, 0.5)]
        xs = np.round(p0[0] + dx * t[:, None] + nxp * offs[None, :]).astype(int); ys = np.round(p0[1] + dy * t[:, None] + nyp * offs[None, :]).astype(int)
        zz = (d0 + (d1 - d0) * t)[:, None] - 0.03 * np.sqrt(np.maximum(0, 1 - (offs / max(hw, 0.5)) ** 2))[None, :]
        cyl = np.sqrt(np.maximum(0.0, 1 - (offs / max(hw, 0.5)) ** 2)); inten = np.minimum(1.0, 0.28 + 0.72 * (0.35 + 0.65 * cyl)) * fade
        ok = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h)
        xs, ys, zz = xs[ok], ys[ok], zz[ok]; I = np.broadcast_to(inten[None, :], ok.shape)[ok]; tt = np.broadcast_to(t[:, None], ok.shape)[ok]
        better = zz < self.z[ys, xs]; xs, ys, zz, I, tt = xs[better], ys[better], zz[better], I[better], tt[better]
        if len(xs) == 0: return
        col = np.where((tt < 0.5)[:, None], color[0][None, :], color[1][None, :]); self.z[ys, xs] = zz; self.rgb[ys, xs] = np.minimum(255.0, col * I[:, None]); self.hit[ys, xs] = True


def oh_pairs(at, L, cutoff=1.3):
    """O-H pairs closer than cutoff under periodic boundaries with their minimum-image vectors (scipy KD-trees: ~20x faster than ase's neighbor_list); falls back to ase"""
    try:
        from scipy.spatial import cKDTree
        pos = at.get_scaled_positions(wrap=True) * L; pos = np.where(pos >= L, pos - L, pos); iO = np.where(at.numbers == 8)[0]; iH = np.where(at.numbers == 1)[0]
        if len(iO) == 0 or len(iH) == 0: return []
        m = cKDTree(pos[iO], boxsize=L).sparse_distance_matrix(cKDTree(pos[iH], boxsize=L), cutoff, output_type="ndarray")
        out = []
        for a, b in zip(m["i"], m["j"]):
            v = pos[iH[b]] - pos[iO[a]]; v -= L * np.round(v / L); out.append((int(iO[a]), int(iH[b]), v))
        return out
    except Exception:
        from ase.neighborlist import neighbor_list
        i_, j_, vec = neighbor_list("ijD", at, cutoff); return [(i, j, v) for i, j, v in zip(i_, j_, vec) if at.numbers[i] == 8 and at.numbers[j] == 1]


def rasterise(at, yaw=0.0, tilt=18.0, width=140, height=44, zoom=1.3, view_dir=None, ortho=False, bonds=True, box=True, clip=0.0, hi=1.0, ssaa=1, mode="braille", show_O=True, show_H=True, light=True, depth_fade=True):
    sub = {"braille": (2, 4), "quad": (2, 2), "half": (1, 2)}[mode]; px, py = width * sub[0] * ssaa, height * sub[1] * ssaa
    cell = np.asarray(at.cell.array); L = np.diag(cell).astype(float); pos = at.get_scaled_positions(wrap=True) * L - L / 2
    ya, ta = np.radians(yaw), np.radians(tilt); Ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]]); Rx = np.array([[1, 0, 0], [0, np.cos(ta), -np.sin(ta)], [0, np.sin(ta), np.cos(ta)]]); R = Rx @ Ry
    if view_dir is not None:
        fwd = np.asarray(view_dir, float); fwd /= np.linalg.norm(fwd); up0 = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.9 else np.array([1.0, 0.0, 0.0]); right = np.cross(up0, fwd); right /= np.linalg.norm(right)
        R = np.array([right, np.cross(fwd, right), fwd]); R = np.array([[np.cos(ya), np.sin(ya), 0], [-np.sin(ya), np.cos(ya), 0], [0, 0, 1]]) @ R
    P = pos @ R.T; half = np.linalg.norm(L) / 2; D = 2.6 * half; f = zoom * 0.46 * min(px, py * 1.0) / half
    sc = lambda p: (px / 2 + f * p[..., 0] * ((D / (D - p[..., 2])) if not ortho else 1.0), py / 2 - f * p[..., 1] * ((D / (D - p[..., 2])) if not ortho else 1.0))
    zmin, zmax = P[:, 2].min(), P[:, 2].max(); near = (P[:, 2] - zmin) / max(zmax - zmin, 1e-9); rs = Raster(px, py)
    if box:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * L / 2 @ R.T
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs((np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])[i] - np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])[j])) > 1.5) == 1:
                    a, b = sc(corners[i]), sc(corners[j]); rs.line(a, b, -corners[i][2] + 50, -corners[j][2] + 50, (np.array([70.0, 85.0, 130.0]),) * 2, hw=0.3, fade=0.8)
    if bonds:
        pairs = oh_pairs(at, L)
        for i, j, v in pairs:
            if near[i] >= clip and near[i] <= hi and show_O and show_H:
                a0 = P[i]; a1 = a0 + v @ R.T; s0 = sc(a0); s1 = sc(a1); fa = (0.50 + 0.50 * min(max(near[i], 0), 1)) if depth_fade else 1.0
                rs.line(s0, s1, -a0[2], -a1[2], (np.array([230.0, 150.0, 140.0]), np.array([170.0, 225.0, 245.0])), hw=max(0.5, 0.05 * f * (1 if ortho else D / (D - a0[2])) / ssaa) * ssaa, fade=fa)
    for k in np.argsort(P[:, 2]):
        if near[k] < clip or near[k] > hi: continue
        if (at.numbers[k] == 8 and not show_O) or (at.numbers[k] == 1 and not show_H): continue
        p = P[k]; s = 1.0 if ortho else D / (D - p[2]); cx, cy = sc(p); Z = at.numbers[k]; fade = (0.45 + 0.55 * near[k] ** 1.1) if depth_fade else 1.0
        rs.sphere(float(cx), float(cy), -p[2], RAD[Z] * f * s, RAD[Z], COL[Z], fade, lit=light)
    return rs


def pack(rs, width, height):
    """pixels (rows*4, cols*2) -> braille code points and one averaged colour per character cell, all array operations"""
    hit = rs.hit.reshape(height, 4, width, 2); rgb = rs.rgb.reshape(height, 4, width, 2, 3); cnt = hit.sum(axis=(1, 3)); mask = np.where(hit, BITS[None, :, None, :], 0).sum(axis=(1, 3))
    col = (rgb * hit[..., None]).sum(axis=(1, 3)) / np.maximum(cnt, 1)[..., None]; return mask, np.clip(col * 1.12, 0, 255).astype(np.uint8)


def render_lit(at, yaw=0.0, step=None, width=140, height=44, note="", mode="braille", **kw):
    rs = rasterise(at, yaw=yaw, width=width, height=height, mode=mode, **kw)
    if mode != "braille": return _blocks(at, rs, yaw, step, width, height, note, mode)
    mask, col = pack(rs, width, height); cells = []; lines = []
    q = (col >> 3) << 3                                       # 5 bits per channel: neighbouring cells share a colour code, so runs merge
    for r in range(height):
        rc = []; line = []; prev = None
        for c in range(width):
            m = int(mask[r, c])
            if not m: rc.append(" "); cur = None; ch = " "
            else: cur = (int(q[r, c, 0]), int(q[r, c, 1]), int(q[r, c, 2])); ch = chr(0x2800 + m); rc.append(f"\033[38;2;{cur[0]};{cur[1]};{cur[2]}m{ch}\033[0m")
            if cur != prev: line.append(("\033[0m" if prev is not None else "") + (f"\033[38;2;{cur[0]};{cur[1]};{cur[2]}m" if cur is not None else "")); prev = cur
            line.append(ch)
        lines.append("│" + "".join(line) + "\033[0m│"); cells.append(rc)
    nO = int((at.numbers == 8).sum()); nH = int((at.numbers == 1).sum()); info = "  ".join(f"{k} {v:.4g}" for k, v in at.info.items() if k in ("T_inst", "T_K", "P_grace_GPa", "P_GPa") and isinstance(v, (int, float)))
    head = f" {len(at)} atoms (O {nO}, H {nH})  lit braille  yaw {yaw % 360:5.1f}  " + (f"step {step}  " if step is not None else "") + info + ("  " + note if note else "")
    mdview.LAST["head"] = head.ljust(width + 2); mdview.LAST["cells"] = cells; mdview.LAST["width"] = width
    return "\033[H\033[1m" + head.ljust(width + 2) + "\033[0m\n┌" + "─" * width + "┐\n" + "\n".join(lines) + "\n└" + "─" * width + "┘"


QUAD = [" ", "▘", "▝", "▀", "▖", "▌", "▞", "▛", "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█"]


def _fg(c): return f"\033[38;2;{int(c[0])};{int(c[1])};{int(c[2])}m"


def _blocks(at, rs, yaw, step, width, height, note, mode):
    """block-character output with solid fills (no dot gaps). 'half': two colours per cell (top pixel as foreground, bottom pixel as background) = true per-pixel colour; 'quad': 2x2 solid quadrants, one colour per cell."""
    gam = lambda c: 255.0 * (np.clip(c, 0, 255) / 255.0) ** 0.85; cells = []; lines = []
    if mode == "half":
        top, bot = rs.rgb[0::2], rs.rgb[1::2]; ht, hb = rs.hit[0::2], rs.hit[1::2]
        for r in range(height):
            rc = []
            for c in range(width):
                a, b = ht[r, c], hb[r, c]
                if a and b: rc.append(_fg(gam(top[r, c])) + f"\033[48;2;{int(gam(bot[r, c])[0])};{int(gam(bot[r, c])[1])};{int(gam(bot[r, c])[2])}m▀\033[0m")
                elif a: rc.append(_fg(gam(top[r, c])) + "▀\033[0m")
                elif b: rc.append(_fg(gam(bot[r, c])) + "▄\033[0m")
                else: rc.append(" ")
            cells.append(rc); lines.append("│" + "".join(rc) + "│")
    else:
        h4 = rs.hit.reshape(height, 2, width, 2); m = (h4[:, 0, :, 0] * 1 + h4[:, 0, :, 1] * 2 + h4[:, 1, :, 0] * 4 + h4[:, 1, :, 1] * 8).astype(int)
        c4 = rs.rgb.reshape(height, 2, width, 2, 3); cnt = h4.sum(axis=(1, 3)); col = (c4 * h4[..., None]).sum(axis=(1, 3)) / np.maximum(cnt, 1)[..., None]
        for r in range(height):
            rc = [(" " if not m[r, c] else _fg(gam(col[r, c])) + QUAD[m[r, c]] + "\033[0m") for c in range(width)]; cells.append(rc); lines.append("│" + "".join(rc) + "│")
    nO = int((at.numbers == 8).sum()); nH = int((at.numbers == 1).sum()); info = "  ".join(f"{k} {v:.4g}" for k, v in at.info.items() if k in ("T_inst", "T_K", "P_grace_GPa", "P_GPa") and isinstance(v, (int, float)))
    head = f" {len(at)} atoms (O {nO}, H {nH})  lit {mode}  yaw {yaw % 360:5.1f}  " + (f"step {step}  " if step is not None else "") + info + ("  " + note if note else "")
    mdview.LAST["head"] = head.ljust(width + 2); mdview.LAST["cells"] = cells; mdview.LAST["width"] = width
    return "\033[H\033[1m" + head.ljust(width + 2) + "\033[0m\n┌" + "─" * width + "┐\n" + "\n".join(lines) + "\n└" + "─" * width + "┘"
