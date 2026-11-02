#!/usr/bin/env python3
"""Text-mode live view of an MD box: a 2D slab projection of the atoms, drawn with 256-colour half-block characters in the terminal. Needs ase + numpy.
  follow a trajectory that another job is writing:   python mdview.py runs/eos_md/<run>/traj.extxyz --follow
  replay a finished trajectory:                       python mdview.py traj.extxyz --replay 8        (frames per second)
  from inside an MD script (every step, no file needed):
      from mdview import render; dyn.attach(lambda: print(render(atoms, step=dyn.nsteps), flush=True), interval=1)      # or interval=5
Atoms inside a z-slab of --slab A around a plane that sweeps slowly through the box are bright; the rest are dim (depth cue). O = red, H = cyan/white. Use a terminal at least 100 columns wide."""
import argparse, sys, time
import numpy as np

W_DEFAULT, H_DEFAULT = 96, 30            # character columns, character rows (each row shows 2 pixels with half blocks)


def render(at, step=None, width=W_DEFAULT, height=H_DEFAULT, slab=2.5, sweep=True, note=""):
    cell = np.asarray(at.cell.array); L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]]) if np.allclose(cell, np.diag(np.diag(cell))) else cell.sum(axis=0)
    pos = at.get_scaled_positions(wrap=True) * L; z0 = (time.time() * 0.6 % 1.0 if sweep else 0.5) * L[2]
    px, py = width, height * 2; img = np.zeros((py, px, 3), dtype=np.uint8) if False else [[None] * px for _ in range(py)]; dep = np.full((py, px), 1e9)
    dz = np.abs(pos[:, 2] - z0); dz = np.minimum(dz, L[2] - dz)
    for p, z, num in sorted(zip(pos, dz, at.numbers), key=lambda t: -t[1]):
        ix = min(int(p[0] / L[0] * px), px - 1); iy = min(int(p[1] / L[1] * py), py - 1)
        bright = z < slab; col = (255, 70, 60) if num == 8 else (90, 230, 255)
        if not bright: col = tuple(int(c * 0.28) for c in col)
        if z < dep[iy, ix] or bright: img[iy][ix] = col; dep[iy, ix] = z
        if num == 8 and bright and ix + 1 < px: img[iy][ix + 1] = col                      # oxygen drawn two pixels wide
    lines = []
    for r in range(0, py, 2):
        row = ""
        for x in range(px):
            top, bot = img[r][x], img[r + 1][x]
            if top is None and bot is None: row += "\033[2m·\033[0m" if (x % 8 == 0 and r % 8 == 0) else " "
            else:
                fg = top or (0, 0, 0); bg = bot or (0, 0, 0); row += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m▀\033[0m"
        lines.append("│" + row + "│")
    nO = int((at.numbers == 8).sum()); nH = int((at.numbers == 1).sum()); info = "  ".join(f"{k} {v:.4g}" for k, v in at.info.items() if k in ("T_inst", "T_K", "P_grace_GPa", "P_GPa") and isinstance(v, (int, float)))
    head = f" {len(at)} atoms (O {nO}, H {nH})   box {L[0]:.1f}x{L[1]:.1f}x{L[2]:.1f} A   slab z={z0:.1f} +-{slab} A   " + (f"step {step}   " if step is not None else "") + info + ("  " + note if note else "")
    return "\033[H" + "\033[1m" + head.ljust(width + 2) + "\033[0m\n┌" + "─" * width + "┐\n" + "\n".join(lines) + "\n└" + "─" * width + "┘"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("traj"); ap.add_argument("--follow", action="store_true"); ap.add_argument("--replay", type=float, default=0)
    ap.add_argument("--slab", type=float, default=2.5); ap.add_argument("--width", type=int, default=W_DEFAULT); ap.add_argument("--height", type=int, default=H_DEFAULT); a = ap.parse_args()
    from ase.io import read
    sys.stdout.write("\033[2J\033[?25l")
    try:
        if a.replay:
            for i, f in enumerate(read(a.traj, index=":")): print(render(f, step=i, width=a.width, height=a.height, slab=a.slab), flush=True); time.sleep(1.0 / a.replay)
        elif a.follow:
            n = -1
            while True:
                try: f = read(a.traj, index=-1); import os; size = os.path.getsize(a.traj)
                except Exception: time.sleep(1); continue
                if size != n: n = size; print(render(f, width=a.width, height=a.height, slab=a.slab, note="[following]"), flush=True)
                time.sleep(0.5)
        else: print(render(read(a.traj, index=-1), width=a.width, height=a.height, slab=a.slab))
    except KeyboardInterrupt: pass
    finally: sys.stdout.write("\033[?25h\n")
