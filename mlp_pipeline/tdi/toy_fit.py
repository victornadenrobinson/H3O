"""Force-matched spline pair potential (O-O, O-H, H-H) fitted to DFT snapshots: a cheap, more realistic stand-in for GRACE in the toy validation (tdi/toy_ice.py).
usage (Young, repo root, CPU is enough; ~/grace-venv/bin/python has numpy, scipy, ASE):
    python mlp_pipeline/tdi/toy_fit.py runs/train_m.extxyz --valid runs/valid_m.extxyz [--pmin 250 --pmax 700 --tmin 3000] [--rc 3.5 --dr 0.2] -o runs/checks/toyfit
    -> runs/checks/toyfit.json (the potential: send this file, it has no DFT data in it) and toyfit.png (pair curves)
then anywhere:  TOYICE_FIT=runs/checks/toyfit.json python mlp_pipeline/tdi/toy_validate.py ...   (model spec 'toyice' uses the fit)
Model. E = sum over pairs phi_ab(r) + sum_s mu_s N_s, phi_ab = clamped cubic B-spline on [r_lo_ab, rc] with phi(rc) = phi'(rc) = 0 (the two end basis functions are
dropped); below r_lo_ab (shortest distance of that pair type in the data, minus --core-pad) a quadratic core phi(r_lo) - phi'(r_lo) (r_lo - r) + K (r_lo - r)^2,
finite at r = 0 (like GRACE's soft core; only reached in the ideal-gas end of TDI paths). Linear in the spline coefficients and mu_s: one least-squares solve of
forces (weight 1 / --sf eV/A), energies per atom (1 / --se eV) and virial stress (1 / --ss eV/A^3), with a curvature penalty (--smooth).
Limits: pair terms only (no many-body / metallic screening), fitted to DFT free energies TOTEN (electronic entropy at SIGMA = kT is T-dependent: a pair potential
averages it). It is a test bed for the TDI method, not a replacement for GRACE.
"""
import argparse, json, os, sys
import numpy as np

PAIRS = ("OO", "OH", "HH")


def pair_kind(zi, zj):
    return np.where((zi == 8) & (zj == 8), 0, np.where((zi == 1) & (zj == 1), 2, 1))


def neighbour_pairs(at, rc):
    """all pairs i<j (with periodic images) within rc: i, j, r, D (r_j - r_i + image)"""
    cell = np.asarray(at.cell); inv = np.linalg.inv(cell); s = at.positions @ inv
    ds = s[None, :, :] - s[:, None, :]; ds -= np.round(ds)
    width = at.get_volume() / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    nimg = [int(np.floor(rc / w + 0.5)) for w in width]; out = []
    iu = np.triu_indices(len(at), 1)
    for sx in range(-nimg[0], nimg[0] + 1):
        for sy in range(-nimg[1], nimg[1] + 1):
            for sz in range(-nimg[2], nimg[2] + 1):
                D = (ds + np.array([sx, sy, sz])) @ cell; d = np.linalg.norm(D, axis=-1)
                if sx == sy == sz == 0:   # i < j within the cell
                    m = np.zeros_like(d, bool); m[iu] = d[iu] < rc
                elif (sx, sy, sz) > (0, 0, 0):   # each image pair once (the mirror image is the same pair)
                    m = d < rc
                else:
                    continue
                i, j = np.nonzero(m); out.append((i, j, d[m], D[m]))
    if not out:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros((0, 3))
    return tuple(np.concatenate([o[k] for o in out]) for k in range(4))


class Basis:
    """clamped cubic B-spline basis on [r_lo, rc], last two functions dropped (phi = phi' = 0 at rc)"""
    def __init__(self, r_lo, rc, dr):
        from scipy.interpolate import BSpline
        inner = np.arange(r_lo + dr, rc - 1e-9, dr)
        self.t = np.r_[[r_lo] * 4, inner, [rc] * 4]; self.n = len(self.t) - 4 - 2; self.r_lo, self.rc = r_lo, rc
        full = len(self.t) - 4
        self.B = BSpline(self.t, np.eye(full), 3, extrapolate=False); self.dB = self.B.derivative()

    def __call__(self, r):
        """values and derivatives of the kept basis functions at r (r_lo <= r <= rc), shape (len(r), n)"""
        r = np.clip(r, self.r_lo, self.rc)
        v = np.nan_to_num(self.B(r))[:, :self.n]; dv = np.nan_to_num(self.dB(r))[:, :self.n]
        return v, dv


def frame_rows(at, bases, rc, K):
    """design rows (forces 3N, energy 1, stress 6) for one frame: X_c (spline coefficients) and counts N_O, N_H; plus the core contribution (fixed K part)"""
    i, j, r, D = neighbour_pairs(at, rc); z = at.numbers; kind = pair_kind(z[i], z[j]); N = len(at); V = at.get_volume()
    ncoef = [b.n for b in bases]; off = np.r_[0, np.cumsum(ncoef)]; nc = off[-1]
    XF = np.zeros((N, 3, nc)); XE = np.zeros(nc); XS = np.zeros((3, 3, nc)); yE_core = 0.0; yF_core = np.zeros((N, 3)); yS_core = np.zeros((3, 3))
    for k, b in enumerate(bases):
        m = kind == k
        if not m.any():
            continue
        rk, Dk, ik, jk = r[m], D[m], i[m], j[m]; core = rk < b.r_lo
        v, dv = b(rk)   # at r < r_lo: values at r_lo (clipped)
        dl = np.where(core, b.r_lo - rk, 0.0)
        val = v - dl[:, None] * dv            # phi(r) = phi(r_lo) - phi'(r_lo) (r_lo - r) [+ K dl^2]
        der = dv                              # phi'(r) = phi'(r_lo) [- 2 K dl] for r < r_lo; spline derivative otherwise
        XE[off[k]:off[k + 1]] += val.sum(0); yE_core += K * (dl ** 2).sum()
        u = Dk / rk[:, None]                  # unit vector i -> j; force on i = +phi' u, on j = -phi' u
        contrib = der[:, None, :] * u[:, :, None]   # (pairs, 3, n)
        np.add.at(XF[:, :, off[k]:off[k + 1]], ik, contrib); np.add.at(XF[:, :, off[k]:off[k + 1]], jk, -contrib)
        fc = (-2 * K * dl)[:, None] * u; np.add.at(yF_core, ik, fc); np.add.at(yF_core, jk, -fc)
        XS[:, :, off[k]:off[k + 1]] += np.einsum("pa,pb,pn->abn", Dk, u, der) / V
        yS_core += np.einsum("pa,pb,p->ab", Dk, u, -2 * K * dl) / V
    return XF.reshape(3 * N, nc), XE, XS.reshape(9, nc)[[0, 4, 8, 5, 2, 1]], yE_core, yF_core.ravel(), yS_core.flat[[0, 4, 8, 5, 2, 1]], (z == 8).sum(), (z == 1).sum()


def load(path, a):
    from ase.io import iread
    out = []
    for f in iread(path, index=":"):
        P, T = f.info.get("P_GPa", np.nan), f.info.get("T_K", np.nan)
        if a.pmin is not None and not P >= a.pmin: continue
        if a.pmax is not None and not P <= a.pmax: continue
        if a.tmin is not None and not T >= a.tmin: continue
        if a.tmax is not None and not T <= a.tmax: continue
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("train"); ap.add_argument("--valid", default=None); ap.add_argument("-o", default="runs/checks/toyfit")
    ap.add_argument("--rc", type=float, default=3.5); ap.add_argument("--dr", type=float, default=0.2, help="knot spacing (A)")
    ap.add_argument("--core-pad", type=float, default=0.05, help="spline starts this far below the shortest distance in the data")
    ap.add_argument("--K", type=float, default=30.0, help="curvature of the quadratic core below r_lo (eV/A^2): repulsive, finite at r = 0")
    ap.add_argument("--sf", type=float, default=1.0); ap.add_argument("--se", type=float, default=0.01); ap.add_argument("--ss", type=float, default=0.02)
    ap.add_argument("--smooth", type=float, default=1e-3); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--pmin", type=float, default=None); ap.add_argument("--pmax", type=float, default=None); ap.add_argument("--tmin", type=float, default=None); ap.add_argument("--tmax", type=float, default=None)
    a = ap.parse_args()
    tr = load(a.train, a)
    if a.max_frames and len(tr) > a.max_frames:
        rng = np.random.default_rng(0); tr = [tr[k] for k in sorted(rng.choice(len(tr), a.max_frames, replace=False))]
    print(f"{len(tr)} training frames ({a.train})", flush=True)
    rmin = [np.inf] * 3
    for f in tr:
        i, j, r, _ = neighbour_pairs(f, a.rc); kind = pair_kind(f.numbers[i], f.numbers[j])
        for k in range(3):
            if (kind == k).any(): rmin[k] = min(rmin[k], float(r[kind == k].min()))
    r_lo = [max(0.3, (m if np.isfinite(m) else 1.0) - a.core_pad) for m in rmin]
    bases = [Basis(r_lo[k], a.rc, a.dr) for k in range(3)]; nc = sum(b.n for b in bases); ncol = nc + 2
    print("shortest distances in the data (A): " + ", ".join(f"{p} {m:.3f}" for p, m in zip(PAIRS, rmin)) + f"; {nc} spline coefficients", flush=True)
    A = np.zeros((ncol, ncol)); y = np.zeros(ncol)
    for n_, f in enumerate(tr):
        XF, XE, XS, eC, fC, sC, nO, nH = frame_rows(f, bases, a.rc, a.K); N = len(f)
        F = f.get_forces().ravel(); E = f.get_potential_energy(); S = f.get_stress()
        rowsF = np.c_[XF, np.zeros((len(XF), 2))] / a.sf; tF = (F - fC) / a.sf          # forces: X c = F - core
        rowE = np.r_[XE, nO, nH][None, :] / (N * a.se); tE = np.array([(E - eC) / (N * a.se)])
        rowsS = np.c_[XS, np.zeros((6, 2))] / a.ss; tS = (S - sC) / a.ss
        for R, t in ((rowsF, tF), (rowE, tE), (rowsS, tS)):
            A += R.T @ R; y += R.T @ t
        if (n_ + 1) % 100 == 0: print(f"  {n_ + 1} frames", flush=True)
    off = np.r_[0, np.cumsum([b.n for b in bases])]   # curvature penalty on each spline's coefficients
    for k in range(3):
        n = bases[k].n; D2 = np.diff(np.eye(n), 2, axis=0); A[off[k]:off[k + 1], off[k]:off[k + 1]] += a.smooth * len(tr) * D2.T @ D2
    c = np.linalg.solve(A, y)
    fit = dict(rc=a.rc, K=a.K, mu_O=float(c[nc]), mu_H=float(c[nc + 1]), pairs={p: dict(t=bases[k].t.tolist(), c=c[off[k]:off[k + 1]].tolist(), r_lo=r_lo[k], n=bases[k].n) for k, p in enumerate(PAIRS)},
               data=dict(train=os.path.basename(a.train), n_train=len(tr), pmin=a.pmin, pmax=a.pmax, tmin=a.tmin, tmax=a.tmax, rmin=dict(zip(PAIRS, rmin))))
    for name, frames in (("train", tr), ("valid", load(a.valid, a) if a.valid else [])):
        if not frames: continue
        m = evaluate(fit, frames); fit[f"errors_{name}"] = m
        print(f"{name}: {m['frames']} frames  E MAE {m['E_MAE_meV_atom']:.1f} meV/atom  F RMSE {m['F_RMSE_eV_A']:.3f} eV/A  stress MAE {m['S_MAE_GPa']:.1f} GPa  "
              f"(DFT force rms {m['F_rms_dft']:.2f} eV/A); mean signed E error by composition: " + ", ".join(f"{k} {v:+.1f}" for k, v in m["E_signed_by_comp"].items()))
    os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True); json.dump({**fit, "pairs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fit["pairs"].items()}}, open(a.o + ".json", "w"), indent=1)   # without cached spline objects; print("wrote", a.o + ".json")
    try:
        plot(fit, a.o + ".png"); print("wrote", a.o + ".png")
    except Exception as ex:
        print("no plot:", ex)


def evaluate(fit, frames):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from toy_ice import ToyIce
    calc = ToyIce(fit=fit); eE, eF, eS, fr, comp = [], [], [], [], {}
    for f in frames:
        b = f.copy(); b.calc = calc; E, F, S = b.get_potential_energy(), b.get_forces(), b.get_stress()
        z = f.numbers; E += fit["mu_O"] * (z == 8).sum() + fit["mu_H"] * (z == 1).sum()
        de = (E - f.get_potential_energy()) / len(f); eE.append(de); eF.append(((F - f.get_forces()) ** 2).mean()); eS.append(np.abs(S - f.get_stress())[:3].mean() * 160.21766)
        fr.append(np.sqrt((f.get_forces() ** 2).mean())); comp.setdefault(str(f.info.get("config_type", "?")).split("_")[0], []).append(de)
    return dict(frames=len(frames), E_MAE_meV_atom=1e3 * float(np.mean(np.abs(eE))), F_RMSE_eV_A=float(np.sqrt(np.mean(eF))), S_MAE_GPa=float(np.mean(eS)), F_rms_dft=float(np.mean(fr)),
                E_signed_by_comp={k: 1e3 * float(np.mean(v)) for k, v in sorted(comp.items())})


def plot(fit, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from toy_ice import pair_energy
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); r = np.linspace(0.0, fit["rc"], 400)
    for k, (p, col) in enumerate(zip(PAIRS, ("#2a78d6", "#eb6834", "#1baf7a"))):
        e, de = pair_energy(k, r, fit=fit)
        for ax in axs: ax.plot(r, e, color=col, lw=2, label=p); ax.axvline(fit["pairs"][p]["r_lo"], color=col, lw=0.6, ls=":")
    axs[0].set_ylim(-4, 30); axs[1].set_ylim(-3, 3); axs[1].set_xlim(0.6, fit["rc"])
    for ax in axs: ax.axhline(0, color="#8a8a85", lw=0.6); ax.set_xlabel("r (A)"); ax.set_ylabel("phi (eV)"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("toy_fit pair potentials (dotted: shortest distance in the data; below it the quadratic core)", fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
