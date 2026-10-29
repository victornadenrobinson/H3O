"""Melting line of fcc superionic H2O from the tables of Cheng, Bethkenhagen, Pickard, Hamel, Nat. Phys. 17, 1228 (2021),
data set https://github.com/BingqingCheng/superionic-water (read-only clone, not copied into this repo; commit 4886156 used).
usage: python docs/extract_cheng_melt.py /path/to/superionic-water [--phase fcc]
T_m(P) where mu_liquid = mu_solid, per native pressure of the chemical-potential grid, linear interpolation in T between grid temperatures:
  T_K       classical nuclei, PBE level (mu-mlp-cl + mu-mlp2pbe): the one to compare with our classical-nuclei GRACE runs
  T_K_nqe   with nuclear quantum effects (+ mu-nqe); equals the data set's phase-boundary/Tm-<phase>.dat (checked: within 1 K at 50-200 GPa)
Coverage: the liquid table ends near 240 GPa (no liquid mu above), so the line stops at 223 GPa, 5.16e3 K. It does NOT reach 300-500 GPa.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("--phase", default="fcc", choices=["fcc", "bcc"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"cheng2021_tm_{a.phase}.csv")


def grid(name, col=2):
    d = np.loadtxt(os.path.join(a.repo, "chemical-potential", name))
    P = np.array(sorted(set(np.round(d[:, 0], 6)))); T = np.array(sorted(set(np.round(d[:, 1], 6))))
    G = np.full((len(P), len(T)), np.nan)
    for r in d:
        G[np.argmin(abs(P - r[0])), np.argmin(abs(T - r[1]))] = r[col]
    return P, T, G


def mu(ph, nqe):
    P, T, g = grid(f"mu-mlp-cl-{ph}.dat")
    g = g + grid(f"mu-mlp2pbe-{ph}.dat")[2]
    return P, T, g + grid(f"mu-nqe-{ph}.dat")[2] if nqe else g


def crossing(T, d):
    ok = np.isfinite(d); t, d = T[ok], d[ok]
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if not len(s):
        return np.nan
    j = s[0]
    return t[j] - d[j] * (t[j + 1] - t[j]) / (d[j + 1] - d[j])


P, T, _ = grid(f"mu-mlp-cl-{a.phase}.dat")
lines = {k: mu("liquid", k)[2] - mu(a.phase, k)[2] for k in (False, True)}
rows = [(p, crossing(T, lines[False][i]), crossing(T, lines[True][i])) for i, p in enumerate(P)]
rows = [r for r in rows if np.isfinite(r[1])]
with open(out, "w") as f:
    f.write(f"# {a.phase} superionic H2O melting line (mu_liquid = mu_{a.phase}), Cheng et al. Nat. Phys. 17, 1228 (2021) tables; T_K classical PBE, T_K_nqe with NQE. "
            f"Ends at {rows[-1][0]:.0f} GPa (no liquid table above). See docs/extract_cheng_melt.py.\n")
    f.write("P_GPa,T_K,T_K_nqe\n")
    for r in rows:
        f.write(f"{r[0]:.2f},{r[1]:.0f},{r[2]:.0f}\n")
print(len(rows), "rows,", f"P {rows[0][0]:.0f}-{rows[-1][0]:.0f} GPa ->", out)
