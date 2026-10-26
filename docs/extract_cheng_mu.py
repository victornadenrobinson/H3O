"""Derive mu_bcc - mu_fcc (eV per H2O molecule) from the tables of Cheng, Bethkenhagen, Pickard, Hamel, Nat. Phys. 17, 1228 (2021),
data set https://github.com/BingqingCheng/superionic-water (directory chemical-potential; read-only clone, not copied into this repo).
usage: python docs/extract_cheng_mu.py /path/to/superionic-water [--T 2000 3000 4000 5000]
Per README of that data set: mu_PBE(+NQE) = mu-mlp-cl + mu-mlp2pbe + mu-nqe. Columns written (eV/molecule; per atom = /3):
  dmu_mlp   classical nuclei, MLP level      dmu_pbe   + PBE free-energy-perturbation correction (this is what compares to Wilson et al.)
  err_fep   statistical error of the correction only (the TI error is not tabulated)      dmu_pbe_nqe   + nuclear quantum effects
Interpolation: bilinear in (ln P, T) on the native grid (80 pressures 10-800 GPa, 60 temperatures 300-5500 K); NaN where any corner is NaN.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("--T", type=float, nargs="+", default=[2000, 3000, 4000, 5000])
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cheng2021_dmu_bcc_fcc.csv"))
a = ap.parse_args()


def grid(name, col=2):
    d = np.loadtxt(os.path.join(a.repo, "chemical-potential", name))
    P = np.array(sorted(set(np.round(d[:, 0], 6)))); T = np.array(sorted(set(np.round(d[:, 1], 6))))
    G = np.full((len(P), len(T)), np.nan)
    for r in d:
        G[np.argmin(abs(P - r[0])), np.argmin(abs(T - r[1]))] = r[col]
    return P, T, G


def val(P, T, G, p, t):
    lp = np.log(P); i = np.searchsorted(lp, np.log(p)) - 1; j = np.searchsorted(T, t) - 1
    if i < 0 or j < 0 or i + 1 >= len(P) or j + 1 >= len(T):
        return np.nan
    x = (np.log(p) - lp[i]) / (lp[i + 1] - lp[i]); y = (t - T[j]) / (T[j + 1] - T[j]); c = G[i:i + 2, j:j + 2]
    return np.nan if np.isnan(c).any() else (1 - x) * (1 - y) * c[0, 0] + x * (1 - y) * c[1, 0] + (1 - x) * y * c[0, 1] + x * y * c[1, 1]


Q = {ph: dict(cl=grid(f"mu-mlp-cl-{ph}.dat"), corr=grid(f"mu-mlp2pbe-{ph}.dat", 2), err=grid(f"mu-mlp2pbe-{ph}.dat", 3), nqe=grid(f"mu-nqe-{ph}.dat")) for ph in ("bcc", "fcc")}
Pn = Q["bcc"]["cl"][0]
rows = []
for T in a.T:
    for P in Pn[(Pn >= 100) & (Pn <= 800)][::2]:
        g = {k: [val(*Q[ph][k], P, T) for ph in ("bcc", "fcc")] for k in ("cl", "corr", "err", "nqe")}
        dcl, dc, dn, e = g["cl"][0] - g["cl"][1], g["corr"][0] - g["corr"][1], g["nqe"][0] - g["nqe"][1], float(np.hypot(*g["err"]))
        if np.isnan(dcl + dc):
            continue
        rows.append((T, P, dcl, dcl + dc, e, dcl + dc + dn))
with open(a.out, "w") as f:
    f.write("# mu_bcc - mu_fcc in eV per H2O molecule, derived from Cheng et al. Nat. Phys. 2021 data (github.com/BingqingCheng/superionic-water); positive = fcc more stable. See extract_cheng_mu.py.\n")
    f.write("T_K,P_GPa,dmu_mlp_eV_mol,dmu_pbe_eV_mol,err_fep_eV_mol,dmu_pbe_nqe_eV_mol\n")
    for r in rows:
        f.write(f"{r[0]:.0f},{r[1]:.2f},{r[2]:.5f},{r[3]:.5f},{r[4]:.5f},{r[5]:.5f}\n")
print(len(rows), "rows ->", a.out)
