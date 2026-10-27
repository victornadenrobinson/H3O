"""1D short-range test of an MLP: isolated dimers H-H, O-H, O-O in a large box, E(r) - E(r_far) for r = 0.1 ... 3 A.
usage (repo root, GPU or CPU):  ~/grace-venv/bin/python mlp_pipeline/tdi/pair_scan.py grace-ft:runs/grace/ft_m2 [-o runs/checks/pair_scan] [--device cuda]
Checks: (1) inside the dimer minimum the energy must rise monotonically as r shrinks (no second well, no turnover), (2) at r <~ 0.3 A it should approach the
universal ZBL screened nuclear repulsion (shown for reference; at larger r chemical bonding makes the true curve lower than ZBL, so ZBL is NOT a bound there).
An isolated dimer is far from the training data of a dense-phase model, so a bad curve here says the model has no built-in repulsion, not that bulk results are wrong:
the in-host probe (short_range.py) is the relevant test for the free energies. Reference bond lengths: H2 0.74 A, OH 0.97 A, O2 1.21 A."""
import argparse, os, sys
import numpy as np
from ase import Atoms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("-o", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--box", type=float, default=20.0)
a = ap.parse_args()
calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
r = np.concatenate([np.arange(0.10, 0.60, 0.025), np.arange(0.60, 3.01, 0.05)])
Z = {"H": 1, "O": 8}


def zbl(z1, z2, x):
    """universal ZBL screened Coulomb repulsion (eV), x in A"""
    au = 0.46850 / (z1 ** 0.23 + z2 ** 0.23); t = x / au
    phi = 0.18175 * np.exp(-3.19980 * t) + 0.50986 * np.exp(-0.94229 * t) + 0.28022 * np.exp(-0.40290 * t) + 0.02817 * np.exp(-0.20162 * t)
    return 14.399645 * z1 * z2 / x * phi


curves = {}
for pair in ("HH", "OH", "OO"):
    E = []
    for x in list(r) + [6.0]:
        at = Atoms(pair, positions=[[0, 0, 0], [x, 0, 0]], cell=[a.box] * 3, pbc=True); at.center(); at.calc = calc
        E.append(at.get_potential_energy())
    E = np.array(E); e = E[:-1] - E[-1]; curves[pair] = e
    i0 = int(np.argmin(e)); inner = e[:i0 + 1]
    bad = np.where(np.diff(inner) > 1e-6)[0]   # inside the minimum e must decrease with r
    zb = zbl(Z[pair[0]], Z[pair[1]], r)
    print(f"\n{pair}: minimum {e[i0]:.3f} eV at r = {r[i0]:.3f} A; E(r) - E(6 A) at 0.1/0.2/0.3/0.5 A: " + " ".join(f"{np.interp(q, r, e):.2f}" for q in (0.1, 0.2, 0.3, 0.5))
          + f" eV; ZBL there: " + " ".join(f"{zbl(Z[pair[0]], Z[pair[1]], q):.1f}" for q in (0.1, 0.2, 0.3, 0.5)))
    if len(bad):
        print(f"  NOT MONOTONIC inside the minimum: energy falls again as r shrinks near r = {', '.join(f'{r[j]:.3f}' for j in bad[:5])} A  <- spurious well / no core repulsion")
    else:
        print("  monotonically repulsive inside the minimum")
    if e[0] < 0.25 * zb[0]:
        print(f"  at 0.1 A only {e[0]:.1f} eV vs ZBL {zb[0]:.1f} eV: the core is soft (finite), typical of an MLP without a ZBL term")
if a.o:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for x_, (pair, e) in zip(ax, curves.items()):
        x_.plot(r, e, color="#2a78d6", lw=2, label="MLP")
        x_.plot(r, zbl(Z[pair[0]], Z[pair[1]], r), color="#1f1f1e", lw=1.2, ls="--", label="ZBL (nuclear repulsion only)")
        x_.set_yscale("symlog", linthresh=1.0); x_.axhline(0, color="#9a9a94", lw=0.8); x_.set_xlabel("r (A)"); x_.set_title(f"{pair[0]}-{pair[1]} dimer", loc="left")
        x_.legend(frameon=False)
    ax[0].set_ylabel("E(r) - E(6 A) (eV, symlog)")
    os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.o}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {a.o}.pdf / .png")
