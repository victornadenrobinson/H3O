#!/usr/bin/env python3
"""Plot the static H3O formation enthalpy vs pressure: MLP runs (h3o_static.py csv files) against Huang (static SI curve and with ZPE); right panel: dH_f vs composition for each run.
   python mlp_pipeline/plot_h3o_dH.py runs/h3o_static_e20 runs/h3o_static_F200 [-o runs/h3o_dH.png]"""
import argparse, csv, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("-o", default="runs/h3o_dH.png"); a = ap.parse_args()
rd = lambda f: [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140); col = plt.cm.tab10
for k, d in enumerate(a.dirs):
    rows = rd(os.path.join(d, "h3o_static.csv")); lab = os.path.basename(d.rstrip("/")).replace("h3o_static_", "")
    h3 = sorted((float(r["P_GPa"]), float(r["dH_meV_atom"])) for r in rows if abs(float(r["H_per_O"]) - 3) < 1e-6)
    if h3: ax[0].plot(*zip(*h3), "o-", color=col(k), label=f"MLP {lab} (H3O)")
    for x, ls in ((2.75, "--"), (2.5, ":")):
        s = sorted((float(r["P_GPa"]), float(r["dH_meV_atom"])) for r in rows if abs(float(r["H_per_O"]) - x) < 1e-6)
        if s: ax[0].plot(*zip(*s), ls, color=col(k), alpha=0.6, label=f"MLP {lab} (H{x:g}O)")
    for P in sorted({float(r["P_GPa"]) for r in rows}):
        s = sorted((float(r["X"]), float(r["dH_meV_atom"])) for r in rows if float(r["P_GPa"]) == P)
        ax[1].plot(*zip(*s), "o-", color=col(k), alpha=0.25 + 0.75 * (P - 300) / 500, ms=3, label=f"{lab} {P:.0f} GPa" if P in (300, 500, 800) else None)
for f, lab, st in (("docs/data/huang_h3o_dH_static_SI.csv", "Huang static (SI)", "k-"), ("docs/data/huang_h3o_dH.csv", "Huang incl. ZPE", "k--")):
    r = rd(f); ax[0].plot([float(x["P_GPa"]) for x in r], [float(x["dH_meV_atom"]) for x in r], st, label=lab)
ax[0].axhline(0, color="gray", lw=0.6); ax[0].set(xlabel="P (GPa)", ylabel="dH_f vs H2O + H (meV/atom)", title="H3O formation enthalpy (static)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
ax[1].axhline(0, color="gray", lw=0.6); ax[1].set(xlabel="X = extra H / (extra H + 1) (H3O = 0.5)", ylabel="dH_f (meV/atom)", title="dH_f vs composition (darker = higher P)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(a.o); print("wrote", a.o)
