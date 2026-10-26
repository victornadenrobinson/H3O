"""Plot G_fcc - G_bcc (meV/atom) versus pressure from every source in data/dg_fcc_bcc.csv, one panel per temperature.
usage: python plot_dg_vs_P.py [csv] [--temps 2000 3000 5000] [--tol 250]
Rows within +-tol K of a requested temperature go into that panel (the actual T is written in the legend). Negative = fcc stable.
Our own MLP-TDI results go into the same CSV with source 'this work ...' and are drawn with filled markers.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dg_fcc_bcc.csv"))
ap.add_argument("--quantity", default="G", help="G (default), PV, U, -TS or E_tot")
ap.add_argument("--temps", type=float, nargs="*", default=[2000, 3000, 5000]); ap.add_argument("--tol", type=float, default=250)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "dg_fcc_bcc_vs_P.png"))
a = ap.parse_args()
df = pd.read_csv(a.csv, comment="#")
if "quantity" in df:
    df = df[df.quantity == a.quantity]
if df.empty:
    raise SystemExit(f"{a.csv} has no data rows yet: fill it from the papers (see the header comments), then rerun")
df["dG"] = np.where(df.sign.str.strip() == "bcc-fcc", -df.dG_meV_atom, df.dG_meV_atom)   # always plotted as fcc - bcc
fig, axs = plt.subplots(1, len(a.temps), figsize=(4.6 * len(a.temps), 4), sharey=True, squeeze=False)
mk = ["o", "s", "^", "D", "v", "P", "X"]; srcs = list(dict.fromkeys(df.source))
for ax, T in zip(axs[0], a.temps):
    d = df[(df.T_K - T).abs() <= a.tol]
    for k, s in enumerate(srcs):
        e = d[d.source == s].sort_values("P_GPa")
        if e.empty: continue
        mine = s.lower().startswith("this work")
        ax.errorbar(e.P_GPa, e.dG, e.err_meV_atom.fillna(0), fmt=mk[k % 7] + "-", ms=5, capsize=2, mfc="auto" if mine else "none", label=f"{s} ({e.T_K.min():.0f}-{e.T_K.max():.0f} K)")
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25, which="both"); ax.set_title(f"T ~ {T:.0f} K"); ax.set_xlabel("P (GPa)"); ax.legend(fontsize=6)
axs[0][0].set_ylabel("G_fcc - G_bcc (meV/atom)   <0: fcc stable")
fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=160); print("wrote", a.out)
