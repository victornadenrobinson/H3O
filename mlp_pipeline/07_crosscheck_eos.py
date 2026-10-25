"""Cross-check mined DFT runs against the hull-CSV EOS (NVT: density is the control variable).

usage: python 07_crosscheck_eos.py mined_v2.csv data-h3o-convex-hull.csv [--out plots] [--min-steps 100] [--max-pstd 20]
For each mined run on the 5000/7000/9000 K isotherms: rho = mass / V; hull P(rho) is interpolated on that
composition+isotherm (no extrapolation); dP = P_mined - P_hull(rho). Writes crosscheck_eos.csv + 2 figures.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module
from common import HULL_COMPS, comp_label, phase_from_path, out as outp, inp

inv = import_module("01_bin_inventory")
AMU_G, M_H, M_O = 1.66053907, 1.00794, 15.9994
KB_EV, EV_A3_GPA = 8.617333e-5, 160.21766
COL = {"h2o": "#d62728", "h2.25o": "#2a78d6", "h3o": "#2e9b4e", "h5o": "#8e44ad", "h7o": "#e66100", "h": "#7f5539"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6b66", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9})


def hull_curves(path):
    h = pd.read_csv(path, low_memory=False)
    out = {}
    for T in inv.ISOTHERMS:
        for suf, ratio in HULL_COMPS.items():
            c = [f"P {T}K {suf}", f"rho {T}K {suf}"]
            if all(x in h for x in c):
                d = h[c].apply(pd.to_numeric, errors="coerce").dropna().sort_values(c[1])
                if len(d) > 2:
                    out[(T, comp_label(ratio))] = (d[c[1]].values, d[c[0]].values)
    return out


def main(a):
    a.out = a.out or outp("plots")
    os.makedirs(a.out, exist_ok=True)
    df = inv.load_inventory(inp(a.inv))
    df = df[(df.N_steps >= a.min_steps) & (df.P_std_GPa.fillna(0) <= a.max_pstd)].copy()
    print("pressure = OUTCAR external + ideal-gas kinetic term (as eos.dat / hull CSV)")
    df["rho"] = (df.N_H * M_H + df.N_O * M_O) * AMU_G / df.True_V_A3
    curves = hull_curves(a.hull)
    rows = []
    for r in df.itertuples():
        T = min(inv.ISOTHERMS, key=lambda t: abs(t - r.True_T_K))
        if abs(T - r.True_T_K) > inv.T_BIN or (T, r.comp_snap) not in curves:
            continue
        rho, P = curves[(T, r.comp_snap)]
        if not (rho[0] <= r.rho <= rho[-1]):
            continue
        rows.append(dict(Path=r.Path, comp=r.comp_snap, T=T, T_run=r.True_T_K, rho=r.rho, P_mined=r.True_P_GPa,
                         P_hull=float(np.interp(r.rho, rho, P)), N_steps=r.N_steps, P_std=r.P_std_GPa))
    x = pd.DataFrame(rows)
    if x.empty:
        raise SystemExit("no mined runs fall inside the hull density ranges")
    x["phase_path"] = x.Path.map(phase_from_path)
    if a.phase_csv:
        ph = pd.read_csv(a.phase_csv)[["Path", "phase"]]
        x = x.merge(ph, on="Path", how="left")
    x["dP"] = x.P_mined - x.P_hull
    x["dP_pct"] = 100 * x.dP / x.P_hull
    x.to_csv(outp("crosscheck_eos.csv"), index=False)
    s = x.groupby(["comp", "T"]).agg(n=("dP", "size"), med_dP=("dP", "median"), mean_dP=("dP", "mean"),
                                     std_dP=("dP", "std"), med_pct=("dP_pct", "median")).round(1)
    print(f"{len(x)} of {len(df)} quality runs lie on a hull isotherm inside its density range")
    print(s.to_string())
    key = "phase" if "phase" in x else "phase_path"
    print(f"\nH2O at 5000/7000 K split by {key}:")
    h = x[(x.comp == "h2o") & x["T"].isin([5000, 7000])]
    print(h.groupby(["T", key]).dP.agg(n="size", med="median", std="std").round(1).to_string())
    print("\nall compositions by", key, ":")
    print(x.groupby(["comp", key]).dP.agg(n="size", med="median", std="std").round(1).to_string())
    print("\noverall median dP = %.1f GPa, median |dP| = %.1f GPa, median %% = %.1f" % (x.dP.median(), x.dP.abs().median(), x.dP_pct.median()))
    bad = x[x.dP.abs() > 3 * x.dP.abs().median() + 20].sort_values("dP", key=abs, ascending=False)
    print(f"{len(bad)} outliers (|dP| > 3 x median + 20 GPa); worst 15:")
    print(bad.head(15)[["Path", "comp", "T", "rho", "P_mined", "P_hull", "dP"]].to_string(index=False))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, T in zip(axs, inv.ISOTHERMS):
        for c, col in COL.items():
            if (T, c) in curves:
                rho, P = curves[(T, c)]
                ax.plot(P, rho, color=col, lw=1.4, marker="s", ms=4, mec=INK, mew=0.4, label=f"{c} hull", zorder=3)
            d = x[(x["T"] == T) & (x.comp == c)]
            ax.scatter(d.P_mined, d.rho, s=9, color=col, alpha=0.35, linewidths=0, zorder=2)
        ax.set_title(f"{T} K", loc="left", color=INK); ax.set_xlabel("P (GPa)"); ax.grid(color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("rho (g/cm$^3$)")
    axs[0].legend(frameon=False, fontsize=8, title="line+squares: hull CSV\ndots: mined runs", title_fontsize=8)
    fig.savefig(os.path.join(a.out, "fig3_eos_overlay.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, T in zip(axs, inv.ISOTHERMS):
        for c, col in COL.items():
            for ph, mk in (("liquid", "o"), ("solid", "s"), ("superionic", "^")):
                d = x[(x["T"] == T) & (x.comp == c) & (x[key] == ph)]
                ax.scatter(d.P_hull, d.dP, s=12, color=col, alpha=0.55, linewidths=0, marker=mk, label=f"{c} {ph}" if T == 5000 else None)
            d = x[(x["T"] == T) & (x.comp == c) & ~x[key].isin(["liquid", "solid", "superionic"])]
            ax.scatter(d.P_hull, d.dP, s=12, color=col, alpha=0.4, linewidths=0, marker="x")
        ax.axhline(0, color=INK, lw=0.8); ax.set_title(f"{T} K", loc="left", color=INK)
        ax.set_xlabel("hull P at same density (GPa)"); ax.grid(color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("P$_{mined}$ - P$_{hull}$ (GPa)"); axs[0].legend(frameon=False, fontsize=7, markerscale=1.5, ncol=2, title="o liquid  s solid  ^ superionic  x other", title_fontsize=7)
    fig.savefig(os.path.join(a.out, "fig4_eos_residuals.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote crosscheck_eos.csv,", os.path.join(a.out, "fig3_eos_overlay.png"), os.path.join(a.out, "fig4_eos_residuals.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("hull")
    ap.add_argument("--out", default=None); ap.add_argument("--min-steps", type=int, default=100)
    ap.add_argument("--max-pstd", type=float, default=20.0)
    ap.add_argument("--phase-csv", default=None, help="phase.csv from 09_label_phase.py (dynamical phase); else path-based tag")
    main(ap.parse_args())
