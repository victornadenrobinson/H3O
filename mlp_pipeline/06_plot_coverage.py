"""Plot data coverage. usage: python 06_plot_coverage.py mined_v2.csv [hull.csv] [--metric steps|outcars] [--min-steps 100] [--out plots]

fig1_pt_by_composition.png : P-T cells (25 GPa x 250 K) per composition, colour = total MD steps (or # OUTCARs);
                             hull EOS targets overlaid (o = has >=1 run within window, x = none).
fig2_coverage_map.png      : composition x T and composition x P heat maps with counts printed in the cells.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from importlib import import_module
from common import out as outp, inp

inv = import_module("01_bin_inventory")

SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6b66", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"])
ORDER = ["h2o", "h2.25o", "h2.5o", "h2.75o", "h3o", "h4o", "h5o", "h6o", "h7o", "h"]
PRETTY = lambda c: {"h": "H$_2$"}.get(c, "H$_{%s}$O" % c[1:-1])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                     "font.size": 9})


def metric_col(df, metric):
    return df.N_steps if metric == "steps" else pd.Series(1, index=df.index)


def fig1(g, tg, metric, out):
    comps = [c for c in ORDER if (g.comp_snap == c).any() or (tg is not None and (tg.comp == c).any())]
    n = len(comps); nc = min(4, n); nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc + 1.2, 3.1 * nr), sharex=True, sharey=True, squeeze=False)
    g = g.assign(m=metric_col(g, metric))
    cell = g.groupby(["comp_snap", "T_bin", "P_bin"]).m.sum().reset_index()
    vmax = max(cell.m.max(), 10)
    norm = LogNorm(vmin=1, vmax=vmax)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, c in zip(axs.flat, comps):
        ax.set_visible(True)
        d = cell[cell.comp_snap == c]
        ax.scatter(d.T_bin, d.P_bin, c=d.m, cmap=SEQ, norm=norm, marker="s", s=26, linewidths=0.4, edgecolors=SURF, zorder=2)
        if tg is not None:
            t = tg[tg.comp == c]
            cov = t[t.n_runs > 0]; unc = t[t.n_runs == 0]
            ax.scatter(cov["T"], cov.P_GPa, facecolors="none", edgecolors=INK, marker="o", s=34, linewidths=0.9, zorder=3)
            ax.scatter(unc["T"], unc.P_GPa, c="#c2410c", marker="x", s=30, linewidths=1.1, zorder=3)
        nr_ = int((g.comp_snap == c).sum()); ns = int(g[g.comp_snap == c].N_steps.sum()) if "N_steps" in g else 0
        ax.set_title(f"{PRETTY(c)}   {nr_:,} OUTCARs · {ns:,} steps", fontsize=9, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("T (K)")
    for ax in axs[:, 0]:
        ax.set_ylabel("P (GPa)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=axs, shrink=0.8, pad=0.015)
    cb.set_label("total MD steps per cell" if metric == "steps" else "# OUTCARs per cell")
    cb.outline.set_visible(False)
    if tg is not None:
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], marker="o", mfc="none", mec=INK, ls="", label="hull EOS target with data"),
                            Line2D([], [], marker="x", color="#c2410c", ls="", label="hull EOS target, no data")],
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.45, -0.01))
    fig.suptitle("Mined DFT data in P-T space (cells: 25 GPa × 250 K)", x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(out, "fig1_pt_by_composition.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2(g, metric, out):
    comps = [c for c in ORDER if (g.comp_snap == c).any()]
    g = g.assign(m=metric_col(g, metric), Pc=np.clip(np.floor(g.True_P_GPa / 100) * 100, 0, 1500).astype(int))
    tb = g.T_bin.value_counts(); tcols = sorted(tb[tb >= 30].index)
    fig, axs = plt.subplots(1, 2, figsize=(15, 0.55 * len(comps) + 2.2), gridspec_kw=dict(width_ratios=[len(tcols), 16]))
    for ax, key, cols, xl in ((axs[0], "T_bin", tcols, "T (K, 250 K bins)"), (axs[1], "Pc", list(range(0, 1600, 100)), "P (GPa, 100 GPa bins; 1500 = 1500+)")):
        M = g.pivot_table(index="comp_snap", columns=key, values="m", aggfunc="sum", fill_value=0).reindex(index=comps, columns=cols, fill_value=0)
        N = g.pivot_table(index="comp_snap", columns=key, values="Path", aggfunc="count", fill_value=0).reindex(index=comps, columns=cols, fill_value=0)
        im = ax.imshow(np.where(M.values > 0, M.values, np.nan), cmap=SEQ, norm=LogNorm(vmin=1, vmax=max(M.values.max(), 10)), aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=0 if len(cols) < 20 else 60)
        ax.set_yticks(range(len(comps))); ax.set_yticklabels([PRETTY(c) for c in comps])
        ax.set_xlabel(xl)
        for i in range(len(comps)):
            for j in range(len(cols)):
                if N.values[i, j]:
                    v = M.values[i, j]
                    dark = np.log10(max(v, 1)) / np.log10(max(M.values.max(), 10)) > 0.55
                    ax.text(j, i, f"{N.values[i, j]:,}", ha="center", va="center", fontsize=7, color="white" if dark else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.012)
    cb.set_label("total MD steps" if metric == "steps" else "# OUTCARs"); cb.outline.set_visible(False)
    fig.suptitle("Where the data is: composition × T and composition × P (colour = total MD steps, number = # OUTCARs; blank = none)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(out, "fig2_coverage_map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("hull", nargs="?")
    ap.add_argument("--metric", choices=["steps", "outcars"], default="steps")
    ap.add_argument("--min-steps", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or outp("plots")
    os.makedirs(a.out, exist_ok=True)
    df = inv.load_inventory(inp(a.inv))
    if "N_steps" not in df:
        df["N_steps"] = 1
    g = df[df.N_steps >= a.min_steps].copy()
    tg = None
    if a.hull:
        tg = inv.hull_targets(a.hull)
        tg["n_runs"] = [len(g[(g.comp_snap == r.comp) & ((g.True_T_K - r.T).abs() <= inv.T_BIN) & ((g.True_P_GPa - r.P_GPa).abs() <= inv.P_BIN)]) for r in tg.itertuples()]
    fig1(g, tg, a.metric, a.out); fig2(g, a.metric, a.out)
    print("wrote", os.path.join(a.out, "fig1_pt_by_composition.png"), os.path.join(a.out, "fig2_coverage_map.png"))
