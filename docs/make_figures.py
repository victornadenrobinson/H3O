"""Figures from the tabulated outputs of 01_bin_inventory.py / 07_crosscheck_eos.py (docs/data/*.csv).
Numbers were transcribed from the Perlmutter terminal output (mined_h3o.csv, lambda_* runs excluded, runs with <100 steps excluded
from the T x P and composition x T tables). Re-draw from the real CSVs with 06_plot_coverage.py / 07_crosscheck_eos.py.
usage: python docs/make_figures.py"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

D = os.path.join(os.path.dirname(__file__), "data"); F = os.path.join(os.path.dirname(__file__), "figures")
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6b66", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"])
COL = {"h2o": "#d62728", "h2.25o": "#2a78d6", "h3o": "#2e9b4e", "h5o": "#8e44ad", "h7o": "#e66100", "h": "#7f5539"}
lab = lambda c: {"h": "H$_2$"}.get(c, "H$_{%s}$O" % c[1:-1])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9})


def heat(ax, M, rows, cols, xl, yl, cbl):
    im = ax.imshow(np.where(M > 0, M, np.nan), cmap=SEQ, norm=LogNorm(vmin=1, vmax=M.max()), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]: ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=7, color="white" if np.log10(M[i, j]) / np.log10(M.max()) > 0.55 else INK)
    return im


tp = pd.read_csv(f"{D}/inventory_T_P.csv").set_index("T_K")
fig, ax = plt.subplots(figsize=(11, 4.2))
im = heat(ax, tp.values, [f"{t:,}" for t in tp.index], [("1600+" if c == "1600" else c) for c in tp.columns], "pressure bin start (GPa; 100 GPa bins)", "T (K, 250 K bins)", "")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("# OUTCAR runs (≥100 steps, no λ runs)"); cb.outline.set_visible(False)
ax.set_title("Mined DFT runs in T–P space (T bins with ≥30 runs)", loc="left", color=INK)
fig.savefig(f"{F}/inventory_T_P.png", dpi=160, bbox_inches="tight"); plt.close(fig)

ct = pd.read_csv(f"{D}/inventory_comp_T.csv").set_index("comp")
fig, ax = plt.subplots(figsize=(9, 4.4))
im = heat(ax, ct.values, [lab(c) if not c.startswith("h0") else c for c in ct.index], [f"{int(c):,}" for c in ct.columns], "T (K)", "composition (snapped H:O ratio)", "")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("# OUTCAR runs"); cb.outline.set_visible(False)
ax.set_title("Mined runs by composition and temperature (blank = none)", loc="left", color=INK)
fig.savefig(f"{F}/inventory_composition_T.png", dpi=160, bbox_inches="tight"); plt.close(fig)

hc = pd.read_csv(f"{D}/hull_coverage.csv")
comps = [c for c in COL if c in set(hc.comp)]; Ts = sorted(hc.T_K.unique())
fig, ax = plt.subplots(figsize=(7.5, 4))
M = np.array([[hc[(hc.comp == c) & (hc.T_K == t)].runs.iloc[0] for t in Ts] for c in comps])
im = heat(ax, M, [lab(c) for c in comps], [f"{t:,} K" for t in Ts], "isotherm of the hull EOS", "", "")
for i, c in enumerate(comps):
    for j, t in enumerate(Ts):
        r = hc[(hc.comp == c) & (hc.T_K == t)].iloc[0]
        ax.text(j, i + 0.32, f"{r.covered}/{r.targets} targets", ha="center", va="center", fontsize=6.5, color="white" if np.log10(M[i, j]) / np.log10(M.max()) > 0.55 else INK)
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("# runs within ±25 GPa and ±250 K of a target"); cb.outline.set_visible(False)
ax.set_title(f"Hull EOS target coverage: {hc.covered.sum()} of {hc.targets.sum()} targets have data", loc="left", color=INK)
fig.savefig(f"{F}/hull_target_coverage.png", dpi=160, bbox_inches="tight"); plt.close(fig)

ec = pd.read_csv(f"{D}/eos_crosscheck.csv")
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, t in zip(axs, sorted(ec.T_K.unique())):
    d = ec[ec.T_K == t].set_index("comp").reindex(comps)
    x = np.arange(len(comps))
    ax.axhline(0, color=INK, lw=0.8)
    ax.errorbar(x, d.med_dP_GPa, yerr=d.std_dP_GPa, fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=3, zorder=2)
    ax.scatter(x, d.med_dP_GPa, c=[COL[c] for c in comps], s=46, edgecolors=INK, linewidths=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([lab(c) for c in comps]); ax.set_title(f"{t:,} K", loc="left", color=INK); ax.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("P$_{mined}$ − P$_{hull}$ at equal density (GPa)\nmedian ± std")
fig.suptitle("Mined DFT vs hull EOS at equal density (λ runs excluded; pressure = external + N k$_B$T/V)", x=0.01, ha="left", fontsize=10)
fig.savefig(f"{F}/eos_crosscheck.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(os.listdir(F)))
