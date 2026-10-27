"""Figures for report/paper.pdf, drawn only from data that is in this repository.

Inputs (report/data/):
  dg_fcc_bcc.csv          Wilson 2013 / Cheng 2021 / this work: G_fcc - G_bcc of H2O (PR #2, docs/data)
  dft_hull_sheet.csv      liquid mixing hull of the DFT sheet (data-h3o-convex-hull.csv, columns 1-15, rows 0-4)
  tdi_summary_values.csv  numbers transcribed from docs/CURRENT_ISSUES.md and HANDOFF.md (TDI raw runs are not in git)

Panels whose data are not in the repository are drawn as labelled placeholders (dashed box); nothing is invented.
Run: python3 report/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(HERE, "data", f)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "red": "#e34948"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "legend.fontsize": 6.8,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def tag(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def grid(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def placeholder(ax, title, need):
    """Dashed grey box: this panel's data are not in the repository."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linestyle((0, (4, 3)))
        s.set_color("#9a9992")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f4f3ef")
    ax.text(0.5, 0.68, "DATA NOT\nIN REPO", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=MUTED, transform=ax.transAxes)
    ax.text(0.5, 0.42, need, ha="center", va="center", fontsize=5.6, color=MUTED, transform=ax.transAxes, linespacing=1.3)
    ax.set_title(title, fontsize=7, color=MUTED, pad=4)


V = pd.read_csv(D("tdi_summary_values.csv"), comment="#")


# ---------------------------------------------------------------- Fig. 1: ghost-proton insertion, 500 GPa 6000 K n=1
def fig1():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.7, 1, 1, 1], "wspace": 0.3})
    ax = axs[0]
    parts = [("$<10^{-2}$", 150), ("$10^{-2}$\n–0.1", 320), ("0.1\n–1", 2480)]
    base, xs = 0.0, np.arange(len(parts) + 1)
    for i, (lab, v) in enumerate(parts):
        ax.bar(i, v, bottom=base, width=0.62, color=C["blue"], edgecolor="#fcfcfb", lw=1.2)
        ax.text(i, base + v + 60, f"+{v}", ha="center", va="bottom", fontsize=6.8, color=INK)
        base += v
    tot = V[V.kind == "integral_total_TDI"].iloc[0]
    ax.bar(3, tot.mu_meV, width=0.62, color=C["orange"], edgecolor="#fcfcfb", lw=1.2)
    ax.errorbar(3, tot.mu_meV, tot.err_meV, color=INK, lw=0.8, capsize=2)
    ax.text(3, tot.mu_meV + 150, f"{tot.mu_meV:.0f}", ha="center", fontsize=6.8)
    w = V[V.kind == "widom_check"].iloc[0]
    ax.errorbar(3.42, w.mu_meV, w.err_meV, fmt="D", color=INK, mfc="white", ms=4, lw=0.8, capsize=2)
    ax.text(3.5, w.mu_meV - 900, "Widom\ncheck", fontsize=6, ha="left", va="top", color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in parts] + ["total\nTDI"], fontsize=6.5)
    ax.set_xlabel("$\\lambda$ range", fontsize=7)
    ax.set_xlim(-0.6, 4.1)
    ax.set_ylim(0, 3800)
    ax.set_ylabel("reversible work of one proton (meV)")
    grid(ax)
    ax.set_title("500 GPa, 6000 K, $n=1$", fontsize=7.5)
    tag(ax, "a")
    for a, t, need in [(axs[1], "integrand", "per-lambda\n<U_full−U_host>\n(runs/tdi_long)"),
                       (axs[2], "running integral", "tdi/collect.py output\n(first vs second half)"),
                       (axs[3], "$\\mu_H$ breakdown", "tdi/mu_breakdown.py\n(reference terms)")]:
        placeholder(a, t, need)
    for a, s in zip(axs[1:], "bcd"):
        tag(a, s)
    fig.savefig(os.path.join(OUT, "fig1_insertion.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_insertion.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2: mu_H - G_H(pure) vs x
def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    states = [(500, 5500), (500, 6000), (400, 5000)]
    for ax, (P, T) in zip(axs, states):
        m = (V.P_GPa == P) & (V.T_K == T)
        ins, cr, fit = V[m & (V.kind == "insertion_pcorr")], V[m & (V.kind == "crystal_direct")], V[m & (V.kind == "linfit_extrap")]
        ax.axhline(0, color=INK, lw=0.7)
        if len(ins):
            ax.plot(ins.x, ins.mu_meV, "s", color=C["aqua"], ms=5, mec="#fcfcfb", mew=0.8, label="ghost insertion, $n=1,2,4$\n(pressure-corrected)")
        if len(fit):
            ax.plot(fit.x, fit.mu_meV, "o", mfc="white", mec=C["blue"], mew=1.2, ms=6, label="linear fit of $n=1,2,4$,\nextrapolated to $x=0.25$")
        if len(cr):
            ax.plot(cr.x, cr.mu_meV, "o", color=C["blue"], ms=6, mec="#fcfcfb", mew=0.8, label="crystal TDI, H$_{2.25}$O")
        ax.set_title(f"{P} GPa, {T:,} K")
        ax.set_xlim(-0.01, 0.28)
        ax.set_xlabel("added H per O, $x$")
        grid(ax)
        if len(ins) == 0:
            ax.text(0.04, 0.06, "insertion values for this state\nnot in repo notes", transform=ax.transAxes,
                    fontsize=6.2, color=MUTED, va="bottom")
        ax.legend(loc="upper left", fontsize=6)
    axs[0].set_ylabel("$\\mu_H - G_H^{pure}$ (meV per added H)")
    axs[0].set_ylim(-800, 560)
    for a, s in zip(axs, "abc"):
        tag(a, s)
    fig.savefig(os.path.join(OUT, "fig2_solubility.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_solubility.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: P / T / van 't Hoff (no data in repo)
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw={"wspace": 0.18})
    for a, s, t in zip(axs, "abc", ["$\\mu_0$ vs pressure", "$\\mu_0$ vs temperature", "van 't Hoff plot"]):
        placeholder(a, t, "dilute-limit mu_H on a (P, T) grid:\ninsertion runs, 300–500 GPa\n× 5000–7000 K\n(tdi/insertion_model.py)")
        tag(a, s)
    fig.savefig(os.path.join(OUT, "fig3_vanthoff.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_vanthoff.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: liquid hull (DFT sheet) + melting placeholders
def fig4():
    h = pd.read_csv(D("dft_hull_sheet.csv"))
    x = h.x.values
    n = np.where(x < 1, x / (1 - x), np.inf)          # sheet x = n/(1+n), n = H per O - 2
    xa = np.where(np.isfinite(n), n / (3 + n), 1.0)   # added-H atom fraction n/(3+n)
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.28)
    cols = {9000: C["blue"], 7000: C["orange"], 5000: C["aqua"]}
    sets = [("A", "set A (400 GPa assumed*)"), ("B", "set B (350 GPa)")]
    for j, (k, title) in enumerate(sets):
        ax = fig.add_subplot(gs[0, j])
        for T, col in [(9000, "9000K"), (7000, "7000K"), (5000, "5000K_liq")]:
            y, e = h[f"{k}_{col}"].values, h[f"{k}_{col}_err"].values
            ok = np.isfinite(y)
            if k == "B" and T == 5000:          # 4th row is a placeholder 0 in the sheet
                ok &= np.array([True, True, True, False, False])
            ax.errorbar(xa[ok], y[ok], e[ok], fmt="o-", color=cols[T], ms=4, lw=1, capsize=1.5, mec="#fcfcfb", mew=0.6,
                        label=f"{T:,} K" + (" (liquid)" if T == 5000 else ""))
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(title, fontsize=6.8)
        ax.set_xlabel("added-H atom fraction $n/(3+n)$")
        grid(ax)
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel("$\\Delta G_{mix}$ (meV/atom)")
        ax.set_ylim(-215, 60)
        ax.set_yticks([-150, -100, -50, 0, 50])
        ax.legend(loc="lower center", fontsize=5.8, ncol=3, columnspacing=0.8, handlelength=1.4)
        tag(ax, "ab"[j])
    placeholder(fig.add_subplot(gs[0, 2]), "GRACE MLP hull", "liquid TDI hull, 300–500 GPa\n(young_tdi_liquid/_hull;\nsummarize.py _hull table)")
    tag(fig.axes[-1], "c")
    for j, P in enumerate([300, 400, 500]):
        ax = fig.add_subplot(gs[1, j])
        placeholder(ax, f"$G_{{liquid}}-G_{{fcc}}$ at {P} GPa", "H$_2$O and H$_{2.25}$O solid–liquid TDI\n(young_tdi_melt; not summarised)")
        tag(ax, "def"[j])
    fig.savefig(os.path.join(OUT, "fig4_melting.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig4_melting.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Extended Data Fig. 2: fcc - bcc benchmark
def edfig2():
    d = pd.read_csv(D("dg_fcc_bcc.csv"), comment="#")
    d = d[d.quantity == "G"].copy()
    d["fcc_bcc"] = np.where(d.sign == "bcc-fcc", -d.dG_meV_atom, d.dG_meV_atom)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, T in zip(axs, [3000, 5000]):
        s = d[d.T_K == T]
        w = s[s.source.str.startswith("Wilson")]
        ch = s[s.source.str.startswith("Cheng")]
        ax.errorbar(ch.P_GPa, ch.fcc_bcc, ch.err_meV_atom, fmt="-o", color=C["violet"], ms=3.5, lw=1, capsize=1.5,
                    label="Cheng 2021 (NNP + FEP to PBE)")
        ax.errorbar(w.P_GPa, w.fcc_bcc, w.err_meV_atom, fmt="D", color=C["orange"], ms=5, capsize=2, mec="#fcfcfb", mew=0.6,
                    label="Wilson 2013 (DFT TI)")
        t = s[s.source.str.startswith("this work")]
        big = t[t.source.str.contains("108/128") & ~t.source.str.contains("H2.25O")]
        ax.errorbar(big.P_GPa, big.fcc_bcc, big.err_meV_atom, fmt="o", color=C["blue"], ms=6, capsize=2, mec="#fcfcfb", mew=0.8,
                    label="this work, GRACE ft_m2, 108/128 O")
        small = t[t.source.str.contains("32/54")]
        ax.errorbar(small.P_GPa, small.fcc_bcc, small.err_meV_atom, fmt="o", mfc="white", mec=C["blue"], mew=1.2, color=C["blue"], ms=6,
                    capsize=2, label="this work, 32/54 O (still drifting)")
        hh = t[t.source.str.contains("H2.25O")]
        ax.errorbar(hh.P_GPa, hh.fcc_bcc, hh.err_meV_atom, fmt="^", color=C["aqua"], ms=6, capsize=2, mec="#fcfcfb", mew=0.8,
                    label="this work, H$_{2.25}$O (not H$_2$O)")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlim(80, 560)
        ax.set_title(f"{T:,} K")
        ax.set_xlabel("pressure (GPa)")
        grid(ax)
    axs[0].set_ylabel("$G_{fcc}-G_{bcc}$ (meV/atom)")
    axs[0].legend(loc="lower left", fontsize=6)
    for a, s in zip(axs, "ab"):
        tag(a, s)
    fig.savefig(os.path.join(OUT, "edfig2_fcc_bcc.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "edfig2_fcc_bcc.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); edfig2()
    print("wrote", sorted(os.listdir(OUT)))
