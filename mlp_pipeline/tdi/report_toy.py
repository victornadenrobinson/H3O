"""Report of every toy-model result (pair potentials standing in for GRACE): what the toy tests of the TDI / insertion pipeline found, the dg scan over (P, T),
the quick method checks, and a schematic H2O-H phase diagram drawn from them. Plain python3 + numpy + matplotlib (+ scipy for the fitted pair curves).
usage (repo root):  python3 mlp_pipeline/tdi/report_toy.py [-o docs/report_tdi_toymodel] [--png-dir docs/figures]     -> <o>.pdf and <png-dir>/toy_<figure>.png
Inputs: docs/data/toyfit_t4.json (fitted pair potential), docs/data/toyscan_t4.csv, docs/data/toychecks_t4.csv; the validation numbers (T0-T5, MBAR) are the
results recorded in docs/CURRENT_ISSUES.md 2b (tables below cite them); GRACE dg points: summarize / insertion_model of 2026-10-28 (n -> 0, no P correction basis).
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK, INK2, MUTED, GRID = "#1f1f1e", "#52514d", "#8a897f", "#e4e3dd"
TOY, GRACE = "#2a78d6", "#eb6834"   # categorical slots 1, 2
PRES = {300: "#86b6ef", 400: "#3987e5", 500: "#1c5cab", 600: "#0d366b"}   # ordinal blue ramp (steps 250 / 400 / 550 / 700)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

# GRACE ft_m2 insertion, dilute limit (n -> 0), meV per H (CURRENT_ISSUES / insertion_model, 2026-10-28)
GRACE_DG = [(300, 5000, -324, 29), (400, 5000, -399, 51), (400, 6000, -318, 38), (400, 6500, -306, 41),
            (500, 5000, -591, 39), (500, 5500, -503, 38), (500, 6000, -552, 35), (500, 6500, -548, 42), (500, 7000, -403, 38)]
# validation tests (CURRENT_ISSUES 2b and the fitted-toy compare run): name, result, error, pass
VALID = [
    ("T0 bookkeeping: crystal route = insertion route = exact formulas", "analytic toy", "exact to 1e-13 eV", "", "PASS"),
    ("T3 method check: insertion into pure H - G_H - finite-size term, 3000 K", "analytic toy", "-6.5", "15", "PASS"),
    ("T3 method check, 5000 K", "analytic toy", "-3.4", "28", "PASS"),
    ("T4 insertion n=8 - crystal route [G(H2.25O) - G(H2O) - 8 G_H]/8, 21 lambda", "analytic toy", "+10", "57", "PASS"),
    ("T1 insertion integral - Widom/BAR (H2O host), 3000 / 5000 K", "analytic toy", "+45 / +36", "40 / 29", "PASS"),
    ("T2 insertion integral - Widom/BAR (pure-H host), 3000 / 5000 K", "analytic toy", "-16 / -40", "23 / 28", "PASS"),
    ("T1 insertion integral - Widom/BAR (H2O host), 5000 / 6000 K", "fitted toy", "+4.5 / +5.1", "15.6 / 17.6", "PASS"),
    ("T2 insertion integral - Widom/BAR (pure-H host), 5000 K", "fitted toy", "+12.1", "15.3", "PASS"),
    ("T3 method check, 5000 K (expected -7.9)", "fitted toy", "+1.7 vs expected", "16.8", "PASS"),
    ("T5 dg pipeline - dg Widom/BAR, 5000 / 6000 K", "fitted toy", "+41.9 / +31.1", "24.6 / 31.1", "PASS (1.7 / 1.0 sigma)"),
    ("T5 slope d(dg)/dT 5000-6000 K: pipeline -0.069, Widom -0.058 meV/K", "fitted toy", "-0.011", "0.040", "PASS"),
    ("MBAR over the lambda series - TI (21-lambda insertion runs)", "analytic toy", "-39 .. +7", "12-26", "within errors"),
    ("Gauss-Kronrod - Gauss quadrature, H2.25O crystal (per atom)", "analytic toy", "-14.1", "", "use 21 lambda"),
]


def read_csv(f):
    rows = [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]
    return rows


def page_text(pdf, title, lines):
    fig = plt.figure(figsize=(11, 8.5)); fig.text(0.06, 0.93, title, fontsize=16, color=INK, weight="bold")
    y = 0.87
    for l in lines:
        bold = l.startswith("## ")
        fig.text(0.06, y, l[3:] if bold else l, fontsize=11 if bold else 9.5, color=INK if bold else INK2, weight="bold" if bold else "normal", wrap=True)
        y -= 0.034 if bold else 0.027
    pdf.savefig(fig); plt.close(fig)


def fig_pairs(fit):
    from tdi.toy_ice import pair_energy
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r = np.linspace(0.5, fit["rc"], 400)
    for k, (lab, c) in enumerate((("O-O", "#1c5cab"), ("O-H", GRACE), ("H-H", "#1baf7a"))):
        e, de = pair_energy(k, r, fit=fit)
        ax[0].plot(r, e, color=c, lw=2, label=lab); ax[1].plot(r, -de, color=c, lw=2, label=lab)
    ax[0].set(xlabel="r (A)", ylabel="pair energy (eV)", ylim=(-1, 15), title="fitted pair potentials (toyfit_t4)"); ax[0].legend()
    ax[1].set(xlabel="r (A)", ylabel="pair force -dphi/dr (eV/A)", ylim=(-3, 40), title="pair forces")
    et, ev = fit["errors_train"], fit["errors_valid"]; d = fit["data"]
    fig.text(0.06, 0.005, f"Force-matched to {d['n_train']} DFT frames ({d['pmin']:.0f}-{d['pmax']:.0f} GPa, T >= {d['tmin']:.0f} K; frames stay in scratch, not in this public repo). "
             f"Errors train / validation: energy MAE {et['E_MAE_meV_atom']:.0f} / {ev['E_MAE_meV_atom']:.0f} meV/atom, force RMSE {et['F_RMSE_eV_A']:.2f} / {ev['F_RMSE_eV_A']:.2f} eV/A "
             f"(DFT force rms {et['F_rms_dft']:.1f}), stress MAE {et['S_MAE_GPa']:.1f} / {ev['S_MAE_GPa']:.1f} GPa.", fontsize=8, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); return fig


def fig_validation():
    fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes([0.03, 0.05, 0.94, 0.85]); ax.axis("off")
    fig.text(0.04, 0.93, "Pipeline validation on the toy models (meV per added H unless noted)", fontsize=13, color=INK, weight="bold")
    t = ax.table(cellText=[list(v) for v in VALID], colLabels=["test", "model", "result", "+-", "verdict"], loc="upper left", cellLoc="left", colWidths=[0.52, 0.1, 0.13, 0.09, 0.16])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.55)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID); c.set_text_props(color=INK if i else INK2, weight="bold" if i == 0 else "normal")
    fig.text(0.04, 0.12, "T0 is exact (no statistics): the reference free energies (Einstein crystal, ideal-gas H, ln N!, thermal wavelength, PV, ghost placement) of the crystal\n"
             "route and the insertion route are identical, so no kT-sized bookkeeping offset is possible. T3/T4 are two-route checks of the lambda integral.\n"
             "Widom/BAR references (T1, T2, T5) are themselves only good to ~50 meV in the dense host (4 -> 12 ps moved them by up to 80 meV).", fontsize=8.5, color=INK2)
    return fig


def fig_checks(rows):
    m = {r["run"]: (float(r["mu_meV"]), float(r["err_meV"])) for r in rows}
    hyp = lambda a, b: float(np.hypot(a, b))
    chk = [("path: lambda = x^2 minus x^4\n(expect 0)", m["path"][0] - m["ref"][0], hyp(m["path"][1], m["ref"][1])),
           ("size: 108 O minus 32 O host\n(expect ~0)", m["size"][0] - m["ref"][0], hyp(m["size"][1], m["ref"][1])),
           ("linearity: n = 8 average minus\n(H2O + H2.25O hosts)/2 (expect ~0)", m["ins8"][0] - (m["ref"][0] + m["rev"][0]) / 2, hyp(m["ins8"][1], hyp(m["ref"][1], m["rev"][1]) / 2))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(chk))[::-1]
    ax[0].axvline(0, color=INK, lw=1)
    ax[0].errorbar([c[1] for c in chk], y, xerr=[2 * c[2] for c in chk], fmt="none", ecolor="#86b6ef", elinewidth=6, alpha=0.6, label="2 sigma")
    ax[0].errorbar([c[1] for c in chk], y, xerr=[c[2] for c in chk], fmt="o", color=TOY, ms=8, capsize=0, elinewidth=2, label="1 sigma")
    for yy, c in zip(y, chk):
        ax[0].text(c[1], yy - 0.3, f"{c[1]:+.0f} +- {c[2]:.0f}", ha="center", color=INK2, fontsize=8.5)
    ax[0].set_yticks(y, [c[0] for c in chk]); ax[0].set(xlabel="difference (meV per added H)", ylim=(-0.6, 2.9), title="method checks, toy, 450 GPa 5000 K: all within 1 sigma"); ax[0].legend(loc="upper right", ncols=2)
    xs = [2.0, 2.25]; mu = [m["ref"][0], m["rev"][0]]; er = [m["ref"][1], m["rev"][1]]
    ax[1].errorbar(xs, mu, er, fmt="o", color=TOY, ms=8, capsize=3, label="insertion of 1 H into H(2+a)O")
    ax[1].errorbar([2.125], [m["ins8"][0]], [m["ins8"][1]], fmt="s", color=GRACE, ms=8, capsize=3, label="average over 8 added H (H2O -> H2.25O)")
    s = (m["rev"][0] - m["ref"][0]) / 0.25; se = hyp(m["rev"][1], m["ref"][1]) / 0.25
    ax[1].plot(xs, mu, color=TOY, lw=1, ls="--")
    ax[1].set(xlabel="H per O of the host", ylabel="mu_H (meV, absolute)", title=f"mu_H rises as the ice fills:\n{s:+.0f} +- {se:.0f} meV per (H per O)"); ax[1].legend(loc="upper left")
    fig.tight_layout(); return fig


def fig_scan(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    a = ax[0]; a.axhline(0, color=INK, lw=1)
    for P, c in PRES.items():
        pts = sorted((float(r["T_K"]), float(r["dg_meV"]), float(r["err_meV"])) for r in rows if int(r["P_GPa"]) == P and r["dg_meV"] not in ("melted", ""))
        mel = sorted(float(r["T_K"]) for r in rows if int(r["P_GPa"]) == P and r["dg_meV"] == "melted")
        if pts:
            T, d, e = map(np.array, zip(*pts)); a.errorbar(T, d, e, fmt="o-", color=c, ms=7, lw=2, capsize=2, label=f"toy {P} GPa")
            a.annotate(f"{P}", (T[-1], d[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, va="center", fontsize=8.5)
        if mel:
            a.plot(mel, [640] * len(mel), "x", color=c, ms=8, mew=2)
    for P, T, d, e in GRACE_DG:
        a.errorbar(T + 60, d, e, fmt="D", color=GRACE, mfc="white" if P != 500 else GRACE, ms=6, capsize=2, label=None)
    a.plot([], [], "D", color=GRACE, label="GRACE (filled: 500 GPa; open: 300/400)"); a.plot([], [], "x", color=MUTED, mew=2, label="toy ice melted (top row)")
    a.set(xlabel="T (K)", ylabel="dg = mu_H(ice) - G_H(fluid H)  (meV per H)", ylim=(-700, 700), title="dilute H uptake: below 0 = ice takes up H")
    a.legend(loc="upper right", bbox_to_anchor=(1, 0.92), fontsize=7.5)
    b = ax[1]
    ok = [r for r in rows if r["dg_meV"] not in ("melted", "")]
    T = np.array([float(r["T_K"]) for r in ok]); P = np.array([float(r["P_GPa"]) for r in ok]); d = np.array([float(r["dg_meV"]) for r in ok])
    sc = b.scatter(T, P, c=d, cmap="RdBu_r", vmin=-500, vmax=500, s=170, edgecolor="white", linewidth=1.5, zorder=3)
    for t_, p_, d_ in zip(T, P, d):
        b.text(t_, p_ + 16, f"{d_:+.0f}", ha="center", color=INK2, fontsize=7.5)
    mel = [r for r in rows if r["dg_meV"] == "melted"]
    b.scatter([float(r["T_K"]) for r in mel], [float(r["P_GPa"]) for r in mel], marker="x", s=60, color=MUTED, linewidth=2, zorder=3, label="toy ice melted")
    b.set(xlabel="T (K)", ylabel="P (GPa)", xlim=(1500, 7500), ylim=(260, 650), title="toy dg over (P, T): red = rejects H, blue = takes H, x = ice melted")
    fig.colorbar(sc, ax=b, label="dg (meV per H)")
    fig.tight_layout(); return fig


def fig_phase():
    """schematic H2O - H phase diagram at ~500 GPa (atom fraction of H); shapes from the trends, not computed boundaries"""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x0 = 2 / 3   # H2O
    xa = lambda a: (2 + a) / (3 + a)
    Tm, Te = 7100.0, 700.0   # GRACE / Cheng melting of SI H2O near 500 GPa; eutectic near the (low) melting T of pure H
    # solid solution limit a*(T): ~0 when cold (dg > 0), grows when hot (dg < 0), largest just below melting, then the solidus
    Tx, amax = 6300.0, 1.0   # largest solubility (H3O) just below melting
    Ts = np.linspace(Te, Tx, 200)
    a0 = 0.35   # cold solubility at ~500 GPa: dg ~ -130 meV (toy), ~ -500 (GRACE) at 2000-5000 K, mu_H rising ~330 meV per H/O -> a ~ 0.4
    a_solv = a0 + (amax - a0) * np.clip((Ts - 4500) / (Tx - 4500), 0, 1) ** 1.6
    xs_solv = xa(a_solv)
    Tsol = np.linspace(Tx, Tm, 60); xs_sol = np.interp(Tsol, [Tx, Tm], [xs_solv[-1], x0]) + 0.006 * np.sin(np.linspace(0, np.pi, 60))   # solidus
    xl = np.linspace(x0, 0.995, 300); Tliq = Te + (Tm - Te) * ((0.995 - xl) / (0.995 - x0)) ** 0.25   # liquidus: melting lowered as H is added; stays H-richer than the solidus
    ax.fill_betweenx(np.r_[Ts, Tsol], x0, np.r_[xs_solv, xs_sol], color="#cde2fb", lw=0)
    ax.fill_between(xl, Tliq, 9000, color="#fbe3ec", lw=0)
    ax.plot(xs_solv, Ts, color=TOY, lw=2.2); ax.plot(xs_sol, Tsol, color=TOY, lw=2.2)
    ax.plot(xl, Tliq, color="#d55181", lw=2.2)
    ax.plot([x0, 1.0], [Te, Te], color=INK2, lw=1.2, ls="--")
    ax.plot([x0], [Tm], "o", color=INK, ms=6); ax.annotate(f"SI H2O melts ~{Tm:.0f} K\n(GRACE / Cheng 2021, ~500 GPa)", (x0, Tm), xytext=(12, 8), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.text(0.671, 2200, "superionic\nH(2+a)O\nsolid\nsolution", color="#1c5cab", fontsize=9.5, weight="bold")
    ax.text(0.84, 8200, "fluid (H2O-H fully miscible)", color="#a8335f", fontsize=11, weight="bold", ha="center")
    ax.text(0.86, 3500, "SI H(2+a)O  +  H-rich fluid", color=INK2, fontsize=10.5, ha="center")
    ax.text(0.86, Te - 380, "eutectic ~ melting of pure H (low; shape below it not explored)", color=INK2, fontsize=8, ha="center")
    ax.annotate("finite solubility even when cold at 500 GPa\n(dg < 0 from 2000 K), growing near melting\n(at 400 GPa: ~0 when cold, toy dg > 0)", (xs_solv[170], Ts[170]), xytext=(60, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=INK2), fontsize=8.5, color=INK2)
    ax.annotate("liquidus: dissolved H lowers\nthe melting (if H prefers the liquid)", (xl[150], Tliq[150]), xytext=(30, 25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=INK2), fontsize=8.5, color=INK2)
    ax.annotate("", xy=(0.722, 4300), xytext=(0.697, 4300), arrowprops=dict(arrowstyle="->", color=TOY, lw=1.5))
    ax.text(0.724, 4300, "higher P: more H dissolves (dg/dP < 0)", color=TOY, fontsize=8, va="center")
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ticks = [(0, "H2O"), (0.5, "H2.5O"), (1, "H3O"), (2, "H4O"), (4, "H6O")]
    top.set_xticks([xa(a) for a, _ in ticks], [l for _, l in ticks]); top.tick_params(colors=INK2)
    ax.set(xlim=(x0 - 0.005, 1.0), ylim=(0, 9000), xlabel="atom fraction of H, (2+a)/(3+a)   (right edge: pure H)", ylabel="T (K)",
           title="SCHEMATIC H2O - H phase diagram near 500 GPa (shapes from the dg trends; boundaries not computed)")
    ax.grid(False)
    fig.text(0.06, 0.01, "Built from: toy scan (dg < 0 at 500-600 GPa from 2000 K, > 0 at 300-400 GPa below ~5000 K; most negative near melting), toy checks (mu_H rises ~330 meV per H/O as the ice fills: finite solubility), GRACE dg < 0 at 5000-7000 K, "
             "GRACE liquid hull (miscible). Not computed: solidus / liquidus positions, the partition coefficient (needed for the sign of the melting-point change), the eutectic.", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); return fig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="docs/report_tdi_toymodel"); ap.add_argument("--png-dir", default="docs/figures", help="figures as <dir>/toy_<name>.png"); a = ap.parse_args()
    fit = json.load(open("docs/data/toyfit_t4.json")); scan = read_csv("docs/data/toyscan_t4.csv"); chk = read_csv("docs/data/toychecks_t4.csv")
    figs = {"pairs": fig_pairs(fit), "validation": fig_validation(), "checks": fig_checks(chk), "scan": fig_scan(scan), "phase": fig_phase()}
    ok = [r for r in scan if r["dg_meV"] not in ("melted", "")]
    with PdfPages(a.o + ".pdf") as pdf:
        page_text(pdf, "TDI / insertion pipeline: toy-model report", [
            "## What the toy models are",
            "Pair potentials that run the unchanged production pipeline (run_real.py, young_tdi_par.sh) on CPU: 'analytic toy' (Born-Mayer O-O, Morse O-H,",
            "soft H-H) and 'fitted toy' (spline pairs force-matched to the DFT frames, toyfit_t4). Both give a superionic fcc H2O at 400-500 GPa (O vibrates,",
            "H diffuses). They are cheap enough for independent references (Widom / deletion / BAR) and for long runs, so they test the METHOD, not GRACE.",
            "## Findings",
            "1. Bookkeeping is exact (T0): crystal route = insertion route = analytic formulas to 1e-13 eV. No kT-sized offset is possible.",
            "2. The lambda integral agrees with independent routes: method check (insertion into pure H = G_H) passes at 3000 / 5000 K, insertion n = 8 =",
            "    crystal route, insertion = Widom/BAR within ~1-2 sigma, MBAR = TI. 10 Gauss lambda values are too coarse for crystals (-14 meV/atom).",
            "3. Quick checks (450 GPa, 5000 K): path independence (x^2 vs x^4), finite size (108 vs 32 O), linearity in composition: all within 1 sigma.",
            f"4. Scan of dg = mu_H(ice) - G_H over {len(ok)} solid (P, T) states: PRESSURE decides: +530..+550 at 300 GPa, +100..+215 at 400 GPa (2000-5000 K),",
            "    -86..-135 at 500 GPa and -325..-490 at 600 GPa from 2000 K up: dg = 0 near 460 GPa, nearly T independent below ~5000 K. T matters only close",
            "    to melting (400 GPa: -207 at 6000 K, 500 GPa: -416 at 7000 K): the softening lattice takes up H. Pipeline and Widom/BAR agree on the T trend (T5).",
            "5. GRACE: dg -300..-600 meV at 5000-7000 K (favourable), dg/dP -1.5 meV/GPa (toy: about -3); its T slope is not resolved (+80 +- 26 per 1000 K",
            "    with all points, +16 +- 36 at 500 GPa only). The toy says the sign of the T trend should be negative; GRACE needs lower-T points to test it.",
            "## Caveats",
            "The fitted toy errs by 40 meV/atom and 1.5 eV/A: its dg values are not predictions for water, only the trends and the method checks carry over.",
            "Toy ice melts at a lower T than GRACE ice (400 GPa: between 6000 and 7000 K; 300 GPa: below 4000 K). 300 GPa and T < 4000 K are outside its fit data.",
            "500 GPa 2000/3000 K and 600 GPa were rerun after a ghost-placement fix (dense pure H: the inserted H goes to the roomiest point tried)."])
        for f in figs.values():
            pdf.savefig(f)
    for n, f in figs.items():
        f.savefig(os.path.join(a.png_dir, f"toy_{n}.png"), dpi=130); plt.close(f)
    print("wrote", a.o + ".pdf", "and", ", ".join(os.path.join(a.png_dir, f"toy_{n}.png") for n in figs))


if __name__ == "__main__":
    main()
