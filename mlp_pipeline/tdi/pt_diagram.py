"""P-T diagrams of superionic H2O and H uptake from what we have computed: GRACE (TDI fcc-bcc, TDI melting, free-NVT melting of H2O / H2.25O / H2.5O crystals,
insertion dg) and the toy pair potential (melt check and dg scan). Three figures: GRACE only, toy only, both. Boundaries are interpolated sign changes or
midpoints of solid / melted brackets (free NVT of a perfect crystal can superheat: those are upper bounds), not fitted phase lines.
usage (repo root, python3 + numpy + matplotlib):  python3 mlp_pipeline/tdi/pt_diagram.py [-o docs/figures/pt]   -> <o>_grace.png, <o>_toy.png, <o>_both.png, <o>.pdf
Inputs: docs/data/grace_pt_points.csv, docs/data/toyscan_t4.csv, docs/data/cheng2021_tm_fcc.csv.
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK, INK2, MUTED, GRID = "#1f1f1e", "#52514d", "#8a897f", "#e4e3dd"
TAKES, REJECTS = "#2a78d6", "#e34948"   # dg < 0 (ice takes up H) / dg > 0
COMP = {2.0: INK, 2.25: "#eb6834", 2.5: "#d55181"}   # melting of H2O / H2.25O / H2.5O
FCCBCC = "#6250d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def rows(f):
    return list(csv.DictReader(l for l in open(f) if not l.startswith("#")))


def crossings(pts, start_positive=False):
    """pts: {P: [(T, value)]} -> [(P, T of the first sign change)]; start_positive: only where the lowest-T value is > 0 (bcc -> fcc on heating;
    the near-melting flips of G_fcc - G_bcc are within 1-2 sigma of zero and are not used)"""
    out = []
    for P, s in sorted(pts.items()):
        s = sorted(s)
        if start_positive and s[0][1] <= 0: continue
        for (t0, v0), (t1, v1) in zip(s, s[1:]):
            if v0 * v1 < 0:
                out.append((P, t0 + (t1 - t0) * v0 / (v0 - v1))); break
    return out


def brackets(solid, melted):
    """{P: [T solid]}, {P: [T melted]} -> [(P, midpoint, half width)] where both exist"""
    out = []
    for P in sorted(set(solid) & set(melted)):
        lo, hi = max(solid[P]), min(m for m in melted[P])
        if hi > lo: out.append((P, (lo + hi) / 2, (hi - lo) / 2))
    return out


def grace(ax, label=True, alpha=1.0):
    g = rows("docs/data/grace_pt_points.csv")
    fb = {}
    for r in g:
        if r["kind"] == "fccbcc" and float(r["H_per_O"]) == 2.0: fb.setdefault(float(r["P_GPa"]), []).append((float(r["T_K"]), float(r["value"])))
    for P, s in fb.items():
        for T, v in s:
            ax.plot(P, T, "s" if v > 0 else "o", ms=4, color=FCCBCC, mfc="white" if v > 0 else FCCBCC, alpha=0.55 * alpha, mew=0.8)
    cb = crossings(fb, start_positive=True)
    if cb: ax.plot(*zip(*cb), "-", color=FCCBCC, lw=2, alpha=alpha, label="GRACE bcc | fcc (TDI sign change)" if label else None)
    ch = rows("docs/data/cheng2021_tm_fcc.csv")
    ax.plot([float(r["P_GPa"]) for r in ch], [float(r["T_K"]) for r in ch], "--", color=MUTED, lw=1.5, label="Cheng 2021 fcc melting (PBE, classical)" if label else None)
    tm = sorted((float(r["P_GPa"]), float(r["value"])) for r in g if r["kind"] == "melt_tdi")
    ax.plot(*zip(*tm), "-*", color=COMP[2.0], ms=12, lw=2, alpha=alpha, label="GRACE H2O melting (TDI)" if label else None)
    for h in (2.25, 2.5):
        sol, mel = {}, {}
        for r in g:
            if r["kind"] == "melt_nvt" and float(r["H_per_O"]) == h:
                (sol if r["value"] == "solid" else mel).setdefault(float(r["P_GPa"]), []).append(float(r["T_K"]))
        b = brackets(sol, mel)
        if b:
            P, T, w = map(np.array, zip(*b))
            ax.errorbar(P, T, w, fmt="-D", color=COMP[h], ms=6, lw=2, capsize=3, alpha=alpha, label=f"GRACE H{h:g}O melting (free NVT bracket)" if label else None)
        for P, Ts in sol.items(): ax.plot([P] * len(Ts), Ts, "^", color=COMP[h], mfc="white", ms=5, alpha=0.6 * alpha)
        for P, Ts in mel.items(): ax.plot([P] * len(Ts), Ts, "x", color=COMP[h], ms=6, alpha=0.6 * alpha)
    ins = [(float(r["P_GPa"]), float(r["T_K"]), float(r["value"])) for r in g if r["kind"] == "insert"]
    for P, T, v in ins:
        ax.plot(P + 8, T, "o", ms=9, color=TAKES if v < 0 else REJECTS, mec="white", mew=1.2, alpha=alpha)
    if label: ax.plot([], [], "o", color=TAKES, ms=9, label="GRACE dilute H uptake favourable (dg < 0)")
    ax.text(170, 1900, "bcc", color=FCCBCC, fontsize=11, weight="bold"); ax.text(330, 3200, "fcc", color=FCCBCC, fontsize=11, weight="bold")


def toy(ax, label=True, ls="--"):
    s = rows("docs/data/toyscan_t4.csv")
    sol, mel, dg = {}, {}, {}
    for r in s:
        P, T = float(r["P_GPa"]), float(r["T_K"])
        if r["dg_meV"] == "melted": mel.setdefault(P, []).append(T); continue
        sol.setdefault(P, []).append(T); dg.setdefault(P, []).append((T, float(r["dg_meV"])))
        ax.plot(P - 8, T, "D", ms=7, color=TAKES if float(r["dg_meV"]) < 0 else REJECTS, mec="white", mew=1)
    for P, Ts in mel.items(): ax.plot([P - 8] * len(Ts), Ts, "x", color=INK2, ms=7, mew=1.5)
    b = brackets(sol, mel)
    if b:
        P, T, w = map(np.array, zip(*b)); ax.errorbar(P, T, w, fmt=ls, color=INK2, lw=2, capsize=3, label="toy H2O melting (1-ps melt check bracket)" if label else None)
    byT = {}
    for P, s_ in dg.items():
        for T, v in s_: byT.setdefault(T, []).append((P, v))
    z = sorted([(P, T) for T, P in crossings(byT)] + crossings(dg), key=lambda pt: pt[1])   # sign changes along P at fixed T, and along T at fixed P
    if z: ax.plot(*zip(*z), ls, color=TAKES, lw=2.5, marker="o", label="toy dg = 0 (H uptake favourable to the right / above)" if label else None)
    if label:
        ax.plot([], [], "D", color=REJECTS, ms=7, label="toy dg > 0 (ice rejects H)"); ax.plot([], [], "D", color=TAKES, ms=7, label="toy dg < 0 (ice takes up H)")
        ax.plot([], [], "x", color=INK2, ms=7, mew=1.5, label="toy ice melted")


def finish(ax, title):
    ax.set(xlabel="P (GPa)", ylabel="T (K)", xlim=(80, 640), ylim=(1500, 8000), title=title); ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="docs/figures/pt"); a = ap.parse_args()
    figs = {}
    fig, ax = plt.subplots(figsize=(12, 6.5)); grace(ax); finish(ax, "GRACE (ft_m2): superionic H2O phases, melting and H uptake"); figs["grace"] = fig
    fig, ax = plt.subplots(figsize=(12, 6.5)); toy(ax, ls="-"); finish(ax, "Toy pair potential: melting and the sign of dg = mu_H(ice) - G_H"); figs["toy"] = fig
    fig, ax = plt.subplots(figsize=(12, 6.5)); grace(ax, alpha=0.9); toy(ax); finish(ax, "GRACE (solid lines) and toy (dashed): melting and H uptake"); figs["both"] = fig
    note = ("Free-NVT brackets: highest T solid / lowest T melted in 6-ps runs of a perfect crystal (superheating possible: upper bounds). TDI melting H2O from the\n"
            "liquid-fcc G crossing (400, 500 GPa only). fcc-bcc: squares = bcc lower G, dots = fcc lower G; near melting the two are within 1-2 sigma. GRACE dg points:\n"
            "insertion of 1 H (some without the pressure correction). Toy (fitted pair potential, errors 40 meV/atom): trends only; its ice melts lower than GRACE's.")
    with PdfPages(a.o + ".pdf") as pdf:
        for n, f in figs.items():
            f.text(0.06, 0.005, note, fontsize=6.8, color=INK2); f.tight_layout(rect=(0, 0.06, 1, 1)); pdf.savefig(f); f.savefig(f"{a.o}_{n}.png", dpi=130); plt.close(f)
    print("wrote", a.o + ".pdf", "and", ", ".join(f"{a.o}_{n}.png" for n in figs))


if __name__ == "__main__":
    main()
