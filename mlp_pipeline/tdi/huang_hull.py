"""Huang-style formation free energy of H(2+n)O from H2O + n H(fluid), per atom, vs X = n / (n + 1) (Huang et al.: X = N_H / (N_H + N_H2O), H3O at X = 0.5),
for the superionic fcc solid solution at finite T from our GRACE TDI: regular-solution form from insertion (mu_H - G_H = mu0 + s n, n = added H per O):
   dG(n) per formula unit = mu0 n + s n^2 / 2,  per atom / (3 + n);  minimum at n* = -mu0 / s (the solubility), dG(n*) = -mu0^2 / (2 s).
Direct crystal TDI points (H2.25O, H2.5O; summarize mixing rows, melted / mismatched ones left out) per atom = n * (average per added H) / (3 + n).
Compared with Huang's DFT enthalpy of the ordered H3O phase INCLUDING zero-point energy (their figure, read off; our numbers are classical, no ZPE). Different objects: theirs is an ordered crystal at 0 K, ours a
disordered superionic solution at 4000-7000 K (G, including entropy); an ordered H3O phase below our curve would be the stable one.
usage: python3 mlp_pipeline/tdi/huang_hull.py [-o docs/figures/huang_style_hull.png]
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# direct fcc crystal TDI, average dG per added H (meV), not melted, same dt / grid (summarize mixing table 2026-10-28)
CRYSTAL = [(400, 4000, 0.25, 276.1, 19.2), (400, 5000, 0.25, 223.7, 26.2), (400, 5000, 0.25, 252.0, 38.1), (400, 5000, 0.25, 198.3, 22.5),
           (500, 4000, 0.25, 217.8, 18.3), (500, 4000, 0.5, 507.1, 10.7), (500, 5000, 0.25, 41.1, 32.9), (500, 5000, 0.25, 107.9, 20.0), (500, 5500, 0.25, 141.1, 45.1),
           (500, 6000, 0.25, 56.1, 27.1), (500, 6000, 0.25, 33.0, 30.6), (500, 6000, 0.25, 40.7, 22.1),
           (600, 4000, 0.25, 80.9, 20.9), (600, 4000, 0.5, 451.1, 15.1), (600, 5000, 0.25, 39.3, 21.7), (600, 6000, 0.25, -36.0, 25.4)]
INK2, GRID = "#52514d", "#e4e3dd"


def rows(f):
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="docs/figures/huang_style_hull.png"); a = ap.parse_args()
    fit = rows("docs/data/insertion_regsol_fit.csv"); hu = rows("docs/data/huang_h3o_dH.csv")
    Ts = sorted({r["T_K"] for r in fit} | {c[1] for c in CRYSTAL}); cm = plt.get_cmap("plasma"); col = lambda T: cm(0.05 + 0.8 * (T - 4000) / 3000)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
    n = np.linspace(0, 1.0, 200); X = n / (n + 1)
    for axi, P in zip(ax[:3], (400, 500, 600)):
        for r in [r for r in fit if r["P_GPa"] == P]:
            g = (r["mu0"] * n + r["s"] * n ** 2 / 2) / (3 + n); c = col(r["T_K"]); sol = n <= 0.5
            axi.plot(X[sol], g[sol], "-", color=c, lw=2, label=f"{r['T_K']:.0f} K (insertion, regular solution)")
            axi.plot(X[~sol], g[~sol], ":", color=c, lw=1.5)
            ns = -r["mu0"] / r["s"]; axi.plot(ns / (ns + 1), -r["mu0"] ** 2 / (2 * r["s"]) / (3 + ns), "o", color=c, mec="k", ms=7)
        for (p, T, nn, d, e) in CRYSTAL:
            if p == P:
                axi.errorbar(nn / (nn + 1), nn * d / (3 + nn), nn * e / (3 + nn), fmt="s", color=col(T), mec="k", ms=6, capsize=2)
        h = [x for x in hu if x["P_GPa"] == P]
        if h: axi.plot(0.5, h[0]["dH_meV_atom"], "*", color="#2a78d6", ms=15, mec="k", label="Huang: ordered H3O, DFT dH incl. ZPE (0 K)")
        axi.axhline(0, color="k", lw=0.8); axi.set(xlim=(0, 0.55), ylim=(-80, 120), xlabel="X = N_H / (N_H + N_H2O)   (H3O at 0.5)", title=f"{P} GPa")
        axi.grid(color=GRID); axi.legend(fontsize=6.5, loc="upper left")
        sec = axi.secondary_xaxis("top", functions=(lambda x: 2 + x / np.maximum(1 - x, 1e-9), lambda h: (h - 2) / (h - 1)))
        sec.set_xticks([2, 2.1, 2.25, 2.5, 3]); sec.set_xlabel("H per O", fontsize=8)
    ax[0].set_ylabel("formation free energy per atom (meV/atom)\nH2O + n H(fluid) -> H(2+n)O")
    b = ax[3]
    for T in sorted({r["T_K"] for r in fit}):
        s = sorted((r["P_GPa"], -r["mu0"] ** 2 / (2 * r["s"]) / (3 - r["mu0"] / r["s"])) for r in fit if r["T_K"] == T)
        b.plot(*zip(*s), "o-", color=col(T), label=f"our minimum, {T:.0f} K (superionic solid solution, G)")
    b.plot([x["P_GPa"] for x in hu], [x["dH_meV_atom"] for x in hu], "*-", color="#2a78d6", ms=10, label="Huang: H3O, DFT dH incl. ZPE (0 K)")
    b.axhline(0, color="k", lw=0.8); b.set(xlabel="P (GPa)", ylabel="meV/atom", title="our minimum vs Huang H3O"); b.grid(color=GRID); b.legend(fontsize=6.5)
    fig.suptitle("Huang-style hull at finite T: H2O + n H -> H(2+n)O. Lines: regular-solution form from insertion (solid to H2.5O, dotted beyond: extrapolated); dots: minimum = solubility; "
                 "squares: direct crystal TDI (H2.25O, H2.5O). Huang: ordered H3O at 0 K WITH zero-point energy (ours: classical nuclei) - a different object.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.o, dpi=130); print("wrote", a.o)
    for r in fit:
        ns = -r["mu0"] / r["s"]; print(f"{r['P_GPa']:.0f} GPa {r['T_K']:.0f} K: solubility n* = {ns:.3f} H per O (H{2 + ns:.2f}O, X = {ns / (1 + ns):.3f}), minimum {-r['mu0'] ** 2 / (2 * r['s']) / (3 + ns):6.1f} meV/atom")


if __name__ == "__main__":
    main()
