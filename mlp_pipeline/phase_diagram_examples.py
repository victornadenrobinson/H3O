#!/usr/bin/env python3
"""SCHEMATIC T-x phase diagrams of H2O + excess H at one pressure, computed (not drawn) from simple model free energies by the lower-convex-hull construction
that the real diagram will use on TDI / PWidom free energies. Numbers are illustrative only (not results).
   python mlp_pipeline/phase_diagram_examples.py -o docs/figures/phase_diagram_scenarios.png
Composition: c = fraction of the H2O -> H axis (c = 0 H2O, x_H = 2/3; c = 1 pure H), x_H = 2/3 + c/3. Free energies per atom in meV, relative to the pure liquids (linear terms do not change the hull).
Models: liquid  G_L(c) = kT [c ln c + (1-c) ln(1-c)] + W c (1-c)          (ideal + regular-solution mixing; W < 0 favours mixing, W > 2 kT_c gives a miscibility gap)
        SI-H2O  G_S = dS (T - Tm) at c = 0 (line compound), or a solid-solution curve with its own large W_s (limited H uptake into the superionic H sublattice)
        H3O     G_C = G_L(c_C) + dS_C (T - T_C) at c_C = 0.25 (T_C = melting point of the compound against the liquid of its own composition)
Pure H is fluid over the whole T range shown, so there is no solid on the H side."""
import argparse
import numpy as np

kB = 0.08617   # meV/K
TM, DS = 6000.0, 1.2 * kB          # SI-H2O O-sublattice melting T and entropy of melting per atom (illustrative)


def mix(c, T, W):
    c = np.clip(c, 1e-9, 1 - 1e-9)
    return kB * T * (c * np.log(c) + (1 - c) * np.log(1 - c)) + W * c * (1 - c)


def phases(sc, T):
    """list of (name, c array, G array) at temperature T for scenario sc"""
    cl = np.linspace(0, 1, 401)
    W = {"a": -100.0, "b": -100.0, "c": -100.0, "d": -100.0, "e": 1000.0}[sc]
    out = [("L", cl, mix(cl, T, W))]
    if sc == "d":                                             # SI-H2O dissolves a little H: solid-solution curve (W_s 400 meV + 650 meV per unit c: H-rich solid unfavourable)
        cs = np.linspace(0, 1, 401); out.append(("SI-H2O(H)", cs, DS * (T - TM) + mix(cs, T, 400.0) + 650.0 * cs))   # + linear penalty: no solid on the H side
    else:
        out.append(("SI-H2O", np.array([0.0]), np.array([DS * (T - TM)])))
    if sc in ("b", "c"):                                      # H3O compound: congruent (b) or hidden melting point -> peritectic (c)
        TC, DSC = (5300.0, 1.0 * kB) if sc == "b" else (4650.0, 1.3 * kB)
        out.append(("H3O", np.array([0.25]), np.array([mix(0.25, T, W) + DSC * (T - TC)])))
    return out


def lower_hull(pts):
    """pts: list of (c, G, phase, index). Monotone-chain lower hull, sorted by c then G."""
    pts = sorted(pts, key=lambda p: (p[0], p[1])); pts = [p for i, p in enumerate(pts) if i == 0 or p[0] != pts[i - 1][0]]; h = []   # same c: keep the lowest G only
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0: h.pop()
        h.append(p)
    return h


def labels_at(sc, T, cq):
    ph = phases(sc, T); pts = [(c, g, n, i) for n, cc, gg in ph for i, (c, g) in enumerate(zip(cc, gg))]
    h = lower_hull(pts); lab = []
    for c in cq:
        k = max(0, min(len(h) - 2, np.searchsorted([p[0] for p in h], c) - 1)); a, b = h[k], h[k + 1]
        if a[2] == b[2] and abs(a[3] - b[3]) == 1: lab.append(a[2])
        elif a[2] == b[2] == "L": lab.append("L1 + L2")
        else: lab.append(" + ".join(sorted({a[2], b[2]}, key=lambda s: ["SI-H2O", "SI-H2O(H)", "H3O", "L"].index(s))))
    return lab, h, ph


REGION_COL = {"L": "#dbe9fb", "SI-H2O + L": "#fbe1d6", "SI-H2O(H) + L": "#fbe1d6", "SI-H2O(H)": "#f3c3ad", "H3O + L": "#d3f0e4",
              "SI-H2O + H3O": "#fcebc6", "L1 + L2": "#e6e1f8", "SI-H2O + L1 + L2": "#e6e1f8"}
TITLES = {"a": "(a) miscible fluid, no compound:\n    liquidus falls from Tm toward the H side",
          "b": "(b) stable H3O, congruent melting:\n    two eutectics",
          "c": "(c) H3O melts incongruently (peritectic):\n    H3O -> SI-H2O + H-rich fluid",
          "d": "(d) SI-H2O takes up some H:\n    solid solution + liquidus",
          "e": "(e) fluid miscibility gap:\n    monotectic"}


def hulls_figure(out, Ts=(6500.0, 5500.0, 4300.0, 3000.0)):
    """rows = scenarios (a)-(e), columns = temperatures: G(c) of every phase and the lower convex hull; straight hull pieces = two-phase coexistence (common tangent)"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#888", "xtick.color": "#555", "ytick.color": "#555"})
    col = {"L": "#2a78d6", "SI-H2O": "#eb6834", "SI-H2O(H)": "#eb6834", "H3O": "#008300"}
    xt = [3 * n / (n + 1) - 2 for n in (2, 3, 4, 6)] + [1.0]; xtl = ["H2O", "H3O", "H4O", "H6O", "H"]
    cq = np.linspace(0.0015, 0.9985, 600)
    fig, axs = plt.subplots(5, len(Ts), figsize=(4.1 * len(Ts), 17), constrained_layout=True)
    for r, sc in enumerate("abcde"):
        for k, T in enumerate(Ts):
            ax = axs[r, k]; lab, h, ph = labels_at(sc, T, cq)
            for n, cc, gg in ph:
                if len(cc) > 1:
                    m = gg < 400; ax.plot(cc[m], gg[m], color=col[n], lw=1.8, label="liquid" if n == "L" else n + " (solid solution)")
                else: ax.plot(cc, gg, "o", color=col[n], ms=8, mec="white", mew=1.4, label=n, zorder=5)
            hc = np.array([p[0] for p in h]); hg = np.array([p[1] for p in h])
            for i in range(len(h) - 1):                       # two-phase pieces of the hull: thick grey; one-phase pieces lie on a curve
                a_, b_ = h[i], h[i + 1]
                if not (a_[2] == b_[2] and abs(a_[3] - b_[3]) == 1): ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color="#222", lw=2.4, alpha=0.55, solid_capstyle="butt", zorder=4)
            ax.plot(hc, hg, color="#222", lw=0.8, ls="--", zorder=4)
            seg = []; prev = None                              # phase sequence along x, e.g. "SI-H2O + L | L"
            for c, l in zip(cq, lab):
                if l != prev: seg.append([l, c]); prev = l
            txt = "   |   ".join(f"{l}" for l, _ in seg); bnd = ", ".join(f"{2 / 3 + c / 3:.3f}" for _, c in seg[1:])
            ax.set_title(f"{T:.0f} K:  {txt}" + (f"\n    boundaries at x_H = {bnd}" if bnd else ""), loc="left", fontsize=8, color="#222")
            lo = min(min(hg), min(g.min() for _, _, g in ph)) - 25; hi = max(30, max((g.max() for n, c, g in ph if len(c) == 1), default=0) + 30)
            ax.set_ylim(lo, hi); ax.set_xlim(-0.03, 1.02); ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.grid(True, color="#eee", lw=0.6)
            if k == 0: ax.set_ylabel(TITLES[sc].split(":")[0] + "\nG per atom (meV, rel.)", fontsize=9)
            if r == 0 and k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower left")
            if r == 2 and k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower left")
            if r == 3 and k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Convex hulls behind the schematic phase diagrams (fixed P; rows = scenarios a-e, columns = T). Thick grey = two-phase tie line (common tangent); x_H = H/(H+O) atom fraction.\n"
                 "Illustrative model free energies, NOT results.", fontsize=10, color="#222")
    fig.savefig(out, dpi=130); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="phase_diagram_scenarios.png"); ap.add_argument("--hulls", help="also write the G(x) hull panels (scenarios x temperatures) to this file"); a = ap.parse_args()
    if a.hulls: hulls_figure(a.hulls)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888", "xtick.color": "#555", "ytick.color": "#555"})
    cq = np.linspace(0.0015, 0.9985, 500); Tq = np.linspace(2000, 7000, 260)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4), constrained_layout=True)
    xt = [3 * n / (n + 1) - 2 for n in (2, 3, 4, 6)] + [1.0]; xtl = ["H2O", "H3O", "H4O", "H6O", "pure H"]   # c = 3 x_H - 2, x_H = n / (n + 1) for HnO
    for ax, sc in zip(axs.flat, "abcde"):
        grid = [labels_at(sc, T, cq)[0] for T in Tq]; names = sorted({l for row in grid for l in row}, key=lambda s: list(REGION_COL).index(s) if s in REGION_COL else 99)
        idx = np.array([[names.index(l) for l in row] for row in grid])
        from matplotlib.colors import ListedColormap
        ax.pcolormesh(cq, Tq, idx, cmap=ListedColormap([REGION_COL.get(n, "#eeeeee") for n in names]), vmin=-0.5, vmax=len(names) - 0.5, shading="auto", rasterized=True)
        for k in range(len(names)):
            ax.contour(cq, Tq, (idx == k).astype(float), levels=[0.5], colors="#333", linewidths=1.0)
        for k, n in enumerate(names):                         # label at the centroid of the largest part of each region
            m = idx == k
            if m.sum() < 60 or n == "SI-H2O(H)": continue
            ii, jj = np.nonzero(m); ax.text(cq[int(np.median(jj))], Tq[int(np.median(ii))], n, ha="center", va="center", fontsize=8.5, color="#222")
        if sc == "d": ax.annotate("SI-H2O(H)\n(solid solution)", xy=(0.03, 3600), xytext=(0.12, 2500), fontsize=8.5, color="#222", arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
        if sc in ("b", "c"):                                  # the compound is a vertical line where it is on the hull
            on = [T for T in Tq if any(p[2] == "H3O" for p in labels_at(sc, T, cq[:2])[1])]
            if on: ax.plot([0.25, 0.25], [min(on), max(on)], color="#008300", lw=2.4, solid_capstyle="butt"); ax.text(0.255, min(on) + 120, "H3O", color="#006400", fontsize=8.5)
        ax.axhline(TM, color="#999", lw=0.6, ls=":"); ax.text(0.99, TM + 60, "Tm(SI-H2O)", ha="right", fontsize=7.5, color="#777")
        ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_xlim(0, 1); ax.set_ylim(Tq[0], Tq[-1]); ax.set_ylabel("T (K)"); ax.set_xlabel("composition (H2O -> H)")
        ax.set_title(TITLES[sc], loc="left", fontsize=9.5, color="#222")
    # (f): the construction itself, scenario (c) just below the peritectic: G(c) of each phase and the lower hull (common tangents)
    ax = axs[1, 2]; T0 = 4300.0; lab, h, ph = labels_at("c", T0, cq)
    col = {"L": "#2a78d6", "SI-H2O": "#eb6834", "H3O": "#008300"}
    for n, cc, gg in ph:
        if len(cc) > 1: ax.plot(cc, gg, color=col[n], lw=2, label="liquid")
        else: ax.plot(cc, gg, "o", color=col[n], ms=9, mec="white", mew=1.5, label=n, zorder=5)
    hc = np.array([p[0] for p in h]); hg = np.array([p[1] for p in h]); ax.plot(hc, hg, color="#222", lw=1.1, ls="--", label="lower convex hull", zorder=4)
    ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_xlim(-0.02, 1.0); ax.set_xlabel("composition (H2O -> H)"); ax.set_ylabel("G per atom (meV, relative)")
    ax.grid(True, color="#eee", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.set_title(f"(f) how it is computed: G(x) at {T0:.0f} K for case (c);\n    straight hull segments = two-phase coexistence", loc="left", fontsize=9.5, color="#222")
    fig.suptitle("Schematic H2O + H phase diagrams at one pressure (illustrative model free energies, NOT results)", fontsize=11, color="#222")
    fig.savefig(a.o, dpi=150); print("wrote", a.o)


if __name__ == "__main__":
    main()
