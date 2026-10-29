"""H in superionic fcc H2O by three methods, per (P, T): (top) the chemical potential of H relative to pure H, (bottom) the formation free energy (hull) of H(2+n)O.
  insertion TDI (ghost protons, lambda path): dilute limit mu0 (n -> 0) and the regular-solution fit through insertion n = 1, 2, 4 and the crystals (docs/data/insertion_regsol_fit.csv):
      average cost per added H  mu0 + s n (dotted), next-H cost (differential) mu0 + 2 s n (dashed); hull g = (mu0 n + s n^2) / (3 + n)
  crystal TDI (whole H(2+n)O crystals): AVERAGE cost per added H over 0 -> n (open triangles, huang_hull.CRYSTAL); hull points n * avg / (3 + n)
  pwidom (proxy Widom / deletion + BAR, no lambda path): DIFFERENTIAL cost of one more H in an H2O or H(2+n)O host (squares; open = unconverged, x = host melted);
      docs/data/pwidom_scan_*.csv and, on Young, runs/pwidom*/scan.jsonl (run pwidom_scan.py --rescore <dir> first: robust pressure correction); hull: trapezoid integral of the differential points from n = 0 (dilute mu0 from the
      insertion TDI where the pwidom H2O host is unconverged), so  g(n) = int_0^n mu dn' / (3 + n); drawn only with >= 2 pwidom compositions above n = 0
      (pwidom_scan.py --hpers), since a straight line from mu0 to one point can be far off (400 GPa 5000 K: mu(n) must rise steeply at small n, see the crystal average)
All in meV, relative to H2O (superionic fcc) + n H (pure fluid H) at the same P, T; pwidom pressure-corrected, the fits as in report.py. Classical nuclei, GRACE ft_m2.
Solubility = where the differential cost crosses 0 = the hull minimum.
usage: python3 mlp_pipeline/tdi/fig_methods.py [-o docs/figures/mu_hull_methods.png] [--points "400 5000;500 5000;..."]
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from huang_hull import CRYSTAL

C_INS, C_CRY, C_PW = "#52514d", "#1c5cab", "#d55181"


def rows(f):
    return [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]


def pwidom_points():
    pts = {}
    for f in sorted(glob.glob("docs/data/pwidom_scan_*.csv")):
        for r in rows(f):
            pts[(float(r["P_GPa"]), float(r["T_K"]), round(float(r["n"]), 4))] = dict(dg=float(r["dg_meV"]), err=float(r["err_meV"]), flag=r["flag"])
    for f in sorted(glob.glob("runs/pwidom*/scan.jsonl"), key=os.path.getmtime):   # Young: newer scans override
        for l in open(f):
            r = json.loads(l); n = r.get("n"); h = r["host"]
            if n is None: n = 0.25 if "2.25" in h else (0.0 if h == "H2O" else float(h[1:-1]) - 2)
            fl = r["flags"]; flag = "melted" if "MELTED" in fl else ("unconverged" if ("UNCONVERGED" in fl or "few effective" in fl) else "ok")
            pts[(float(r["P"]), float(r["T"]), round(float(n), 4))] = dict(dg=r["dg"], err=r["err"], flag=flag)
    return pts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="docs/figures/mu_hull_methods.png"); ap.add_argument("--points", default=None); a = ap.parse_args()
    fit = {(float(r["P_GPa"]), float(r["T_K"])): {k: float(v) for k, v in r.items()} for r in rows("docs/data/insertion_regsol_fit.csv")}
    pw = pwidom_points()
    allpt = sorted({(P, T) for P, T, _ in pw} | {(float(p), float(t)) for p, t, *_ in CRYSTAL if (float(p), float(t)) in fit or any(k[:2] == (p, t) for k in pw)})
    want = [tuple(map(float, p.split())) for p in a.points.split(";")] if a.points else [pt for pt in allpt if (pt in fit) + any(k[:2] == pt for k in pw) + any((c[0], c[1]) == pt for c in CRYSTAL) >= 2]
    fig, ax = plt.subplots(2, len(want), figsize=(3.6 * len(want), 8.2), squeeze=False, sharex=True)
    n = np.linspace(0, 0.55, 200)
    for j, (P, T) in enumerate(want):
        top, bot = ax[0, j], ax[1, j]; f = fit.get((P, T))
        if f:
            top.errorbar([0], [f["mu0"]], [f["mu0_err"]], fmt="o", color=C_INS, ms=7, capsize=3, label="insertion TDI: dilute mu0 (n -> 0)", zorder=5)
            top.plot(n, f["mu0"] + f["s"] * n, ":", color=C_INS, lw=1.5, label="insertion fit: average per added H")
            top.plot(n, f["mu0"] + 2 * f["s"] * n, "--", color=C_INS, lw=1.5, label="insertion fit: next-H cost (differential)")
            bot.plot(n, (f["mu0"] * n + f["s"] * n ** 2) / (3 + n), "-", color=C_INS, lw=1.5, label="insertion fit (regular solution)")
            ns = -f["mu0"] / (2 * f["s"]); bot.plot(ns, -f["mu0"] ** 2 / (4 * f["s"]) / (3 + ns), "o", color=C_INS, ms=5)
        for p, t, nn, avg, err in CRYSTAL:
            if (p, t) == (P, T):
                top.errorbar(nn, avg, err, fmt="^", mfc="white", mec=C_CRY, color=C_CRY, ms=8, capsize=3, label="crystal TDI: average per added H (0 -> n)")
                bot.errorbar(nn, nn * avg / (3 + nn), nn * err / (3 + nn), fmt="^", mfc="white", mec=C_CRY, color=C_CRY, ms=8, capsize=3, label="crystal TDI (whole crystal)")
        mine = sorted((k[2], v) for k, v in pw.items() if k[:2] == (P, T))
        for nn, v in mine:
            mk = "x" if v["flag"] == "melted" else "s"
            top.errorbar(nn, v["dg"], v["err"], fmt=mk, ms=8, capsize=3, color=C_PW, mfc=C_PW if v["flag"] == "ok" else "white", mec=C_PW,
                         label={"ok": "pwidom: next-H cost (differential)", "unconverged": "pwidom: unconverged (Widom/deletion disagree)", "melted": "pwidom: host melted"}[v["flag"]])
        good = [(nn, v["dg"]) for nn, v in mine if v["flag"] == "ok"]
        if f and good and not any(nn == 0 for nn, _ in good): good = [(0.0, f["mu0"])] + good   # dilute end from the insertion TDI
        good = sorted(good)
        allp = sorted([(nn, v["dg"]) for nn, v in mine if v["flag"] != "melted"])
        if f and allp and not any(nn == 0 for nn, _ in allp): allp = [(0.0, f["mu0"])] + allp
        if len([g for g in allp if g[0] > 0]) >= 2 and len([g for g in good if g[0] > 0]) < len([g for g in allp if g[0] > 0]):   # incl. unconverged points: dashed, open
            x = np.array([g[0] for g in allp]); y = np.array([g[1] for g in allp]); cum = np.r_[0, np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)]
            bot.plot(x, cum / (3 + x), "s--", color=C_PW, mfc="white", ms=6, lw=1.2, alpha=0.8, label="pwidom integral incl. unconverged points")
            top.plot(x[1:], cum[1:] / x[1:], "--", color=C_PW, lw=1.0, alpha=0.6, label="pwidom-implied AVERAGE 0 -> n (compare with the blue triangles)")
        if len([g for g in good if g[0] > 0]) >= 2:   # at least two pwidom compositions above 0 (an n-scan): one point would only give a straight line from mu0, which can be badly off
            x = np.array([g[0] for g in good]); y = np.array([g[1] for g in good]); cum = np.r_[0, np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)]
            bot.plot(x, cum / (3 + x), "s-", color=C_PW, ms=7, lw=1.5, label="pwidom: integral of the next-H cost (from insertion mu0)")
            top.plot(x[1:], cum[1:] / x[1:], "-", color=C_PW, lw=1.2, alpha=0.7, label="pwidom-implied AVERAGE 0 -> n (compare with the blue triangles)")
        top.axhline(0, color="k", lw=0.8); bot.axhline(0, color="k", lw=0.8)
        sec = top.secondary_xaxis("top", functions=(lambda x: x * 108, lambda k: k / 108)); sec.set_xticks([0, 6, 12, 18, 27, 36, 54]); sec.tick_params(labelsize=7)
        if j == 0: sec.set_xlabel("extra H per 108-O cell", fontsize=8)
        top.set(title=f"{P:.0f} GPa, {T:.0f} K", ylim=(-1200, 1000)); top.title.set_position((0.5, 1.12)); bot.set(xlabel="n = extra H per O  (H(2+n)O)", ylim=(-25, 90))
        top.grid(alpha=0.3); bot.grid(alpha=0.3)
        if j == 0:
            top.set_ylabel("mu_H - G_H(pure H)  (meV per H)"); bot.set_ylabel("formation free energy  (meV/atom)\nH2O + n H(fluid) -> H(2+n)O")
    for row in ax:
        h, l = [], []
        for x in row:
            for hh, ll in zip(*x.get_legend_handles_labels()):
                if ll not in l: h.append(hh); l.append(ll)
        row[-1].legend(h, l, fontsize=6.5, loc="upper right")
    fig.suptitle("H in superionic fcc H2O, GRACE (classical nuclei): insertion TDI (grey), crystal TDI (blue, averages), pwidom (red, next-H cost). "
                 "Solubility = where the next-H cost crosses 0 = hull minimum.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(a.o, dpi=130); print("wrote", a.o, "points:", want)


if __name__ == "__main__":
    main()
