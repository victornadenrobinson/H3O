#!/usr/bin/env python3
"""Plot Perlmutter pwidom runs in the style of docs/figures/mu_hull_methods.png: top = dg(n) = mu_H(host n) - mu_H(pure H) per added H [meV] (next-H cost), bottom = formation free energy of H(2+n)O
[meV/atom] = trapezoid integral of dg from n = 0, divided by (3 + n). One column per run (P, T). Solubility = where dg crosses 0 = hull minimum.
Markers: filled = no flags, open = UNCONVERGED / LOW KISH, x = HOST MELTED (excluded from the hull line). Old Young m2 scan points (docs/data/pwidom_scan_*.csv) are overlaid in grey for the same (P, T) with --old.
  python3 mlp_pipeline/plot_pwidom_hull.py runs/pwidom_new_mnf3e3_500GPa_5000K [more run dirs] [--labels "mn_f_lr3e-3_e5" ...] [--old docs/data/pwidom_scan_2026-10-29.csv] [--err-scale 4] -o runs/pwidom_hull.png
Needs matplotlib and the scan code (mlp_pipeline/tdi/pwidom_scan.py) like pwidom_hull.py."""
import argparse, csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pwidom_hull as ph

C_NEW, C_OLD = "#d55181", "#6b6b6b"


def old_points(files, P, T):
    pts = {}
    for f in files:
        for r in csv.DictReader(l for l in open(f) if not l.startswith("#")):
            try:
                if abs(float(r["P_GPa"]) - P) < 1 and abs(float(r["T_K"]) - T) < 1: pts[round(float(r["n"]), 4)] = dict(n=float(r["n"]), dg=float(r["dg_meV"]), err=float(r["err_meV"]), flags=[] if r["flag"] == "ok" else [r["flag"].upper()])
            except (KeyError, ValueError): continue
    return sorted(pts.values(), key=lambda x: x["n"])


def draw(ax, pts, color, label, es, size=7):
    for kind in ("ok", "open", "melt"):
        sel = [p for p in pts if (("MELTED" in p["flags"]) if kind == "melt" else (bool(p["flags"]) if kind == "open" else not p["flags"]))]
        if kind == "open": sel = [p for p in sel if "MELTED" not in p["flags"]]
        if not sel: continue
        x = [p["n"] for p in sel]; y = [p["dg"] for p in sel]; e = [p["err"] * es for p in sel]
        if kind == "melt": ax.errorbar(x, y, e, fmt="x", color=color, ms=size, capsize=2, label=f"{label} host melted")
        else: ax.errorbar(x, y, e, fmt="s", mfc=color if kind == "ok" else "white", mec=color, color=color, ms=size, capsize=2, label=f"{label} " + ("next-H cost" if kind == "ok" else "unconverged"))


def hull_line(ax, pts, color, ls):
    use = [(p["n"], p["dg"], p["err"]) for p in pts if "MELTED" not in p["flags"]]
    if not use: return
    r = ph.hull(use, 1.0, 0); ax.plot(r["n"], r["per_atom"], ls, color=color, marker="s", ms=5, mfc="white" if any(p["flags"] for p in pts if "MELTED" not in p["flags"]) else color, lw=1.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("runs", nargs="+"); ap.add_argument("--labels", nargs="*", default=[]); ap.add_argument("--old", nargs="*", default=None)
    ap.add_argument("--err-scale", type=float, default=1.0); ap.add_argument("-o", default="runs/pwidom_hull.png"); ap.add_argument("--ylim", type=float, default=0); a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    olds = a.old if a.old is not None else sorted(glob.glob("docs/data/pwidom_scan_*.csv")); nc = len(a.runs)
    fig, ax = plt.subplots(2, nc, figsize=(5.6 * nc, 8.2), squeeze=False, constrained_layout=True)
    for c, run in enumerate(a.runs):
        m, rows = ph.load_run(run); P, T = float(m["P"]), float(m["T"]); lab = a.labels[c] if c < len(a.labels) else os.path.basename(os.path.normpath(run)).replace("pwidom_", ""); top, bot = ax[0, c], ax[1, c]
        old = old_points(olds, P, T)
        if old: draw(top, old, C_OLD, "old m2 (Young)", a.err_scale, 5); hull_line(bot, old, C_OLD, ":")
        draw(top, rows, C_NEW, "new", a.err_scale); hull_line(bot, rows, C_NEW, "-")
        top.axhline(0, color="k", lw=0.8); bot.axhline(0, color="k", lw=0.8); top.set_title(f"{P:g} GPa, {T:g} K\n{lab}", fontsize=10); top.set_ylabel("mu_H(host) - mu_H(pure H)  (meV per H)"); bot.set_ylabel("formation free energy  H2O + n H(fluid) -> H(2+n)O  (meV/atom)")
        bot.set_xlabel("n = extra H per O  (H(2+n)O)"); top.grid(alpha=0.25); bot.grid(alpha=0.25); top.legend(fontsize=7, loc="upper left")
        sec = top.secondary_xaxis("top", functions=(lambda x: x * 108, lambda x: x / 108)); sec.set_xlabel("extra H per 108-O cell", fontsize=8)
        if a.ylim: top.set_ylim(-a.ylim, a.ylim)
        use = [(r["n"], r["dg"], r["err"]) for r in rows if "MELTED" not in r["flags"]]
        if use:
            b = ph.hull(use, a.err_scale, 0)["base"]; bot.text(0.03, 0.94, ("hull min n* = %.3f, depth %.1f meV/atom" % (b["n_star"], b["depth_atom"])) if b["bracketed"] else ("hull NOT bracketed: depth >= %.1f meV/atom (n <= %.3f)" % (b["depth_atom"], b["n_star"])), transform=bot.transAxes, fontsize=8, color=C_NEW)
    os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True); fig.savefig(a.o, dpi=130); print("figure ->", a.o)


if __name__ == "__main__":
    main()
