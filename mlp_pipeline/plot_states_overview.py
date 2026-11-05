#!/usr/bin/env python3
"""Overview plots of every PWidom state in runs/pw_states (replicates pooled per host):
  figure 1  one column per (P, T): top dg(n) = mu_H(host) - mu_H(pure H) per added H; bottom: formation free energy per atom of H(2+n)O with the hull line. Open markers = flagged unconverged / low Kish, x = host MELTED (excluded from the hull).
  figure 2  dg versus T at fixed n, one panel per pressure: does dg fall toward zero with T, and where does the host melt (x)? --extrapolate adds a straight-line fit of the NON-melted points per n to higher T.
   python3 mlp_pipeline/plot_states_overview.py [--root runs/pw_states] [--extra "label=dir1,dir2" ...] [-o runs/states_overview] [--extrapolate]
Statistical errors only; a state built from a T-mixed EOS is a starting-point state (see mix_eos.py): check P_host in morning_review.py."""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdi"))
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/pw_states"); ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("-o", default="runs/states_overview"); ap.add_argument("--extrapolate", action="store_true"); a = ap.parse_args()


def load_state(dirs):
    import pwidom_hull as ph
    runs = []
    for d in dirs:
        if os.path.isfile(os.path.join(d, "manifest.json")):
            try: runs.append(ph.load_run(d)[1])
            except (SystemExit, Exception): pass
    return (ph.pool_rows(runs) if len(runs) > 1 else runs[0]) if runs else []


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, pwidom_hull as ph
    states = {}
    for sd in sorted(glob.glob(os.path.join(a.root, "*GPa_*K"))):
        m = re.search(r"(\d+)GPa_(\d+)K", os.path.basename(sd))
        rows = load_state(sorted(glob.glob(sd + "/xh*_s??")))
        if m and rows: states[(int(m.group(1)), int(m.group(2)))] = (os.path.basename(sd), rows)
    extra = {}
    for e in a.extra:
        lab, _, ds = e.partition("="); rows = load_state(ds.split(","))
        if rows: extra[lab] = rows
    cols = [(f"{P} GPa, {T} K", rows) for (P, T), (_, rows) in sorted(states.items())] + list(extra.items())
    if not cols: sys.exit("no finished states found")
    fig, ax = plt.subplots(2, len(cols), figsize=(3.9 * len(cols), 7.4), squeeze=False, constrained_layout=True)
    for c, (lab, rows) in enumerate(cols):
        top, bot = ax[0, c], ax[1, c]; rows = sorted(rows, key=lambda r: r["n"]); ok = [r for r in rows if "MELTED" not in r["flags"]]
        for r in rows:
            melt, flag = "MELTED" in r["flags"], bool(r["flags"])
            top.errorbar(r["n"], r["dg"], r["err"], fmt="x" if melt else "s", ms=6, color="#d55181", mfc="white" if (flag and not melt) else "#d55181", capsize=2)
        if ok:
            h = ph.hull([(r["n"], r["dg"], r["err"]) for r in ok], 1.0, 0); bot.plot(h["n"], h["per_atom"], "-s", color="#d55181", ms=4)
        top.axhline(0, color="k", lw=0.8); bot.axhline(0, color="k", lw=0.8); top.set_title(lab, fontsize=10); top.grid(alpha=0.25); bot.grid(alpha=0.25); bot.set_xlabel("n = extra H per O")
        if c == 0: top.set_ylabel("dg (meV per added H)"); bot.set_ylabel("formation free energy (meV/atom)")
    fig.suptitle("PWidom, all states: dg(n) and hull (open = flagged, x = host melted)", fontsize=11); fig.savefig(a.o + "_hulls.png", dpi=120); print("figure ->", a.o + "_hulls.png")
    Ps = sorted({P for P, _ in states})
    fig, ax = plt.subplots(1, len(Ps), figsize=(5.2 * len(Ps), 4.6), squeeze=False, constrained_layout=True); cm = plt.get_cmap("viridis")
    for k, P in enumerate(Ps):
        axk = ax[0, k]; Ts = sorted(T for p, T in states if p == P); ns = sorted({round(r["n"], 3) for T in Ts for r in states[(P, T)][1]})
        for i, n in enumerate(ns):
            pts = [(T, r["dg"], r["err"], "MELTED" in r["flags"]) for T in Ts for r in states[(P, T)][1] if abs(r["n"] - n) < 1e-3]
            if not pts: continue
            col = cm(i / max(1, len(ns) - 1)); axk.errorbar([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], fmt="-", color=col, lw=1, label=f"n = {n:g}")
            for T, y, e, melt in pts: axk.plot(T, y, "x" if melt else "o", color=col, ms=7)
            good = [(T, y) for T, y, e, melt in pts if not melt]
            if a.extrapolate and len(good) >= 2:
                x, y = zip(*good); cf = np.polyfit(x, y, 1); xs = np.linspace(min(x), max(x) + 3000, 20); axk.plot(xs, np.polyval(cf, xs), ":", color=col, lw=1)
        axk.axhline(0, color="k", lw=0.8); axk.set_title(f"{P} GPa"); axk.set_xlabel("T (K)"); axk.grid(alpha=0.25); axk.legend(fontsize=7)
        if k == 0: axk.set_ylabel("dg at fixed n (meV per H); x = host melted")
    fig.savefig(a.o + "_dg_vs_T.png", dpi=120); print("figure ->", a.o + "_dg_vs_T.png")


if __name__ == "__main__":
    main()
