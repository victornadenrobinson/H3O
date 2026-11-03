#!/usr/bin/env python3
"""One figure with the training information of every fine-tune: runs/grace/ft_*/seed/1/log.txt (TRAIN(TEST) lines with energy / force errors and timestamps) + recipe.txt.
Panels: (a) test energy RMSE vs updates (INITIAL = the foundation model before fine-tuning is plotted at 0)   (b) test force RMSE vs updates   (c) test/train energy-error ratio vs updates (overfitting)
(d) test energy RMSE vs wall time   (e) final test energy RMSE vs learning rate (colour = tier, marker = frozen/unfrozen)   (f) training speed (updates/s) and cost (GPU hours) per run, plus the legend.
Errors are the training-time test split of each run's OWN validation tier: runs on different tiers are not strictly comparable (score on a common file with baseline_D.py / compare_D.py for that).
   python3 mlp_pipeline/ft_curves.py [--grace runs/grace] [--out runs/ft_curves.png] [--only lc_ mn_f] [--skip nBase] [--term]
--term also prints one line per run with a sparkline of the test energy RMSE. Needs matplotlib (the grace env has it)."""
import argparse, glob, os, re, sys, datetime, math
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--grace", default="runs/grace"); ap.add_argument("--out", default="runs/ft_curves.png")
ap.add_argument("--only", nargs="*", default=[], help="substrings: keep runs whose name contains any"); ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--term", action="store_true"); ap.add_argument("--dpi", type=int, default=130); a = ap.parse_args()
PAIR = re.compile(r"([\w/]+):\s*([0-9.eE+-]+)\s*\(([0-9.eE+-]+)\)"); TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def parse(d):
    name = os.path.basename(d)[3:]; rec = {}
    try: txt = open(d + "/recipe.txt").read()
    except Exception: txt = ""
    m = re.search(r"train (\S+) valid (\S+) updates (\d+) weights E(\S+) F(\S+) S(\S+) (frozen|naive)(?: lr ([0-9.eE+-]+))?(?: batch (\d+))?", txt)
    if m: rec = dict(train=m.group(1), valid=m.group(2), updates=int(m.group(3)), wE=m.group(4), wF=m.group(5), wS=m.group(6), mode=m.group(7), lr=float(m.group(8) or 1e-3), batch=int(m.group(9) or 2))
    else: rec = dict(train="?", valid="?", updates=0, wE="?", wF="?", wS="?", mode="frozen", lr=1e-3, batch=2)
    pts = []; t0 = None
    try: lines = open(d + "/seed/1/log.txt", errors="replace").read().splitlines()
    except Exception: return None
    for l in lines:
        mt = TS.match(l)
        if mt and t0 is None: t0 = datetime.datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S")
        if "TRAIN(TEST):" not in l: continue
        head, tail = l.split("TRAIN(TEST):", 1); it = re.search(r"#(\d+)\s*/\s*(\d+)", head); t = datetime.datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S") if mt else None
        vals = {k: (float(x), float(y)) for k, x, y in PAIR.findall(tail)}
        pts.append(dict(it=int(it.group(1)) if it else 0, n=int(it.group(2)) if it else 0, t=(t - t0).total_seconds() / 3600.0 if (t and t0) else float("nan"), v=vals, init=("INITIAL" in head)))
    if not pts: return None
    starts = [i for i, p in enumerate(pts) if p["init"]]
    if len(starts) > 1:                                    # the log holds several training passes (a restart appended to the same log.txt): use the LAST pass, time zero at its INITIAL
        print(f"WARNING {name}: {len(starts)} training passes in log.txt; plotting the last one only")
        pts = pts[starts[-1]:]; t_ref = pts[0]["t"]
        for p in pts: p["t"] = p["t"] - t_ref
    return dict(name=name, rec=rec, pts=pts, done=os.path.exists(d + "/seed/1/final_model/saved_model.pb"))


runs = []
for d in sorted(glob.glob(a.grace + "/ft_*")):
    nm = os.path.basename(d)[3:]
    if a.only and not any(s in nm for s in a.only): continue
    if any(s in nm for s in a.skip): continue
    r = parse(d)
    if r: runs.append(r)
if not runs: sys.exit(f"no runs with a log under {a.grace}")
def series(r, key, which):          # which 0 = train, 1 = test -> (updates, value) with INITIAL at 0
    n_tot = max((p["n"] for p in r["pts"]), default=0) or 1; U = r["rec"]["updates"] or n_tot; out = []
    for p in r["pts"]:
        if key in p["v"]: out.append((0.0 if p["init"] else U * p["it"] / n_tot, p["v"][key][which], p["t"]))
    return out
SPARK = "▁▂▃▄▅▆▇█"
def spark(vals):
    vals = [math.log(v) for v in vals if v > 0]
    if len(vals) < 2: return ""
    lo, hi = min(vals), max(vals); return "".join(SPARK[min(7, int((v - lo) / (hi - lo + 1e-12) * 8))] for v in vals)
rows = []
for r in runs:
    e_te = series(r, "rmse/depa", 1); e_tr = series(r, "rmse/depa", 0); f_te = series(r, "rmse/f_comp", 1); f_tr = series(r, "rmse/f_comp", 0)
    fin = [x for x in e_te if x[0] > 0] or e_te; hrs = max((x[2] for x in e_te if x[2] == x[2]), default=float("nan")); upd = max((x[0] for x in e_te), default=0)
    r.update(e_te=e_te, e_tr=e_tr, f_te=f_te, f_tr=f_tr, hrs=hrs, upd=upd)
    rows.append((r["name"], "DONE" if r["done"] else "run", r["rec"], upd, (fin[-1][1] * 1000 if fin else float("nan")), (e_tr[-1][1] * 1000 if e_tr else float("nan")), (f_te[-1][1] if f_te else float("nan")), hrs, (upd / (hrs * 3600) if hrs == hrs and hrs > 0 else float("nan")), spark([x[1] for x in e_te])))
print(f"{'run':24s}{'state':6s}{'tier':>9s}{'mode':>8s}{'lr':>8s}{'upd':>8s}{'test E':>8s}{'train E':>8s}{'test F':>8s}{'hours':>7s}{'upd/s':>7s}  test-E trace (log)")
for n, st, rc, upd, te, tr, f, h, sp, tr_ in sorted(rows, key=lambda x: x[4] if x[4] == x[4] else 1e9):
    print(f"{n:24s}{st:6s}{rc['train']:>9s}{rc['mode']:>8s}{rc['lr']:>8.0e}{upd:8.0f}{te:8.1f}{tr:8.1f}{f:8.2f}{h:7.1f}{sp:>7.2f}  {tr_}" if h == h else f"{n:24s}{st:6s}{rc['train']:>9s}{rc['mode']:>8s}{rc['lr']:>8.0e}{upd:8.0f}{te:8.1f}{tr:8.1f}{f:8.2f}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
tiers = sorted({r["rec"]["train"] for r in runs}); cmap = plt.get_cmap("tab10"); tcol = {t: cmap(i % 10) for i, t in enumerate(tiers)}
rc20 = plt.get_cmap("tab20"); rcol = {r["name"]: rc20(i % 20) for i, r in enumerate(sorted(runs, key=lambda r: (r["rec"]["train"], r["rec"]["lr"], r["name"])))}
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True); A = ax.ravel()
for r in runs:
    c = rcol[r["name"]]; tc = tcol[r["rec"]["train"]]; ls = "--" if r["rec"]["mode"] == "naive" else "-"; lw = 1.6 if r["done"] else 1.0; al = 0.9 if r["done"] else 0.55
    for axi, key, which, scale in ((0, "rmse/depa", 1, 1000), (1, "rmse/f_comp", 1, 1)):
        s = series(r, key, which)
        if s:
            A[axi].plot([x[0] for x in s], [x[1] * scale for x in s], ls, color=c, lw=lw, alpha=al)
            if axi == 0: A[0].annotate(r["name"], (s[-1][0], s[-1][1] * scale), fontsize=6, color=c, xytext=(3, 0), textcoords="offset points", va="center")
    te = {x[0]: x[1] for x in r["e_te"]}; tr = {x[0]: x[1] for x in r["e_tr"]}; xs = sorted(k for k in te if k in tr and k > 0 and tr[k] > 0)
    if xs: A[2].plot(xs, [te[k] / tr[k] for k in xs], ls, color=c, lw=lw, alpha=al)
    s = [x for x in r["e_te"] if x[2] == x[2]]
    if s:
        A[3].plot([x[2] for x in s], [x[1] * 1000 for x in s], ls, color=c, lw=lw, alpha=al); A[3].annotate(r["name"], (s[-1][2], s[-1][1] * 1000), fontsize=6, color=c, xytext=(3, 0), textcoords="offset points", va="center")
    fin = [x for x in r["e_te"] if x[0] > 0]
    if fin: A[4].scatter(r["rec"]["lr"], fin[-1][1] * 1000, color=tc, marker="^" if r["rec"]["mode"] == "naive" else "o", s=55 if r["done"] else 30, alpha=0.9, edgecolor="k", lw=0.5); A[4].annotate(r["name"], (r["rec"]["lr"], fin[-1][1] * 1000), fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.8)
init = [x for r in runs for x in r["e_te"] if x[0] == 0]
if init: A[0].axhline(sum(x[1] for x in init) / len(init) * 1000, color="gray", lw=0.8, ls=":"); A[0].text(0.01, 0.97, "foundation model before fine-tuning", transform=A[0].transAxes, fontsize=7, va="top", color="gray")
A[0].set_yscale("log"); A[0].set_title("(a) test energy RMSE vs updates"); A[0].set_ylabel("meV / atom"); A[1].set_yscale("log"); A[1].set_title("(b) test force-component RMSE vs updates"); A[1].set_ylabel("eV / Å")
A[2].axhline(1.0, color="gray", lw=0.8, ls=":"); A[2].set_title("(c) test / train energy error (>1: overfitting)"); A[2].set_yscale("log")
A[3].set_yscale("log"); A[3].set_title("(d) test energy RMSE vs wall time"); A[3].set_xlabel("hours"); A[3].set_ylabel("meV / atom")
A[4].set_xscale("log"); A[4].set_yscale("log"); A[4].set_title("(e) final test energy RMSE vs learning rate"); A[4].set_xlabel("learning rate"); A[4].set_ylabel("meV / atom")
for i in (0, 1, 2): A[i].set_xlabel("updates")
for x in A[:5]: x.grid(True, which="both", alpha=0.25)
sp = A[5]; sp.axis("off"); names = [x for x in rows if x[8] == x[8]]
handles = [Line2D([0], [0], color="k", ls="-", label="frozen (last layers)"), Line2D([0], [0], color="k", ls="--", label="unfrozen (naive)"), Line2D([0], [0], color="k", lw=1, alpha=0.5, label="faint = still running")] + [Line2D([0], [0], marker="^" if False else "o", color="w", markerfacecolor=tcol[t], markeredgecolor="k", markersize=8, label=f"(e) tier {t}") for t in tiers]
sp.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, title="line colour = run (labelled at the end); (e) colour = tier, triangle = unfrozen")
lines = [f"{r[0][:24]:24s}{r[8]:5.1f}/s {r[7]:5.1f} h  test E {r[4]:5.1f}" for r in sorted(rows, key=lambda x: x[4] if x[4] == x[4] else 1e9) if r[8] == r[8]]
sp.text(0.0, 0.50, "(f) speed, cost, final test E [meV/atom]\n" + "\n".join(lines[:26]), family="monospace", fontsize=6.5, va="top", transform=sp.transAxes)
fig.suptitle(f"GRACE fine-tunes: {len(runs)} runs ({sum(1 for r in runs if r['done'])} finished)  -  errors on each run's own validation tier", fontsize=12)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); fig.savefig(a.out, dpi=a.dpi); print("figure ->", a.out)
