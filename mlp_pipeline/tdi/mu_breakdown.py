"""Where does mu_H come from? Breakdown of insertion runs (no MD: node files and stage0 json only).
usage (repo root, a python with numpy + matplotlib, e.g. ~/grace-venv/bin/python):
    python mlp_pipeline/tdi/mu_breakdown.py runs/tdi_long/insert_500GPa_6000K_n1 runs/tdi/insert_500GPa_5500K_n1 runs/tdi/insert_500GPa_5500K_n4 [-o runs/checks/mu_breakdown]
Per run, in meV per added H:
  ref_conf  = [kT ln((N_H+n)!/N_H!) - n kT ln V] / n      (ideal-gas placement of the ghost among N_H identical protons)
  ref_kin   = 3 kT ln Lambda_H
  TDI pieces: sum_k w_k <U_full - U_host>_k / n over the lambda values in [0, 1e-4), [1e-4, 1e-2), [1e-2, 0.1), [0.1, 1]
  P_corr    = pressure correction to P_target (0 if the stage0 json predates it), -G_H = pure H at the same P, T (largest run found next to it, or --gh)
  total     = mu_H - G_H.   'drift' = the same pieces from the 2nd half of every series minus the 1st half: which lambda range is still moving.
  'trapezoid in x' = the integral from a piecewise-linear interpolation of the integrand in x = lambda^(1/power) (what Gauss-Legendre integrates), ends held constant:
  a crude second quadrature from the same data; a large difference from Gauss-Legendre means the integrand is not resolved by the lambda grid.
Figure (<-o>.pdf / .png): (a) <dU/dlambda> per added H vs lambda, (b) the integrand in x, (c) running integral from lambda = 0 (solid: all samples; dashed / dotted: 1st / 2nd half),
(d) the breakdown bars."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze

BINS = [(0.0, 1e-4), (1e-4, 1e-2), (1e-2, 0.1), (0.1, 1.0 + 1e-12)]
BLAB = ["lambda<1e-4", "1e-4..1e-2", "1e-2..0.1", "0.1..1"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order (one colour per run, never cycled past 6)


def load(base):
    st = json.load(open(base + "_stage0.json")); A = st["args"]; lam, w, _ = analyze.rule_nodes(A)
    ser = [np.load(f"{base}_node{k}.npy") if os.path.exists(f"{base}_node{k}.npy") else np.array([]) for k in range(A["nodes"])]
    return st, A, lam, w, ser


def stale(base, A, lam):
    """stage0.json no longer describes the node files (rewritten by a later launch with other settings): same test as summarize.py"""
    if os.path.exists(f"{base}_node{A['nodes']}.npy"):
        return True
    for k in range(A["nodes"]):
        try:
            z = np.load(f"{base}_node{k}_state.npz")
            if "lam" in z.files and abs(float(z["lam"]) - lam[k]) > 1e-9 * max(1.0, lam[k]):
                return True
        except (OSError, ValueError, KeyError):
            pass
    return False


def G_pure_H(d, P, T):
    """G per atom (meV) of the pure-H run at (P, T) with the most samples, in d first, else in its sibling runs/tdi* directories (gcalc needs ase.units)"""
    from tdi.gcalc import G_from_nodes
    for dd in [d] + sorted(x for x in glob.glob(os.path.join(os.path.dirname(os.path.abspath(d)), "tdi*")) if os.path.abspath(x) != os.path.abspath(d) and not x.endswith("tdi_test")):
        best = None
        for f in glob.glob(os.path.join(dd, f"H_{P:.0f}GPa_{T:.0f}K*_stage0.json")):
            base = f[:-len("_stage0.json")]
            try:
                st, A, lam, w, ser = load(base)
            except Exception:
                continue
            if stale(base, A, lam) or min(len(s) for s in ser) < 25:
                continue
            n = min(len(s) for s in ser)
            if best is None or n > best[0]:
                G, Ge = G_from_nodes(st, lam, w, [s[:n] for s in ser], st["N"])[:2]; best = (n, G, Ge, os.path.join(os.path.basename(dd), os.path.basename(base)))
        if best:
            return best
    return None


def breakdown(base, gh):
    st, A, lam, w, ser = load(base); n = st["N"]; T = st["T"]; kt = 8.617333262e-5 * T
    if st["phase"] != "insert":
        raise SystemExit(f"{base}: not an insertion run")
    L = min(len(s) for s in ser)
    if L < 50:
        raise SystemExit(f"{base}: a lambda value has fewer than 50 samples")
    ser = [s[:L] for s in ser]
    st5 = [analyze.block_stats(s, nblocks=5) for s in ser]; m = np.array([x[0] for x in st5]); e = np.array([x[1] for x in st5])
    h1 = np.array([s[:L // 2].mean() for s in ser]); h2 = np.array([s[L // 2:].mean() for s in ser])
    c, c1, c2 = (w * v / n * 1e3 for v in (m, h1, h2)); ce = w * e / n * 1e3   # meV per added H per lambda value
    out = dict(tag=os.path.join(os.path.basename(os.path.dirname(base)), os.path.basename(base)), n=n, T=T, P=st["P_target"], kT=kt * 1e3, ps=L * A["dt"] * A["every"] / 1000,
               ref_conf=st["f_ref_cons_eV"] / n * 1e3, ref_kin=st["f_kin_eV"] / n * 1e3, P_corr=st.get("dG_P_eV", np.nan) / n * 1e3)
    for (lo, hi), lab in zip(BINS, BLAB):
        sel = (lam >= lo) & (lam < hi)
        out[lab] = float(c[sel].sum()); out[lab + " err"] = float(np.sqrt((ce[sel] ** 2).sum())); out[lab + " drift"] = float((c2 - c1)[sel].sum())
    out["TDI"] = float(c.sum()); out["TDI err"] = float(np.sqrt((ce ** 2).sum()))
    p = float(A["power"]); x = lam ** (1 / p); o = np.argsort(x); g = p * x ** (p - 1) * m / n * 1e3   # integrand in x (meV per H per unit x)
    xs = np.concatenate([[0.0], x[o], [1.0]]); gs = np.concatenate([[g[o][0]], g[o], [g[o][-1]]])
    out["TDI trapezoid in x"] = float(np.sum(np.diff(xs) * (gs[1:] + gs[:-1]) / 2))
    G_H = gh if gh is not None else G_pure_H(os.path.dirname(base), st["P_target"], T)
    if isinstance(G_H, tuple):
        out["G_H"], out["G_H src"] = G_H[1], f"{G_H[3]} ({G_H[0]} samples)"
    else:
        out["G_H"], out["G_H src"] = (G_H, "--gh") if G_H is not None else (np.nan, "none found")
    out["mu_H"] = out["ref_conf"] + out["ref_kin"] + out["TDI"] + (0.0 if np.isnan(out["P_corr"]) else out["P_corr"])
    out["total"] = out["mu_H"] - out["G_H"]
    return out, dict(lam=lam, m=m / n, e=e / n, x=x, g=g, ge=p * x ** (p - 1) * e / n * 1e3, c=c, c1=c1, c2=c2)


ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+", help="<dir>/<tag> of insertion runs"); ap.add_argument("-o", default="runs/checks/mu_breakdown")
ap.add_argument("--gh", type=float, default=None, help="G of pure H (meV/atom) instead of looking it up"); a = ap.parse_args()
res = [breakdown(r, a.gh) for r in a.runs]
keys = ["ref_conf", "ref_kin"] + BLAB + ["TDI", "TDI trapezoid in x", "P_corr", "mu_H", "G_H", "total"]
print("meV per added H (kT in the header); err = statistical, drift = 2nd half - 1st half of each series")
for o, _ in res:
    print(f"\n{o['tag']}  (n {o['n']}, {o['P']:.0f} GPa, {o['T']:.0f} K, kT {o['kT']:.0f} meV, {o['ps']:.2f} ps per lambda; G_H from {o['G_H src']})")
    for k in keys:
        extra = f"   +- {o[k + ' err']:7.1f}   drift {o[k + ' drift']:+8.1f}" if k + " err" in o and k + " drift" in o else (f"   +- {o['TDI err']:7.1f}" if k == "TDI" else "")
        print(f"  {k:20s} {o[k]:10.1f}{extra}")
    print(f"  Gauss-Legendre minus trapezoid-in-x: {o['TDI'] - o['TDI trapezoid in x']:+.1f} (large = integrand not resolved by the lambda grid)")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not available: table only")
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5)); (A1, A2), (A3, A4) = ax
for i, (o, d) in enumerate(res):
    col = SERIES[i % len(SERIES)]; lab = f"{o['tag']} ({o['ps']:.1f} ps)"; s = np.argsort(d["lam"])
    A1.errorbar(d["lam"][s], d["m"][s], yerr=d["e"][s], color=col, lw=2, marker="o", ms=5, capsize=2, label=lab)
    A2.errorbar(d["x"][s], d["g"][s], yerr=d["ge"][s], color=col, lw=2, marker="o", ms=5, capsize=2, label=lab)
    lx = np.concatenate([[1e-12], d["lam"][s]])
    for cc, ls in ((d["c"], "-"), (d["c1"], "--"), (d["c2"], ":")):
        A3.step(lx, np.concatenate([[0.0], np.cumsum(cc[s])]), where="post", color=col, lw=2 if ls == "-" else 1.2, ls=ls, label=lab if ls == "-" else None)
A1.set_xscale("log"); A1.set_yscale("symlog", linthresh=1.0); A1.axhline(0, color=MUTED, lw=0.8)
A1.set_xlabel("lambda"); A1.set_ylabel("<U_full - U_host> per added H (eV, symlog)"); A1.set_title("(a) integrand vs lambda", loc="left")
A2.axhline(0, color=MUTED, lw=0.8); A2.set_xlabel("x = lambda^(1/power)"); A2.set_ylabel("integrand in x (meV per H per unit x)"); A2.set_title("(b) what Gauss-Legendre integrates: smooth = resolved", loc="left")
A3.set_xscale("log"); A3.set_xlim(1e-10, 1.2); A3.axhline(0, color=MUTED, lw=0.8)
A3.set_xlabel("lambda (upper limit)"); A3.set_ylabel("running TDI integral (meV per added H)"); A3.set_title("(c) running integral; dashed = 1st half, dotted = 2nd half", loc="left")
comp = ["ref_conf", "ref_kin"] + BLAB + ["P_corr", "G_H", "total"]; yy = np.arange(len(comp)); bh = 0.8 / len(res)
for i, (o, _) in enumerate(res):
    vals = [-o["G_H"] if k == "G_H" else (0.0 if np.isnan(o[k]) else o[k]) for k in comp]
    errs = [o.get(k + " err", 0.0) for k in comp]
    A4.barh(yy + (i - (len(res) - 1) / 2) * bh, vals, height=bh * 0.9, xerr=errs, color=SERIES[i % len(SERIES)], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
A4.set_yticks(yy); A4.set_yticklabels(["ref: placement", "ref: 3kT ln Lambda"] + ["TDI " + b for b in BLAB] + ["P correction", "- G_H (pure)", "TOTAL mu_H - G_H"]); A4.invert_yaxis()
A4.axvline(0, color=MUTED, lw=0.8); A4.set_xlabel("meV per added H"); A4.set_title("(d) breakdown of mu_H - G_H", loc="left")
h, l = A1.get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=min(3, len(res)), frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94)); os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{a.o}.{ext}", dpi=150)
print(f"\nwrote {a.o}.pdf / .png")
