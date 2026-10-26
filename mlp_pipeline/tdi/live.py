"""Live text dashboard (and optional PNG) for running TDI jobs; safe on partial runs. Run in any shell that sees the lustre files:
    watch -n 60 -c "~/grace-venv/bin/python mlp_pipeline/tdi/live.py runs/tdi fcc_400GPa_3000K bcc_400GPa_3000K"      # or add --loop 60 --png
Per phase: progress, per-node integrand (meV/atom, whole-cell eV in brackets), weighted contribution, drift between the first and second half of each node's series in sigma
(|drift| > 2 means that node is not equilibrated), a sparkline of the integrand, a PREVIEW integral that works with whatever nodes already have data (linear interpolation in x,
constant beyond the end nodes) next to the final Gauss-Legendre estimate G(t), and G_A - G_B when two phases are given."""
import argparse, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.run_real import G_from_nodes, stats, refs

BARS = "▁▂▃▄▅▆▇█"


def load(d, tag):
    f = os.path.join(d, f"{tag}_stage0.json")
    if not os.path.exists(f):
        return None
    st = json.load(open(f)); A = st["args"]; lam, w = analyze.gauss_legendre(A["nodes"], A["power"])
    ser = []
    for k in range(A["nodes"]):
        g = os.path.join(d, f"{tag}_node{k}.npy")
        try:
            ser.append(np.load(g) if os.path.exists(g) else np.array([]))
        except Exception:
            ser.append(np.array([]))
    return st, A, lam, w, ser


def block_means(s, per_ps, blk=0.25, maxn=16):
    n = max(1, int(round(blk * per_ps))); nb = len(s) // n
    b = [np.mean(s[i * n:(i + 1) * n]) for i in range(nb)]
    return b[-maxn:]


def spark(v):
    v = np.asarray(v, float); ok = np.isfinite(v)
    if not ok.any():
        return ""
    lo, hi = v[ok].min(), v[ok].max(); sc = (v - lo) / (hi - lo + 1e-12)
    return "".join(BARS[int(round(s * 7))] if o else " " for s, o in zip(sc, ok))


def preview(st, A, lam, ser, N):
    """G with the integral over x = lambda^(1/p) of the piecewise-linear integrand through the nodes that have >= 25 samples (constant beyond the end nodes)"""
    ok = [k for k in range(len(lam)) if len(ser[k]) >= 25]
    if len(ok) < 2:
        return float("nan")
    x = lam ** (1 / A["power"]); xs = np.linspace(0, 1, 4001); m = np.array([np.mean(ser[k]) for k in ok])
    g = np.interp(xs, x[ok], m) * A["power"] * xs ** (A["power"] - 1)
    dF = float(np.sum((g[1:] + g[:-1]) / 2 * np.diff(xs)))
    kt = refs.kT(st["T"])
    return (st["f_ref_cons_eV"] + dF - kt * np.log(st["V"]) + st["f_kin_eV"] + st["PV_eV"]) / N * 1e3


def show(d, tags):
    res = {}
    for tag in tags:
        L = load(d, tag)
        if L is None:
            print(f"{tag}: no stage0 json in {d} yet"); continue
        st, A, lam, w, ser = L; N = st["N"]; per_ps = 1000 / A["dt"] / A["every"]; n = [len(s) for s in ser]; target = A["ps"] * per_ps
        print(f"\n== {tag}: N {N}, k {st['kE']:.2f}, {A['nodes']} nodes, target {A['ps']} ps/node; done {100 * sum(n) / (target * len(n)):.0f}%  ({time.strftime('%H:%M:%S')})")
        print("  node     lambda     ps    integrand meV/atom  [eV cell]  +- sem    contribution  drift(sigma)  time trace (0.25 ps blocks)")
        mm, cc = [], []
        for k, s in enumerate(ser):
            if len(s) < 10:
                print(f"  {k:3d}  {lam[k]:10.3e}  {len(s) / per_ps:5.2f}   (waiting)"); mm.append(np.nan); cc.append(np.nan); continue
            m, e = stats(s) if len(s) >= 25 else (float(np.mean(s)), float("nan"))
            h = len(s) // 2; a1, a2 = np.array(s[:h]), np.array(s[h:])
            drift = (a2.mean() - a1.mean()) / (np.hypot(stats(a1)[1], stats(a2)[1]) + 1e-12) if h >= 25 else float("nan")
            mm.append(m / N * 1e3); cc.append(w[k] * m / N * 1e3)
            print(f"  {k:3d}  {lam[k]:10.3e}  {len(s) / per_ps:5.2f}   {m / N * 1e3:12.1f}  [{m:9.1f}]  {e / N * 1e3:7.2f}   {cc[-1]:11.2f}   {drift:+6.1f}  {spark(block_means(s, per_ps)):<18s}{'  <-- not equilibrated?' if abs(drift) > 2 else ''}")
        print(f"  integrand vs node: {spark(mm)}    (lambda increasing ->)")
        print(f"  PREVIEW (interpolated, whatever data exists): G = {preview(st, A, lam, ser, N):9.2f} meV/atom")
        have = min(n) / per_ps
        if have >= 0.25:
            ts = np.arange(0.25, have + 1e-9, 0.25); G = [G_from_nodes(st, lam, w, [s[:int(round(t * per_ps))] for s in ser], N)[:2] for t in ts]
            res[tag] = (ts, np.array([g[0] for g in G]), np.array([g[1] for g in G]))
            print(f"  G(t) trace {spark([g for g, _ in G])}   ", end="")
            print("  Gauss-Legendre G(t), meV/atom:  " + "   ".join(f"{t:.2f} ps: {g:.1f}+-{e:.1f}" for t, (g, e) in list(zip(ts, G))[-5:]))
        else:
            print("  Gauss-Legendre G(t): waiting until every node has 0.25 ps")
    if len(tags) == 2 and all(t in res for t in tags):
        (t1, g1, e1), (t2, g2, e2) = res[tags[0]], res[tags[1]]; k = min(len(t1), len(t2))
        print(f"\n== {tags[0]} - {tags[1]}  (meV/atom; literature at 400 GPa 3000 K: -22.1):")
        print(f"   trace {spark(g1[:k] - g2[:k])}   " + "   ".join(f"{t1[i]:.2f} ps: {g1[i] - g2[i]:+.1f}+-{np.hypot(e1[i], e2[i]):.1f}" for i in range(max(0, k - 6), k)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tags", nargs="+"); ap.add_argument("--loop", type=float, default=0, help="seconds between refreshes (0: once)")
    ap.add_argument("--png", action="store_true", help="also rewrite <dir>/tdi_plots.png (plot_tdi.py) at every refresh")
    a = ap.parse_args()
    while True:
        if a.loop:
            print("\033[2J\033[H", end="")
        show(a.dir, a.tags)
        if a.png:
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_tdi.py"), a.dir, *a.tags], capture_output=True)
        if not a.loop:
            break
        time.sleep(a.loop)
