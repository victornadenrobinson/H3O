"""Plots from the TDI node files; aware of nodes that have not finished (or not started).
usage: python3 mlp_pipeline/tdi/plot_tdi.py runs/tdi fcc_400GPa_3000K [bcc_400GPa_3000K] [-o out.png] [--lit -22.1]
Panels: (1) <U_MLP - E_Einstein> per node vs lambda (log x): filled markers = nodes with data (size grows with progress), grey crosses = not started;
(2) the same vs x = lambda^(1/p), bars = weight x integrand (= contribution to the integral);
(3) G(t): solid = Gauss-Legendre over all nodes (only once every node has reached time t), dashed = PREVIEW using the nodes that have reached t
    (linear interpolation in x between them, constant beyond the end nodes: only as good as the nodes you have);
(4) G_A - G_B vs time (solid GL / dashed preview) with the literature value. Title says how many nodes are complete."""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi.run_real import G_from_nodes, stats
from tdi.live import load, preview


def curves(L, step=0.25):
    st, A, lam, w, ser = L; N = st["N"]; per_ps = 1000 / A["dt"] / A["every"]
    tmax = max(len(s) for s in ser) / per_ps; ts = np.arange(step, tmax + 1e-9, step); full, prev = [], []
    for t in ts:
        n = int(round(t * per_ps)); sub = [s[:n] for s in ser]
        prev.append(preview(st, A, lam, sub, N))
        full.append(G_from_nodes(st, lam, w, sub, N)[:2] if all(len(s) >= n for s in ser) else (np.nan, np.nan))
    return ts, np.array(prev), np.array([f[0] for f in full]), np.array([f[1] for f in full])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tags", nargs="+"); ap.add_argument("-o", default=None); ap.add_argument("--lit", type=float, default=-22.1)
    a = ap.parse_args()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); col = ["C0", "C3"]; cur = {}; info = []
    for n, tag in enumerate(a.tags):
        L = load(a.dir, tag)
        if L is None:
            print("no stage0 json for", tag); continue
        st, A, lam, w, ser = L; N = st["N"]; per_ps = 1000 / A["dt"] / A["every"]; target = A["ps"] * per_ps; x = lam ** (1 / A["power"])
        ndone = sum(len(s) >= target for s in ser); info.append(f"{tag}: {ndone}/{len(ser)} nodes complete")
        have = [k for k, s in enumerate(ser) if len(s) >= 10]; none = [k for k in range(len(ser)) if k not in have]
        m = np.array([stats(ser[k])[0] if len(ser[k]) >= 25 else np.mean(ser[k]) for k in have]) / N * 1e3
        e = np.array([stats(ser[k])[1] if len(ser[k]) >= 25 else np.nan for k in have]) / N * 1e3
        fr = np.array([min(1.0, len(ser[k]) / target) for k in have])
        for axx, xx in ((ax[0, 0], lam), (ax[0, 1], x)):
            axx.plot(xx[have], m, "-", color=col[n], alpha=.5, lw=1)
            axx.errorbar(xx[have], m, e, fmt="none", color=col[n], alpha=.6)
            axx.scatter(xx[have], m, s=25 + 90 * fr, color=col[n], label=tag, zorder=3)
            axx.scatter(xx[none], np.zeros(len(none)), marker="x", color="grey", zorder=2)
        ax[0, 1].bar(x[have], w[have] * m, width=0.4 * np.gradient(x)[have], color=col[n], alpha=.25)
        cur[tag] = curves(L)
        ts, prev, full, fe = cur[tag]
        if len(ts):
            ax[1, 0].plot(ts, prev, "--", color=col[n], alpha=.8); ax[1, 0].errorbar(ts, full, fe, fmt="o-", color=col[n], label=tag)
    ax[0, 0].set(xscale="log", xlabel="lambda (grey x: node not started)", ylabel="<U_MLP - E_Einstein>  (meV/atom)", title="integrand per node (marker size ~ progress)"); ax[0, 0].axhline(0, c="k", lw=.5); ax[0, 0].legend()
    ax[0, 1].set(xlabel="x = lambda^(1/p)", ylabel="meV/atom (line: integrand, bars: weight x integrand)", title="integrand in the quadrature variable"); ax[0, 1].axhline(0, c="k", lw=.5)
    ax[1, 0].set(xlabel="simulation time per node (ps)", ylabel="G (meV/atom)", title="G incl. P_target V   (solid: all nodes, dashed: preview from available nodes)"); ax[1, 0].legend()
    if len(a.tags) == 2 and all(t in cur for t in a.tags):
        (t1, p1, f1, e1), (t2, p2, f2, e2) = cur[a.tags[0]], cur[a.tags[1]]; k = min(len(t1), len(t2))
        ax[1, 1].plot(t1[:k], (p1 - p2)[:k], "--", color="k", alpha=.7, label="preview"); ax[1, 1].errorbar(t1[:k], (f1 - f2)[:k], np.hypot(e1, e2)[:k], fmt="o-", color="k", label="all nodes")
        ax[1, 1].axhline(a.lit, c="g", ls="--", label=f"literature {a.lit}"); ax[1, 1].axhline(0, c="k", lw=.5); ax[1, 1].legend()
        ax[1, 1].set(xlabel="simulation time per node (ps)", ylabel=f"{a.tags[0]} - {a.tags[1]}  (meV/atom)", title="free energy difference vs time")
    else:
        ax[1, 1].axis("off")
    fig.suptitle("   |   ".join(info)); fig.tight_layout(); out = a.o or os.path.join(a.dir, "tdi_plots.png"); fig.savefig(out, dpi=110); print("wrote", out, "|", "; ".join(info))
