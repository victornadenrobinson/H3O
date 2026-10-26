"""Plots from the TDI node files (works on partial runs).
usage: python3 mlp_pipeline/tdi/plot_tdi.py runs/tdi fcc_400GPa_3000K [bcc_400GPa_3000K] [-o out.png] [--lit -22.1]
Panels: (1) <U_MLP - E_Einstein> per node vs lambda (log x), (2) the same vs x = lambda^(1/p) with the quadrature weight shading (area = contribution),
(3) G(t) per phase (meV/atom), (4) G_fcc - G_bcc vs time with the literature value. Needs matplotlib."""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.run_real import G_from_nodes, stats


def load(d, tag):
    st = json.load(open(os.path.join(d, f"{tag}_stage0.json"))); A = st["args"]
    lam, w = analyze.gauss_legendre(A["nodes"], A["power"])
    ser = [np.load(os.path.join(d, f"{tag}_node{k}.npy")) if os.path.exists(os.path.join(d, f"{tag}_node{k}.npy")) else np.array([]) for k in range(A["nodes"])]
    return st, A, lam, w, ser


ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tags", nargs="+"); ap.add_argument("-o", default=None); ap.add_argument("--lit", type=float, default=-22.1)
a = ap.parse_args()
fig, ax = plt.subplots(2, 2, figsize=(13, 9)); col = {0: "C0", 1: "C3"}; Gt = {}
for n, tag in enumerate(a.tags):
    st, A, lam, w, ser = load(a.dir, tag); N = st["N"]; per_ps = 1000 / A["dt"] / A["every"]; have = min(len(s) for s in ser) / per_ps
    ms = [stats(s) for s in ser]; m = np.array([x[0] for x in ms]); e = np.array([x[1] for x in ms]); x = lam ** (1 / A["power"])
    ax[0, 0].errorbar(lam, m / N * 1e3, e / N * 1e3, fmt="o-", color=col[n], label=tag)
    ax[0, 1].errorbar(x, m / N * 1e3, e / N * 1e3, fmt="o-", color=col[n], label=tag)
    ax[0, 1].bar(x, w * m / N * 1e3, width=0.5 * np.gradient(x), color=col[n], alpha=0.25)
    ts = np.arange(0.25, have + 1e-9, 0.25)
    if len(ts):
        G = [G_from_nodes(st, lam, w, [s[:int(round(t * per_ps))] for s in ser], N)[:2] for t in ts]
        Gt[tag] = (ts, np.array([g[0] for g in G]), np.array([g[1] for g in G]))
        ax[1, 0].errorbar(ts, Gt[tag][1], Gt[tag][2], fmt="o-", color=col[n], label=tag)
ax[0, 0].set(xscale="symlog", xlabel="lambda", ylabel="<U_MLP - E_Einstein>  (meV/atom)", title="integrand per node (error bars: block SEM)"); ax[0, 0].axhline(0, c="k", lw=.5)
ax[0, 1].set(xlabel="x = lambda^(1/p)", ylabel="meV/atom   (line: integrand; bars: weight x integrand)", title="integrand in the quadrature variable"); ax[0, 1].axhline(0, c="k", lw=.5)
ax[1, 0].set(xlabel="simulation time per node (ps)", ylabel="G (meV/atom)", title="G including P_target V"); ax[1, 0].legend(); ax[0, 0].legend()
if len(a.tags) == 2 and all(t in Gt for t in a.tags):
    (t1, g1, e1), (t2, g2, e2) = Gt[a.tags[0]], Gt[a.tags[1]]; k = min(len(t1), len(t2))
    ax[1, 1].errorbar(t1[:k], g1[:k] - g2[:k], np.hypot(e1[:k], e2[:k]), fmt="o-", color="k")
    ax[1, 1].axhline(a.lit, c="g", ls="--", label=f"literature {a.lit} (Cheng 2021 / Wilson 2013)"); ax[1, 1].axhline(0, c="k", lw=.5); ax[1, 1].legend()
    ax[1, 1].set(xlabel="simulation time per node (ps)", ylabel=f"{a.tags[0]} - {a.tags[1]}  (meV/atom)", title="free energy difference vs time")
else:
    ax[1, 1].axis("off")
fig.tight_layout(); out = a.o or os.path.join(a.dir, "tdi_plots.png"); fig.savefig(out, dpi=110); print("wrote", out)
