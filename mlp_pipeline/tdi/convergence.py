"""Time convergence table: G(t) of every run in a directory, using only the first t ps of every lambda series (same integration as summarize.py / plot_all.py), at t = 0.5, 1, 2, 3, 5, 7.5, 10, ... ps and at the full length.
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/convergence.py [dir ...] [--times 1 2 3 5 7.5 10]      default dir: runs/tdi_long
Per run: G(t) +- statistical error (meV/atom; insertion: per added H), then G(t) - G(final), and err*sqrt(t) (constant if the error falls as 1/sqrt(t): independent samples). A run is converged in time when G(t) - G(final)
stays inside the error for the later times; a steady slope means the equilibration or a slow degree of freedom (H hopping, O lattice) is still evolving. Runs of different length in the same directory are all listed."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.gcalc import G_from_nodes

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("--times", type=float, nargs="*", default=[0.5, 1, 2, 3, 5, 7.5, 10, 15, 20]); a = ap.parse_args()
for d in (a.dirs or ["runs/tdi_long"]):
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; st = json.load(open(f0)); A = st.get("args", {})
        lam, w, _ = analyze.rule_nodes(A); per_ps = 1000 / A["dt"] / A["every"]; ser = []
        for k in range(A["nodes"]):
            f = os.path.join(d, f"{tag}_node{k}.npy")
            try: ser.append(np.load(f) if os.path.exists(f) else np.array([]))
            except Exception: ser.append(np.array([]))
        for k in range(A["nodes"]):   # failed lambda value with negligible weight: next lambda's series (as summarize.py)
            if len(ser[k]) == 0 and w[k] < 1e-4:
                j = next((j for j in range(k + 1, A["nodes"]) if len(ser[j]) > 0), None)
                if j is not None: ser[k] = ser[j].copy()
        n = min(len(x) for x in ser)
        if n < 0.5 * per_ps: print(f"{d}/{tag}: fewer than 0.5 ps in at least one lambda value ({sum(len(x) > 0 for x in ser)}/{A['nodes']} lambda values have samples)"); continue
        tend = n / per_ps; ts = [t for t in a.times if t < tend - 1e-6] + [tend]
        G = [G_from_nodes(st, lam, w, [x[:int(round(t * per_ps))] for x in ser], st["N"])[:2] for t in ts]; gf = G[-1][0]
        print(f"\\n{d}/{tag}  (N {st['N']}, dt {A['dt']:g}, {A['nodes']} lambda, target {A['ps']:g} ps, done {tend:.2f} ps)")
        print("   t (ps):          " + "".join(f"{t:9.2f}" for t in ts)); print("   G (meV/atom):    " + "".join(f"{g[0]:9.2f}" for g in G))
        print("   +- error:        " + "".join(f"{g[1]:9.2f}" for g in G)); print("   G - G(final):    " + "".join(f"{g[0] - gf:9.2f}" for g in G))
        print("   err*sqrt(t):     " + "".join(f"{g[1] * t ** .5:9.2f}" for g, t in zip(G, ts)))
