"""Integrate the node files written by `run_real.py --node K` and print G (meV/atom) versus simulation time per node; safe to run while the nodes are still going.
usage: python mlp_pipeline/tdi/collect.py runs/tdi bcc_400GPa_3000K [--every-ps 0.25] [--nodes 8 --power 4] [--dt 0.5 --sample-every 2]
Writes <dir>/tdi_<tag>.json in the format compare.py reads (history = the time points printed)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.run_real import G_from_nodes

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tag"); ap.add_argument("--every-ps", type=float, default=0.25)
ap.add_argument("--truncate", nargs="*", default=[], help="K:PS ... keep only the first PS ps of node K (e.g. a node whose dynamics blew up after PS ps)")
a = ap.parse_args()
st = json.load(open(os.path.join(a.dir, f"{a.tag}_stage0.json"))); A = st["args"]
lam, w = analyze.gauss_legendre(A["nodes"], A["power"])
ser = []
for k in range(A["nodes"]):
    f = os.path.join(a.dir, f"{a.tag}_node{k}.npy")
    ser.append(np.load(f) if os.path.exists(f) else np.array([]))
per_ps = 1000 / A["dt"] / A["every"]           # recorded samples per ps
for t in a.truncate:
    k_, ps_ = t.split(":"); ser[int(k_)] = ser[int(k_)][:int(float(ps_) * per_ps)]; print(f"node {k_} truncated to the first {ps_} ps ({len(ser[int(k_)])} samples)")
have = [len(s) / per_ps for s in ser]
print(f"{a.tag}: samples per node (ps): " + "  ".join(f"{h:.2f}" for h in have))
tmax = min(have)
if tmax < a.every_ps:
    sys.exit(f"not all nodes have {a.every_ps} ps yet")
hist = []
for t in np.arange(a.every_ps, tmax + 1e-9, a.every_ps):
    G, Ge, dF, m, s = G_from_nodes(st, lam, w, [x[:int(round(t * per_ps))] for x in ser], st["N"])
    hist.append(dict(t_ps=float(t), G_meV_atom=G, G_err_meV_atom=Ge, dF_meV_atom=dF)); print(f"  t/node {t:5.2f} ps   G {G:10.2f} +- {Ge:5.2f} meV/atom   (dF_TDI {dF:9.2f})")
B = "▁▂▃▄▅▆▇█"; g = [h["G_meV_atom"] for h in hist]
print("G(t) trace: " + "".join(B[int(round((v - min(g)) / (max(g) - min(g) + 1e-12) * 7))] for v in g))
print("per-node <U_MLP - E_Einstein> (eV, whole cell) over all samples:")
for k in range(A["nodes"]):
    print(f"   node {k} lam {lam[k]:.3e}  weight {w[k]:.3e}  {m[k]:12.3f} +- {s[k]:8.3f}   contribution {w[k] * m[k] / st['N'] * 1e3:9.2f} meV/atom")
json.dump(dict(st, lam=lam.tolist(), weights=w.tolist(), dudl_mean=m.tolist(), dudl_sem=s.tolist(), hist=hist, G_meV_atom=hist[-1]["G_meV_atom"], G_err_meV_atom=hist[-1]["G_err_meV_atom"]),
          open(os.path.join(a.dir, f"tdi_{a.tag}.json"), "w"), indent=1)
