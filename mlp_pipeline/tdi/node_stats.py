"""Per-lambda health of a TDI run from its node series alone (no GPU, no ASE): is each <dU/dlambda> converged, and is its mean carried by a few rare samples?
usage (repo root):  python3 mlp_pipeline/tdi/node_stats.py runs/tdi insert_500GPa_6000K_n1 [more tags ...]
Columns per lambda value: weight, samples, mean +- 5-block error (eV, whole cell), median, max, 'top1%' = share of sum(x - median) carried by the top 1% of samples
(heavy tail: a short run under-samples the rare large values and its mean is biased low), drift = mean(2nd half) - mean(1st half) in units of the error,
contribution w * mean to G (meV per atom, or per added H for insertion), and the closest H-H / H-O contacts of the last health line in <tag>_node<k>.log.
Flags: TAIL (top1% > 0.3), DRIFT (|drift| > 3 errors), FEW (< 200 samples). Lambda values are listed in increasing order."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze

d, tags = sys.argv[1], sys.argv[2:]
for tag in tags:
    f0 = os.path.join(d, f"{tag}_stage0.json")
    if not os.path.exists(f0):
        print(f"\n{tag}: no {f0}"); continue
    st = json.load(open(f0)); A = st["args"]; lam, w, _ = analyze.rule_nodes(A); N = st["N"]
    unit = "meV per added H" if st["phase"] == "insert" else "meV/atom"
    print(f"\n{d}/{tag}: {A['nodes']} lambda ({A.get('rule') or 'gl'}), N {N}; contribution in {unit}")
    print(f"{'k':>3} {'lambda':>10} {'weight':>9} {'n':>6} {'mean':>12} {'err':>9} {'median':>11} {'max':>11} {'top1%':>6} {'drift/err':>9} {'contrib':>9}  {'dHH':>5} {'dHO':>5}  flags")
    tot = 0.0
    for k in np.argsort(lam):
        fn = os.path.join(d, f"{tag}_node{k}.npy")
        if not os.path.exists(fn):
            print(f"{k:>3} {lam[k]:10.3e} {w[k]:9.2e}   (no samples)"); continue
        s = np.load(fn)
        if len(s) < 25:
            print(f"{k:>3} {lam[k]:10.3e} {w[k]:9.2e} {len(s):6d}   (too few samples)"); continue
        m, e = analyze.block_stats(s, nblocks=5); med = float(np.median(s))
        x = np.sort(s - med)[::-1]; top = x[:max(1, len(x) // 100)].sum() / max(np.abs(x).sum(), 1e-300)
        h = len(s) // 2; dr = (s[h:].mean() - s[:h].mean()) / e if e > 0 else float("nan")
        c = w[k] * m / N * 1e3; tot += c
        dhh = dho = float("nan"); lf = os.path.join(d, f"{tag}_node{k}.log")
        if os.path.exists(lf):
            hits = re.findall(r"min d\(H-H\) ([0-9.na]+) d\(H-O\) ([0-9.na]+) A", open(lf, errors="replace").read())
            if hits: dhh, dho = (float(v) for v in hits[-1])
        flags = " ".join(t for t, c_ in (("TAIL", top > 0.3), ("DRIFT", abs(dr) > 3), ("FEW", len(s) < 200)) if c_)
        print(f"{k:>3} {lam[k]:10.3e} {w[k]:9.2e} {len(s):6d} {m:12.3f} {e:9.3f} {med:11.3f} {s.max():11.3f} {top:6.2f} {dr:9.1f} {c:9.2f}  {dhh:5.2f} {dho:5.2f}  {flags}")
    print(f"    sum of contributions (TDI integral only): {tot:.2f} {unit}")
