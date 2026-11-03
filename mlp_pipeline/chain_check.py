#!/usr/bin/env python3
"""Test B1 (paired insertion energy error) from a baseline_D.py CSV: set-B chain frames hold the SAME growing configuration at several n (n_add), so
   error of the insertion energy per added H = [D(n2) - D(n1)] / (n2 - n1), D = E_DFT - E_model (free energy, per cell). This is the quantity that shifts mu_H.
   python3 mlp_pipeline/chain_check.py runs/D_<model>.csv [runs/D_other.csv ...] [--set B] [--keep-valid]
   Needs a CSV made with the current baseline_D.py (columns run_id, host, n_add, method). Pairs are formed between consecutive n_add of the same (host, run_id)
   chain; only frames present in the scored file count (valid_n has few chain frames: score the chain frames of the whole snapdb for more pairs, but train-split pairs are optimistic).
   Criterion from docs/MODEL_TEST_SUITE.md B1: mean error <= 100 meV, sd <= 200 meV per added H, no trend with n."""
import argparse, collections, csv, math, os
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--set", default="B"); ap.add_argument("--split", default=""); ap.add_argument("--dE-max", type=float, default=0.0, help="drop pairs whose DFT insertion energy per H exceeds this (eV): unrelaxed close contacts"); a = ap.parse_args()
def stats(x):
    m = sum(x) / len(x); sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan"); return m, sd
for p in a.files:
    rows = [r for r in csv.DictReader(open(p)) if r["set"] == a.set and r.get("n_add", "") not in ("", None) and (not a.split or r.get("split", "") == a.split)]
    if not rows: print(f"{p}: no frames of set {a.set} with n_add (old CSV? rerun baseline_D.py)"); continue
    ch = collections.defaultdict(list)
    for r in rows: ch[(r["host"], r["run_id"])].append((int(float(r["n_add"])), int(r["nH"]), float(r["D_F"]), float(r["F"])))
    err = []; pairs = []; by_n = collections.defaultdict(list)
    for k, v in ch.items():
        v.sort()
        for (n1, h1, d1, f1), (n2, h2, d2, f2) in zip(v, v[1:]):
            if h2 > h1:
                de = (f2 - f1) / (h2 - h1)                                  # DFT insertion energy per added H [eV] (raw free-energy difference)
                if a.dE_max and de > a.dE_max: continue
                e = (d2 - d1) / (h2 - h1) * 1000.0; err.append(e); by_n[n1].append(e); pairs.append((de, e))      # meV per added H
    print(f"\n{os.path.basename(p)}: {len(rows)} chain frames in {len(ch)} chains -> {len(err)} consecutive pairs")
    if len(err) < 2: continue
    m, sd = stats(err); print(f"   error of the insertion energy per added H [meV]: mean {m:+.0f}  sd {sd:.0f}   (criterion: |mean| <= 100, sd <= 200)  {'PASS' if abs(m) <= 100 and sd <= 200 else 'FAIL'}")
    xs = sorted(by_n)
    if len(xs) >= 3:
        mx = sum(xs) / len(xs); my = sum(sum(by_n[x]) / len(by_n[x]) for x in xs) / len(xs); sl = sum((x - mx) * (sum(by_n[x]) / len(by_n[x]) - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
        print(f"   trend with n: {sl * 10:+.1f} meV per +10 n   |  by starting n: " + "  ".join(f"n={x}:{sum(by_n[x]) / len(by_n[x]):+.0f}({len(by_n[x])})" for x in xs[:12]))
    if len(pairs) >= 8:
        pairs.sort(); q = len(pairs) // 4 or 1
        print("   error vs size of the DFT insertion energy per H (quartiles of the raw dE_DFT):")
        for i in range(4):
            sub = pairs[i * q:(i + 1) * q if i < 3 else len(pairs)]; m, sd = stats([e for _, e in sub])
            print(f"      dE_DFT {sub[0][0]:+8.2f} .. {sub[-1][0]:+8.2f} eV   mean error {m:+7.0f}  sd {sd:5.0f} meV  (n={len(sub)})")
