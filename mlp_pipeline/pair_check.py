#!/usr/bin/env python3
"""Tests A2 / B1 / B4 on the Widom-style pairs: the insertion-energy error per added H at the states we simulate.
Input: baseline_D.py CSVs (columns run_id, role, set, nH, T_K, D_F) made on the labelled pair frames (set P / C2). For each group (run_id) with a reference frame (role host or full) and variants:
   insertion pair  host -> host+H  (role prox / lowphi):  dD = D(host+H) - D(host)      eV per added H   (D = E_DFT - E_model, free energy)
   deletion pair   full -> full-H  (role del):             dD = D(full) - D(full-H)      the same quantity, for the removed H
Per state (cache, from the run_id) and pooled: mean dD, sd, and the Gaussian-corrected composition error  c = mean(dD) - var(dD)/(2 kT)   [meV per H]   (B4: forward estimate; the model's own samples).
Pass (docs/MODEL_TEST_SUITE.md A2/B4): |c| <= 50 meV/H  (kT = 0.43 eV at 5000 K: 50 meV = 0.12 kT, a 12 % factor in solubility).  --rows prints every pair.
   python3 mlp_pipeline/pair_check.py runs/Dall_<model>.csv [more.csv] [--split valid] [--mode random]"""
import argparse, collections, csv, math, os
KB = 8.617333e-5
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--split", default=""); ap.add_argument("--mode", default="", help="acq_mode filter (random = unbiased sample)"); ap.add_argument("--rows", action="store_true"); a = ap.parse_args()
def ms(x):
    m = sum(x) / len(x); sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan"); return m, sd
for p in a.files:
    rows = [r for r in csv.DictReader(open(p)) if r.get("role", "") and (not a.split or r.get("split", "") == a.split) and (not a.mode or r.get("acq_mode", "") == a.mode)]
    if not rows: print(f"{p}: no pair frames (role column empty: score the labelled pair set with the current baseline_D.py)"); continue
    g = collections.defaultdict(dict)
    for r in rows: g[r["run_id"]].setdefault(r["role"], []).append(float(r["D_F"]))
    T = {r["run_id"]: float(r["T_K"]) for r in rows}; res = collections.defaultdict(list)
    for rid, d in g.items():
        st = rid.split(":")[0]
        if "host" in d:
            for v in d.get("prox", []) + d.get("lowphi", []): res[(st, "ins")].append((v - d["host"][0], T[rid]))
        if "full" in d:
            for v in d.get("del", []): res[(st, "del")].append((d["full"][0] - v, T[rid]))
    print(f"\n{os.path.basename(p)}   dD per added H [meV] (mean, sd), c = mean - var/2kT [meV], n pairs")
    print(f"   {'state':34s}{'kind':5s}{'n':>4s}{'mean':>8s}{'sd':>7s}{'c':>8s}")
    pooled = []
    for (st, kind), lst in sorted(res.items()):
        x = [v for v, _ in lst]; kT = KB * lst[0][1]; m, sd = ms(x); c = m - (sd ** 2) / (2 * kT) if sd == sd else m; pooled += [(v, kT) for v, _ in lst]
        print(f"   {st[:34]:34s}{kind:5s}{len(x):4d}{m * 1000:8.0f}{sd * 1000:7.0f}{c * 1000:8.0f}")
        if a.rows: print("      " + " ".join(f"{v * 1000:+.0f}" for v in x))
    if pooled:
        x = [v for v, _ in pooled]; m, sd = ms(x); kT = sum(k for _, k in pooled) / len(pooled); c = m - (sd ** 2) / (2 * kT) if sd == sd else m
        print(f"   {'ALL':34s}{'':5s}{len(x):4d}{m * 1000:8.0f}{sd * 1000:7.0f}{c * 1000:8.0f}   {'PASS' if abs(c) <= 0.05 else 'FAIL'} (|c| <= 50 meV/H)")
