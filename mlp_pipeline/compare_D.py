#!/usr/bin/env python3
"""Compare fine-tunes (and the old model) from their D_*.csv files (baseline_D.py output on the SAME file). Green = best in the column.
   python3 mlp_pipeline/compare_D.py runs/D_*.csv [--sets A C C2 D_melt E H] """
import argparse, collections, csv, math, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--sets", nargs="+", default=["A", "C", "C2", "D_melt", "E", "H"]); ap.add_argument("--min-group", type=int, default=6); ap.add_argument("--keep-small-pureh", action="store_true", help="keep pure-H rows with < 500 atoms (invalid labels)"); a = ap.parse_args()
def sdD(rows):
    g = collections.defaultdict(list)
    for r in rows: g[(r["set"], int(round(float(r["T_K"]) / 1000)) if r["T_K"] not in ("", "nan") else -1, r["nO"], r["nH"])].append(float(r["D_F"]))
    ss = dof = 0
    for v in g.values():
        if len(v) >= a.min_group: x = np.array(v); ss += ((x - x.mean()) ** 2).sum(); dof += len(v) - 1
    return math.sqrt(ss / dof) if dof else float("nan")
res = {}
for p in a.files:
    rows = list(csv.DictReader(open(p))); rows = rows if a.keep_small_pureh else [r for r in rows if int(r["nO"]) > 0 or int(r["natoms"]) >= 500]; name = os.path.basename(p)[2:-4]; n = np.array([float(r["natoms"]) for r in rows]); e = np.array([float(r["D_F"]) for r in rows]) / n * 1000
    fr = np.array([float(r["f_rmse"]) for r in rows]); fm = np.array([float(r["f_rms"]) for r in rows]); sr = np.array([float(r.get("s_rmse_GPa", "nan") or "nan") for r in rows])
    d = {"sd(D) all": sdD(rows), "E rmse": float(np.sqrt((e ** 2).mean())), "E bias": float(e.mean()), "F rmse": float(np.sqrt((fr ** 2 * n).sum() / n.sum())), "F rel%": 100 * float(fr.mean() / fm.mean()), "S rmse": float(np.sqrt(np.nanmean(sr ** 2)))}
    for s in a.sets:
        sub = [r for r in rows if r["set"] == s]; d["sd " + s] = sdD(sub) if sub else float("nan")
    res[name] = d; res[name]["_n"] = len(rows)
cols = ["sd(D) all"] + ["sd " + s for s in a.sets] + ["E rmse", "E bias", "F rmse", "F rel%", "S rmse"]; unit = {"E rmse": "meV/at", "E bias": "meV/at", "F rmse": "eV/A", "F rel%": "%", "S rmse": "GPa"}
best = {c: min((abs(v[c]), k) for k, v in res.items() if v[c] == v[c])[1] for c in cols if any(v[c] == v[c] for v in res.values())}
G, Z = "\033[32;1m", "\033[0m"
print("sd(D) = within-group scatter of E_DFT - E_model per cell [eV]; E/F/S = raw RMSE vs DFT (E per atom).  n frames: " + ", ".join(sorted({str(v['_n']) for v in res.values()})))
print(f"{'model':26s}" + "".join(f"{c:>11s}" for c in cols)); print(f"{'':26s}" + "".join(f"{unit.get(c, 'eV/cell'):>11s}" for c in cols))
for k in sorted(res, key=lambda k: res[k]["sd(D) all"]):
    print(f"{k:26s}" + "".join((G if best.get(c) == k else "") + f"{res[k][c]:11.2f}" + (Z if best.get(c) == k else "") for c in cols))
