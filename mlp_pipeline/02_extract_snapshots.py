"""Extract decorrelated frames from selected OUTCARs to NESTED extxyz datasets (small <= medium <= large).

usage: python 02_extract_snapshots.py inventory_binned.csv snapshots [--tiers s=1x1,m=3x3,l=6x5] [--workers 32] [--root ~/work]
Writes snapshots_s.extxyz, snapshots_m.extxyz, snapshots_l.extxyz. A tier "RxF" = up to R runs per (composition, T bin,
100 GPa bin) and up to F frames per run. Runs/frames are ranked by a spread-out order (middle first, then fill the gaps),
so every smaller tier is a subset of the larger ones. Frames: 2nd half of each run, >= MIN_STEP_GAP MD steps apart.
Stress is stored as full 3x3 (ASE convention, eV/A^3, excludes the ions' kinetic term).
"""
import argparse
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from ase.io import read, write
from common import progress

MIN_STEP_GAP = 50


def spread_order(n):
    """Order 0..n-1 so that every prefix is spread out (van der Corput, starting mid-range)."""
    order, i = [], 0
    while len(order) < n and i < 64 * n:
        f, k, w = 0.0, i, 0.5
        while k:
            f += w * (k & 1); k >>= 1; w /= 2
        j = int(((f + 0.5) % 1.0) * n)
        if j not in order:
            order.append(j)
        i += 1
    return order + [j for j in range(n) if j not in order]


def pick(path, n, root=""):
    try:
        frames = read(os.path.join(root, path), index=":", format="vasp-out")
    except Exception as e:
        return path, [], str(e)
    L = len(frames)
    lo = L // 2
    if L - lo < 1:
        return path, [], "empty"
    k = min(n, max(1, (L - lo) // MIN_STEP_GAP))
    idx = np.unique(np.linspace(lo, L - 1, k).astype(int))
    out = []
    for rank, j in enumerate(spread_order(len(idx))):
        i = idx[j]
        a = frames[i]
        a.info.update(source=path, frame=int(i), frame_rank=rank)
        a.calc.results["stress"] = a.get_stress(voigt=False)
        out.append(a)
    return path, out, None


def pick_star(args):
    return pick(*args)  # (path, n, root)


def select(df, a):
    """Quality-filtered, state-balanced run selection. Uses the P-corrected inventory from 01 (inventory_binned.csv)."""
    d = df[df.good & (df.comp_snap != "unknown")]
    d = d[d.True_P_GPa.between(a.pmin, a.pmax) & (d.True_T_K >= a.tmin) & (d.P_std_GPa.fillna(0) <= a.max_pstd)]
    lam = d.Path.str.contains("lambda_", case=False)
    keep = d.Path.str.contains(a.lambda_keep) if a.lambda_keep else pd.Series(False, index=d.index)
    d = d[~lam | keep]
    d = d.assign(run=d.Path.map(lambda p: p.rsplit("/", 2)[0]), P100=(d.True_P_GPa // 100 * 100).astype(int))
    d = d.sort_values("Path").drop_duplicates("run")  # one OUTCAR per run directory (out1/out2... are a chain)
    out = []
    for _, g in d.groupby(["comp_snap", "T_bin", "P100"]):
        g = g.sort_values("True_P_GPa")
        order = spread_order(len(g))[:a.max_runs]
        out.append(g.iloc[order].assign(run_rank=range(len(order))))
    return pd.concat(out) if out else d.iloc[:0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out", help="output prefix, e.g. snapshots")
    ap.add_argument("--tiers", default="s=1x1,m=3x3,l=6x5", help="name=RUNSxFRAMES, nested (small first)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--root", default="~/work", help="dir that inventory paths are relative to")
    ap.add_argument("--pmin", type=float, default=200.0); ap.add_argument("--pmax", type=float, default=1450.0)
    ap.add_argument("--tmin", type=float, default=2000.0)
    ap.add_argument("--max-pstd", type=float, default=20.0)
    ap.add_argument("--lambda-keep", default=None, help="regex of lambda_* paths to keep (default: drop all lambda_* runs)")
    a = ap.parse_args()
    tiers = [(n, *map(int, v.split("x"))) for n, v in (t.split("=") for t in a.tiers.split(","))]
    a.max_runs = max(t[1] for t in tiers); a.per_run = max(t[2] for t in tiers)
    root = os.path.abspath(os.path.expanduser(a.root))
    df = select(pd.read_csv(a.inv), a)
    meta = df.set_index("Path")
    print(f"{len(df)} runs selected in {df.groupby(['comp_snap', 'T_bin']).ngroups} (composition, T) states")
    print(df.pivot_table(index="comp_snap", columns="T_bin", values="Path", aggfunc="count", fill_value=0).to_string())
    allf = []
    with Pool(a.workers) as p:
        for path, fr, err in progress(p.imap(pick_star, [(x, a.per_run, root) for x in df.Path]), total=len(df), desc="extracting"):
            if err:
                print("skip", path, err); continue
            m = meta.loc[path]
            for f in fr:
                f.info.update(config_type=f"{m.comp_snap}_{int(m.T_bin)}K", T_K=float(m.True_T_K), P_GPa=float(m.True_P_GPa),
                              run_rank=int(m.run_rank))
            allf += fr
    for name, nr, nf in tiers:
        sub = [f for f in allf if f.info["run_rank"] < nr and f.info["frame_rank"] < nf]
        fn = f"{a.out}_{name}.extxyz"
        write(fn, sub, format="extxyz")
        print(f"tier {name} ({nr} runs x {nf} frames): {len(sub)} frames -> {fn}")
