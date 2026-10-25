"""Extract decorrelated frames from selected OUTCARs to extxyz.

usage: python 02_extract_snapshots.py inventory_binned.csv out.extxyz [--per-run 5] [--workers 32]
Frames: evenly spaced from the 2nd half of each run, >= MIN_STEP_GAP MD steps apart.
Stress is stored as full 3x3 (ASE convention, eV/A^3).
"""
import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from ase.io import read, write
from common import progress

MIN_STEP_GAP = 50


def pick(path, n):
    try:
        frames = read(path, index=":", format="vasp-out")
    except Exception as e:
        return path, [], str(e)
    L = len(frames)
    lo = L // 2
    if L - lo < 1:
        return path, [], "empty"
    k = min(n, max(1, (L - lo) // MIN_STEP_GAP))
    idx = np.unique(np.linspace(lo, L - 1, k).astype(int))
    out = []
    for i in idx:
        a = frames[i]
        a.info.update(source=path, frame=int(i), comp=0)
        a.calc.results["stress"] = a.get_stress(voigt=False)
        out.append(a)
    return path, out, None


def pick_star(args):
    return pick(*args)


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
        idx = np.unique(np.linspace(0, len(g) - 1, min(a.max_runs, len(g))).astype(int))
        out.append(g.iloc[idx])
    return pd.concat(out) if out else d.iloc[:0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out")
    ap.add_argument("--per-run", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pmin", type=float, default=200.0); ap.add_argument("--pmax", type=float, default=1450.0)
    ap.add_argument("--tmin", type=float, default=2000.0)
    ap.add_argument("--max-runs", type=int, default=6, help="runs per (composition, T bin, 100 GPa bin)")
    ap.add_argument("--max-pstd", type=float, default=20.0)
    ap.add_argument("--lambda-keep", default=None, help="regex of lambda_* paths to keep (default: drop all lambda_* runs)")
    a = ap.parse_args()
    df = select(pd.read_csv(a.inv), a)
    meta = df.set_index("Path")
    print(f"{len(df)} runs selected in {df.groupby(['comp_snap', 'T_bin']).ngroups} (composition, T) states")
    print(df.pivot_table(index="comp_snap", columns="T_bin", values="Path", aggfunc="count", fill_value=0).to_string())
    allf = []
    with Pool(a.workers) as p:
        for path, fr, err in progress(p.imap(pick_star, [(x, a.per_run) for x in df.Path]), total=len(df), desc="extracting"):
            if err:
                print("skip", path, err); continue
            m = meta.loc[path]
            for f in fr:
                f.info.update(config_type=f"{m.comp_snap}_{int(m.T_bin)}K", T_K=float(m.True_T_K), P_GPa=float(m.True_P_GPa))
            allf += fr
    write(a.out, allf, format="extxyz")
    print(f"wrote {len(allf)} frames -> {a.out}")
