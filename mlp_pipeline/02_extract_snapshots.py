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

MIN_STEP_GAP = 50
P_MIN, P_MAX = 250.0, 1425.0
T_TARGETS = (5000, 7000, 9000)
T_TOL = 250.0
MAX_RUNS_PER_STATE = 6


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


def select(df):
    d = df[df.comp != "unknown"]
    d = d[d.True_P_GPa.between(P_MIN, P_MAX)]
    near = d.True_T_K.map(lambda t: min(abs(t - T) for T in T_TARGETS))
    d = d[near <= T_TOL]
    return d.groupby(["comp", "P_bin", "T_bin"], group_keys=False).head(MAX_RUNS_PER_STATE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out")
    ap.add_argument("--per-run", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    df = select(pd.read_csv(a.inv))
    meta = df.set_index("Path")
    print(f"{len(df)} runs selected")
    allf = []
    with Pool(a.workers) as p:
        for path, fr, err in p.starmap(pick, [(x, a.per_run) for x in df.Path]):
            if err:
                print("skip", path, err); continue
            m = meta.loc[path]
            for f in fr:
                f.info.update(config_type=m.comp, T_K=float(m.True_T_K), P_GPa=float(m.True_P_GPa))
            allf += fr
    write(a.out, allf, format="extxyz")
    print(f"wrote {len(allf)} frames -> {a.out}")
