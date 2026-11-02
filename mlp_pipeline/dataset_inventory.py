"""Inventory of the DFT data behind the training sets: what is in the tiers, what is in the pool of mined runs, and where the pool is barely used.

usage (machine that has runs/inventory_binned.csv and the extxyz tiers; repo root):
  python mlp_pipeline/dataset_inventory.py inventory_binned.csv --tiers train_m.extxyz valid_m.extxyz train_l.extxyz valid_l.extxyz [--out runs/dataset_inventory]
Prints (and writes as CSV into --out):
  1. pool: non-lambda production runs per composition x T bin: runs, total MD steps, cell sizes, P range
  2. each tier file: frames, runs, atoms per cell, frames per composition and per T bin
  3. per state (composition, T bin, 100 GPa bin): pool runs, pool MD steps, frames in each tier, and the ratio of pool steps to tier frames
  4. states with plenty of pool data but few tier frames (the cheap places to add data: extract more frames from runs we already have)
Pool = the 01_bin_inventory.load_inventory table, i.e. lambda_* TDI runs are already excluded; run `frame_batch.py --count` separately for the lambda = 1 frames of the TDI campaigns.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def tier_table(path):
    from ase.io import read
    fr = read(path, index=":")
    rows = []
    for f in fr:
        i = f.info; ct = str(i.get("config_type", "?"))
        comp, _, t = ct.rpartition("_")
        try: tb = int(t.rstrip("K"))
        except ValueError: comp, tb = ct, -1
        P = i.get("P_GPa", np.nan)
        rows.append(dict(comp=comp, T_bin=tb, P100=int(round(P / 100.0) * 100) if np.isfinite(P) else -1, source=str(i.get("source", "?")), N=len(f), nH=int((f.numbers == 1).sum()), nO=int((f.numbers == 8).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inventory", nargs="?", help="inventory_binned.csv (omit to inventory only the tiers)")
    ap.add_argument("--tiers", nargs="*", default=[]); ap.add_argument("--out"); ap.add_argument("--min-pool-runs", type=int, default=20, help="a state counts as 'well covered in the pool' above this many runs")
    a = ap.parse_args()
    out = a.out
    if out: os.makedirs(out, exist_ok=True)
    pool = None
    if a.inventory:
        from importlib import import_module
        pool = import_module("01_bin_inventory").load_inventory(a.inventory)
        pool = pool.assign(P100=(np.round(pool.True_P_GPa / 100) * 100).astype(int), steps=pool.get("N_steps", pd.Series(np.nan, index=pool.index)))
        print("\n== pool: production runs (no lambda_* TDI runs) per composition x T bin ==")
        g = pool.groupby(["comp_snap", "T_bin"]).agg(runs=("Path", "size"), steps=("steps", "sum"), N_atoms=("N_ions", "median"), P_min=("True_P_GPa", "min"), P_max=("True_P_GPa", "max")).round(0).reset_index()
        print(g.to_string(index=False))
        if out: g.to_csv(os.path.join(out, "pool_by_comp_T.csv"), index=False)
    tiers = {}
    for t in a.tiers:
        name = os.path.basename(t).replace(".extxyz", ""); df = tier_table(t); tiers[name] = df
        print(f"\n== tier {name}: {len(df)} frames from {df.source.nunique()} runs, atoms per cell median {df.N.median():.0f} (min {df.N.min()}, max {df.N.max()}) ==")
        print("frames per composition:", df.comp.value_counts().sort_index().to_dict())
        print("frames per T bin:      ", df.T_bin.value_counts().sort_index().to_dict())
        print("frames per 100 GPa bin:", df.P100.value_counts().sort_index().to_dict())
        dil = df[(df.nO > 0) & (df.nH / df.nO.clip(lower=1) > 2.0) & (df.nH / df.nO.clip(lower=1) < 2.2)]
        print(f"frames with 2.0 < H/O < 2.2 (dilute H in H2O; snapped to 'h2o' by the composition rule): {len(dil)}")
    if pool is not None and tiers:
        key = ["comp_snap", "T_bin", "P100"]
        st = pool.groupby(key).agg(pool_runs=("Path", "size"), pool_steps=("steps", "sum")).reset_index().rename(columns={"comp_snap": "comp"})
        for name, df in tiers.items():
            c = df.groupby(["comp", "T_bin", "P100"]).size().rename(f"frames_{name}").reset_index(); st = st.merge(c, how="left", on=["comp", "T_bin", "P100"])
        st = st.fillna(0)
        cols = [c for c in st.columns if c.startswith("frames_")]
        st["tier_frames_total"] = st[cols].sum(axis=1)
        if out: st.to_csv(os.path.join(out, "state_coverage.csv"), index=False)
        gap = st[(st.pool_runs >= a.min_pool_runs)].sort_values("pool_steps", ascending=False)
        gap = gap.assign(steps_per_tier_frame=(gap.pool_steps / gap.tier_frames_total.replace(0, np.nan)).round(0))
        print(f"\n== states with >= {a.min_pool_runs} pool runs and the ratio of pool MD steps to training frames (large = barely used) ==")
        print(gap.head(40).to_string(index=False))
        print(f"\nstates in the pool: {len(st)}; with no frame in any tier: {int((st.tier_frames_total == 0).sum())}; pool runs total {int(st.pool_runs.sum())}; tier frames total {int(st.tier_frames_total.sum())}")


if __name__ == "__main__":
    main()
