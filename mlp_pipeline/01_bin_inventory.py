"""Bin mined runs by composition / P / T and check coverage of hull-CSV EOS targets.

usage: python 01_bin_inventory.py mined_v2.csv data-h3o-convex-hull.csv [--min-steps 100]
Composition comes from OUTCAR ion counts (00_mine_outcars.py). `comp` = exact H:O ratio,
`comp_snap` = nearest of KNOWN ratios within SNAP_TOL (off-stoichiometric cells like 33H16O -> h2o).
"""
import argparse
import numpy as np
import pandas as pd
from common import parse_composition, comp_label, HULL_COMPS

P_BIN, T_BIN = 25.0, 250.0
P_MIN, P_MAX = 250.0, 1425.0
ISOTHERMS = (5000, 7000, 9000)
KNOWN = [2.0, 2.25, 2.5, 2.75, 3.0, 5.0, 7.0, float("inf")]
SNAP_TOL = 0.04  # relative


def snap(r):
    if r != r:
        return r
    if np.isinf(r):
        return r
    best = min(KNOWN[:-1], key=lambda k: abs(r - k) / k)
    return best if abs(r - best) / best <= SNAP_TOL else r


def load_inventory(path):
    df = pd.read_csv(path)
    for c in ("True_T_K", "True_P_GPa", "True_V_A3"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["True_T_K", "True_P_GPa"])
    if "N_H" in df:  # composition from OUTCAR ion counts
        df["Other"] = df.Other.fillna("")
        n = df.N_H.fillna(0) + df.N_O.fillna(0)
        df = df[(df.Other == "") & (n > 0)].copy()
        df["ratio"] = np.where(df.N_O > 0, df.N_H / df.N_O.replace(0, np.nan), np.inf).round(3)
    else:
        df["ratio"] = df.Path.map(parse_composition)
    df["comp"] = df.ratio.map(comp_label)
    df["ratio_snap"] = df.ratio.map(snap)
    df["comp_snap"] = df.ratio_snap.map(comp_label)
    df["P_bin"] = (np.round(df.True_P_GPa / P_BIN) * P_BIN).astype(int)
    df["T_bin"] = (np.round(df.True_T_K / T_BIN) * T_BIN).astype(int)
    return df


def hull_targets(path):
    h = pd.read_csv(path, low_memory=False)
    rows = []
    for T in ISOTHERMS:
        for suf, ratio in HULL_COMPS.items():
            col = f"P {T}K {suf}"
            if col in h:
                for p in pd.to_numeric(h[col], errors="coerce").dropna():
                    rows.append((T, comp_label(ratio), ratio, float(p)))
    return pd.DataFrame(rows, columns=["T", "comp", "ratio", "P_GPa"])


def main(inv_path, hull_path, min_steps):
    df = load_inventory(inv_path)
    df["good"] = (df.get("N_steps", 1e9) >= min_steps)
    df.to_csv("inventory_binned.csv", index=False)
    print(f"{len(df)} H/O-only runs; {int((~df.good).sum())} have < {min_steps} steps (excluded from coverage)")
    print(f"{int((df.True_P_GPa <= 0).sum())} runs with mean P <= 0")
    print("\nP (GPa) pct 0/5/25/50/75/95/100:", df.True_P_GPa.quantile([0, .05, .25, .5, .75, .95, 1]).round(0).tolist())
    print("\nruns per composition (exact -> snapped):")
    print(pd.concat([df.comp.value_counts().rename("exact"), df.comp_snap.value_counts().rename("snapped")], axis=1).fillna(0).astype(int).to_string())

    g = df[df.good].copy()
    g["Pc"] = np.clip(np.floor(g.True_P_GPa / 100) * 100, -100, 1600).astype(int)
    big = g.T_bin.value_counts()
    print("\n=== runs by T x P(100 GPa; -100 = '<0', 1600 = '1600+'), T bins with >=30 runs ===")
    print(g[g.T_bin.isin(big[big >= 30].index)].pivot_table(index="T_bin", columns="Pc", values="Path", aggfunc="count", fill_value=0).to_string())
    print("\n=== runs by snapped composition x T (T bins >=30 runs) ===")
    print(g[g.T_bin.isin(big[big >= 30].index)].pivot_table(index="comp_snap", columns="T_bin", values="Path", aggfunc="count", fill_value=0).to_string())

    if hull_path:
        tg = hull_targets(hull_path)
        tg["n_runs"] = [len(g[(g.comp_snap == r.comp) & ((g.True_T_K - r.T).abs() <= T_BIN) & ((g.True_P_GPa - r.P_GPa).abs() <= P_BIN)])
                        for r in tg.itertuples()]
        tg.to_csv("hull_target_coverage.csv", index=False)
        print(f"\nHull EOS targets covered: {(tg.n_runs > 0).sum()} / {len(tg)}  (P +-{P_BIN:.0f} GPa, T +-{T_BIN:.0f} K)")
        s = tg.groupby(["comp", "T"]).agg(targets=("n_runs", "size"), covered=("n_runs", lambda x: int((x > 0).sum())), runs=("n_runs", "sum"))
        print(s.to_string())
        print("full per-target table -> hull_target_coverage.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("hull", nargs="?")
    ap.add_argument("--min-steps", type=int, default=100)
    a = ap.parse_args()
    main(a.inv, a.hull, a.min_steps)
