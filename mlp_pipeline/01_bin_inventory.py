"""Bin mined runs by composition / P / T and check coverage of hull-CSV EOS targets.

usage: python 01_bin_inventory.py mined_inventory_ground_truth.csv data-h3o-convex-hull.csv
"""
import sys
import numpy as np
import pandas as pd
from common import parse_composition, comp_label, HULL_COMPS

P_BIN, T_BIN = 25.0, 250.0
P_MIN, P_MAX = 250.0, 1425.0
ISOTHERMS = (5000, 7000, 9000)


def load_inventory(path):
    df = pd.read_csv(path)
    for c in ("True_T_K", "True_P_GPa", "True_V_A3"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["True_T_K", "True_P_GPa"])
    print(f"{int((df.True_P_GPa <= 0).sum())} of {len(df)} runs have P<=0 -> pressure mining likely failed; re-run 00_mine_outcars.py")
    if df.True_P_GPa.median() > 3000:
        print("WARNING: median pressure >3000 -> looks like kB; dividing by 10")
        df["True_P_GPa"] *= 0.1
    if "N_H" in df:  # inventory from 00_mine_outcars.py: composition from OUTCAR ion counts
        df["Other"] = df.Other.fillna("")
        n = df.N_H.fillna(0) + df.N_O.fillna(0)
        df = df[(df.Other == "") & (n > 0)].copy()
        df["ratio"] = np.where(df.N_O > 0, df.N_H / df.N_O.replace(0, np.nan), np.inf).round(3)
    else:
        df["ratio"] = df.Path.map(parse_composition)
    df["comp"] = df.ratio.map(comp_label)
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


def main(inv_path, hull_path=None):
    df = load_inventory(inv_path)
    df.to_csv("inventory_binned.csv", index=False)
    print(f"{len(df)} runs; unknown composition: {(df.comp == 'unknown').sum()}")
    print("\nP (GPa) percentiles:", df.True_P_GPa.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).tolist())
    print("T (K) percentiles:", df.True_T_K.quantile([0, .05, .25, .5, .75, .95, 1]).round(0).tolist())
    print("P<=0 or V missing:", int((df.True_P_GPa <= 0).sum()), int(df.True_V_A3.isna().sum()))
    print("\nruns per T bin (all):\n", df.T_bin.value_counts().sort_index().to_string())
    print("\nruns per 100-GPa bin (all):\n", (np.floor(df.True_P_GPa / 100) * 100).astype(int).value_counts().sort_index().to_string())
    print("\nruns per composition:\n", df.comp.value_counts().to_string())
    print("\n=== ALL T x P(100 GPa bins), runs ===")
    df["P100"] = (np.floor(df.True_P_GPa / 100) * 100).astype(int)
    print(df.pivot_table(index="T_bin", columns="P100", values="Path", aggfunc="count", fill_value=0).to_string())
    unk = df[df.comp == "unknown"]
    if len(unk):
        unk.Path.sample(min(30, len(unk)), random_state=0).to_csv("unknown_comp_sample.txt", index=False, header=False)
        print("\nsample of unknown-composition paths -> unknown_comp_sample.txt")
    for T in ISOTHERMS:
        s = df[(df.T_bin == T) & df.True_P_GPa.between(P_MIN, P_MAX)]
        print(f"\n=== {T} K, {P_MIN:.0f}-{P_MAX:.0f} GPa (runs per comp x P bin) ===")
        if len(s):
            print(s.pivot_table(index="comp", columns="P_bin", values="Path", aggfunc="count", fill_value=0).to_string())
        else:
            print("(none)")
    if hull_path:
        tg = hull_targets(hull_path)
        counts = []
        for r in tg.itertuples():
            m = df[(df.comp == r.comp) & ((df.True_T_K - r.T).abs() <= T_BIN) & ((df.True_P_GPa - r.P_GPa).abs() <= P_BIN)]
            counts.append(len(m))
        tg["n_runs"] = counts
        tg.to_csv("hull_target_coverage.csv", index=False)
        print(f"\nHull EOS targets covered: {(tg.n_runs > 0).sum()} / {len(tg)}")
        print("Uncovered:\n", tg[tg.n_runs == 0].to_string(index=False))
        print("\nNote: hull has no H2.5O / H2.75O EOS columns.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
