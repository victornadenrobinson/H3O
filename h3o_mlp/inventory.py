"""Ingest the mined VASP inventory, bin it in (x, P, T) and check coverage."""
import re

import numpy as np
import pandas as pd

P_BIN_GPA = 25.0
T_BIN_K = 250.0

_COMP = [
    (re.compile(r"pure[-_]?h2(?![0-9.]*o)", re.I), 1.0),
    (re.compile(r"h(\d+(?:\.\d+)?)o", re.I), None),
]


def hydrogen_fraction_from_path(path):
    """Fraction x of H2 in the mixture (1-x) H2O + x H2, parsed from a path.

    H2O -> 0.0, H3O -> 1/3, H5O -> 0.6, pure H2 -> 1.0; NaN if unrecognised.
    """
    path = str(path)
    if _COMP[0][0].search(path):
        return 1.0
    m = _COMP[1][0].search(path)
    if not m:
        return np.nan
    h = float(m.group(1))
    excess_h2 = (h - 2.0) / 2.0
    return excess_h2 / (1.0 + excess_h2)


def load_inventory(csv_path):
    df = pd.read_csv(csv_path)
    for c in ("True_T_K", "True_P_GPa", "True_V_A3"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["True_T_K", "True_P_GPa"]).copy()
    df["x"] = df["Path"].map(hydrogen_fraction_from_path)
    return bin_inventory(df)


def bin_inventory(df, p_bin=P_BIN_GPA, t_bin=T_BIN_K):
    df = df.copy()
    df["P_bin"] = np.round(df["True_P_GPa"] / p_bin) * p_bin
    df["T_bin"] = np.round(df["True_T_K"] / t_bin) * t_bin
    df["x_bin"] = df["x"].round(2)
    return df


def coverage(inv, hull_csv, x_col, p_col, t_col, p_bin=P_BIN_GPA, t_bin=T_BIN_K,
             isotherms=(5000, 7000, 9000)):
    """Report, per hull (x, P, T) state, how many mined runs fall in the same bin."""
    hull = pd.read_csv(hull_csv, usecols=[x_col, p_col, t_col]).dropna()
    hull.columns = ["x", "P", "T"]
    hull = hull[hull["T"].round().isin(isotherms)]
    hull = bin_inventory(
        hull.rename(columns={"P": "True_P_GPa", "T": "True_T_K"}), p_bin, t_bin
    ).drop_duplicates(["x_bin", "P_bin", "T_bin"])
    counts = inv.groupby(["x_bin", "P_bin", "T_bin"]).size().rename("n_runs").reset_index()
    out = hull[["x_bin", "P_bin", "T_bin"]].merge(
        counts, on=["x_bin", "P_bin", "T_bin"], how="left"
    )
    out["n_runs"] = out["n_runs"].fillna(0).astype(int)
    return out
