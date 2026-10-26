"""Write small, committable summaries of the DFT inventory so they can be read from git instead of re-querying the big files.
usage (on the machine that has inventory_binned.csv, repo root):  python mlp_pipeline/18b_inventory_summary.py inventory_binned.csv
Writes docs/data/inventory_runs_slim.csv (one row per run: no OUTCAR content) and docs/data/inventory_eos_fits.csv
(per composition x path phase tag x T bin with >= 5 runs: number of runs, P and V/atom ranges, V/atom at P = 100..800 GPa from a quadratic fit of ln P against V/atom, fit rms in ln P)
and docs/data/inventory_counts.csv (runs per composition x phase bucket x T bin x P bin of 100 GPa)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
from common import path_phase_tag, phase_bucket

inv = sys.argv[1] if len(sys.argv) > 1 else "inventory_binned.csv"
df = import_module("01_bin_inventory").load_inventory(inv)
df = df.assign(vpa=df.True_V_A3 / df.N_ions, phase_tag=df.Path.map(path_phase_tag), bucket=df.Path.map(phase_bucket), P_bin100=(np.round(df.True_P_GPa / 100) * 100).astype(int))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data")   # optional 2nd argument: output directory
keep = [c for c in ("Path", "comp_snap", "N_ions", "True_T_K", "T_bin", "True_P_GPa", "True_V_A3", "vpa", "phase_tag", "bucket") if c in df.columns]
df[keep].round(4).to_csv(os.path.join(out, "inventory_runs_slim.csv"), index=False)
df.groupby(["comp_snap", "bucket", "T_bin", "P_bin100"]).size().rename("runs").reset_index().to_csv(os.path.join(out, "inventory_counts.csv"), index=False)
rows = []
for (comp, tag, T), g in df.dropna(subset=["vpa"]).groupby(["comp_snap", "phase_tag", "T_bin"]):
    g = g[np.isfinite(g.vpa) & np.isfinite(g.True_P_GPa)]   # a non-finite P or V in one bin made polyfit fail (SVD did not converge)
    if len(g) < 5 or g.vpa.nunique() < 3:
        continue
    c = np.polyfit(g.vpa, np.log(g.True_P_GPa.clip(lower=1)), 2); res = float(np.std(np.log(g.True_P_GPa.clip(lower=1)) - np.polyval(c, g.vpa)))
    grid = np.linspace(g.vpa.min(), g.vpa.max(), 400); Pg = np.exp(np.polyval(c, grid)); r = dict(comp=comp, phase_tag=tag, T_bin=T, runs=len(g), P_min=g.True_P_GPa.min(), P_max=g.True_P_GPa.max(), vpa_min=g.vpa.min(), vpa_max=g.vpa.max(), fit_rms_lnP=res)
    for P in range(100, 900, 100):
        r[f"vpa_at_{P}GPa"] = float(np.interp(-P, -Pg, grid)) if g.True_P_GPa.min() <= P <= g.True_P_GPa.max() else np.nan
    rows.append(r)
pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "inventory_eos_fits.csv"), index=False)
print("wrote", [os.path.join(out, x) for x in ("inventory_runs_slim.csv", "inventory_counts.csv", "inventory_eos_fits.csv")])
