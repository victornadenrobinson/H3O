"""Plot MLP NVT-MD EOS points (from 17_eos_md.py) against the DFT runs of the same phase and temperature.
usage: python 18_plot_eos_md.py inventory_binned.csv runs/eos_md/ft_s_bcc_3000K.csv [runs/eos_md/ft_m_bcc_3000K.csv ...] [--phase bcc --T 3000] [--zero-shot-csv ...]
Left: P vs V/atom (DFT runs grey, quadratic fit of ln P line, MLP points with block error bars). Right: P_MLP - P_DFTfit at the same V/atom.
"""
import argparse, os
from importlib import import_module
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import out as outp

ap = argparse.ArgumentParser(); ap.add_argument("inv"); ap.add_argument("csv", nargs="+")
ap.add_argument("--phase", default="bcc"); ap.add_argument("--T", type=float, default=3000); ap.add_argument("--out", default=None)
a = ap.parse_args()
df = import_module("01_bin_inventory").load_inventory(a.inv)
df = df[(df.comp_snap == "h2o") & (df.T_bin == int(round(a.T / 1000.0)) * 1000) & df.Path.str.contains(a.phase, case=False) & ~df.Path.str.contains("liquid|solid|melt", case=False)]
df = df.assign(vpa=df.True_V_A3 / df.N_ions).dropna(subset=["vpa"])
c = np.polyfit(df.vpa, np.log(df.True_P_GPa.clip(lower=1)), 2); fit = lambda v: np.exp(np.polyval(c, v))
g = np.linspace(df.vpa.min(), df.vpa.max(), 200)
fig, (A, B) = plt.subplots(1, 2, figsize=(10, 4.2))
A.scatter(df.vpa, df.True_P_GPa, s=8, c="#b8b8b2", label=f"DFT runs ({len(df)})"); A.plot(g, fit(g), color="#1a1a19", lw=1, label="DFT fit")
cols = ["#2a78d6", "#d62728", "#2e9b4e", "#e66100"]
for k, f in enumerate(a.csv):
    r = pd.read_csv(f); lab = os.path.basename(f).replace(".csv", "")
    A.errorbar(r.vpa, r.P_GPa, r.P_sem, fmt="o", color=cols[k % 4], ms=5, capsize=2, label=lab)
    B.errorbar(fit(r.vpa), r.P_GPa - fit(r.vpa), r.P_sem, fmt="o-", color=cols[k % 4], ms=5, capsize=2, label=lab)
    print(lab); print(r.assign(dP=r.P_GPa - fit(r.vpa))[["vpa", "P_GPa", "P_sem", "dP", "T_mean", "D_H_cm2s", "D_O_cm2s", "msd_O_end"]].round(4).to_string(index=False))
A.set_xlabel("V / atom (A^3)"); A.set_ylabel("P (GPa), incl. N kB T / V"); A.set_yscale("log"); A.legend(fontsize=7)
B.axhline(0, color="#1a1a19", lw=.6); B.set_xlabel("P_DFT-fit at the same V (GPa)"); B.set_ylabel("P_MLP - P_DFT (GPa)"); B.legend(fontsize=7)
fig.suptitle(f"{a.phase} H2O, {a.T:.0f} K: MLP NVT-MD vs DFT"); fig.tight_layout()
fn = a.out or outp(f"eos_md_{a.phase}_{a.T:.0f}K.png"); fig.savefig(fn, dpi=160); print("wrote", fn)
