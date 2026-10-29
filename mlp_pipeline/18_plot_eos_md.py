"""Plot MLP NVT-MD EOS points (from 17_eos_md.py) against the DFT runs of the same phase and temperature.
usage: python 18_plot_eos_md.py inventory_binned.csv runs/eos_md/ft_s_bcc_3000K.csv [runs/eos_md/ft_m_bcc_3000K.csv ...] [--phase bcc --T 3000] [--zero-shot-csv ...]
Left: V/atom versus P (P on x; DFT runs grey, quadratic fit of ln P, MLP points). Middle: P_MLP - P_DFTfit at the same V. Right: volume error (V_MLP - V_DFT)/V_DFT in % at the same pressure.
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
g = np.linspace(df.vpa.min(), df.vpa.max(), 400); Pg = fit(g)
Vdft = lambda P: np.interp(P, Pg[::-1], g[::-1])            # DFT fit inverted: V/atom at a given P (P falls with V)
fig, (A, B, C) = plt.subplots(1, 3, figsize=(14, 4.2))
A.scatter(df.True_P_GPa, df.vpa, s=12, c="#9a9a94", label=f"DFT runs ({len(df)})"); A.plot(Pg, g, color="#1a1a19", lw=1, label="DFT fit")
cols = ["#2a78d6", "#d62728", "#2e9b4e", "#e66100"]
for k, f in enumerate(a.csv):
    r = pd.read_csv(f); lab = os.path.basename(f).replace(".csv", "")
    dP = r.P_GPa - fit(r.vpa)                                 # pressure error at the same volume
    dV = (r.vpa - Vdft(r.P_GPa)) / Vdft(r.P_GPa) * 100       # volume error at the same (MLP) pressure, % of the DFT volume
    A.errorbar(r.P_GPa, r.vpa, xerr=r.P_sem, fmt=["s", "D", "^", "v"][k % 4], mfc="none", mew=1.6, color=cols[k % 4], ms=9, capsize=2, label=lab, zorder=5)   # open symbols: the DFT dots stay visible underneath
    B.errorbar(fit(r.vpa), dP, r.P_sem, fmt="o-", color=cols[k % 4], ms=5, capsize=2, label=lab)
    C.plot(r.P_GPa, dV, "o-", color=cols[k % 4], ms=5, label=lab)
    print(lab); print(r.assign(dP=dP, dV_pct=dV)[["vpa", "P_GPa", "P_sem", "dP", "dV_pct", "T_mean", "D_H_cm2s", "D_O_cm2s", "msd_O_end"]].round(4).to_string(index=False))
A.set_xlabel("P (GPa), incl. N kB T / V"); A.set_ylabel("V / atom (A^3)"); A.legend(fontsize=7)
B.axhline(0, color="#1a1a19", lw=.6); B.set_xlabel("P (GPa): DFT fit at the same V"); B.set_ylabel("P_MLP - P_DFT at the same V (GPa)"); B.legend(fontsize=7)
C.axhline(0, color="#1a1a19", lw=.6); C.set_xlabel("P_MLP (GPa)"); C.set_ylabel("(V_MLP - V_DFT) / V_DFT at the same P (%)"); C.legend(fontsize=7)
fig.suptitle(f"{a.phase} H2O, {a.T:.0f} K: MLP NVT-MD vs DFT")
fig.tight_layout()
fn = a.out or outp(f"eos_md_{a.phase}_{a.T:.0f}K.png"); fig.savefig(fn, dpi=160); print("wrote", fn)
