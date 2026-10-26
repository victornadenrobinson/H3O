"""Summarise the MLP equation-of-state scan (runs/eos_scan/eos_*.csv written by 17_eos_md.py via young_eos_scan.sbatch).
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/22_eos_summary.py [dir] [-o docs/data/mlp_eos.csv]
Part 1: every run: composition (H per O), branch (solid / liquid / fluid), T, target pressure of the DFT-EOS volume, MLP pressure at that volume and the difference (MLP minus DFT EOS), O structure label, Lindemann ratio, O diffusion and a
verdict (solid branch must keep the fcc O lattice with D_O < 1e-5 cm2/s; liquid branch must have melted: label 'other' or D_O > 1e-5; a run that fails its branch is kept in the table but left out of the fits).
Part 2: per (composition, branch, T) the MLP volume V/atom at which the MLP pressure equals 100, 200, ..., 1200 GPa (cubic fit of ln P against ln V, inverted; 'in_range' False = outside the measured pressures: do not use)."""
import argparse, glob, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="?", default="runs/eos_scan"); ap.add_argument("-o", default=None); a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.dir, "eos_*.csv"))):
    try: rows.append(pd.read_csv(f).iloc[-1].to_dict())
    except Exception: pass
if not rows: raise SystemExit(f"no eos_*.csv in {a.dir}")
df = pd.DataFrame(rows)
for c, v in (("branch", "solid"), ("hper", 2.0), ("P_target", np.nan), ("O_struct", ""), ("delta_L", np.nan), ("D_O_cm2s", np.nan)):
    if c not in df: df[c] = v
def verdict(r):
    if r.phase == "H": return "fluid"
    solid_like = (r.O_struct == r.phase) and (r.delta_L < 0.2) and (r.D_O_cm2s < 1e-5)
    return ("solid" if solid_like else "MELTED") if r.branch == "solid" else ("NOT MELTED" if solid_like else "liquid")
df["verdict"] = [verdict(r) for r in df.itertuples()]
df["comp"] = np.where(df.phase == "H", np.inf, df.hper); df["dP_GPa"] = df.P_GPa - df.P_target
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 2000)
cols = ["hper", "phase", "branch", "T_set", "P_target", "vpa", "P_GPa", "dP_GPa", "P_sem", "O_struct", "delta_L", "D_O_cm2s", "D_H_cm2s", "verdict"]
print("every run (dP = MLP pressure at the DFT-EOS volume minus the target pressure):")
print(df.sort_values(["phase", "hper", "branch", "T_set", "P_target"])[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
ok = df[df.verdict.isin(["solid", "liquid", "fluid"])]
print("\nMLP minus DFT-EOS pressure per group (mean and largest |dP| over the targets), and how many runs failed their branch:")
g = []
for (ph, h, b, T), d in df.groupby(["phase", "hper", "branch", "T_set"]):
    dd = d[d.verdict.isin(["solid", "liquid", "fluid"])]
    g.append(dict(phase=ph, hper=h, branch=b, T=T, runs=len(d), failed_branch=len(d) - len(dd), dP_mean=dd.dP_GPa.mean(), dP_absmax=dd.dP_GPa.abs().max() if len(dd) else np.nan))
print(pd.DataFrame(g).round(1).to_string(index=False))
eos = []
for (ph, h, b, T), d in ok.groupby(["phase", "hper", "branch", "T_set"]):
    d = d.sort_values("vpa")
    if len(d) < 4: continue
    x, y = np.log(d.vpa.values), np.log(d.P_GPa.clip(lower=1).values); c = np.polyfit(x, y, min(3, len(d) - 1)); grid = np.linspace(x.min(), x.max(), 2000); Pg = np.exp(np.polyval(c, grid))
    for P in range(100, 1300, 100):
        inr = bool(Pg.min() <= P <= Pg.max())
        v = float(np.exp(np.interp(-np.log(P), -np.log(Pg), grid))) if inr else float("nan")
        eos.append(dict(phase=ph, hper=h, branch=b, T_K=T, P_GPa=P, vpa_mlp=v, in_range=inr, P_min_measured=Pg.min(), P_max_measured=Pg.max()))
eos = pd.DataFrame(eos)
if not eos.empty:
    print("\nMLP volume (A^3/atom) at the target pressures (nan = outside the measured range):")
    print(eos.pivot_table(index=["phase", "hper", "branch", "T_K"], columns="P_GPa", values="vpa_mlp").round(3).to_string())
    if a.o: eos.to_csv(a.o, index=False); print("wrote", a.o)
