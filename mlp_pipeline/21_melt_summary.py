"""Summarise the stability scan (runs/melt/*.csv written by 17_eos_md.py): per phase, pressure and temperature the O-sublattice label, Lindemann ratio of the time-averaged lattice,
O and H diffusion constants and the mean pressure. usage: python3 mlp_pipeline/21_melt_summary.py runs/melt
Verdict in the last column (tdi/verdict.py, the rule shared with plot_pt.py / melt_table.py / summarize.py): 'solid' if the label equals the starting phase, D_O < 1e-5 cm2/s and delta_L < 0.2;
'marginal' the same with 0.2 <= delta_L < 0.5; 'MELTED' otherwise; 'unknown' too few frames (superheating caveat: see young_melt_scan.sbatch)."""
import glob, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdi.verdict import verdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs/melt"
rows = []
for f in sorted(glob.glob(os.path.join(d, "melt_*.csv"))):
    r = pd.read_csv(f).iloc[-1].to_dict(); m = re.match(r"melt_(bcc|fcc)_(\d+)GPa_(\d+)K", os.path.basename(f))
    r.update(phase_start=m.group(1), P_target=int(m.group(2)), T_target=int(m.group(3))); rows.append(r)
if not rows:
    sys.exit(f"no melt_*.csv in {d}")
df = pd.DataFrame(rows)
df["verdict"] = [verdict(r, phase=r["phase_start"]).replace("melted", "MELTED") for _, r in df.iterrows()]
cols = ["phase_start", "P_target", "T_target", "P_GPa", "T_mean", "O_struct", "CN1", "CN2", "delta_L", "D_O_cm2s", "D_H_cm2s", "msd_O_end", "ps", "verdict"]
pd.set_option("display.width", 220)
print(df[[c for c in cols if c in df.columns]].sort_values(["phase_start", "P_target", "T_target"]).round(4).to_string(index=False))
