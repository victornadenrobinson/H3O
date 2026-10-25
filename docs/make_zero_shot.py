"""Zero-shot benchmark figures from the numbers pasted in the session (Young A100, valid_m, cells >= 8.5 A). Indicative: 11-30 frames/model."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)

# force RMSE range (eV/A) over the T bins, from PROJECT.md zero-shot table / MACE scan notes (ranges, not per-state values)
M = [("GRACE-2L-SMAX-OMAT-medium", 1.0, 1.6), ("GRACE-2L-SMAX-OMAT-large", 1.1, 1.8), ("MatterSim 5M", 0.9, 1.8),
     ("MatterSim 1M", 1.1, 2.1), ("GRACE-1L-SMAX-OMAT-large", 1.7, 2.5), ("GRACE-3L-OMAT-large", 1.9, 3.5),
     ("MACE-OMAT-0", 3.9, 22), ("MACE mpa-0 (H2O / H-rich)", 1.6, 15), ("GRACE-2L-OMAT-large-ft-E", 6, 13),
     ("MACE mh-1 (3 PBE heads)", 0.56, 19.8)]
pd.DataFrame(M, columns=["model", "F_RMSE_min", "F_RMSE_max"]).to_csv("data/zero_shot_force_ranges.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.6))
col = lambda n: "#1b7837" if "SMAX" in n else "#2166ac" if "MatterSim" in n else "#b2182b"
for i, (n, lo, hi) in enumerate(M[::-1]):
    ax.plot([lo, hi], [i, i], lw=7, solid_capstyle="butt", color=col(n), alpha=.85)
ax.set_yticks(range(len(M))); ax.set_yticklabels([m[0] for m in M[::-1]], fontsize=8); ax.set_xscale("log")
ax.axvline(1, color="grey", ls=":", lw=1); ax.set_xlabel("force RMSE vs VASP (eV/Å), range over states")
ax.set_title("Zero-shot force error, 300-1400 GPa, 3000-13000 K"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig("figures/zero_shot_force_ranges.png", dpi=160); plt.close(fig)

# MACE mh-1, per state, three PBE heads (pasted tables; loop order omat_pbe, mp_pbe_refit_add, oc20_usemppbe)
st = ["h2.25o_3000K","h2.25o_5000K","h2.25o_7000K","h2.25o_9000K","h2.5o_3000K","h2.75o_3000K","h2o_13000K","h2o_3000K","h2o_7000K","h2o_9000K","h5o_7000K"]
F = {"omat_pbe":[0.871,1.216,5.287,19.832,5.611,8.158,6.027,0.600,0.799,3.119,1.259],
     "mp_pbe_refit_add":[0.873,1.227,4.962,19.099,5.221,7.749,5.710,0.598,0.800,2.975,1.269],
     "oc20_usemppbe":[0.864,1.235,5.108,18.999,5.499,8.219,5.776,0.565,0.776,3.010,1.177]}
dP = {"omat_pbe":[-99.8,-147.5,801.6,4064.3,860.2,1379.0,644.8,-104.1,-71.0,213.9,-213.1],
      "mp_pbe_refit_add":[-92.9,-135.5,746.5,4018.7,786.5,1340.8,601.5,-96.8,-65.6,192.3,-206.4],
      "oc20_usemppbe":[-59.0,-90.2,862.6,4348.4,968.8,1601.9,716.0,-63.0,-47.6,235.9,-132.5]}
rows = [dict(state=s, head=h, F_RMSE=F[h][i], dP_GPa=dP[h][i]) for h in F for i, s in enumerate(st)]
pd.DataFrame(rows).to_csv("data/mace_mh1_zero_shot.csv", index=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True); y = np.arange(len(st)); w = .27
for k, (h, c) in enumerate(zip(F, ["#1b7837", "#762a83", "#e08214"])):
    a.barh(y + (k - 1) * w, F[h], w, color=c, label=h); b.barh(y + (k - 1) * w, np.sign(dP[h]) * np.log10(1 + np.abs(dP[h])), w, color=c)
a.set_yticks(y); a.set_yticklabels(st, fontsize=8); a.invert_yaxis(); a.set_xlabel("force RMSE (eV/Å)"); a.legend(fontsize=8)
b.set_xlabel("pressure error, signed log10(1+|ΔP/GPa|)   (2 = 100 GPa, 3 = 1000 GPa)"); b.axvline(0, color="k", lw=.6)
fig.suptitle("MACE mh-1 PBE heads, zero-shot: every head fails on hot, dense, H-rich states"); fig.tight_layout()
fig.savefig("figures/mace_mh1_zero_shot.png", dpi=160); plt.close(fig)
