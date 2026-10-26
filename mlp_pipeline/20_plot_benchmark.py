"""Compare models on the same validation frames from the CSVs written by 04_benchmark.py.
usage: python 20_plot_benchmark.py bench_A.csv bench_B.csv ... [--labels "zero-shot" "FT small" "FT medium"] [--out runs/bench_compare.png]
Panels: energy error vs T (box: per-frame signed error, meV/atom, with kB*T guides), force RMSE vs T, pressure error vs P_DFT, pressure parity,
energy parity (per atom, after the per-element shift), and a summary table. Needs the columns e_signed / P_virial_* (04_benchmark.py from 2026-10-26 on);
older CSVs still give the force, stress and dP panels.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import out as outp

KB = 8.617333e-2   # meV/K
ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--labels", nargs="*"); ap.add_argument("--out", default=None)
a = ap.parse_args()
labs = a.labels if a.labels and len(a.labels) == len(a.csv) else [os.path.basename(f).replace("bench_", "").replace(".csv", "") for f in a.csv]
dfs = []
for f, l in zip(a.csv, labs):
    d = pd.read_csv(f); d["model"] = l; d["F_rmse"] = np.sqrt(d.f_se); dfs.append(d)
cols = ["#8a8a85", "#2a78d6", "#d62728", "#2e9b4e", "#e66100"]
Ts = sorted(set(pd.concat(dfs).T.dropna().astype(int)) if False else set(int(t) for d in dfs for t in d["T"].dropna().unique()))
has_e = all("e_signed" in d for d in dfs); has_p = all("P_virial_mlp" in d for d in dfs)
fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
w = 0.8 / len(dfs)
for k, d in enumerate(dfs):
    pos = np.arange(len(Ts)) + (k - (len(dfs) - 1) / 2) * w
    if has_e:
        data = [d[d["T"] == t].e_signed.values * 1e3 for t in Ts]
        b = ax[0].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for p in b["boxes"]: p.set_facecolor(cols[k % 5]); p.set_alpha(.7)
    data = [d[d["T"] == t].F_rmse.values for t in Ts]
    b = ax[1].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
    for p in b["boxes"]: p.set_facecolor(cols[k % 5]); p.set_alpha(.7)
for i in (0, 1):
    ax[i].set_xticks(range(len(Ts))); ax[i].set_xticklabels([f"{t}" for t in Ts]); ax[i].set_xlabel("T (K)")
if has_e:
    ax[0].plot(range(len(Ts)), [KB * t for t in Ts], "k:", lw=1, label="k_B T"); ax[0].plot(range(len(Ts)), [-KB * t for t in Ts], "k:", lw=1)
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_ylabel("energy error, model - DFT (meV/atom)"); ax[0].set_ylim(-400, 400); ax[0].legend(fontsize=7)
else:
    ax[0].text(.5, .5, "no e_signed column:\nre-run 04_benchmark.py", ha="center", transform=ax[0].transAxes)
ax[1].set_ylabel("force RMSE per frame (eV/A)"); ax[1].set_yscale("log")
for k, d in enumerate(dfs):
    c = cols[k % 5]
    ax[2].scatter(d.P_dft, d.dP, s=10, color=c, alpha=.7, label=labs[k])
    if has_p:
        ax[3].scatter(d.P_virial_dft, d.P_virial_mlp, s=10, color=c, alpha=.7, label=labs[k])
    if has_e:
        ax[4].scatter(d.E_dft, d.E_mlp_shifted, s=10, color=c, alpha=.7, label=labs[k])
ax[2].axhline(0, color="k", lw=.6); ax[2].set_xlabel("P_DFT (GPa, incl. kinetic term)"); ax[2].set_ylabel("P_model - P_DFT (GPa)"); ax[2].legend(fontsize=7)
if has_p:
    lo, hi = min(d.P_virial_dft.min() for d in dfs), max(d.P_virial_dft.max() for d in dfs); ax[3].plot([lo, hi], [lo, hi], "k-", lw=.8)
    ax[3].set_xlabel("virial pressure, DFT (GPa)"); ax[3].set_ylabel("virial pressure, model (GPa)"); ax[3].legend(fontsize=7)
if has_e:
    lo, hi = min(d.E_dft.min() for d in dfs), max(d.E_dft.max() for d in dfs); ax[4].plot([lo, hi], [lo, hi], "k-", lw=.8)
    ax[4].set_xlabel("E / atom, DFT (eV)"); ax[4].set_ylabel("E / atom, model after shift (eV)")
rows = []
for k, d in enumerate(dfs):
    r = dict(model=labs[k], frames=len(d), F_RMSE_eV_A=float(np.sqrt(d.f_se.mean())), S_MAE_GPa=float(d.s_ae.mean()), dP_mean_GPa=float(d.dP.mean()), dP_MAE_GPa=float(d.dP.abs().mean()))
    if has_e:
        r.update(E_MAE_meV=float(d.e_ae.mean() * 1e3), E_bias_meV=float(d.e_signed.mean() * 1e3))
        r["E_MAE_over_kT"] = float((d.e_ae * 1e3 / (KB * d["T"])).mean())
    rows.append(r)
tab = pd.DataFrame(rows).round(3); print(tab.to_string(index=False))
ax[5].axis("off"); ax[5].text(0, 1, tab.set_index("model").T.to_string(), family="monospace", fontsize=7, va="top")
fig.tight_layout(); fn = a.out or outp("bench_compare.png"); fig.savefig(fn, dpi=150); print("wrote", fn)
