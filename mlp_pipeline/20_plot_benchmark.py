"""Compare models on the same validation frames from the CSVs written by 04_benchmark.py.
usage: python 20_plot_benchmark.py bench_A.csv bench_B.csv ... [--labels "zero-shot" "FT small" "FT medium"] [--out runs/bench_compare.png]
Top row: energy error and force RMSE versus T (boxes per model), summary table. Then one column per model: pressure error versus P_DFT and energy parity
(per atom, after the per-element shift), points coloured by composition (viridis: pure H -> H2O). Needs the columns e_signed / P_virial_* (04_benchmark.py from 2026-10-26 on);
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
cols = ["#8a8a85", "#2a78d6", "#d62728", "#2e9b4e", "#e66100", "#7f5539"]
Ts = sorted(set(int(t) for d in dfs for t in d["T"].dropna().unique()))
has_e = all("e_signed" in d for d in dfs); has_p = all("P_virial_mlp" in d for d in dfs)
for d in dfs:
    d["cmp"] = d.comp.astype(str).str.split("_").str[0]
ORDER = ["h", "h7o", "h5o", "h3o", "h2.75o", "h2.5o", "h2.25o", "h2o"]      # from pure H to H2O
present = [c for c in ORDER if any((d.cmp == c).any() for d in dfs)] + sorted(set(c for d in dfs for c in d.cmp) - set(ORDER))
cc = {c: plt.cm.viridis(i / max(len(present) - 1, 1)) for i, c in enumerate(present)}
nm = len(dfs); nc = max(3, nm)
fig = plt.figure(figsize=(5.2 * nc, 15)); gs = fig.add_gridspec(3, nc)
A0, A1, A2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
w = 0.8 / nm
for k, d in enumerate(dfs):
    pos = np.arange(len(Ts)) + (k - (nm - 1) / 2) * w
    for axx, col, scale in ((A0, "e_signed", 1e3), (A1, "F_rmse", 1.0)):
        if col not in d:
            continue
        b = axx.boxplot([d[d["T"] == t][col].values * scale for t in Ts], positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for p in b["boxes"]:
            p.set_facecolor(cols[k % 6]); p.set_alpha(.7)
for axx in (A0, A1):
    axx.set_xticks(range(len(Ts))); axx.set_xticklabels([f"{t}" for t in Ts]); axx.set_xlabel("T (K)")
if has_e:
    A0.plot(range(len(Ts)), [KB * t for t in Ts], "k:", lw=1, label="+-k_B T"); A0.plot(range(len(Ts)), [-KB * t for t in Ts], "k:", lw=1)
    A0.axhline(0, color="k", lw=.6); A0.set_ylabel("energy error, model - DFT (meV/atom)"); A0.legend(fontsize=7)
    lim = max(np.abs(np.concatenate([d.e_signed.values for d in dfs])) * 1e3); A0.set_ylim(-min(1.2 * lim, 500), min(1.2 * lim, 500))
else:
    A0.text(.5, .5, "no e_signed column:\nre-run 04_benchmark.py", ha="center", transform=A0.transAxes)
A1.set_ylabel("force RMSE per frame (eV/A)"); A1.set_yscale("log")
for k in range(nm):
    A0.plot([], [], "s", color=cols[k % 6], label=labs[k]); 
A1.legend([plt.Rectangle((0, 0), 1, 1, fc=cols[k % 6], alpha=.7) for k in range(nm)], labs, fontsize=7)
rows = []
for k, d in enumerate(dfs):
    r = dict(model=labs[k], frames=len(d), F_RMSE_eV_A=float(np.sqrt(d.f_se.mean())), S_MAE_GPa=float(d.s_ae.mean()), dP_mean_GPa=float(d.dP.mean()), dP_MAE_GPa=float(d.dP.abs().mean()))
    if has_e:
        r.update(E_MAE_meV=float(d.e_ae.mean() * 1e3), E_bias_meV=float(d.e_signed.mean() * 1e3))
        r["E_MAE_over_kT"] = float((d.e_ae * 1e3 / (KB * d["T"])).mean())
    rows.append(r)
tab = pd.DataFrame(rows).round(3); print(tab.to_string(index=False))
A2.axis("off"); A2.text(0, 1, tab.set_index("model").T.to_string(), family="monospace", fontsize=7, va="top")
# rows 2 and 3: one column per model, coloured by composition
ymax = max(np.abs(d.dP).max() for d in dfs)
for k, d in enumerate(dfs):
    B = fig.add_subplot(gs[1, k]); C = fig.add_subplot(gs[2, k])
    for c in present:
        m = d[d.cmp == c]
        if len(m):
            B.scatter(m.P_dft, m.dP, s=16, color=cc[c], alpha=.85, label=c, edgecolor="none")
            if has_e:
                C.scatter(m.E_dft, m.E_mlp_shifted, s=16, color=cc[c], alpha=.85, edgecolor="none")
    B.axhline(0, color="k", lw=.6); B.set_ylim(-1.05 * ymax, 1.05 * ymax); B.set_title(labs[k]); B.set_xlabel("P_DFT (GPa, incl. kinetic term)"); B.set_ylabel("P_model - P_DFT (GPa)")
    if has_e:
        lo, hi = min(x.E_dft.min() for x in dfs), max(x.E_dft.max() for x in dfs); C.plot([lo, hi], [lo, hi], "k-", lw=.8)
        C.set_xlabel("E / atom, DFT (eV)"); C.set_ylabel("E / atom, model after the per-element shift (eV)"); C.set_title(labs[k])
    if k == 0:
        B.legend(fontsize=7, title="composition (H:O)", ncol=2)
fig.tight_layout(); fn = a.out or outp("bench_compare.png"); fig.savefig(fn, dpi=140); print("wrote", fn)
