#!/usr/bin/env python3
"""Baseline D = E_DFT - E_model per cell on the new labels, for THREE candidate DFT energies, and force errors. GPU node, one process.
  python mlp_pipeline/baseline_D.py snapdb/labelled.extxyz snapdb_v2/labelled.extxyz --model grace-ft:runs/grace/ft_m2/seed/1/final_model --out baseline_D.csv
The three DFT energies per frame (labelled.extxyz stores F = TOTEN as the label and E_sigma0_eV):
   F   free energy TOTEN                      (consistent with the forces, what collect writes)
   E0  energy(sigma->0)  = F + TS/2
   E   energy without entropy = 2*E0 - F = F + TS
sd(D) within groups of identical state (set, T, nO, nH) shows which of them the model actually follows: the electronic-entropy term varies by several eV per cell between frames
(outcar_entropy.py), so a model fitted to the wrong one has a per-cell scatter of about that size. Forces are the same for all three (VASP forces derive from F)."""
import argparse, collections, csv, os, sys
import numpy as np
from ase.io import read
from gamebar import Bar
EV_A3_GPA = 160.21766208

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="baseline_D.csv")
ap.add_argument("--min-group", type=int, default=6); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
frames = []
for p in a.files:
    for f in read(p, index=":"):
        if "E_sigma0_eV" not in f.info and "energy_sigma0" in f.info: f.info["E_sigma0_eV"] = f.info["energy_sigma0"]        # tier files (build_tier.py) use the name of the old tiers
        if "E_sigma0_eV" in f.info: frames.append(f)
if a.limit: frames = frames[:a.limit]
frames.sort(key=len)          # same-size frames back to back: the compiled GRACE kernels are reused (a new size can trigger a recompile)
bar = Bar(len(frames), "GRACE-D"); rows = []; n = 0
if True:
    for f in frames:
        n += 1
        F = f.get_potential_energy(); E0 = float(f.info["E_sigma0_eV"]); Ens = 2 * E0 - F; Fd = f.get_forces()
        g = f.copy(); g.calc = calc; Em = float(g.get_potential_energy()); Fm = g.get_forces()
        try: Sd = np.asarray(f.get_stress()) * EV_A3_GPA; Sm = np.asarray(g.get_stress()) * EV_A3_GPA; s_rmse = float(np.sqrt(((Sd - Sm) ** 2).mean())); s_rms = float(np.sqrt((Sd ** 2).mean()))   # Voigt 6, GPa
        except Exception: s_rmse = s_rms = float("nan")
        s = str(f.info.get("db_set", "?")); T = float(f.info.get("T_K", np.nan)); nO = int((f.numbers == 8).sum()); nH = int((f.numbers == 1).sum())
        rows.append(dict(db_id=f.info.get("db_id", ""), run_id=str(f.info.get("run_id", "")), role=str(f.info.get("widom_role", "")), acq_mode=str(f.info.get("acq_mode", "")), host=str(f.info.get("host_source", "")) + "#" + str(f.info.get("host_frame", "")), n_add=f.info.get("n_add", ""), method=str(f.info.get("method", "")), set=s, split=f.info.get("split", ""), T_K=T, nO=nO, nH=nH, natoms=len(f), E_model=Em, F=F, E0=E0, E=Ens, ts=Ens - F,
                         D_F=F - Em, D_E0=E0 - Em, D_E=Ens - Em, f_rmse=float(np.sqrt(((Fd - Fm) ** 2).mean())), f_rms=float(np.sqrt((Fd ** 2).mean())), s_rmse_GPa=s_rmse, s_rms_GPa=s_rms))
        bar.update(n, f"set {s}  {len(f)} atoms")
bar.done("all frames scored")
w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); print(f"{len(rows)} frames -> {a.out}")
grp = collections.defaultdict(list)
for r in rows: grp[(r["set"], int(round(r["T_K"] / 1000.0)) * 1000 if r["T_K"] == r["T_K"] else -1, r["nO"], r["nH"])].append(r)
pooled = {k: [0.0, 0] for k in ("D_F", "D_E0", "D_E")}
for v in grp.values():
    if len(v) < a.min_group: continue
    for k in pooled:
        x = np.array([r[k] for r in v]); pooled[k][0] += ((x - x.mean()) ** 2).sum(); pooled[k][1] += len(v) - 1
print(f"\npooled within-group sd(D) per cell [eV] over groups of identical (set, T, nO, nH) with >= {a.min_group} frames "
      f"({sum(1 for v in grp.values() if len(v) >= a.min_group)} groups):")
for k, (ss, dof) in pooled.items(): print(f"   {k:5s} {np.sqrt(ss / max(dof, 1)):7.3f}")
print("\nper set (frames, force RMSE / force RMS, sd of D_F / D_E0 / D_E within the set's groups):")
for s in sorted({r["set"] for r in rows}):
    v = [r for r in rows if r["set"] == s]; sd = {}
    for k in ("D_F", "D_E0", "D_E"):
        ss = dof = 0
        for key, g_ in grp.items():
            if key[0] == s and len(g_) >= a.min_group: x = np.array([r[k] for r in g_]); ss += ((x - x.mean()) ** 2).sum(); dof += len(g_) - 1
        sd[k] = np.sqrt(ss / dof) if dof else float("nan")
    print(f"   {s:8s}{len(v):6d}   force {np.mean([r['f_rmse'] for r in v]):.3f}/{np.mean([r['f_rms'] for r in v]):.3f} eV/A ({100 * np.mean([r['f_rmse'] / r['f_rms'] for r in v]):.1f}%)   sd D_F {sd['D_F']:.2f}  D_E0 {sd['D_E0']:.2f}  D_E {sd['D_E']:.2f}")

# ---- the usual MLP test-set table: energy MAE / RMSE per atom, force RMSE per component, stress RMSE (all raw errors vs DFT, offsets included; sd(D) above removes the group mean) ----
def _row(v):
    e = np.array([r["D_F"] / r["natoms"] for r in v]) * 1000.0; fr = np.array([r["f_rmse"] for r in v]); fm = np.array([r["f_rms"] for r in v]); sr = np.array([r["s_rmse_GPa"] for r in v]); sm = np.array([r["s_rms_GPa"] for r in v])
    na = np.array([r["natoms"] for r in v], float)
    # force RMSE pooled over ALL components (each frame weighted by its number of components), like the training log's rmse/f_comp
    fp = float(np.sqrt((fr ** 2 * na).sum() / na.sum()))
    return len(v), np.sqrt((e ** 2).mean()), np.abs(e).mean(), e.mean(), fp, 100.0 * fr.mean() / fm.mean(), np.sqrt(np.nanmean(sr ** 2)), 100.0 * np.nanmean(sr) / max(np.nanmean(sm), 1e-9)
print("\ntest-set errors per set (raw vs DFT: E in meV/atom of the FREE energy, F per component in eV/A, stress RMSE over the 6 Voigt components in GPa):")
print(f"   {'set':8s}{'n':>6s}{'E RMSE':>9s}{'E MAE':>8s}{'E bias':>8s}{'F RMSE':>9s}{'F rel%':>8s}{'S RMSE':>9s}{'S rel%':>8s}")
for s_ in sorted({r["set"] for r in rows}) + ["ALL"]:
    v = rows if s_ == "ALL" else [r for r in rows if r["set"] == s_]; n_, er, ea, eb, fp, fr_, sr_, srr = _row(v)
    print(f"   {s_:8s}{n_:6d}{er:9.2f}{ea:8.2f}{eb:+8.2f}{fp:9.3f}{fr_:8.1f}{sr_:9.2f}{srr:8.1f}")
