"""State-matched pressure check: evaluate a model on the DFT frames of one phase/temperature near given volumes.
usage: python 19_dft_frame_pressure.py grace-ft:runs/grace/ft_m train_m.extxyz valid_m.extxyz --phase bcc --T 3000 --vpa 2.529 2.295 2.125 [--tol 0.12 --n 12]
For every frame: P_DFT and P_MLP are both  -tr(sigma)/3 + N kB T/V  (T = the run's mean T), so their difference is -tr(dsigma)/3, the
same dP the benchmark prints, but restricted to the volumes and phase of the MD EOS test. Frames are picked by path words (bcc/fcc).
If dP here is about 0 while 17_eos_md.py gives -20 GPa, the model is fine on DFT-sampled configurations and the bias lives in its OWN ensemble
(not in a pressure bookkeeping error); if dP here is also about -20, the model is biased on the DFT data itself.
"""
import argparse, os
from importlib import import_module
import numpy as np, pandas as pd
from ase import units
from ase.io import read
from common import inp, phase_bucket

EV_A3_GPA = 160.21766
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("data", nargs="+")
ap.add_argument("--phase", default="bcc"); ap.add_argument("--T", type=float, default=3000); ap.add_argument("--tol-T", type=float, default=250)
ap.add_argument("--vpa", type=float, nargs="+", required=True); ap.add_argument("--tol", type=float, default=0.12, help="A^3/atom window around each volume")
ap.add_argument("--n", type=int, default=40, help="max frames per volume")
ap.add_argument("--md-p", type=float, nargs="*", default=None, help="MD pressures (GPa) in the order of --vpa: print their difference to the local DFT fit and the equivalent volume error")
ap.add_argument("--inventory", default=None, help="inventory csv: also print the DFT run-level mean P of runs of this phase near each volume (the population the EOS fit is made of)"); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
frames = [f for p in a.data for f in read(inp(p), index=":")]
frames = [f for f in frames if str(f.info.get("config_type", "")).startswith("h2o_") and abs(f.info.get("T_K", 0) - a.T) <= a.tol_T
          and phase_bucket(str(f.info.get("source", ""))) == a.phase]
print(f"{len(frames)} {a.phase} h2o frames within {a.tol_T:.0f} K of {a.T:.0f} K, V/atom {min(f.get_volume() / len(f) for f in frames):.2f}-{max(f.get_volume() / len(f) for f in frames):.2f}" if frames else "no matching frames")
calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
rows = []
for v0 in a.vpa:
    sel = sorted([f for f in frames if abs(f.get_volume() / len(f) - v0) <= a.tol], key=lambda f: abs(f.get_volume() / len(f) - v0))[:a.n]
    for f in sel:
        N, V, T = len(f), f.get_volume(), f.info["T_K"]; kin = N * units.kB * T / V * EV_A3_GPA
        Pd = -np.trace(f.get_stress(voigt=False)) / 3 * EV_A3_GPA + kin
        g = f.copy(); g.calc = calc; Pm = -np.trace(g.get_stress(voigt=False)) / 3 * EV_A3_GPA + kin
        rows.append(dict(vpa_target=v0, vpa=V / N, T=T, P_dft=Pd, P_run=float(f.info.get("P_GPa", np.nan)), P_mlp=Pm, dP=Pm - Pd))
r = pd.DataFrame(rows)
if r.empty:
    raise SystemExit("no frame near the requested volumes: widen --tol or --tol-T")
t = r.groupby("vpa_target").agg(n=("dP", "size"), vpa=("vpa", "mean"), T=("T", "mean"), P_dft=("P_dft", "mean"), P_run=("P_run", "mean"), P_mlp=("P_mlp", "mean"), dP=("dP", "mean"), dP_std=("dP", "std")).round(2)
print(t.to_string())
print("P_dft = frame stress + kinetic term; P_run = the run-level mean P stored with the frame (inventory convention). If they differ by more than a few GPa on average, the frame stress and the inventory EOS are not the same quantity.")
if a.inventory:
    inv = import_module("01_bin_inventory").load_inventory(a.inventory)
    inv = inv[(inv.comp_snap == "h2o") & ((inv.True_T_K - a.T).abs() <= a.tol_T) & (inv.Path.map(phase_bucket) == a.phase)].assign(vpa=lambda d: d.True_V_A3 / d.N_ions)
    print(f"\nDFT runs of the same phase and T in the inventory ({len(inv)}): run-level mean P near each volume (window +-{a.tol:.2f} A^3/atom)")
    rr = []
    for k, v0 in enumerate(a.vpa):
        w = inv[(inv.vpa - v0).abs() <= a.tol]
        row = dict(vpa_target=v0, n_runs=len(w), P_median=w.True_P_GPa.median(), P_std=w.True_P_GPa.std())
        if len(w) >= 4:   # local linear fit P(V) of the DFT population, evaluated at the exact MD volume (window means are blurred by the steep P(V))
            c = np.polyfit(w.vpa - v0, w.True_P_GPa, 1); res = w.True_P_GPa - np.polyval(c, w.vpa - v0)
            row.update(P_lin_at_V=c[1], slope_GPa_per_A3=c[0], resid_std=res.std(ddof=2), P_lin_sem=res.std(ddof=2) / np.sqrt(len(w)))
            if a.md_p and k < len(a.md_p):
                row.update(P_md=a.md_p[k], dP=a.md_p[k] - c[1], dV_pct=(a.md_p[k] - c[1]) / abs(c[0]) / v0 * 100)
        rr.append(row)
    print(pd.DataFrame(rr).round(2).to_string(index=False))
    print("dV_pct = (P_md - P_lin) / |dP/dV| / V: the volume shift that accounts for the pressure difference.")
print("overall mean dP = %.2f GPa over %d frames" % (r.dP.mean(), len(r)))
