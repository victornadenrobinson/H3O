"""Extrapolation check on saved TDI snapshots: per-atom GRACE extrapolation grade gamma (gamma <= 1: environment similar to the model's training data).
usage (GPU, repo root, Young):  ~/grace-venv/bin/python mlp_pipeline/tdi/uq_check.py grace-ft:runs/grace/ft_m2 runs/tdi/bcc_400GPa_5000K_node*_traj.extxyz [--every 3] [--max 40]
Prints, per file (= per lambda node), the mean/max over frames of the per-frame max gamma and the fraction of atoms with gamma > 1; also the MLP energies if --compare MODEL2 is given
(difference between two fine-tuned models on the same frames = a second, independent indicator)."""
import argparse, glob, os, sys, time
import numpy as np
os.environ["GRACE_UQ"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
from ase.io import read

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("files", nargs="+"); ap.add_argument("--every", type=int, default=3, help="use every n-th frame")
ap.add_argument("--max", type=int, default=40, help="max frames per file"); ap.add_argument("--compare", default=None, help="second model spec: report energy/force differences between the two models")
ap.add_argument("--device", default="cuda"); a = ap.parse_args()
bm = import_module("04_benchmark"); calc = bm.get_calc_spec(a.model, a.device); calc2 = bm.get_calc_spec(a.compare, a.device) if a.compare else None
files = sorted(f for pat in a.files for f in glob.glob(pat)); t0 = time.time(); allg = []
print(f"{'file':60s} {'frames':>6s} {'<max g>':>8s} {'max g':>7s} {'atoms g>1':>10s}" + ("   dE (meV/atom) rms   dF rms (eV/A)" if calc2 else ""))
for f in files:
    fr = read(f, ":")[::a.every][:a.max]; gm, frac, dE, dF = [], [], [], []
    for at in fr:
        at.calc = calc; e = at.get_potential_energy(); F = at.get_forces(); g = getattr(calc, "results", {}).get("gamma")
        if g is not None:
            gm.append(float(np.max(g))); frac.append(float(np.mean(np.asarray(g) > 1)))
        if calc2:
            b = at.copy(); b.calc = calc2; dE.append((b.get_potential_energy() - e) / len(at) * 1e3); dF.append(np.sqrt(np.mean((b.get_forces() - F) ** 2)))
    allg += gm
    print(f"{os.path.basename(f):60s} {len(fr):6d} {np.mean(gm) if gm else float('nan'):8.2f} {np.max(gm) if gm else float('nan'):7.2f} {100 * np.mean(frac) if frac else float('nan'):9.2f}%"
          + (f"   {np.sqrt(np.mean(np.square(dE))):8.2f}   {np.sqrt(np.mean(np.square(dF))):8.3f}" if calc2 else ""), flush=True)
if not allg:
    print("no gamma returned by the calculator: this gracemaker version or model has no UQ output (check `calc.results` keys)")
print(f"{(time.time() - t0) / 60:.1f} min")
