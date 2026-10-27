"""Did the liquid TDI runs stay liquid? O self-diffusion from the saved configurations of the highest lambda values (closest to the real system): D_O from the O mean-squared displacement between the stored frames of the production run
(<tag>_node<k>_traj.extxyz, a frame every 20 recorded samples; displacements between consecutive frames are taken with the minimum-image convention and summed, so wrapping does not matter).
usage (repo root, needs ASE: ~/grace-venv/bin/python):  python mlp_pipeline/tdi/liquid_check.py [dir ...] [--lambdas 2] [--dmin 1e-5]      default dir runs/tdi, tags liquid_*
       solid side (insertion hosts, crystals): ... liquid_check.py runs/tdi runs/tdi_long --glob 'insert_*' --lambdas 3     (verdict solid / MELTED?: D_O and O Lindemann ratio)
D_O = MSD / (6 t) over the second half of the stored frames, in cm2/s (1 A^2/ps = 1e-4 cm2/s). Verdict: 'liquid' if D_O > --dmin (the free-NVT rule of tdi/verdict.py), 'frozen?' if below: the lattice may have re-formed after the hot step or the run is too short
to see diffusion (a 2 ps run at D_O = 1e-5 cm2/s moves O by only sqrt(6 D t) = 0.35 A: the test is blind at low D). Also printed: the O MSD at the end (A^2; a liquid shows several A^2, a crystal about 0.1) and the number of frames."""
import argparse, glob, json, os, sys
import numpy as np
from ase.io import read
from ase.geometry import find_mic

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("--lambdas", type=int, default=2, help="how many of the highest lambda values to check"); ap.add_argument("--dmin", type=float, default=1e-5)
ap.add_argument("--glob", default="liquid_*", help="tags to check: liquid_* (did it stay liquid?), or insert_* / fcc_* / bcc_* (did the host / crystal stay SOLID at the high-lambda nodes?)")
ap.add_argument("--lind-max", type=float, default=0.2, help="solid runs: O Lindemann ratio above this (or D_O > --dmin) = MELTED?"); a = ap.parse_args()
want_liquid = a.glob.startswith("liquid")
print(f"{'run':44s} {'lambda':>6s} {'frames':>6s} {'span ps':>8s} {'MSD_O end A^2':>14s} {'D_O cm2/s':>11s} {'Lindemann':>9s}  verdict")
for d in (a.dirs or ["runs/tdi"]):
    for f0 in sorted(glob.glob(os.path.join(d, a.glob + "_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; A = json.load(open(f0)).get("args", {}); nodes = int(A["nodes"]); dtf = A["dt"] * A["every"] * 20 / 1000   # ps between stored frames
        for k in range(nodes - 1, max(nodes - 1 - a.lambdas, -1), -1):
            f = os.path.join(d, f"{tag}_node{k}_traj.extxyz")
            if not os.path.exists(f): print(f"{d}/{tag:{43 - len(d)}s} {k:6d}  no trajectory file"); continue
            try: fr = read(f, ":")
            except Exception as ex: print(f"{tag} node {k}: cannot read ({ex})"); continue
            iO = np.where(fr[0].numbers == 8)[0]
            if len(fr) < 4 or len(iO) == 0: print(f"{tag} node {k}: {len(fr)} frames: too few"); continue
            R = np.zeros((len(fr), len(iO), 3))
            for i in range(1, len(fr)): R[i] = R[i - 1] + find_mic(fr[i].positions[iO] - fr[i - 1].positions[iO], fr[0].cell, True)[0]
            R -= R.mean(1, keepdims=True)   # remove the O centre-of-mass drift (insertion hosts and fluids have no fixed COM)
            msd = ((R - R[0]) ** 2).sum(-1).mean(1); t = np.arange(len(fr)) * dtf; h = len(fr) // 2
            slope = np.polyfit(t[h:], msd[h:], 1)[0]; D = slope / 6 * 1e-4
            u = R - R.mean(0); d1 = (4 * fr[0].get_volume() / len(iO)) ** (1 / 3) / np.sqrt(2)   # O-O nearest-neighbour distance of an fcc lattice at this O density
            lind = float(np.sqrt((u ** 2).sum(-1).mean()) / d1)
            verdict = ("liquid" if D > a.dmin else "frozen?") if want_liquid else ("solid" if D <= a.dmin and lind <= a.lind_max else "MELTED?")
            print(f"{d + '/' + tag:44s} {k:6d} {len(fr):6d} {t[-1]:8.2f} {msd[-1]:14.2f} {D:11.2e} {lind:9.3f}  {verdict}")
