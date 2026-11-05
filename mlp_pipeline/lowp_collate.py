#!/usr/bin/env python3
"""Collate the low-P MD folders and the relaxed hydrate frames into ONE extxyz of geometries to label with DFT (feeds dgcheck_jobs.sh through FRAMES_EXTXYZ).
   python mlp_pipeline/lowp_collate.py [--md runs/eos_lowp3] [--hyd runs/hydrates] [--n 2] [--nH 1] [-o runs/lowp3/dft_frames.extxyz]
Per MD volume folder: skipped (and listed) if the run is not sane: median T more than 25 % off the set T, median P above --pmax GPa (exploded), or too few frames. Otherwise --n frames spread over the
production trajectory (pure H: --nH, the cells are 686 atoms). Frames with a too-short contact (H-H < 0.45, O-H < 0.75 A) are dropped. All hydrate frames are kept (they were relaxed, contacts are checked too).
info tags: set (liquid / cold / hydrate / pureH), src (folder), T_set, vpa, P_md (median GPa, a monitoring value only)."""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from ase.io import read, write
from ase.geometry import get_distances
ap = argparse.ArgumentParser(); ap.add_argument("--md", default="runs/eos_lowp3"); ap.add_argument("--hyd", default="runs/hydrates"); ap.add_argument("--n", type=int, default=2); ap.add_argument("--nH", type=int, default=1)
ap.add_argument("--pmax", type=float, default=300.0); ap.add_argument("-o", default="runs/lowp3/dft_frames.extxyz"); a = ap.parse_args()


def contacts_ok(at):
    d = at.get_all_distances(mic=True); np.fill_diagonal(d, 9.9); Z = at.numbers; hh = d[np.ix_(Z == 1, Z == 1)].min() if (Z == 1).sum() > 1 else 9.9
    oh = d[np.ix_(Z == 8, Z == 1)].min() if (Z == 8).any() and (Z == 1).any() else 9.9
    return hh > 0.45 and oh > 0.75


out, skipped, counts = [], [], {}
for d in sorted(glob.glob(os.path.join(a.md, "*_v*"))):
    name = os.path.basename(d); m = re.search(r"_(\d+)K_v([0-9.]+)$", name); tr = os.path.join(d, "traj.extxyz"); th = os.path.join(d, "thermo.csv")
    if not (m and os.path.exists(tr) and os.path.exists(th)): skipped.append((name, "no traj/thermo")); continue
    T0, vpa = float(m.group(1)), float(m.group(2)); t = pd.read_csv(th)
    Tm, Pm = t["T"].median(), t["P_GPa"].median()
    if abs(Tm / T0 - 1) > 0.25: skipped.append((name, f"median T {Tm:.3g} K vs set {T0:.0f}")); continue
    if abs(Pm) > a.pmax: skipped.append((name, f"median P {Pm:.3g} GPa")); continue
    fr = read(tr, ":")
    if len(fr) < 4: skipped.append((name, f"only {len(fr)} frames")); continue
    pure = "_H_" in name; n = a.nH if pure else a.n; kind = "pureH" if pure else ("cold" if "_solid_" in name else "liquid")
    idx = np.unique(np.linspace(len(fr) // 2, len(fr) - 1, n).astype(int)); k = 0
    for i in idx:
        at = fr[i]
        if not contacts_ok(at): continue
        at.info.update(set=kind, src=name, T_set=T0, vpa=vpa, P_md=float(Pm), run_id=f"{name}_{i}"); out.append(at); k += 1
    counts[kind] = counts.get(kind, 0) + k
    if k == 0: skipped.append((name, "all chosen frames had too-short contacts"))
for f in sorted(glob.glob(os.path.join(a.hyd, "*_relaxed.extxyz"))):
    nm = os.path.basename(f).split("_")[0]; k = 0
    for j, at in enumerate(read(f, ":")):
        if not contacts_ok(at): skipped.append((f"{nm} frame {j}", "short contact")); continue
        at.info.update(set="hydrate", src=nm, run_id=f"{nm}_{j}"); out.append(at); k += 1
    counts["hydrate_" + nm] = k
os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True); write(a.o, out)
print(f"wrote {len(out)} geometries to {a.o}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
print("atoms per structure: " + ", ".join(f"{n} atoms x{c}" for n, c in sorted(zip(*np.unique([len(x) for x in out], return_counts=True)))))
for n, why in skipped: print(f"  SKIPPED {n}: {why}")
