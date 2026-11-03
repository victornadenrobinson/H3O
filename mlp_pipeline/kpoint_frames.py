#!/usr/bin/env python3
"""Test F2 (k-point sensitivity of the DFT labels): choose geometries from the DFT-MD frames (dft_md_labelled.py output) for 2x2x2 / 3x3x3 / Gamma(vasp_std) re-labelling. Geometry only, no VASP inputs.
Systems at one pressure: pure H (729 H), H2O (108 O + 216 H), H2.083O (225 H), H2.25O (243 H); the composition error needs pure H, H2O and H2.25O (H2.083O is the optional middle point).
Each frame carries its Gamma free energy (the cached label) as info E_gamma_eV, so no Gamma rerun is needed; k111 re-runs a few with vasp_std as a sanity check of the Gamma energies.
  python mlp_pipeline/kpoint_frames.py runs/dft_md_500.extxyz --out runs/kpt [--n222 4] [--n333 1] [--n111 1] [--set M500]
writes runs/kpt/frames_k222.extxyz (4 per system), frames_k333.extxyz (1 per key system, a subset of the k222 frames), frames_k111.extxyz (1 per key system, the same frames) and plan.txt"""
import argparse, os
import numpy as np
from ase.io import read, write
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("frames"); ap.add_argument("--out", default="runs/kpt"); ap.add_argument("--set", default="M500")
ap.add_argument("--n222", type=int, default=4); ap.add_argument("--n333", type=int, default=1); ap.add_argument("--n111", type=int, default=1); a = ap.parse_args()
SYS = {"pureH": (0, 729), "H2O": (108, 216), "H2p083O": (108, 225), "H2p25O": (108, 243)}; KEY = ("pureH", "H2O", "H2p25O")
fr = [f for f in read(a.frames, index=":") if str(f.info.get("db_set")) == a.set]; os.makedirs(a.out, exist_ok=True); sel = {}
for name, (nO, nH) in SYS.items():
    pool = [(i, f) for i, f in enumerate(fr) if int((f.numbers == 8).sum()) == nO and int((f.numbers == 1).sum()) == nH and str(f.info.get("stage", "host")) == "host"]
    if not pool: print(f"WARNING: no {name} frames in set {a.set}"); continue
    pick = np.unique(np.linspace(0, len(pool) - 1, min(a.n222, len(pool))).round().astype(int)); sel[name] = []
    for j in pick:
        i, f = pool[j]; g = f.copy(); E = float(f.get_potential_energy()); g.calc = None; g.info = dict(T_K=float(f.info.get("T_K", 5000.0)), config_type=f"kpt_{name}", kpt_sys=name, kpt_key=f"{name}:{i}", E_gamma_eV=E, P_target=float(f.info.get("P_target", float("nan"))), nO=nO, nH=nH)
        sel[name].append(g)
def dump(fn, frames):
    if os.path.exists(fn): os.remove(fn)
    write(fn, frames)
all222 = [g for n in SYS for g in sel.get(n, [])]; dump(os.path.join(a.out, "frames_k222.extxyz"), all222)
sub = {n: sel[n][: max(a.n333, a.n111)] for n in KEY if n in sel}
dump(os.path.join(a.out, "frames_k333.extxyz"), [g for n in sub for g in sub[n][: a.n333]]); dump(os.path.join(a.out, "frames_k111.extxyz"), [g for n in sub for g in sub[n][: a.n111]])
txt = "\n".join(f"{n:8s} {len(sel[n])} frames, {len(sel[n][0])} atoms" for n in sel) + f"\nk222: {len(all222)} points; k333: {sum(len(v[:a.n333]) for v in sub.values())}; k111 (vasp_std Gamma): {sum(len(v[:a.n111]) for v in sub.values())}\n"
open(os.path.join(a.out, "plan.txt"), "w").write(txt); print(txt, end="")
