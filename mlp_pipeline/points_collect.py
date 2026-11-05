#!/usr/bin/env python3
"""Collect finished single-point VASP directories made by prepare_points.py (e.g. runs/dgcheck/points_0 ... points_5) into ONE labelled extxyz, for baseline_D.py / pair_check.py.
   python3 mlp_pipeline/points_collect.py runs/dgcheck/points_* --out runs/dgcheck/labelled.extxyz [--db-set PD]            (needs ase: use $PY)
Per finished point (OUTCAR found directly or via DONE.json -> attempts/<stamp>/OUTCAR): the structure in the OUTCAR's own atom order (VASP sorts species into POTCAR order; it is checked to equal the source frame up to that order) with the source frame's info (run_id, widom_role, T_K, ...), energy = FREE energy (TOTEN, what the training labels are),
E_sigma0_eV = energy(sigma->0) from the same OUTCAR, forces and stress. Unfinished or unreadable points are counted and skipped, so it is safe to run while jobs are still going.
Health: points whose SCF never reached EDIFF (NELM hit; no "aborting loop because EDIFF is reached" in the OUTCAR) are SKIPPED (--keep-unconverged keeps them, flagged scf_converged=False);
at the end a table per composition (nO, nH): frames, DFT pressure range, largest |force|, E/atom spread, and outliers (|E/atom - median| > 6 MAD and > 0.1 eV, or |F| > --fwarn)."""
import argparse, csv, glob, json, os, sys
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

import numpy as np
from ase.geometry import get_distances


def same_structure(src, o, tol=2e-3):
    """True if o (OUTCAR atoms, VASP/POTCAR species order) is the same structure as src up to the atom order and periodic wrapping: same composition, and every atom of o has a src atom of the same species within tol (min-image)"""
    if sorted(src.numbers) != sorted(o.numbers) or not np.allclose(src.cell.array, o.cell.array, atol=1e-5): return False
    for z in set(o.numbers):
        a, b = src.positions[src.numbers == z], o.positions[o.numbers == z]
        D = get_distances(b, a, cell=src.cell, pbc=True)[1]
        if D.min(axis=1).max() > tol: return False
    return True


ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--db-set", default="PD")
ap.add_argument("--keep-unconverged", action="store_true"); ap.add_argument("--fwarn", type=float, default=50.0, help="flag frames with max |force| above this (eV/A)"); a = ap.parse_args()


def scf_converged(path):
    with open(path, "rb") as fh: return b"aborting loop because EDIFF is reached" in fh.read()


def health(frames):
    import collections
    g = collections.defaultdict(list)
    for f in frames: g[(int((f.numbers == 8).sum()), int((f.numbers == 1).sum()))].append(f)
    print(f"\n{'nO':>4s} {'nH':>5s} {'frames':>6s} {'P_DFT GPa':>15s} {'max|F|':>7s} {'E/atom eV':>17s}  outliers")
    flagged = 0
    for (nO, nH), fs in sorted(g.items()):
        e = np.array([f.get_potential_energy() / len(f) for f in fs]); P = np.array([-np.trace(f.get_stress(voigt=False)) / 3 * 160.21766 for f in fs])
        fm = np.array([np.abs(f.get_forces()).max() for f in fs]); med = np.median(e); mad = np.median(np.abs(e - med)) or 1e-9
        out = [i for i in range(len(fs)) if (abs(e[i] - med) > 6 * mad and abs(e[i] - med) > 0.1) or fm[i] > a.fwarn]; flagged += len(out)
        print(f"{nO:4d} {nH:5d} {len(fs):6d} {P.min():7.0f}..{P.max():<7.0f} {fm.max():7.1f} {e.min():8.3f}..{e.max():<8.3f}  " + ", ".join(f"{os.path.relpath(fs[i].info['point_dir'])} (E/at {e[i]:.3f}, |F| {fm[i]:.0f})" for i in out[:4]) + (f" +{len(out) - 4}" if len(out) > 4 else ""))
    print(f"health: {flagged} flagged frames" + (" (check them before training)" if flagged else ", none"))
out, ok, missing, bad, unconv = [], 0, 0, 0, []
for pdir in sorted({d for g in a.dirs for d in glob.glob(g)}):
    if not os.path.isfile(os.path.join(pdir, "manifest.json")): continue
    man = json.load(open(os.path.join(pdir, "manifest.json"))); src = read(man["source_frames"], ":"); meta = list(csv.DictReader(open(os.path.join(pdir, "frames_meta.csv"))))
    for r in meta:
        d = os.path.join(pdir, r["id"]); oc = os.path.join(d, "OUTCAR"); dj = os.path.join(d, "DONE.json")
        if os.path.exists(dj):
            att = json.load(open(dj)).get("attempt"); oc = os.path.join(d, att, "OUTCAR") if att else oc
        elif os.path.isdir(os.path.join(d, "attempts")) or not os.path.exists(oc): missing += 1; continue
        try:
            conv = scf_converged(oc)
            if not conv:
                unconv.append(d)
                if not a.keep_unconverged: continue
            o = read(oc, index=-1, format="vasp-out"); F = float(o.get_potential_energy(force_consistent=True)); E0 = float(o.get_potential_energy())
            sf = src[int(r["frame"])]
            if not same_structure(sf, o): raise ValueError("OUTCAR structure is not the source frame (species/positions differ beyond the atom order)")
            at = o.copy(); at.pbc = True; at.info = dict(sf.info)                                 # the OUTCAR's own atom order, so the forces belong to the right atoms (VASP sorts species into POTCAR order)
            at.calc = SinglePointCalculator(at, energy=F, free_energy=F, forces=o.get_forces(), stress=o.get_stress())
            at.info.update(E_sigma0_eV=E0, energy_sigma0=E0, atom_order="vasp"); at.info.setdefault("db_set", a.db_set); at.info["point_dir"] = d; at.info["scf_converged"] = conv
            out.append(at); ok += 1
        except Exception as e:
            bad += 1
            if bad <= 5: print(f"  {d}: {type(e).__name__}: {str(e)[:100]}")
if out: write(a.out, out)
print(f"{ok} labelled frames -> {a.out}; {missing} points not finished, {bad} unreadable, {len(unconv)} SCF not converged" + (" (kept, flagged)" if a.keep_unconverged else " (skipped)") + (": " + ", ".join(os.path.relpath(x) for x in unconv[:5]) + (" ..." if len(unconv) > 5 else "") if unconv else ""))
if out: health(out)
