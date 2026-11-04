#!/usr/bin/env python3
"""Collect finished single-point VASP directories made by prepare_points.py (e.g. runs/dgcheck/points_0 ... points_5) into ONE labelled extxyz, for baseline_D.py / pair_check.py.
   python3 mlp_pipeline/points_collect.py runs/dgcheck/points_* --out runs/dgcheck/labelled.extxyz [--db-set PD]            (needs ase: use $PY)
Per finished point (OUTCAR found directly or via DONE.json -> attempts/<stamp>/OUTCAR): the ORIGINAL frame geometry and its info (run_id, widom_role, T_K, ...) with energy = FREE energy (TOTEN, what the training labels are),
E_sigma0_eV = energy(sigma->0) from the same OUTCAR, forces and stress. Unfinished or unreadable points are counted and skipped, so it is safe to run while jobs are still going."""
import argparse, csv, glob, json, os, sys
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--db-set", default="PD"); a = ap.parse_args()
out, ok, missing, bad = [], 0, 0, 0
for pdir in sorted({d for g in a.dirs for d in glob.glob(g)}):
    if not os.path.isfile(os.path.join(pdir, "manifest.json")): continue
    man = json.load(open(os.path.join(pdir, "manifest.json"))); src = read(man["source_frames"], ":"); meta = list(csv.DictReader(open(os.path.join(pdir, "frames_meta.csv"))))
    for r in meta:
        d = os.path.join(pdir, r["id"]); oc = os.path.join(d, "OUTCAR"); dj = os.path.join(d, "DONE.json")
        if os.path.exists(dj):
            att = json.load(open(dj)).get("attempt"); oc = os.path.join(d, att, "OUTCAR") if att else oc
        elif os.path.isdir(os.path.join(d, "attempts")) or not os.path.exists(oc): missing += 1; continue
        try:
            o = read(oc, index=-1, format="vasp-out"); F = float(o.get_potential_energy(force_consistent=True)); E0 = float(o.get_potential_energy())
            at = src[int(r["frame"])].copy()
            if len(at) != len(o): raise ValueError("atom count differs from the source frame")
            at.calc = SinglePointCalculator(at, energy=F, free_energy=F, forces=o.get_forces(), stress=o.get_stress())
            at.info.update(E_sigma0_eV=E0, energy_sigma0=E0); at.info.setdefault("db_set", a.db_set); at.info["point_dir"] = d
            out.append(at); ok += 1
        except Exception as e:
            bad += 1
            if bad <= 5: print(f"  {d}: {type(e).__name__}: {str(e)[:100]}")
if out: write(a.out, out)
print(f"{ok} labelled frames -> {a.out}; {missing} points not finished, {bad} unreadable")
