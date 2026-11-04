#!/usr/bin/env python3
"""Analyse the k-point sensitivity jobs: delta = E_Gamma - E_mesh per cell for each frame, and the bias this puts into the composition error per added H.
  python3 mlp_pipeline/kpoint_check.py --collect          read the OUTCARs of runs/kpt/*_points -> runs/kpt/energies.csv  (needs ase: use $PY)
  python3 mlp_pipeline/kpoint_check.py                    print the tables from energies.csv
Everything is a free energy (TOTEN). E_Gamma is the cached Gamma label of the same geometry; k111 (vasp_std, 1x1x1) should reproduce it (a check of the pipeline, not of k convergence).
c_bias [meV/H] = [ delta(H2.25O) - delta(H2O) - 27 * delta(pure H)/729 ] / 27 : the shift the Gamma-only labels put into c (E_mesh is the better-converged one; so the converged c = c_Gamma - c_bias).
Error bars: bootstrap over frames (few frames: indicative). Decision rule: |c_bias| < 30 meV/H keep Gamma; otherwise correct the labels or relabel (see the F2 notes)."""
import argparse, collections, csv, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--collect", action="store_true"); ap.add_argument("--dir", default="runs/kpt"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
fn = os.path.join(a.dir, "energies.csv")
if a.collect:
    from ase.io import read
    rows = []
    for pdir in sorted(glob.glob(os.path.join(a.dir, "k*_points"))):
        mesh = os.path.basename(pdir).split("_")[0]; meta = list(csv.DictReader(open(os.path.join(pdir, "frames_meta.csv"))))
        for r in meta:
            d = os.path.join(pdir, r["id"]); oc = os.path.join(d, "OUTCAR"); dj = os.path.join(d, "DONE.json")
            if os.path.exists(dj):
                att = json.load(open(dj)).get("attempt"); oc = os.path.join(d, att, "OUTCAR") if att else oc
            elif os.path.isdir(os.path.join(d, "attempts")): continue
            try: E = float(read(oc, index=-1, format="vasp-out").get_potential_energy(force_consistent=True))
            except Exception: continue
            rows.append(dict(mesh=mesh, key=r["kpt_key"], system=r["kpt_sys"], natoms=r["natoms"], E_mesh=E, E_gamma=float(r["E_gamma_eV"]), T_K=r.get("T_K", "")))
    with open(fn, "w", newline="") as fh:                                  # closed (flushed) before the tables below read it back
        w = csv.DictWriter(fh, fieldnames=["mesh", "key", "system", "natoms", "E_mesh", "E_gamma", "T_K"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} finished points -> {fn}")
rows = list(csv.DictReader(open(fn))); rng = np.random.default_rng(1)
by = collections.defaultdict(list)
for r in rows: by[(r["mesh"], r["system"])].append(dict(key=r["key"], n=int(r["natoms"]), d=float(r["E_gamma"]) - float(r["E_mesh"])))
print("delta = E_Gamma - E_mesh [eV per cell; meV per atom]")
for (m, s), v in sorted(by.items()):
    d = np.array([x["d"] for x in v]); n = v[0]["n"]; print(f"  {m} {s:8s} {len(d)} frames  delta {d.mean():+8.3f} eV  (sd {d.std(ddof=1) if len(d) > 1 else float('nan'):.3f})   = {1000 * d.mean() / n:+7.1f} meV/atom")
for m in ("k222", "k333"):
    need = [by.get((m, s)) for s in ("H2O", "H2p25O", "pureH")]
    if any(x is None for x in need): continue
    def c(samples):
        dO, dH2, dP = (np.mean([x for x in s]) for s in samples); return (dH2 - dO - 27 * dP / 729) / 27
    vals = [[x["d"] for x in by[(m, s)]] for s in ("H2O", "H2p25O", "pureH")]; c0 = c(vals) * 1000
    bs = [c([rng.choice(v, len(v)) for v in vals]) * 1000 for _ in range(a.boot)]
    print(f"c_bias ({m}) = {c0:+.0f} +- {np.std(bs):.0f} meV/H   {'(keep Gamma)' if abs(c0) < 30 else '(Gamma labels shift the composition error: correct or relabel)'}")
if ("k222", "pureH") in by and ("k333", "pureH") in by:
    k2 = {x["key"]: x["d"] for s in ("pureH", "H2O", "H2p25O") for x in by.get(("k222", s), [])}; print("2x2x2 -> 3x3x3 on the same frames (convergence of the mesh): delta_333 - delta_222 [meV/atom]")
    for s in ("pureH", "H2O", "H2p25O"):
        for x in by.get(("k333", s), []):
            if x["key"] in k2: print(f"  {s:8s} {x['key']}  {1000 * (x['d'] - k2[x['key']]) / x['n']:+7.1f}")
if ("k111", "pureH") in by:
    print("vasp_std Gamma vs the cached vasp_gam label (should be ~0):", ", ".join(f"{s} {1000 * np.mean([x['d'] for x in by[('k111', s)]]):+.2f} meV/cell" for s in ("pureH", "H2O", "H2p25O") if ("k111", s) in by))
