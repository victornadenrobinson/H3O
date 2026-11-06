#!/usr/bin/env python3
"""Analyse the k-point sensitivity jobs: delta = E_Gamma - E_mesh per cell for each frame, and the bias this puts into the composition error per added H.
  python3 mlp_pipeline/kpoint_check.py --collect          read the OUTCARs of runs/kpt/*_points -> runs/kpt/energies.csv  (needs ase: use $PY)
  python3 mlp_pipeline/kpoint_check.py                    print the tables from energies.csv
Everything is a free energy (TOTEN). E_Gamma is the cached Gamma label of the same geometry; k111 (vasp_std, 1x1x1) should reproduce it (a check of the pipeline, not of k convergence).
c_bias [meV/H] = [ delta(H2.25O) - delta(H2O) - 27 * delta(pure H)/729 ] / 27 : the shift the Gamma-only labels put into c (E_mesh is the better-converged one; so the converged c = c_Gamma - c_bias).
Forces / stress (--collect also writes forces.csv): per frame, F_mesh - F_Gamma (rms over components, eV/A, and relative to the rms force) and P_mesh - P_Gamma, max |d sigma| (GPa),
against the Gamma labels of the same geometries in --gamma-src (the DFT-MD file kpoint_frames.py picked them from; atoms matched by species + position, since VASP reorders species).
Error bars: bootstrap over frames (few frames: indicative). Decision rule: |c_bias| < 30 meV/H keep Gamma; otherwise correct the labels or relabel (see the F2 notes)."""
import argparse, collections, csv, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--collect", action="store_true"); ap.add_argument("--dir", default="runs/kpt"); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--gamma-src", default="runs/dft_md_500.extxyz", help="DFT-MD frames the k-point geometries came from (Gamma forces / stress)"); ap.add_argument("--set", default="M500")
ap.add_argument("--time-ref", nargs="*", default=[], help="point directories of ordinary Gamma (vasp_gam) labels to time for comparison, e.g. runs/reweight_dft/points_0 (first 20 OUTCARs each)"); a = ap.parse_args()
import re


def timing(oc):
    """elapsed s, SCF steps, irreducible k-points, total cores, atoms from an OUTCAR (None where not found)"""
    t = dict(elapsed=None, scf=0, nkpts=None, cores=None, natoms=None)
    try:
        with open(oc, errors="ignore") as fh:
            for l in fh:
                if "Iteration" in l and "(" in l: t["scf"] += 1
                elif "NKPTS =" in l and t["nkpts"] is None: m = re.search(r"NKPTS =\s*(\d+)", l); t["nkpts"] = int(m.group(1)) if m else None
                elif "NIONS =" in l and t["natoms"] is None: m = re.search(r"NIONS =\s*(\d+)", l); t["natoms"] = int(m.group(1)) if m else None
                elif t["cores"] is None and ("total cores" in l or "mpi-ranks" in l):         # VASP 5: "running on N total cores"; VASP 6: "running N mpi-ranks"
                    m = re.search(r"running\s+on\s+(\d+)\s+total cores", l) or re.search(r"running\s+(\d+)\s+mpi-ranks", l); t["cores"] = int(m.group(1)) if m else None
                elif "Elapsed time (sec):" in l: t["elapsed"] = float(l.split(":")[1])
    except OSError: pass
    return t
GPA = 160.21766


def match(src, o):
    """index array idx with o[k] == src[idx[k]] (same species, min-image distance < 2e-3 A), or None"""
    from ase.geometry import get_distances
    idx = np.full(len(o), -1)
    for z in set(o.numbers):
        io, isrc = np.nonzero(o.numbers == z)[0], np.nonzero(src.numbers == z)[0]
        D = get_distances(o.positions[io], src.positions[isrc], cell=src.cell, pbc=True)[1]; j = D.argmin(axis=1)
        if D[np.arange(len(io)), j].max() > 2e-3: return None
        idx[io] = isrc[j]
    return idx if len(set(idx)) == len(idx) else None
fn = os.path.join(a.dir, "energies.csv")
if a.collect:
    from ase.io import read
    rows = []; frows = []
    src = [f for f in read(a.gamma_src, index=":") if str(f.info.get("db_set")) == a.set] if os.path.exists(a.gamma_src) else []
    if not src: print(f"(no Gamma frames from {a.gamma_src} set {a.set}: forces / stress not compared)")
    for pdir in sorted(glob.glob(os.path.join(a.dir, "k*_points"))):
        mesh = os.path.basename(pdir).split("_")[0]; meta = list(csv.DictReader(open(os.path.join(pdir, "frames_meta.csv"))))
        for r in meta:
            d = os.path.join(pdir, r["id"]); oc = os.path.join(d, "OUTCAR"); dj = os.path.join(d, "DONE.json")
            if os.path.exists(dj):
                att = json.load(open(dj)).get("attempt"); oc = os.path.join(d, att, "OUTCAR") if att else oc
            elif os.path.isdir(os.path.join(d, "attempts")): continue
            try: o = read(oc, index=-1, format="vasp-out"); E = float(o.get_potential_energy(force_consistent=True))
            except Exception: continue
            i = int(r["kpt_key"].split(":")[1])
            if src and i < len(src):
                g = src[i]; idx = match(g, o)
                if idx is not None:
                    fm, fg = o.get_forces(), g.get_forces()[idx]; sm, sg = o.get_stress(voigt=False), g.get_stress(voigt=False)
                    dF = fm - fg; frows.append(dict(mesh=mesh, key=r["kpt_key"], system=r["kpt_sys"], natoms=r["natoms"], dF_rms=float(np.sqrt((dF ** 2).mean())), F_rms=float(np.sqrt((fg ** 2).mean())),
                                                    dF_max=float(np.abs(dF).max()), dP_GPa=float(-(np.trace(sm) - np.trace(sg)) / 3 * GPA), dS_max_GPa=float(np.abs(sm - sg).max() * GPA)))
                else: print(f"  {d}: atoms do not match the Gamma frame {i} of {a.gamma_src}: forces skipped")
            tm = timing(oc)
            rows.append(dict(mesh=mesh, key=r["kpt_key"], system=r["kpt_sys"], natoms=r["natoms"], E_mesh=E, E_gamma=float(r["E_gamma_eV"]), T_K=r.get("T_K", ""),
                             elapsed_s=tm["elapsed"], scf_steps=tm["scf"], nkpts=tm["nkpts"], cores=tm["cores"]))
    with open(fn, "w", newline="") as fh:                                  # closed (flushed) before the tables below read it back
        w = csv.DictWriter(fh, fieldnames=["mesh", "key", "system", "natoms", "E_mesh", "E_gamma", "T_K", "elapsed_s", "scf_steps", "nkpts", "cores"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} finished points -> {fn}")
    if frows:
        with open(os.path.join(a.dir, "forces.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(frows[0])); w.writeheader(); w.writerows(frows)
        print(f"{len(frows)} force / stress comparisons -> {os.path.join(a.dir, 'forces.csv')}")
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
ff = os.path.join(a.dir, "forces.csv")
if os.path.exists(ff):
    fr_ = list(csv.DictReader(open(ff))); g = collections.defaultdict(list)
    for r in fr_: g[(r["mesh"], r["system"])].append(r)
    print("forces / stress, mesh minus Gamma on the same geometry (mean over frames; dP = P_mesh - P_Gamma)")
    print(f"  {'mesh':5s} {'system':8s} {'n':>2s} {'dF rms eV/A':>12s} {'of F rms':>9s} {'dF max':>7s} {'dP GPa':>8s} {'max|dsig| GPa':>14s}")
    for (m, s_), v in sorted(g.items()):
        f = lambda k: np.mean([float(x[k]) for x in v])
        print(f"  {m:5s} {s_:8s} {len(v):2d} {f('dF_rms'):12.3f} {100 * f('dF_rms') / f('F_rms'):8.1f}% {f('dF_max'):7.2f} {f('dP_GPa'):+8.2f} {f('dS_max_GPa'):14.2f}")
tr = [r for r in rows if r.get("elapsed_s") not in (None, "", "None")]
if tr:
    print("timing per point (from the OUTCARs; core-s = elapsed x cores; per SCF step removes the effect of different SCF counts)")
    print(f"  {'mesh':5s} {'system':8s} {'n':>2s} {'atoms':>5s} {'nkpts':>5s} {'cores':>5s} {'elapsed s':>10s} {'SCF':>5s} {'s/SCF':>7s}")
    tg = collections.defaultdict(list)
    for r in tr: tg[(r["mesh"], r["system"])].append(r)
    per = {}
    for (m, s_), v in sorted(tg.items()):
        f = lambda k: np.mean([float(x[k]) for x in v if x.get(k) not in (None, "", "None")])
        el, sc = f("elapsed_s"), f("scf_steps"); per[(m, s_)] = (el, el / sc if sc else float("nan"))
        print(f"  {m:5s} {s_:8s} {len(v):2d} {v[0]['natoms']:>5s} {f('nkpts'):5.0f} {f('cores'):5.0f} {el:10.0f} {sc:5.0f} {el / sc if sc else float('nan'):7.1f}")
    for s_ in ("H2O", "H2p25O", "pureH"):
        if ("k111", s_) in per and ("k222", s_) in per:
            print(f"  k222 / k111 (both vasp_std) {s_:8s}: elapsed x{per[('k222', s_)][0] / per[('k111', s_)][0]:.1f}, per SCF step x{per[('k222', s_)][1] / per[('k111', s_)][1]:.1f}")
for d in a.time_ref:                                    # ordinary production Gamma labels (vasp_gam), for scale
    ocs = sorted(glob.glob(os.path.join(d, "*", "OUTCAR")) + glob.glob(os.path.join(d, "*", "attempts", "*", "OUTCAR")))[:20]
    ts = [timing(o) for o in ocs]; ts = [t for t in ts if t["elapsed"]]
    if ts: print(f"  reference {d}: {len(ts)} OUTCARs, atoms {min(t['natoms'] or 0 for t in ts)}-{max(t['natoms'] or 0 for t in ts)}, cores {ts[0]['cores']}, nkpts {ts[0]['nkpts']}, "
                 f"elapsed {np.mean([t['elapsed'] for t in ts]):.0f} s (median {np.median([t['elapsed'] for t in ts]):.0f}), SCF {np.mean([t['scf'] for t in ts]):.0f}, s/SCF {np.mean([t['elapsed'] / t['scf'] for t in ts if t['scf']]):.1f}")
    else: print(f"  reference {d}: no finished OUTCARs with timing")
