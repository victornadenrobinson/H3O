#!/usr/bin/env python3
"""Recover the DFT labels of the sampled frames of the VASP-driven PWidom runs (tdi/dft_pwidom.py workers).
Every VASP call of a worker is cached as <work>/cache/<sha256(fingerprint + species + positions + cell + pbc)>.npz = {energy, free_energy (= TOTEN), forces, stress}; the cache key hides the positions, but the MD
checkpoints (pw_host_md.npz, pw_full_md.npz) keep the positions of the sampled frames (one per --every-fs inside the sampling window). We recompute the key of each saved frame and read its label: free, no VASP.
  python mlp_pipeline/dft_md_labelled.py --plan runs/dft_pwidom_400GPa_5000K_4x64/dft_plan.json --out dft_md_400.extxyz            (all workers of a prepared plan)
  python mlp_pipeline/dft_md_labelled.py --worker runs/dft_pwidom_500GPa_5000K_1/H2O_rep0 --config <vasp.json> --out dft_md_500.extxyz
  --dry counts frames and labels only. Needs the repo's tdi package (PYTHONPATH=mlp_pipeline) for the config fingerprint; --fp STRING overrides it (tests).
Output = labelled frames (energy = free energy TOTEN like the snapdb labels; energy_sigma0 is NOT available and is set to the free energy, flagged sigma0_missing=1), info: kind=dft_md, db_set=M<P in GPa> (M400, M500, M600), run_id, stage host|full, T_K, P_target, nO, nH, split
(frame-level hash, --valid-frac). Frames are autocorrelated within a state: train on a subset, keep the rest for the DFT-ensemble test (baseline_D.py then gives D on the true ensemble)."""
import argparse, glob, hashlib, json, os, re, sys
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--plan"); ap.add_argument("--worker", nargs="*", default=[]); ap.add_argument("--config", help="vasp config json (for --worker)"); ap.add_argument("--fp", default=None)
ap.add_argument("--out", default="dft_md_labelled.extxyz"); ap.add_argument("--valid-frac", type=float, default=0.25); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()


def key(fp, numbers, positions, cell, pbc):
    d = hashlib.sha256(fp.encode())
    for arr in (numbers, positions, cell, pbc): d.update(np.ascontiguousarray(arr).tobytes())          # same order as VaspEngine.calculate
    return d.hexdigest()


jobs = []                       # (worker dir, config path)
if a.plan:
    plan = json.load(open(a.plan))
    for w in plan["workers"]:
        args = w["args"]; jobs.append((w["directory"], args[args.index("--model") + 1].split(":", 1)[1]))
for w in a.worker: jobs.append((w, a.config))
if not jobs: sys.exit("give --plan or --worker")
if os.path.exists(a.out) and not a.dry: os.remove(a.out)
tot = found = 0
for d, cfg in jobs:
    name = os.path.basename(os.path.normpath(d)); m = re.search(r"(\d+)GPa", d + (cfg or "")); P = float(m.group(1)) if m else float("nan")
    try:
        if a.fp: fp = a.fp; work = os.path.join(d, "vasp_work")
        else:
            from tdi.vasp_backend import config, fingerprint
            fp = fingerprint(cfg); work = config(cfg)["work"]
    except Exception as e: print(f"{name}: cannot read config/fingerprint ({e})"); continue
    cache = os.path.join(work, "cache"); n_here = f_here = 0
    for stage, fn in (("host", "pw_host_md.npz"), ("full", "pw_full_md.npz")):
        p = os.path.join(d, fn)
        if not os.path.exists(p): continue
        z = np.load(p, allow_pickle=False); md = json.loads(str(z["metadata"])); fr = z["frames"]; numbers = z["numbers"]; cell = z["cell"]; pbc = z["pbc"]
        nO = int((numbers == 8).sum()); nH = int((numbers == 1).sum()); T = float(md["T"]); n_here += len(fr)
        for i, x in enumerate(fr):
            f = os.path.join(cache, key(fp, numbers, x, cell, pbc) + ".npz")
            if not os.path.exists(f): continue
            f_here += 1
            if a.dry: continue
            r = np.load(f, allow_pickle=False); at = Atoms(numbers=numbers, positions=x, cell=cell, pbc=pbc); E = float(r["free_energy"])
            at.calc = SinglePointCalculator(at, energy=E, free_energy=E, forces=r["forces"], stress=r["stress"])
            rid = f"dftmd:{name}:{stage}{i}"; sp = "valid" if int(hashlib.md5(rid.encode()).hexdigest(), 16) % 1000 < 1000 * a.valid_frac else "train"
            at.info.update(kind="dft_md", db_set=f"M{int(P)}" if P == P else "M", run_id=rid, source=d, stage=stage, T_K=T, P_target=P, nO=nO, nH=nH, split=sp, energy_sigma0=E, sigma0_missing=1, config_type=f"dftmd_{nH}H{nO}O_{int(T)}K")
            write(a.out, at, append=True)
    print(f"{name:26s} P {P:5.0f} GPa: {f_here}/{n_here} saved frames have a cached label"); tot += n_here; found += f_here
print(f"{found}/{tot} frames labelled" + ("" if a.dry else f" -> {a.out}"))
