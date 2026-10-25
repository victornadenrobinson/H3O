"""Integrity / format check for extxyz datasets from 02/03.

usage: python 10_check_snapshots.py snapshots_s.extxyz [snapshots_m.extxyz snapshots_l.extxyz ...]   (bare names are looked up in runs/)
Checks per file: parses, species, energy/forces/stress present + finite + right shape, cell, pbc, required info keys, duplicates,
min frame spacing within a run, force/stress outliers, stress sign/units (frame virial P + N kB T/V vs stored run-mean P_GPa).
Across files: nested subsets (each file's frames contained in the next listed one); train/valid sharing no run directory.
Exit code 1 if any FAIL.
"""
import os, sys
from collections import Counter
import numpy as np
from ase.io import read
from common import inp

KB_EV, EV_A3_GPA = 8.617333e-5, 160.21766
NEED_INFO = ("source", "frame", "config_type", "T_K", "P_GPa")
SOFT_INFO = ("frame_rank", "run_rank", "energy_sigma0")
SPECIES = {"H", "O"}
F_WARN, F_FAIL = 50.0, 500.0   # eV/A max |force component|
MIN_GAP = 50
fails, warns = [], []


def bad(msg): fails.append(msg); print("  FAIL:", msg)
def warn(msg): warns.append(msg); print("  WARN:", msg)


def check(path):
    print(f"\n== {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    try:
        frames = read(path, index=":")
    except Exception as e:
        bad(f"cannot parse: {e}"); return None
    n = len(frames); print(f"  {n} frames")
    import re
    with open(path) as fh:
        fh.readline(); hdr = fh.readline()
    if not re.search(r'stress="(\s*-?[\d.eE+-]+){9}"', hdr): bad("header has no 9-component stress")
    if "forces:R:3" not in hdr: bad("header has no forces:R:3 column")
    if not re.search(r"\benergy=", hdr): bad("header has no energy=")
    if n == 0:
        bad("no frames"); return None
    missing = Counter(); nonfinite = 0; bad_shape = 0; keys = set(); dp = []; fmax = []; smax = []
    for i, a in enumerate(frames):
        res = a.calc.results if a.calc is not None else {}
        for k in ("energy", "forces", "stress"):
            if k not in res: missing[k] += 1
        for k in NEED_INFO + SOFT_INFO:
            if k not in a.info: missing["info:" + k] += 1
        if not set(a.get_chemical_symbols()) <= SPECIES: missing["species!=H,O"] += 1
        if "forces" in res:
            f = np.asarray(res["forces"])
            if f.shape != (len(a), 3): bad_shape += 1
            if not np.isfinite(f).all(): nonfinite += 1
            else: fmax.append(np.abs(f).max())
        if "stress" in res:
            try:
                s = a.get_stress(voigt=False)   # extxyz round-trip gives Voigt(6); normalise to 3x3
            except Exception:
                s = np.zeros(0); bad_shape += 1
            if s.shape == (3, 3) and not np.isfinite(s).all(): nonfinite += 1
            elif s.shape == (3, 3) and all(k in a.info for k in ("T_K", "P_GPa")):
                V = a.get_volume()
                p_vir = -np.trace(s) / 3 * EV_A3_GPA
                kin = len(a) * KB_EV * a.info["T_K"] / V * EV_A3_GPA
                dp.append(p_vir + kin - a.info["P_GPa"]); smax.append(np.abs(s).max() * EV_A3_GPA)
        if "energy" in res and not np.isfinite(res["energy"]): nonfinite += 1
        if abs(np.linalg.det(np.array(a.cell))) < 1e-3: missing["cell singular"] += 1
        if not np.all(a.pbc): missing["pbc not all True"] += 1
    for k, v in sorted(missing.items()):
        (warn if k.split(":")[-1] in SOFT_INFO else bad)(f"{v} frames missing/invalid: {k}")
    if bad_shape: bad(f"{bad_shape} arrays with wrong shape")
    if nonfinite: bad(f"{nonfinite} frames with NaN/inf")
    # duplicates and spacing
    ids = [(a.info.get("source"), a.info.get("frame")) for a in frames]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup: bad(f"{len(dup)} duplicated (source, frame), e.g. {dup[0]}")
    by = {}
    for s_, f_ in ids:
        by.setdefault(s_, []).append(f_)
    close = sum(1 for v in by.values() if len(v) > 1 and np.min(np.diff(sorted(v))) < MIN_GAP)
    if close: warn(f"{close} runs have frames closer than {MIN_GAP} MD steps")
    # numbers
    if fmax:
        fm = np.array(fmax)
        print(f"  max |F| per frame (eV/A): median {np.median(fm):.2f}, 99% {np.percentile(fm, 99):.2f}, max {fm.max():.2f}")
        if fm.max() > F_FAIL: bad(f"force > {F_FAIL} eV/A (max {fm.max():.1f})")
        elif fm.max() > F_WARN: warn(f"{int((fm > F_WARN).sum())} frames with force > {F_WARN} eV/A (max {fm.max():.1f})")
    if dp:
        d = np.array(dp)
        print(f"  frame virial P + N kB T/V - stored run-mean P (GPa): median {np.median(d):.1f}, |.| 95% {np.percentile(np.abs(d), 95):.1f}")
        if abs(np.median(d)) > 5: bad(f"stress sign/units look wrong: median offset {np.median(d):.1f} GPa")
        elif np.percentile(np.abs(d), 95) > 60: warn("some frames' instantaneous P differ >60 GPa from their run mean")
    e_pa = np.array([a.get_potential_energy() / len(a) for a in frames if "energy" in (a.calc.results if a.calc else {})])
    if len(e_pa): print(f"  energy/atom (eV): {e_pa.min():.2f} .. {e_pa.max():.2f}")
    sizes = Counter(len(a) for a in frames); print("  atoms/frame:", dict(sorted(sizes.items())))
    ct = Counter(a.info.get("config_type") for a in frames)
    print("  config_type (top 12):", dict(ct.most_common(12)), f"... {len(ct)} total")
    return {(a.info.get("source"), a.info.get("frame")) for a in frames}, {os.path.dirname(a.info.get("source", "")) for a in frames}


def main(paths):
    sets = []
    for p in paths:
        r = check(inp(p)); sets.append((p, r))
    print("\n== cross-file")
    ok = [(p, r) for p, r in sets if r]
    for (p1, r1), (p2, r2) in zip(ok, ok[1:]):
        b1, b2 = os.path.basename(p1), os.path.basename(p2)
        if b1.startswith("train") and b2.startswith("valid") or b1.startswith("valid") and b2.startswith("train"):
            shared = r1[1] & r2[1]
            (bad if shared else print)(f"{len(shared)} run directories shared between {b1} and {b2}" if shared else f"  no run directory shared between {b1} and {b2}")
        elif len(r1[0] - r2[0]):
            warn(f"{b1} has {len(r1[0] - r2[0])} frames not in {b2} (tiers should be nested)")
        else:
            print(f"  {b1} is a subset of {b2}")
    print(f"\n{'FAIL' if fails else 'PASS'}: {len(fails)} failures, {len(warns)} warnings")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
