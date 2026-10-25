"""Integrity / format check for extxyz datasets from 02/03.

usage: python 10_check_snapshots.py snapshots_s.extxyz [snapshots_m.extxyz snapshots_l.extxyz ...] [--workers 32]   (bare names are looked up in runs/)
Checks per file: parses, species, energy/forces/stress present + finite + right shape, cell, pbc, required info keys, duplicates,
min frame spacing within a run, force/stress outliers, stress sign/units (frame virial P + N kB T/V vs stored run-mean P_GPa).
Across files: nested subsets (each file's frames contained in the next listed one); train/valid sharing no run directory.
Exit code 1 if any FAIL.
"""
import argparse, os, re, sys
from io import StringIO
from multiprocessing import Pool
from collections import Counter
import numpy as np
from ase.io import read
from common import inp, progress

KB_EV, EV_A3_GPA = 8.617333e-5, 160.21766
NEED_INFO = ("source", "frame", "config_type", "T_K", "P_GPa")
SOFT_INFO = ("frame_rank", "run_rank", "energy_sigma0")
SPECIES = {"H", "O"}
F_WARN, F_FAIL = 50.0, 500.0   # eV/A max |force component|
MIN_GAP = 50
fails, warns = [], []


def bad(msg): fails.append(msg); print("  FAIL:", msg)
def warn(msg): warns.append(msg); print("  WARN:", msg)


def frame_offsets(lines):
    """Line index where each frame starts (extxyz: natoms line, comment line, natoms atom lines)."""
    offs, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        offs.append(i); i += int(lines[i]) + 2
    return offs


def analyze(text):
    """Parse one chunk of frames and return mergeable partial statistics."""
    frames = read(StringIO(text), format="extxyz", index=":")
    P = dict(n=len(frames), missing=Counter(), nonfinite=0, bad_shape=0, ids=[], fmax=[], dp=[], e_pa=[], sizes=Counter(), ct=Counter())
    for a in frames:
        res = a.calc.results if a.calc is not None else {}
        for k in ("energy", "forces", "stress"):
            if k not in res: P["missing"][k] += 1
        for k in NEED_INFO + SOFT_INFO:
            if k not in a.info: P["missing"]["info:" + k] += 1
        if not set(a.get_chemical_symbols()) <= SPECIES: P["missing"]["species!=H,O"] += 1
        if "forces" in res:
            f = np.asarray(res["forces"])
            if f.shape != (len(a), 3): P["bad_shape"] += 1
            if not np.isfinite(f).all(): P["nonfinite"] += 1
            else: P["fmax"].append(float(np.abs(f).max()))
        if "stress" in res:
            try:
                s = a.get_stress(voigt=False)   # extxyz round-trip gives Voigt(6); normalise to 3x3
            except Exception:
                s = np.zeros(0); P["bad_shape"] += 1
            if s.shape == (3, 3) and not np.isfinite(s).all(): P["nonfinite"] += 1
            elif s.shape == (3, 3) and all(k in a.info for k in ("T_K", "P_GPa")):
                p_vir = -np.trace(s) / 3 * EV_A3_GPA
                kin = len(a) * KB_EV * a.info["T_K"] / a.get_volume() * EV_A3_GPA
                P["dp"].append(float(p_vir + kin - a.info["P_GPa"]))
        if "energy" in res:
            if np.isfinite(res["energy"]): P["e_pa"].append(float(res["energy"]) / len(a))
            else: P["nonfinite"] += 1
        if abs(np.linalg.det(np.array(a.cell))) < 1e-3: P["missing"]["cell singular"] += 1
        if not np.all(a.pbc): P["missing"]["pbc not all True"] += 1
        P["ids"].append((a.info.get("source"), int(a.info.get("frame", -1))))
        P["sizes"][len(a)] += 1; P["ct"][a.info.get("config_type")] += 1
    return P


def merge(parts):
    M = dict(n=0, missing=Counter(), nonfinite=0, bad_shape=0, ids=[], fmax=[], dp=[], e_pa=[], sizes=Counter(), ct=Counter())
    for P in parts:
        for k, v in P.items():
            if isinstance(v, Counter): M[k].update(v)
            elif isinstance(v, list): M[k] += v
            else: M[k] += v
    return M


def check(path, pool, workers):
    if not os.path.exists(path):
        print(f"\n== {path}"); bad("file not found (bare names are looked up in the current dir, then in runs/)"); return None
    print(f"\n== {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    with open(path) as fh:
        lines = fh.readlines()
    try:
        offs = frame_offsets(lines)
    except Exception as e:
        bad(f"cannot split into frames: {e}"); return None
    n = len(offs); print(f"  {n} frames")
    if n == 0:
        bad("no frames"); return None
    hdr = lines[1]
    if not re.search(r'stress="(\s*-?[\d.eE+-]+){9}"', hdr): bad("header has no 9-component stress")
    if "forces:R:3" not in hdr: bad("header has no forces:R:3 column")
    if not re.search(r"\benergy=", hdr): bad("header has no energy=")
    step = max(1, -(-n // (workers * 4)))
    bounds = offs[::step] + [len(lines)]
    chunks = ["".join(lines[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]
    try:
        M = merge(list(progress(pool.imap(analyze, chunks), total=len(chunks), desc="  parsing", leave=False)))
    except Exception as e:
        bad(f"cannot parse: {e}"); return None
    for k, v in sorted(M["missing"].items()):
        (warn if k.split(":")[-1] in SOFT_INFO else bad)(f"{v} frames missing/invalid: {k}")
    if M["bad_shape"]: bad(f"{M['bad_shape']} arrays with wrong shape")
    if M["nonfinite"]: bad(f"{M['nonfinite']} frames with NaN/inf")
    ids = M["ids"]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup: bad(f"{len(dup)} duplicated (source, frame), e.g. {dup[0]}")
    by = {}
    for s_, f_ in ids:
        by.setdefault(s_, []).append(f_)
    close = sum(1 for v in by.values() if len(v) > 1 and np.min(np.diff(sorted(v))) < MIN_GAP)
    if close: warn(f"{close} runs have frames closer than {MIN_GAP} MD steps")
    if M["fmax"]:
        fm = np.array(M["fmax"])
        print(f"  max |F| per frame (eV/A): median {np.median(fm):.2f}, 99% {np.percentile(fm, 99):.2f}, max {fm.max():.2f}")
        if fm.max() > F_FAIL: bad(f"force > {F_FAIL} eV/A (max {fm.max():.1f})")
        elif fm.max() > F_WARN: warn(f"{int((fm > F_WARN).sum())} frames with force > {F_WARN} eV/A (max {fm.max():.1f})")
    if M["dp"]:
        d = np.array(M["dp"])
        print(f"  frame virial P + N kB T/V - stored run-mean P (GPa): median {np.median(d):.1f}, |.| 95% {np.percentile(np.abs(d), 95):.1f}")
        if abs(np.median(d)) > 5: bad(f"stress sign/units look wrong: median offset {np.median(d):.1f} GPa")
        elif np.percentile(np.abs(d), 95) > 60: warn("some frames' instantaneous P differ >60 GPa from their run mean")
    if M["e_pa"]: print(f"  energy/atom (eV): {min(M['e_pa']):.2f} .. {max(M['e_pa']):.2f}")
    print("  atoms/frame:", dict(sorted(M["sizes"].items())))
    print("  config_type (top 12):", dict(M["ct"].most_common(12)), f"... {len(M['ct'])} total")
    return set(ids), {os.path.dirname(s_ or "") for s_, _ in ids}


def main(paths, workers):
    sets = []
    with Pool(workers) as pool:
        for p in paths:
            r = check(inp(p), pool, workers); sets.append((p, r))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    a = ap.parse_args()
    main(a.files, a.workers)
