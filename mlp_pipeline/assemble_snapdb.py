"""Collect every generated snapshot set into one database, then verify and plot it (no DFT, no GPU; run on a login node).
  python mlp_pipeline/assemble_snapdb.py [--leg ~/leg] [--root .] [--out snapdb] [--dry]
Looks for the standard outputs, concatenates each set's files (reading with ASE, so a truncated frame is skipped, not copied), tags the low-P MD trajectories, and calls
snapdb.py build -> verify -> plot. Missing sets are reported, not fatal. Caps and priorities follow docs/DFT_PLAN.md (prio 1 = label first). Edit SETS to change them.
Set -> source:  A ins_grid/*  A_b1 ins_batch1/*  A_part partial/*  B chain/*  C sel_pwidom  D_melt sel_meltcomp  D_eos sel_eos  E runs/eos_lowp/*/traj.extxyz  F toy/{h2o,h2.25o,h2.5o,h3o}  F_H toy/H729  G H729_md  H h3o_sc/*
"""
import argparse, glob, hashlib, os, re, subprocess, sys
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--leg", default=os.path.expanduser("~/leg")); ap.add_argument("--root", default="."); ap.add_argument("--out", default="snapdb"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
L = lambda *p: os.path.join(a.leg, *p); R = lambda *p: os.path.join(a.root, *p)
# name: (file globs, cap (0 = all), prio)
SETS = {
    "B": ([L("chain", "*.extxyz")], 0, 1), "C": ([R("sel_pwidom.extxyz")], 440, 1), "C2": ([L("widom_c2.extxyz")], 0, 1),   # C 440 = C 200 + C3 240 (end-state D statistics come from the same pwidom caches); C2: no cap, host and site geometries of a frame must stay together
    "A": ([L("ins_grid", "ins_*.extxyz")], 520, 2), "A_b1": ([L("ins_batch1", "ins_*.extxyz")], 150, 2), "A_part": ([L("partial", "ins_*.extxyz")], 100, 3),
    "E": ([R("runs", "eos_lowp", "*", "traj.extxyz"), R("runs", "eos_lowp2", "*", "traj.extxyz")], 180, 3), "H": ([L("h3o_sc", "*.extxyz")], 220, 3),
    "D_melt": ([R("sel_meltcomp.extxyz")], 144, 4), "D_eos": ([R("sel_eos.extxyz")], 100, 4)   # D_melt 144 = D + D2 (4000/6000 K fill), "G": ([L("H729_md.extxyz"), L("H729_md_s2.extxyz")], 30, 4),
    "F": ([L("toy", n + ".extxyz") for n in ("h2o", "h2.25o", "h2.5o", "h3o")], 220, 5), "F_H": ([L("toy", "H729.extxyz")], 32, 5)}


def tag_lowp(f, path):
    d = os.path.basename(os.path.dirname(path)); T = re.search(r"(\d+)K", d); nO = int((f.numbers == 8).sum()); nH = int((f.numbers == 1).sum())
    f.info.update(kind="eos_md", run_id=d, source=path, split="valid" if int(hashlib.md5(d.encode()).hexdigest(), 16) % 3 == 0 else "train", nO=nO, nH=nH, T_K=float(T.group(1)) if T else np.nan,
                  vpa=float(f.get_volume() / len(f)), config_type=f"lowP_{nH}H{nO}O_{int(float(T.group(1))) if T else 0}K", method="md"); return f


work = a.out + "_src"; os.makedirs(work, exist_ok=True); specs = []; report = []
for name, (globs, cap, prio) in SETS.items():
    files = sorted({p for g in globs for p in glob.glob(g)}); frames = []; bad = 0
    for p in files:
        try: fr = read(p, index=":")
        except Exception as e: bad += 1; print(f"  skipped unreadable {p}: {str(e)[:60]}"); continue
        for f in fr:
            if name == "E": f = tag_lowp(f, p)
            f.calc = None; frames.append(f)
    report.append((name, len(files), len(frames), bad, cap, prio))
    if frames and not a.dry: out = os.path.join(work, name + ".extxyz"); write(out, frames); specs.append(f"{name}={out}:{cap or ''}:{prio}")
print(f"{'set':8s} {'files':>5s} {'frames':>7s} {'unreadable':>10s} {'cap':>5s} {'prio':>4s}")
for r in report: print(f"{r[0]:8s} {r[1]:5d} {r[2]:7d} {r[3]:10d} {r[4]:5d} {r[5]:4d}" + ("   MISSING" if r[2] == 0 else ""))
if a.dry or not specs: sys.exit(0)
snap = [sys.executable, os.path.join(HERE, "snapdb.py")]
for cmd in (["build", "--out", a.out, "--sets", *specs, "--seed", str(a.seed)], ["verify", a.out], ["plot", a.out]):
    print("\n$ snapdb.py", cmd[0]); r = subprocess.run(snap + cmd)
    if r.returncode and cmd[0] != "verify": sys.exit(r.returncode)
