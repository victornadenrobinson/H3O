"""80/20 train/valid split grouped by source run directory (no frame leakage).
usage: python 03_split.py snapshots.extxyz [seed]"""
import os, sys, random
from ase.io import read, write

frames = read(sys.argv[1], index=":")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
groups = {}
for f in frames:
    groups.setdefault(os.path.dirname(f.info["source"]), []).append(f)
keys = sorted(groups); random.Random(seed).shuffle(keys)
n_tr = int(0.8 * len(frames)); tr, va = [], []
for k in keys:
    (tr if len(tr) < n_tr else va).extend(groups[k])
write("train.extxyz", tr); write("valid.extxyz", va)
print(f"train {len(tr)}  valid {len(va)}  ({len(keys)} run dirs)")
