"""80/20 train/valid split grouped by source run directory (no frame leakage).
usage: python 03_split.py snapshots.extxyz [seed]"""
import os, sys, random
from common import out, inp
from ase.io import read, write

frames = read(inp(sys.argv[1]), index=":")
stem = os.path.basename(sys.argv[1]).rsplit(".", 1)[0]
tag = stem.split("snapshots_", 1)[1] if stem.startswith("snapshots_") else stem
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
groups = {}
for f in frames:
    groups.setdefault(os.path.dirname(f.info["source"]), []).append(f)
keys = sorted(groups); random.Random(seed).shuffle(keys)
n_tr = int(0.8 * len(frames)); tr, va = [], []
for k in keys:
    (tr if len(tr) < n_tr else va).extend(groups[k])
write(out(f"train_{tag}.extxyz"), tr); write(out(f"valid_{tag}.extxyz"), va)
print(f"train {len(tr)}  valid {len(va)}  ({len(keys)} run dirs) -> train_{tag}.extxyz, valid_{tag}.extxyz")
