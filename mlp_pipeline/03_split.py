"""80/20 train/valid split grouped by source run directory (no frame leakage).
usage: python 03_split.py snapshots.extxyz [seed]
       python 03_split.py snapshots_l.extxyz --valid-from valid_m.extxyz
The second form puts EVERY run directory that appears in valid_m into the validation set and all other runs into training (writes train_l_vm / valid_l_vm).
Each tier is otherwise shuffled on its own, so valid_m runs can sit in train_l: use this form whenever a model trained on a larger tier is judged on valid_m."""
import os, sys, random
from common import out, inp
from ase.io import read, write

frames = read(inp(sys.argv[1]), index=":")
stem = os.path.basename(sys.argv[1]).rsplit(".", 1)[0]
tag = stem.split("snapshots_", 1)[1] if stem.startswith("snapshots_") else stem
vfrom = None
if "--valid-from" in sys.argv:
    i = sys.argv.index("--valid-from"); vfrom = sys.argv[i + 1]; del sys.argv[i:i + 2]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
groups = {}
for f in frames:
    groups.setdefault(os.path.dirname(f.info["source"]), []).append(f)
keys = sorted(groups); random.Random(seed).shuffle(keys)
if vfrom:
    vdirs = {os.path.dirname(f.info["source"]) for f in read(inp(vfrom), index=":")}
    tr = [f for k in keys if k not in vdirs for f in groups[k]]; va = [f for k in keys if k in vdirs for f in groups[k]]
    tag += "_vm"
    print(f"validation runs from {vfrom}: {len(vdirs)} run dirs, {sum(k in vdirs for k in keys)} of them present in this tier")
else:
    n_tr = int(0.8 * len(frames)); tr, va = [], []
    for k in keys:
        (tr if len(tr) < n_tr else va).extend(groups[k])
write(out(f"train_{tag}.extxyz"), tr); write(out(f"valid_{tag}.extxyz"), va)
print(f"train {len(tr)}  valid {len(va)}  ({len(keys)} run dirs) -> train_{tag}.extxyz, valid_{tag}.extxyz")
