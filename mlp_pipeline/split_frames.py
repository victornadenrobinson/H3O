#!/usr/bin/env python3
"""Split an extxyz into N chunks, keeping every structure of one run_id (one PWidom frame: host + insertions + deletions) in the same chunk.
   python mlp_pipeline/split_frames.py FRAMES.extxyz N OUTDIR     -> OUTDIR/frames_0.extxyz ... (empty chunks are not written)"""
import collections, os, sys
from ase.io import read, write

src, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
frames = read(src, ":"); groups = collections.OrderedDict()
for i, f in enumerate(frames): groups.setdefault(f.info.get("run_id", f"single{i}"), []).append(f)
bins = [[] for _ in range(k)]
for g in sorted(groups.values(), key=len, reverse=True): min(bins, key=len).extend(g)   # largest groups first, each to the emptiest chunk
for j, b in enumerate(bins):
    if b: write(os.path.join(out, f"frames_{j}.extxyz"), b)
print(f"{len(frames)} geometries in {len(groups)} groups -> chunk sizes {[len(b) for b in bins]}")
