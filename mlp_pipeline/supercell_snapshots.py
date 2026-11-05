#!/usr/bin/env python3
"""Snapshots of an ORDERED crystal (e.g. the hydrogen-hydrate phases C2 / C3, H4O / H6O) for DFT labelling: a supercell large enough for Gamma-only sampling, volumes scaled to sample a pressure range, atoms rattled.
   python3 mlp_pipeline/supercell_snapshots.py STRUCT.cif [STRUCT2.vasp ...] --name C2 --out runs/hydrates/C2.extxyz [--min-atoms 200 --max-atoms 450 --min-edge 8.5] [--scales 0.80 0.86 0.92 1.0 1.06] [--rattle 0.03 0.08] [--seeds 2] [--T 300]
The supercell is the DIAGONAL repeat (n1, n2, n3) whose atom count lies in [min-atoms, max-atoms] with the longest allowed shortest-cell-edge (>= --min-edge A); each frame = volume scale (linear, applied to the cell and positions) x rattle sigma (A) x seed.
Frames carry info config_type=hydrate_<name>, T_K (the electronic temperature used by the DFT writer: its floor is 1000 K), scale and rattle. Molecules are NOT kept rigid by the rattle (a plain Gaussian displacement): keep sigma <= 0.08 A,
or generate thermal frames with GRACE MD at the target volumes instead (17_eos_md.py style) and use this script for the static strained / rattled set.
Check the structure first (ASE reads cif / POSCAR / extxyz); this script does NOT know the chemistry: the stoichiometry, density and bonds are whatever the input file has."""
import argparse, itertools, sys
import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--name", default="X"); ap.add_argument("--out", required=True)
ap.add_argument("--min-atoms", type=int, default=200); ap.add_argument("--max-atoms", type=int, default=450); ap.add_argument("--min-edge", type=float, default=8.5)
ap.add_argument("--scales", type=float, nargs="+", default=[0.80, 0.86, 0.92, 1.0, 1.06]); ap.add_argument("--rattle", type=float, nargs="+", default=[0.03, 0.08]); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--T", type=float, default=300.0); a = ap.parse_args()
frames = []
for fn in a.files:
    u = read(fn); cell = u.cell.array; lens = np.linalg.norm(cell, axis=1); n0 = len(u)
    best = None
    for n in itertools.product(range(1, 9), repeat=3):
        N = n0 * n[0] * n[1] * n[2]
        if not a.min_atoms <= N <= a.max_atoms: continue
        edges = np.array(n) * lens; score = (min(edges) >= a.min_edge, min(edges), -abs(N - 0.5 * (a.min_atoms + a.max_atoms)))
        if best is None or score > best[0]: best = (score, n, N, edges)
    if best is None: sys.exit(f"{fn}: no diagonal supercell with {a.min_atoms}-{a.max_atoms} atoms from {n0} atoms (cell edges {lens.round(2)}): relax --min-atoms / --max-atoms")
    _, n, N, edges = best; sc = u.repeat(n); comp = {s: int((sc.get_chemical_symbols().count(s))) for s in sorted(set(sc.get_chemical_symbols()))}
    print(f"{fn}: {n0} atoms -> supercell {n[0]}x{n[1]}x{n[2]} = {N} atoms {comp}, edges {edges.round(2)} A" + ("" if min(edges) >= a.min_edge else f"   WARNING: shortest edge below {a.min_edge} A"), file=sys.stderr)
    for sc_ in a.scales:
        for sg in a.rattle:
            for s in range(a.seeds):
                f = sc.copy(); f.set_cell(sc.cell.array * sc_, scale_atoms=True); rng = np.random.default_rng(1000 * s + int(100 * sc_) + int(1000 * sg))
                f.positions += rng.normal(0, sg, f.positions.shape); f.wrap()
                f.info.update(config_type=f"hydrate_{a.name}", T_K=a.T, scale=sc_, rattle=sg, source=fn, nH=int((f.numbers == 1).sum()), nO=int((f.numbers == 8).sum())); frames.append(f)
import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); write(a.out, frames); print(f"{len(frames)} frames -> {a.out}", file=sys.stderr)
