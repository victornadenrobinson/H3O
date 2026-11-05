#!/usr/bin/env python3
"""Snapshots of an ORDERED crystal (e.g. the hydrogen-hydrate phases C2 / C3, H4O / H6O) for DFT labelling: a supercell large enough for Gamma-only sampling, volumes scaled to sample a pressure range, atoms rattled.
   python3 mlp_pipeline/supercell_snapshots.py STRUCT.cif [STRUCT2.vasp ...] --name C2 --out runs/hydrates/C2.extxyz [--min-atoms 200 --max-atoms 450 --min-edge 8.5] [--scales 0.80 0.86 0.92 1.0 1.06] [--rattle 0.03 0.08] [--seeds 2] [--T 300]
The supercell is the DIAGONAL repeat (n1, n2, n3) whose atom count lies in [min-atoms, max-atoms] with the longest allowed shortest-cell-edge (>= --min-edge A); each frame = volume scale (linear, applied to the cell and positions) x rattle sigma (A) x seed.
Frames carry info config_type=hydrate_<name>, T_K (the electronic temperature used by the DFT writer: its floor is 1000 K), scale and rattle. The volume scaling keeps molecules rigid (centres of mass scale, bonds do not: --no-rigid for the plain scaling); the RATTLE is a plain Gaussian displacement of every atom, so keep sigma <= 0.08 A,
or generate thermal frames with GRACE MD at the target volumes instead (17_eos_md.py style) and use this script for the static strained / rattled set.
Check the structure first (ASE reads cif / POSCAR / extxyz); this script does NOT know the chemistry: the stoichiometry, density and bonds are whatever the input file has."""
import argparse, itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--name", default="X"); ap.add_argument("--out", required=True)
ap.add_argument("--min-atoms", type=int, default=200); ap.add_argument("--max-atoms", type=int, default=450); ap.add_argument("--min-edge", type=float, default=8.5)
ap.add_argument("--scales", type=float, nargs="+", default=[0.80, 0.86, 0.92, 1.0, 1.06]); ap.add_argument("--rattle", type=float, nargs="+", default=[0.03, 0.08]); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--T", type=float, default=300.0); ap.add_argument("--relax", default=None, help="relax the atomic positions at each scaled volume with this model before rattling (grace-ft:<dir>; needs a GPU node), e.g. grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model"); ap.add_argument("--relax-fmax", type=float, default=0.1); ap.add_argument("--relax-steps", type=int, default=300); ap.add_argument("--device", default="cuda")
ap.add_argument("--min-hh", type=float, default=0.55); ap.add_argument("--min-oh", type=float, default=0.85); ap.add_argument("--no-rigid", dest="rigid", action="store_false", help="scale ALL positions with the cell (shrinks the molecules too); default: rigid molecules"); a = ap.parse_args()
def scaled(at, s):
    f = at.copy(); f.set_cell(at.cell.array * s, scale_atoms=True); return f


def rigid_scale(at, s, oh=1.25, hh=0.95):
    """volume scaling that keeps every MOLECULE rigid: molecules are the connected components of O-H < oh and H-H < hh bonds (H2O, H2); their centres of mass follow the cell (scaled by s), the internal geometry is unchanged.
    A plain scaling of all positions would also shrink the bonds (0.74 A H2 -> 0.64 A at s = 0.86), which real molecules do not do."""
    from ase.neighborlist import neighbor_list
    from ase.geometry import find_mic
    n = len(at); I, J, D = neighbor_list("ijd", at, max(oh, hh)); sym = at.numbers; ok = np.where((sym[I] != sym[J]) & (D < oh) | (sym[I] == 1) & (sym[J] == 1) & (D < hh), True, False); parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(I[ok], J[ok]): parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    f = at.copy(); f.set_cell(at.cell.array * s, scale_atoms=False); pos = at.positions.copy(); m = at.get_masses(); new = pos.copy()
    for g in groups.values():
        rel, _ = find_mic(pos[g] - pos[g[0]], at.cell, True); P = pos[g[0]] + rel; com = (P * m[g][:, None]).sum(0) / m[g].sum(); new[g] = com * s + (P - com)      # centre scaled with the cell, internal vectors kept
    f.set_positions(new); f.wrap(); return f


def relax_positions(at, spec, fmax, steps, device):
    """FIRE relaxation of the atomic positions at the fixed (scaled) cell with the model `spec` (grace-ft:<dir> | grace:<name> | emt for tests): the molecules rebuild their bonds for the new density"""
    from ase.optimize import FIRE
    f = at.copy()
    if spec == "emt":
        from ase.calculators.emt import EMT; f.calc = EMT()
    else:
        from importlib import import_module; f.calc = import_module("04_benchmark").get_calc_spec(spec, device)
    f0 = max(np.linalg.norm(f.get_forces(), axis=1)); FIRE(f, logfile=None).run(fmax=fmax, steps=steps); f1 = max(np.linalg.norm(f.get_forces(), axis=1))
    from ase.neighborlist import neighbor_list
    I, J, D = neighbor_list("ijd", f, 1.3); sy = f.numbers; hh = D[(sy[I] == 1) & (sy[J] == 1)]; oh = D[(sy[I] == 8) & (sy[J] == 1)]
    print(f"  relaxed at this volume: max force {f0:.2f} -> {f1:.2f} eV/A; shortest H-H {hh.min() if len(hh) else float('nan'):.2f} A, O-H {oh.min() if len(oh) else float('nan'):.2f} A (H2 ~0.74, O-H ~1.0: far from these = the model broke the molecules)", file=sys.stderr); f.calc = None; return f


def contacts_ok(at, min_hh, min_oh):
    from ase.neighborlist import neighbor_list
    I, J, D = neighbor_list("ijd", at, max(min_hh, min_oh)); sy = at.numbers
    return not (((sy[I] == 1) & (sy[J] == 1) & (D < min_hh)) | ((sy[I] != sy[J]) & (D < min_oh))).any()


def load(fn):
    """cif / POSCAR / extxyz via ASE; a POSCAR with element labels in the coordinate lines (some exporters) is read by hand; a .vesta file is expanded from its asymmetric unit + SYMOP list"""
    if fn.endswith(".vesta"):
        from vesta_to_poscar import parse
        from ase import Atoms
        cell, ops, atoms = parse(fn); out = []
        for el, x in atoms:
            for t, R in ops:
                y = (R @ x + t) % 1.0
                if not any(e == el and np.linalg.norm(((y - z + 0.5) % 1.0) - 0.5) < 1e-3 for e, z in out): out.append((el, y))
        return Atoms([e for e, _ in out], scaled_positions=[y for _, y in out], cell=cell, pbc=True)
    try: return read(fn)
    except Exception:
        from ase import Atoms
        L = [l.split() for l in open(fn).read().splitlines()]; scale = float(L[1][0]); cell = scale * np.array([[float(x) for x in L[i]] for i in (2, 3, 4)])
        sy, cnt = L[5], [int(x) for x in L[6]]; k = 7 + (L[7][0][0] in "sS")             # optional 'Selective dynamics' line
        cartesian = L[k][0][0] in "cCkK"; rows = [r for r in L[k + 1:k + 1 + sum(cnt)]]; pos = np.array([[float(x) for x in r[-3:]] if not r[0].replace('.', '').replace('-', '').isdigit() else [float(x) for x in r[:3]] for r in rows])
        at = Atoms([s_ for s_, c in zip(sy, cnt) for _ in range(c)], cell=cell, pbc=True)
        (at.set_positions(scale * pos) if cartesian else at.set_scaled_positions(pos)); return at


frames = []
for fn in a.files:
    u = load(fn); cell = u.cell.array; lens = np.linalg.norm(cell, axis=1); n0 = len(u)
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
        f0 = rigid_scale(sc, sc_) if a.rigid else scaled(sc, sc_)
        if a.relax:                                                       # let the bonds / hydrogen bonds respond to the new density: positions only, cell fixed
            f0 = relax_positions(f0, a.relax, a.relax_fmax, a.relax_steps, a.device)
        for sg in a.rattle:
            for s in range(a.seeds):
                f = f0.copy(); rng = np.random.default_rng(1000 * s + int(100 * sc_) + int(1000 * sg))
                base = f.positions.copy()
                for _try in range(100):                                       # redraw a rattle that squeezes any H-H or O-H below the limits (a rare Gaussian tail would otherwise make a 0.4 A H2 bond)
                    f.positions = base + rng.normal(0, sg, base.shape)
                    if sg == 0 or contacts_ok(f, a.min_hh, a.min_oh): break
                else: print(f"WARNING: no rattle within the contact limits after 100 draws (scale {sc_}, sigma {sg})", file=sys.stderr)
                f.wrap()
                f.info.update(config_type=f"hydrate_{a.name}", T_K=a.T, scale=sc_, rattle=sg, source=fn, nH=int((f.numbers == 1).sum()), nO=int((f.numbers == 8).sum())); frames.append(f)
import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); write(a.out, frames); print(f"{len(frames)} frames -> {a.out}", file=sys.stderr)
