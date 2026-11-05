#!/usr/bin/env python3
"""Expand a VESTA file (cell, asymmetric unit, SYMOP list) into the full unit cell and write a POSCAR (the .vesta files of the hydrate phases carry only the asymmetric unit + symmetry operations).
   python3 mlp_pipeline/vesta_to_poscar.py in.vesta out.vasp
SYMOP lines are 't1 t2 t3 r11 r12 r13 r21 r22 r23 r31 r32 r33 flag' (fractional coordinates: x' = R x + t); duplicates (mod 1, 1e-3) are merged. Element symbols are taken from the STRUC lines; species are ordered O then H."""
import sys
import numpy as np
from ase import Atoms
from ase.io import write


def parse(fn):
    lines = open(fn).read().splitlines(); cell = ops = None; atoms = []
    for i, l in enumerate(lines):
        if l.strip() == "CELLP": cell = [float(x) for x in lines[i + 1].split()[:6]]
        if l.strip() == "SYMOP":
            ops = []; j = i + 1
            while True:
                t = lines[j].split()
                if len(t) < 13 or t[0] == "-1.0": break
                v = [float(x) for x in t[:12]]; ops.append((np.array(v[:3]), np.array(v[3:]).reshape(3, 3))); j += 1
        if l.strip() == "STRUC":
            j = i + 1
            while lines[j].split()[0] != "0":
                t = lines[j].split(); atoms.append((t[1], np.array([float(x) for x in t[4:7]]))); j += 2
    return cell, ops, atoms


if __name__ == "__main__":
    cell, ops, atoms = parse(sys.argv[1]); out = []
    for el, x in atoms:
        for t, R in ops:
            y = (R @ x + t) % 1.0
            if not any(e == el and np.linalg.norm(((y - z + 0.5) % 1.0) - 0.5) < 1e-3 for e, z in out): out.append((el, y))
    out.sort(key=lambda a: (a[0] != "O", a[0]))
    at = Atoms([e for e, _ in out], scaled_positions=[y for _, y in out], cell=cell, pbc=True); write(sys.argv[2], at, format="vasp", direct=True, sort=False)
    print(f"{sys.argv[1]}: {len(atoms)} asymmetric atoms x {len(ops)} operations -> {len(at)} atoms {at.get_chemical_formula()} in cell {np.round(at.cell.cellpar(), 3)}")
