"""Short-range probe of the MLP: energy cost dU(r) = U(host + H) - U(host) of an extra proton placed at distance r from a host O or H atom (random directions, several frames).
The ideal-gas reference of every TDI path here (insertion ghost, and all H at small lambda in the crystal / fluid paths) makes the integrand sample such overlaps; a physical
potential is strongly repulsive there. A dU(r) that is not repulsive at r < ~0.7 A (a 'hole': lower than at the bonding distance) biases the free energies, and differently for the
insertion route (one ghost in a fully interacting host) and the crystal route (all H decoupled together), so the two estimates of the mixing free energy would disagree.
usage (repo root, GPU or CPU):  ~/grace-venv/bin/python mlp_pipeline/tdi/short_range.py runs/tdi insert_500GPa_6000K_n1 [--frames 5 --dirs 20 --device cuda]
Frames: the last --frames configurations of <tag>_node<K>_traj.extxyz for the highest lambda node; the n ghosts (last atoms, from the stage0 json) are removed first."""
import argparse, json, os, sys
import numpy as np
from ase import Atoms
from ase.io import read
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
from tdi import analyze

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tag"); ap.add_argument("--frames", type=int, default=5); ap.add_argument("--dirs", type=int, default=20)
ap.add_argument("--device", default="cuda"); ap.add_argument("--r", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
ap.add_argument("--uniform", type=int, default=200, help="also: dU at this many uniformly random points per frame (the lambda -> 0 integrand of the insertion route)"); a = ap.parse_args()
st = json.load(open(os.path.join(a.dir, f"{a.tag}_stage0.json"))); A = st["args"]
lam = analyze.rule_nodes(A)[0]; k = int(np.argmax(lam)); f = os.path.join(a.dir, f"{a.tag}_node{k}_traj.extxyz")
ng = int(st["N"]) if st["phase"] == "insert" else 0
calc = import_module("04_benchmark").get_calc_spec(A["model"], a.device)
rng = np.random.default_rng(0)
frames = read(f, index=":")[-a.frames:]
print(f"{a.tag}: {len(frames)} frames of node {k} (lambda {lam[k]:.3f}), {ng} ghost(s) removed; dU in eV")
rows = {"O": [], "H": []}; uni = []
for fr in frames:
    host = fr[:len(fr) - ng] if ng else fr.copy(); host.calc = calc; e0 = host.get_potential_energy()
    def dU(p):
        b = host + Atoms("H", positions=[p]); b.calc = calc
        return b.get_potential_energy() - e0
    for el, z in (("O", 8), ("H", 1)):
        idx = np.where(host.numbers == z)[0]
        if not len(idx): continue
        for _ in range(a.dirs):
            i = rng.choice(idx); v = rng.normal(size=3); v /= np.linalg.norm(v)
            rows[el].append([dU(host.positions[i] + r * v) for r in a.r])
    uni += [dU(rng.random(3) @ host.cell.array) for _ in range(a.uniform)]
for el, R in rows.items():
    if not R: continue
    R = np.array(R)
    print(f"\nproton at r from a host {el} atom ({len(R)} probes): r (A), dU median, min, max (eV)")
    for j, r in enumerate(a.r):
        print(f"  {r:4.2f}  {np.median(R[:, j]):10.3f} {R[:, j].min():10.3f} {R[:, j].max():10.3f}")
    ref = np.median(R[:, a.r.index(1.0)]) if 1.0 in a.r else np.median(R[:, -1])
    holes = int((R[:, [j for j, r in enumerate(a.r) if r <= 0.6]].min(1) < ref).sum())
    print(f"  probes with dU(r <= 0.6 A) below the median dU at 1.0 A ({ref:.3f} eV): {holes} of {len(R)}" + ("   <- NOT REPULSIVE AT SHORT RANGE" if holes else ""))
u = np.array(uni)
print(f"\nuniform random points ({len(u)}): dU median {np.median(u):.3f}, mean {u.mean():.3f}, min {u.min():.3f}, max {u.max():.3f} eV; fraction below 0: {np.mean(u < 0):.2f}")
