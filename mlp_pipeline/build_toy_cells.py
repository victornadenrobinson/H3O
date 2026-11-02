"""Structure generator on the cheap 'toyice' pair potential: DFT-sized cells (H2O .. H3O, 324-432 atoms; pure H, 729 atoms) with deliberate defects, for DFT labelling.
The toy energies/forces are NOT used as labels (DFT labels them); the toy is only a fast sampler of disordered, correlated geometries (0.01-0.02 s/step on one CPU core).

Pipeline: build crystal (fcc/bcc O sublattice + protons on bonds, or sc pure H) at --vpa, run short toy MD at --T (and at --T-hot for 'hot'), take base snapshots,
then apply ONE primary perturbation per output frame:
  shake   gaussian displacement of every atom, sigma 0.05-0.30 A
  vac     delete 1-4 O or 2-8 H atoms at random
  void    delete the 4-16 atoms nearest a random point (a hole)
  ins     insert 1-27 H into large holes (min distance O 0.85 / H 0.60, as build_insertion_set.py)
  cramp   isotropic compression (volume x 0.80-0.95), uniaxial strain, shear strain
  expand  isotropic expansion (volume x 1.10-1.30): stretched bonds, cavities
  close   1-4 forced close contacts: H-H 0.45-0.70, O-H 0.65-0.90, O-O 1.3-1.8 A
  hot     snapshot of toy MD at --T-hot (strongly disordered)
Frames whose closest H-H / O-H / O-O distance is below the floors (0.40 / 0.55 / 1.10 A) are dropped (resampled), so DFT does not meet nuclear-fusion geometries.
Each frame carries info: kind, param, base_id, split, nO, nH, T_K, vpa, d_min_HH/OH/OO, config_type = toy_<kind>_<sys>.
usage (repo root, CPU only, no GPU):
  python mlp_pipeline/build_toy_cells.py --sys h2o --phase fcc --nrep 3 --vpa 2.12 --per-kind 6 --out runs/toy_cells/h2o_fcc.extxyz
  python mlp_pipeline/build_toy_cells.py --sys H --vpa 1.25 --per-kind 8 --out runs/toy_cells/H729.extxyz
  --sys h2o|h2.25o|h2.5o|h3o|H ; --phase fcc (nrep 3 -> 108 O) or bcc (nrep 4 -> 128 O); H: sc 9^3 = 729 atoms
The base trajectories go whole to one side of a split: --split train|valid just labels the output; use different --seed for the other side.
"""
import argparse, os, sys, time
from types import SimpleNamespace
import numpy as np
from ase import Atoms, units
from ase.geometry import get_distances
from ase.io import write
from ase.neighborlist import neighbor_list

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tdi"))
HPER = {"h2o": 2.0, "h2.25o": 2.25, "h2.5o": 2.5, "h3o": 3.0}
FLOOR = {"HH": 0.40, "OH": 0.55, "OO": 1.10}
ALL_KINDS = ["shake", "vac", "void", "ins", "cramp", "expand", "close", "hot"]


def build(a):
    from importlib import import_module
    if a.sys == "H":
        return import_module("tdi.run_real").build_h_n(a.vpa, a.natoms)
    return import_module("17_eos_md").build(a.phase, a.vpa, a.nrep, a.seed, HPER[a.sys])


def toy_md(at, T, steps, seed, dt=1.0):
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from toy_ice import ToyIce
    at = at.copy(); at.calc = ToyIce(); rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng); Stationary(at)
    Langevin(at, dt * units.fs, temperature_K=T, friction=1 / (100 * units.fs), rng=rng).run(steps)
    at.calc = None; return at


def dmin(at):
    """closest distance per species pair"""
    i, j, d = neighbor_list("ijd", at, 2.2); z = at.numbers; out = {}
    for name, (p, q) in {"HH": (1, 1), "OH": (8, 1), "OO": (8, 8)}.items():
        m = ((z[i] == p) & (z[j] == q)) | ((z[i] == q) & (z[j] == p)); out[name] = float(d[m].min()) if m.any() else 9.9
    return out


def ok(at):
    d = dmin(at); return all(d[k] >= FLOOR[k] for k in FLOOR), d


def delete(at, idx):
    keep = np.ones(len(at), bool); keep[list(idx)] = False; return at[keep]


def p_shake(at, rng):
    s = float(rng.choice([0.05, 0.10, 0.20, 0.30])); b = at.copy(); b.positions += rng.normal(0, s, b.positions.shape); return b, f"sigma={s}"


def p_vac(at, rng):
    sp, n = (8, int(rng.integers(1, 5))) if rng.random() < 0.4 and (at.numbers == 8).any() else (1, int(rng.integers(2, 9)))
    idx = rng.choice(np.where(at.numbers == sp)[0], n, replace=False); return delete(at, idx), f"{'O' if sp == 8 else 'H'}x{n}"


def p_void(at, rng):
    k = int(rng.integers(4, 17)); c = rng.random(3) @ at.cell.array
    _, d = get_distances(c[None], at.positions, cell=at.cell, pbc=True); idx = np.argsort(d[0])[:k]
    return delete(at, idx), f"n={k} r={d[0][idx].max():.2f}"


def p_ins(at, rng):
    from build_insertion_set import insert_H
    n = int(rng.choice([1, 2, 4, 8, 16, 27])); a = SimpleNamespace(cand=400, min_d_O=0.85, min_d_H=0.60, power=4.0)
    return insert_H(at, n, rng, a), f"n={n}"


def p_cramp(at, rng):
    b = at.copy(); u = rng.random()
    if u < 0.5:
        f = float(rng.uniform(0.80, 0.95)); b.set_cell(at.cell.array * f ** (1 / 3), scale_atoms=True); return b, f"vol x{f:.3f}"
    e = float(rng.uniform(0.05, 0.15)); M = np.eye(3)
    if u < 0.75: ax = int(rng.integers(3)); M[ax, ax] = 1 - e; tag = f"uniaxial ax{ax} -{e:.3f}"
    else: M[0, 1] = e; tag = f"shear xy {e:.3f}"
    b.set_cell(at.cell.array @ M, scale_atoms=True); return b, tag


def p_expand(at, rng):
    f = float(rng.uniform(1.10, 1.30)); b = at.copy(); b.set_cell(at.cell.array * f ** (1 / 3), scale_atoms=True); return b, f"vol x{f:.3f}"


def p_close(at, rng):
    b = at.copy(); n = int(rng.integers(1, 5)); tags = []
    for _ in range(n):
        hasO = bool((b.numbers == 8).any()); kind = str(rng.choice(["HH", "OH", "OO"], p=[0.5, 0.35, 0.15])) if hasO else "HH"; lo, hi = {"HH": (0.45, 0.70), "OH": (0.65, 0.90), "OO": (1.3, 1.8)}[kind]; d0 = float(rng.uniform(lo, hi))
        i = int(rng.choice(np.where(b.numbers == (8 if kind == "OO" else 1))[0])); want = 8 if kind in ("OO", "OH") else 1
        if kind == "OH": i = int(rng.choice(np.where(b.numbers == 1)[0]))
        D, L = get_distances(b.positions[i][None], b.positions, cell=b.cell, pbc=True)        # D: minimum-image vectors i -> all atoms, L: their lengths
        r = L[0]; cand = np.where((b.numbers == want) & (r > 0.1))[0]; j = int(cand[np.argmin(r[cand])])   # nearest partner of the right species
        v = D[0][j] / np.linalg.norm(D[0][j]); b.positions[i] = b.positions[j] - v * d0; tags.append(f"{kind}={d0:.2f}")
    return b, ",".join(tags)


PERT = {"shake": p_shake, "vac": p_vac, "void": p_void, "ins": p_ins, "cramp": p_cramp, "expand": p_expand, "close": p_close}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sys", default="h2o", choices=list(HPER) + ["H"]); ap.add_argument("--phase", default="fcc"); ap.add_argument("--nrep", type=int, default=3); ap.add_argument("--natoms", type=int, default=729)
    ap.add_argument("--vpa", type=float, required=True, help="volume per atom, A^3 (about 2.12 H2O at 500 GPa, 1.25 pure H)"); ap.add_argument("--T", type=float, default=5000.0); ap.add_argument("--T-hot", type=float, default=10000.0)
    ap.add_argument("--bases", type=int, default=4, help="base snapshots from the toy MD"); ap.add_argument("--md-steps", type=int, default=600); ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--kinds", default=",".join(ALL_KINDS)); ap.add_argument("--per-kind", type=int, default=6, help="frames per kind per base"); ap.add_argument("--split", default="train"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); kinds = a.kinds.split(","); rng = np.random.default_rng(a.seed)
    at0 = build(a); print(f"{a.sys}: {len(at0)} atoms ({int((at0.numbers == 8).sum())} O, {int((at0.numbers == 1).sum())} H), V/atom {a.vpa}", flush=True)
    t0 = time.time(); bases = []; cur = toy_md(at0, a.T, a.md_steps, a.seed)
    for k in range(a.bases):
        bases.append(cur); cur = toy_md(cur, a.T, a.every, a.seed + 1 + k)
    hots = [toy_md(b, a.T_hot, 200, a.seed + 100 + k) for k, b in enumerate(bases)] if "hot" in kinds else []
    print(f"toy MD done in {time.time() - t0:.0f} s: {len(bases)} bases", flush=True)
    if os.path.exists(a.out): os.remove(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); n_out = 0; drop = {}
    for bi, base in enumerate(bases):
        for kind in kinds:
            made = 0; tries = 0
            while made < a.per_kind and tries < 30 * a.per_kind:
                tries += 1
                try:
                    if kind == "hot": b, par = hots[bi], f"T={a.T_hot:.0f}"                   # one hot frame per base is all the toy MD gives; per-kind is capped below
                    else: b, par = PERT[kind](base, rng)
                except RuntimeError as e: drop[kind] = drop.get(kind, 0) + 1; continue
                good, d = ok(b)
                if not good: drop[kind] = drop.get(kind, 0) + 1; continue
                b = b.copy(); b.wrap()
                b.info.update(kind=kind, param=par, base_id=f"{a.sys}:{a.seed}:{bi}", split=a.split, nO=int((b.numbers == 8).sum()), nH=int((b.numbers == 1).sum()), T_K=a.T, vpa=a.vpa,
                              d_min_HH=d["HH"], d_min_OH=d["OH"], d_min_OO=d["OO"], config_type=f"toy_{kind}_{a.sys}", method="toy")
                write(a.out, b, append=True); made += 1; n_out += 1
                if kind == "hot": break
    print(f"wrote {n_out} frames to {a.out}; dropped by distance floors / placement failures: {drop}")


if __name__ == "__main__":
    main()
