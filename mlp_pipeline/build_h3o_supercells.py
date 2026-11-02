"""Supercells of the relaxed Huang H3O phase (and its guest-H-depleted variants) at many pressures, DFT-sized like the 108-O fcc cells, as DFT-labelling candidates.
Input: the static relaxations of h3o_static.py (runs/h3o_static/P<P>/{H3O,H2.75O,H2.5O,H2.25O}_<P>GPa_relaxed.extxyz, 52-64 atoms, relaxed at <P>).
Supercell: the integer matrix with |det| = round(target / natoms) whose cell has the largest minimum perpendicular width (the 64-atom H3O cell is 3.3 x 5.8 x 5.8 A: a plain
3x2x1 would have a 5 A side at 450 GPa; the search gives ~8-9 A for 384 atoms). Per supercell the frames are
  static      the relaxed structure (0 K, exact lattice: forces ~ 0, tests the phase itself)
  rattle      gaussian displacements sigma 0.05 / 0.10 / 0.20 A
  strain      isotropic volume x 0.94 / 0.97 / 1.03 / 1.06 (pressure variety around each relaxation)
  md          GRACE NVT MD at --T-list (default 1000 2000 3000 4000 K) from the rattled start, one frame per --save-every-ps after --equil-ps (needs --model and a GPU)
  vac/void/ins/close   the defect kinds of build_toy_cells.py on the static supercell (vacancy, hole, extra H, forced close contact)
Frames carry: kind, phase, P_static, T_K, nO, nH, base_id, split (--valid-P: that pressure goes to the validation side whole), config_type = h3osc_<phase>_<kind>, P_grace_GPa for md frames.
usage (repo root; the md part on a GPU node):
  python mlp_pipeline/build_h3o_supercells.py --static-dir runs/h3o_static --model grace-ft:runs/grace/ft_m2/seed/1/final_model --P 300 400 500 600 700 800 --valid-P 450 --out runs/h3o_sc/h3o_sc.extxyz
  --no-md  skips the MD (CPU only); --target 384 sets the atom count; --phases H3O H2.75O H2.5O H2.25O
"""
import argparse, glob, itertools, os, sys, time
from types import SimpleNamespace
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def min_width(cell):
    c = np.asarray(cell); vol = abs(np.linalg.det(c)); return min(vol / np.linalg.norm(np.cross(c[(i + 1) % 3], c[(i + 2) % 3])) for i in range(3))


def best_supercell(cell, m, rng_=2):
    """integer matrix (rows = new vectors in old cell vectors), det = m, with the largest minimum perpendicular width"""
    best = (-1, None); vecs = [np.array(v) for v in itertools.product(range(-rng_, rng_ + 1), repeat=3) if any(v)]
    cell = np.asarray(cell); L = [np.linalg.norm(v @ cell) for v in vecs]; order = np.argsort(L)[:90]; vecs = [vecs[i] for i in order]   # short candidate vectors only
    for a_, b_, c_ in itertools.combinations(vecs, 3):
        M = np.array([a_, b_, c_]); d = round(np.linalg.det(M))
        if abs(d) != m: continue
        if d < 0: M = np.array([b_, a_, c_])
        w = min_width(M @ cell)
        if w > best[0] + 1e-9: best = (w, M)
    return best[1], best[0]


def make_sc(at, target, lo=320, hi=420):
    """supercell multiplicity m with lo <= m * natoms <= hi (else the nearest to target) whose best cell is the most cubic: largest min perpendicular width / V^(1/3) (1 = cube)"""
    from ase.build import make_supercell
    ms = [m for m in range(1, 40) if lo <= m * len(at) <= hi] or [max(1, round(target / len(at)))]; best = None
    for m in ms:
        M, w = best_supercell(at.cell.array, m, 3); sph = w / (m * at.get_volume()) ** (1 / 3)
        if best is None or sph > best[0]: best = (sph, M, w)
    sc = make_supercell(at, best[1]); sc.wrap(); return sc, best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-dir", default="runs/h3o_static"); ap.add_argument("--P", type=float, nargs="*"); ap.add_argument("--phases", nargs="+", default=["H3O", "H2.75O", "H2.5O", "H2.25O"])
    ap.add_argument("--target", type=int, default=384); ap.add_argument("--min-atoms", type=int, default=320); ap.add_argument("--max-atoms", type=int, default=420); ap.add_argument("--valid-P", type=float, nargs="*", default=[]); ap.add_argument("--model"); ap.add_argument("--no-md", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--T-list", type=float, nargs="+", default=[1000, 2000, 3000, 4000]); ap.add_argument("--equil-ps", type=float, default=0.2); ap.add_argument("--save-every-ps", type=float, default=0.1); ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--md-frames", type=int, default=3); ap.add_argument("--ins-n", type=int, nargs="*", default=[], help="also insert n extra H (e.g. 1 2 3 4 5 6) into the static supercell and run a short GRACE-MD at --ins-T (needs --model)"); ap.add_argument("--ins-T", type=float, nargs="+", default=[3000.0]); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    from build_toy_cells import PERT, ok
    from build_insertion_set import get_calc, relax_md, finish
    calc = get_calc(a.model, a.device) if (a.model and not a.no_md) else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out): os.remove(a.out)
    files = []
    for ph in a.phases:
        for f in sorted(glob.glob(os.path.join(a.static_dir, "P*", f"{ph}_*GPa_relaxed.extxyz"))):
            P = float(os.path.basename(f).split("_")[1].replace("GPa", ""))
            if not a.P or P in a.P: files.append((ph, P, f))
    print(f"{len(files)} relaxed structures", flush=True); n = 0
    for ph, P, f in files:
        sc, M, w = make_sc(read(f), a.target, a.min_atoms, a.max_atoms); split = "valid" if P in a.valid_P else "train"; bid = f"h3osc:{ph}:{P:g}"
        base = dict(phase=ph, P_static=P, split=split, base_id=bid, T_K=0.0)
        def emit(at, kind, par="", T=0.0, extra=None):
            nonlocal n
            at = at.copy(); at.calc = None; at.info.update(base, kind=kind, param=par, T_K=T, nO=int((at.numbers == 8).sum()), nH=int((at.numbers == 1).sum()), config_type=f"h3osc_{ph}_{kind}", **(extra or {}))
            write(a.out, at, append=True); n += 1
        print(f"{ph} {P:g} GPa: {len(sc)} atoms, supercell det {round(np.linalg.det(M))}, min width {w:.2f} A = {w / sc.get_volume() ** (1 / 3):.2f} of a cube ({split})", flush=True)
        emit(sc, "static")
        for s in (0.05, 0.10, 0.20):
            b = sc.copy(); b.positions += rng.normal(0, s, b.positions.shape); emit(b, "rattle", f"sigma={s}")
        for fv in (0.94, 0.97, 1.03, 1.06):
            b = sc.copy(); b.set_cell(sc.cell.array * fv ** (1 / 3), scale_atoms=True); emit(b, "strain", f"vol x{fv}")
        for kind in ("vac", "void", "ins", "close"):
            for _ in range(2):
                for _try in range(20):
                    try: b, par = PERT[kind](sc, rng)
                    except RuntimeError: continue
                    if ok(b)[0]: emit(b, kind, par); break
        if calc is not None:
            for T in a.T_list:
                st = sc.copy(); st.positions += rng.normal(0, 0.05, st.positions.shape)
                md_a = SimpleNamespace(T=T, dt=a.dt, equil_ps=a.equil_ps, sample_ps=a.save_every_ps * (a.md_frames - 1), save_every_ps=a.save_every_ps)
                t0 = time.time()
                for j, fr in enumerate(relax_md(st, calc, md_a, rng)):
                    fin = finish(fr, calc, dict(T_K=T))
                    emit(fin, "md", f"T={T:g}", T=T, extra=dict(P_target=P, md_frame=j, P_grace_GPa=fin.info.get("P_grace_GPa", float("nan")), E_grace=fin.info.get("E_grace", float("nan")), T_inst=fin.info.get("T_inst", float("nan"))))
                print(f"   md {T:g} K done in {time.time() - t0:.0f} s", flush=True)
            if a.ins_n:
                from build_insertion_set import insert_H
                ia = SimpleNamespace(cand=400, min_d_O=0.85, min_d_H=0.60, power=4.0); dv = 1.26 * (500.0 / P) ** 0.75   # H partial volume, A^3, scaled from 1.26 at 500 GPa
                for nn in a.ins_n:
                    for T in a.ins_T:
                        b = insert_H(sc, nn, rng, ia); f = ((sc.get_volume() + dv * nn) / sc.get_volume()) ** (1 / 3); b.set_cell(sc.cell.array * f, scale_atoms=True)
                        md_a = SimpleNamespace(T=T, dt=a.dt, equil_ps=a.equil_ps, sample_ps=a.save_every_ps * (a.md_frames - 1), save_every_ps=a.save_every_ps)
                        for j, fr in enumerate(relax_md(b, calc, md_a, rng)):
                            fin = finish(fr, calc, dict(T_K=T)); emit(fin, "ins_md", f"n={nn} T={T:g}", T=T, extra=dict(P_target=P, n_add=nn, md_frame=j, P_grace_GPa=fin.info.get("P_grace_GPa", float("nan")), E_grace=fin.info.get("E_grace", float("nan"))))
                print(f"   insertion MD n={a.ins_n} at T={a.ins_T} done", flush=True)
    print(f"wrote {n} frames to {a.out}")


if __name__ == "__main__":
    main()
