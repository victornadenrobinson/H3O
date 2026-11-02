#!/usr/bin/env python3
"""Score a model (GRACE) on DFT-generated frames of a lambda = 1 (pure DFT) run: positions, forces and the free energy TOTEN are read from the OUTCARs, no new DFT is needed.

  eval     python frame_eval.py eval  CASE_OR_LAMBDA1_DIR --model grace-ft:/path/to/final_model --n 80 --start 0.2 --end 0.5 --out ice.csv [--save-extxyz ice.extxyz]
           per frame: E_DFT - E_model (eV per cell), force RMSE; the frames can be saved as a labelled extxyz (energy = TOTEN, forces) for (re)training
  combine  python frame_eval.py combine ice.csv iceh.csv pureh.csv
           gauge-invariant composition error per added H:  c = [D(ice+H) - D(ice) - n_add D(pure H)/N_H] / n_add,  D = mean(E_DFT - E_model) per cell
Elemental energy offsets cancel in c, so c is the composition-dependent part of the model error on DFT-generated configurations.
"""
import argparse, gzip, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import fe_stages as L


def read_frames(ldir, N, stride):
    """(positions A, forces eV/A, TOTEN eV) of every stride-th ionic step of all OUTCAR segments of ldir (restart duplicates dropped)"""
    out = []; first = True
    for f in L.outcar_files(ldir):
        op = gzip.open if f.endswith(".gz") else open
        pos = frc = te = None; step = 0; seg = []
        with op(f, "rt", errors="ignore") as fh:
            lines = iter(fh)
            for line in lines:
                if "POSITION" in line and "TOTAL-FORCE" in line:
                    next(lines)
                    if step % stride == 0:
                        blk = np.array([[float(v) for v in next(lines).split()[:6]] for _ in range(N)]); pos, frc = blk[:, :3], blk[:, 3:]
                    else:
                        for _ in range(N): next(lines)
                        pos = None
                elif "free  energy   TOTEN" in line:
                    te = float(line.split("=")[1].split()[0])
                    if pos is not None: seg.append((pos, frc, te))
                    step += 1; pos = None
        if not first and seg: seg = seg[1:]
        first = False; out += seg
    return out


def cmd_eval(a):
    from ase import Atoms
    from ase.data import atomic_numbers
    from ase.calculators.singlepoint import SinglePointCalculator
    ldir = a.dir
    if not os.path.isdir(os.path.join(ldir, "out1")) and os.path.isdir(os.path.join(ldir, "lambda_1.000")): ldir = os.path.join(ldir, "lambda_1.000")
    ps = L.read_poscar(os.path.join(ldir, "POSCAR")); numbers = np.repeat([atomic_numbers[n] for n in ps["names"]], ps["counts"]); N = len(numbers)
    fr = read_frames(ldir, N, a.stride); n0 = int(len(fr) * a.start); n1 = int(len(fr) * a.end); fr = fr[n0:n1]
    if len(fr) > a.n: fr = [fr[i] for i in np.linspace(0, len(fr) - 1, a.n).round().astype(int)]
    print(f"{ldir}: {N} atoms {ps['names']} {ps['counts']}, {len(fr)} frames from the window [{a.start}, {a.end}] of the run", flush=True)
    if a.model == "emt":
        from ase.calculators.emt import EMT; calc = EMT()
    else:
        from importlib import import_module
        calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    rows = []; labelled = []
    for i, (pos, frc, te) in enumerate(fr):
        at = Atoms(numbers=numbers, positions=pos, cell=ps["lat"], pbc=True); at.calc = calc
        e = float(at.get_potential_energy()); f = at.get_forces()
        rows.append((i, te, e, te - e, float(np.sqrt(((frc - f) ** 2).mean())), float(np.sqrt((frc ** 2).mean()))))
        if a.save_extxyz:
            lab = Atoms(numbers=numbers, positions=pos, cell=ps["lat"], pbc=True); lab.calc = SinglePointCalculator(lab, energy=te, forces=frc); labelled.append(lab)
        if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(fr)}  D = {np.mean([r[3] for r in rows]):.3f} eV  force rmse {np.mean([r[4] for r in rows]):.3f} eV/A", flush=True)
    R = np.array(rows); D = R[:, 3]
    print(f"E_DFT - E_GRACE per cell: mean {D.mean():.3f} eV ({D.mean() / N * 1e3:.2f} meV/atom), sd {D.std():.3f} eV ({D.std() / N * 1e3:.2f} meV/atom); force RMSE {R[:, 4].mean():.3f} eV/A (rms DFT force {R[:, 5].mean():.3f})")
    if a.out:
        np.savetxt(a.out, R, delimiter=",", header=f"N={N};counts={ps['counts']};names={ps['names']};frame,E_DFT,E_GRACE,D,force_rmse,force_rms_dft", comments="# ")
    if a.save_extxyz:
        from ase.io import write; write(a.save_extxyz, labelled)


def cmd_combine(a):
    def load(f):
        head = open(f).readline(); counts = eval(head.split("counts=")[1].split(";")[0]); R = np.loadtxt(f, delimiter=",", skiprows=1)
        return counts, R[:, 3]
    (c_ice, d_ice), (c_iceh, d_iceh), (c_h, d_h) = load(a.ice), load(a.iceh), load(a.pureh)
    n_add = c_iceh[-1] - c_ice[-1]; nh = c_h[0]
    rng = np.random.default_rng(0)
    def stat(d_ice, d_iceh, d_h):
        return (d_iceh.mean() - d_ice.mean() - n_add * d_h.mean() / nh) / n_add
    c = stat(d_ice, d_iceh, d_h)
    b = [stat(rng.choice(d_ice, len(d_ice)), rng.choice(d_iceh, len(d_iceh)), rng.choice(d_h, len(d_h))) for _ in range(2000)]
    print(f"D(ice) = {d_ice.mean():.3f} (sd {d_ice.std():.3f}), D(ice+H) = {d_iceh.mean():.3f} (sd {d_iceh.std():.3f}), D(pure H) = {d_h.mean():.3f} (sd {d_h.std():.3f}) eV per cell")
    print(f"composition error of the GRACE model on DFT-generated frames: c = {c * 1e3:+.0f} +- {np.std(b) * 1e3:.0f} meV per added H (n_add = {n_add}); "
          f"FEP usable if the per-cell sd of D is below ~1 eV (here {max(d_ice.std(), d_iceh.std(), d_h.std()):.2f} eV)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval"); p.add_argument("dir"); p.add_argument("--model", required=True); p.add_argument("--device", default="cpu"); p.add_argument("--n", type=int, default=80)
    p.add_argument("--stride", type=int, default=20, help="read every stride-th ionic step"); p.add_argument("--start", type=float, default=0.2); p.add_argument("--end", type=float, default=1.0)
    p.add_argument("--out"); p.add_argument("--save-extxyz"); p.set_defaults(f=cmd_eval)
    p = sub.add_parser("combine"); p.add_argument("ice"); p.add_argument("iceh"); p.add_argument("pureh"); p.set_defaults(f=cmd_combine)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
