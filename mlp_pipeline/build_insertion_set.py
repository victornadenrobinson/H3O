"""Build insertion / deletion snapshot sets: host frames (DFT-sampled) + n extra H, optionally relaxed with a short MLP-MD, written as one extxyz for DFT labelling.

Why: the training tiers contain no dilute-H cells and only a handful of H2.5O / H2.75O frames, and nothing links composition n to n+1 in the same cell.
Hosts come from the existing tiers (train_l.extxyz for training sets, valid_l.extxyz for validation sets), so whole runs stay on one side of the split.

usage (repo root, GPU node for the MD; PYTHONPATH must contain mlp_pipeline/):
  python mlp_pipeline/build_insertion_set.py --hosts runs/train_l.extxyz --comp h2o --T-bin 5000 --P 450 550 --natoms 324 \\
      --n-list 1,2,3,4,6,8,12,16,20,24,27 --copies 2 --model grace-ft:runs/grace/ft_m2/seed/1/final_model --equil-ps 0.3 --out runs/ins_h2o_5000K_500GPa_train.extxyz
  --n-list may contain negative numbers (delete |n| random H atoms from an H-rich host, e.g. --comp h2.25o --n-list -1,-2,-4,-8)
  --sample-ps 1 --save-every-ps 0.1 --dt 1.0: after equilibration keep running and save 10 more frames per run (cheap GRACE-MD snapshots to label with VASP; frames of one run share run_id, so keep them on one side of a split)
  --equil-ps 0 writes the raw insertions (no MD; use this for exact n -> n+1 pairs with --chain)
  --chain: one host, one insertion order; frames for n = every value in --n-list from the SAME growing configuration (paired energy differences)
  --dv-per-H A^3: isotropic cell scaling by n * dv before the MD (about 1.26 for 500 GPa) so the pressure stays near target; the final GRACE pressure is stored per frame
  --shard i/k: process every k-th host frame (run k processes on k GPUs); --max-hosts caps the host frames
Output frames carry info: n_add, nH, nO, host_source, host_frame, run_id, split, method, T_K, P_target, P_grace_GPa, E_grace, T_inst.
Labels (VASP free energy, forces, stress) are added by a separate DFT step; this script writes geometries only, and NO VASP inputs or POTCAR data.
"""
import argparse, os, sys, time
import numpy as np
from ase import Atoms, units
from ase.geometry import get_distances
from ase.io import read, write

EV_A3_GPA = 160.21766208


def select_hosts(a):
    out = []
    for path in a.hosts:
        for i, f in enumerate(read(path, index=":")):
            ct = str(f.info.get("config_type", ""))
            if a.comp and not ct.startswith(a.comp + "_"): continue
            if a.T_bin is not None and not ct.endswith(f"_{int(a.T_bin)}K"): continue
            if a.P and not (a.P[0] <= float(f.info.get("P_GPa", np.nan)) <= a.P[1]): continue
            if a.natoms and len(f) != a.natoms: continue
            f.info["host_source"] = str(f.info.get("source", path)); f.info["host_frame"] = int(i); f.info["split"] = os.path.basename(path).replace(".extxyz", "")
            out.append(f)
    return out


def insert_H(at, n, rng, a):
    """add n H atoms one by one at random points, preferring large holes: weight ~ (distance to the nearest atom)^power, hard minimum distances"""
    at = at.copy(); added = 0; tries = 0
    while added < n and tries < 50 * n:
        tries += 1
        frac = rng.random((a.cand, 3)); P = frac @ at.cell.array
        _, d = get_distances(P, at.positions, cell=at.cell, pbc=True)
        isO = at.numbers == 8; isH = at.numbers == 1
        dO = d[:, isO].min(axis=1); dH = d[:, isH].min(axis=1) if isH.any() else np.full(len(P), 9.0)
        ok = (dO >= a.min_d_O) & (dH >= a.min_d_H)
        if not ok.any(): continue
        dmin = np.minimum(dO, dH)[ok]; w = dmin ** a.power; w /= w.sum()
        at += Atoms("H", positions=[P[ok][rng.choice(len(w), p=w)]]); added += 1
    if added < n: raise RuntimeError(f"could only place {added} of {n} H atoms; relax --min-d-O/--min-d-H or the cell is too dense")
    return at


def delete_H(at, n, rng):
    h = np.where(at.numbers == 1)[0]
    if len(h) <= n: raise RuntimeError("not enough H to delete")
    keep = np.ones(len(at), bool); keep[rng.choice(h, n, replace=False)] = False
    return at[keep]


def get_calc(spec, device):
    if spec == "emt":
        from ase.calculators.emt import EMT; return EMT()
    from importlib import import_module
    return import_module("04_benchmark").get_calc_spec(spec, device)


def relax_md(at, calc, a, rng):
    """overlap removal (FIRE), a short small-step Langevin start, equilibration at a.dt, then a sampling stretch; returns a list of frames
    (final equilibrated frame plus one every a.save_every_ps over a.sample_ps). Frames from a run that blew up (T or forces far off) are dropped."""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.optimize import FIRE
    at = at.copy(); at.calc = calc
    if np.abs(at.get_forces()).max() > 20.0:
        FIRE(at, logfile=None, maxstep=0.05).run(fmax=5.0, steps=60)
    MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=rng); Stationary(at)
    def go(dt, steps, every=0, sink=None):
        if steps <= 0: return
        dyn = Langevin(at, dt * units.fs, temperature_K=a.T, friction=1 / (100 * units.fs), rng=rng)
        if every and sink is not None:
            first = [True]
            def grab():
                if first[0]: first[0] = False; return          # ASE calls observers once at step 0; that frame is the equilibrated one already saved
                sink.append(at.copy())
            dyn.attach(grab, interval=every)
        dyn.run(steps)
    go(0.1, 60); go(a.dt, int(round(a.equil_ps * 1000 / a.dt)))
    out = [at.copy()]
    if a.sample_ps > 0:
        every = max(1, int(round(a.save_every_ps * 1000 / a.dt))); go(a.dt, int(round(a.sample_ps * 1000 / a.dt)), every, out)
    good = []
    for f in out:
        f.calc = calc
        try:
            if abs(f.get_temperature() - a.T) < 0.5 * a.T and np.abs(f.get_forces()).max() < 500: good.append(f)
        except Exception: pass
    return good


def finish(at, calc, info):
    at = at.copy()
    if calc is not None:
        at.calc = calc
        try:
            s = at.get_stress(voigt=False); info["P_grace_GPa"] = float(-np.trace(s) / 3 * EV_A3_GPA + len(at) * 8.617333e-5 * info["T_K"] / at.get_volume() * EV_A3_GPA)
            info["E_grace"] = float(at.get_potential_energy()); info["T_inst"] = float(at.get_temperature())
        except Exception as e:
            info["calc_error"] = str(e)[:80]
    at.calc = None; at.info.update(info)
    return at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="+", required=True); ap.add_argument("--comp"); ap.add_argument("--T-bin", type=float); ap.add_argument("--P", nargs=2, type=float); ap.add_argument("--natoms", type=int)
    ap.add_argument("--n-list", default="1,2,3,4,6,8,12,16,20,24,27"); ap.add_argument("--copies", type=int, default=1); ap.add_argument("--chain", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--device", default="cuda"); ap.add_argument("--T", type=float, help="MD temperature K (default: the host's T_K)")
    ap.add_argument("--equil-ps", type=float, default=0.3); ap.add_argument("--dt", type=float, default=0.5, help="MD step in fs; 1.0 is usually fine for cheap sampling at 3000-5000 K, runs that blow up are dropped")
    ap.add_argument("--sample-ps", type=float, default=0.0, help="keep sampling after the equilibration for this long ..."); ap.add_argument("--save-every-ps", type=float, default=0.1, help="... and save a frame every this many ps"); ap.add_argument("--dv-per-H", type=float, default=0.0)
    ap.add_argument("--min-d-O", type=float, default=0.85); ap.add_argument("--min-d-H", type=float, default=0.60); ap.add_argument("--cand", type=int, default=400); ap.add_argument("--power", type=float, default=4.0)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--max-hosts", type=int, default=0); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ns = [int(x) for x in a.n_list.split(",")]
    hosts = select_hosts(a); rng0 = np.random.default_rng(a.seed); rng0.shuffle(hosts)
    if a.max_hosts: hosts = hosts[:a.max_hosts]
    i0, k = (int(x) for x in a.shard.split("/")); hosts = hosts[i0::k]
    print(f"{len(hosts)} host frames selected (shard {a.shard}); n values {ns}; copies {a.copies}; MD {'off' if not a.model or a.equil_ps <= 0 else f'{a.equil_ps} ps at dt {a.dt} fs'}", flush=True)
    calc = get_calc(a.model, a.device) if a.model else None
    use_md = calc is not None and a.equil_ps > 0
    if os.path.exists(a.out): os.remove(a.out)
    count = 0
    for hi, h in enumerate(hosts):
        T = a.T or float(h.info.get("T_K", 5000.0)); a.T = T if a.T is None else a.T
        for c in range(a.copies):
            rng = np.random.default_rng([a.seed, hi, c, i0])
            base = {"host_source": h.info["host_source"], "host_frame": h.info["host_frame"], "split": h.info["split"], "T_K": T, "P_target": float(h.info.get("P_GPa", np.nan)), "run_id": f"{os.path.basename(a.out)}:{i0}:{hi}:{c}"}
            comp = str(h.info.get("config_type", "x")).rsplit("_", 1)[0]
            host = h.copy(); host.calc = None
            if a.chain:
                grown = host; prev = 0; frames = []
                for n in sorted(x for x in ns if x > 0):
                    grown = insert_H(grown, n - prev, rng, a); prev = n
                    frames.append((n, grown))
            else:
                frames = []
                for n in ns:
                    try: frames.append((n, insert_H(host, n, rng, a) if n > 0 else delete_H(host, -n, rng)))
                    except RuntimeError as e: print("skip:", e, flush=True)
            for n, at in frames:
                if a.dv_per_H:
                    f = ((host.get_volume() + a.dv_per_H * n) / host.get_volume()) ** (1 / 3)
                    at = at.copy(); at.set_cell(host.cell.array * f, scale_atoms=True)
                t0 = time.time(); outs = relax_md(at, calc, a, rng) if use_md else [at]
                if use_md:
                    steps = 60 + int(round(a.equil_ps * 1000 / a.dt)) + int(round(a.sample_ps * 1000 / a.dt))
                    print(f"  host {hi} n={n:+d}: {len(at)} atoms, {steps} MD steps in {time.time() - t0:.0f} s ({(time.time() - t0) / steps:.2f} s/step), {len(outs)} frames kept", flush=True)
                for j, fr in enumerate(outs):
                    info = dict(base, n_add=n, nH=int((fr.numbers == 1).sum()), nO=int((fr.numbers == 8).sum()), method=("chain" if a.chain else "md" if use_md else "raw"),
                                md_frame=j, config_type=f"{comp}{n:+d}H_{int(T)}K")
                    write(a.out, finish(fr, calc if use_md else None, info), append=True); count += 1
        if (hi + 1) % 5 == 0: print(f"  host {hi + 1}/{len(hosts)}: {count} frames written", flush=True)
    print(f"wrote {count} frames to {a.out}")


if __name__ == "__main__":
    main()
