"""Atomic, non-pickle Langevin checkpoints at completed integration steps."""
import json
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np


def atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def md_frames(model, at0, a, seed, label):
    """Continue unconstrained ASE Langevin with saved momenta and PCG64 state.

    Saves partial sampling in the same atomic snapshot as the integrator state.
    Signals request a save after the current complete step; SIGKILL cannot flush.
    """
    import ase
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from tdi.pwidom import EV_A3_GPA, model_fingerprint, thermostat_friction, make_dynamics

    if at0.constraints:
        raise ValueError('PWidom checkpoints currently support unconstrained Langevin only')
    interval = a.checkpoint_seconds
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('checkpoint-seconds must be finite and positive')
    suffix = '_host_md.npz' if label == 'host' else '_full_md.npz'
    destination = Path(a._checkpoint_prefix + suffix)
    source = Path(a.resume_from + suffix) if a.resume_from else None
    if destination.exists() and not (getattr(a, "resume_in_place", False) and source and destination.resolve() == source.resolve()):
        raise ValueError(f'Checkpoint exists: {destination}; resume into a fresh --out')
    eq = int(a.equil_ps*1000/a.dt)
    every = max(1, int(round(a.every_fs/a.dt)))
    nframes = int(a.ps*1000/a.every_fs)
    total = eq + nframes*every
    friction = thermostat_friction(a); bussi = getattr(a, 'thermostat', 'langevin') == 'bussi'
    if source and source.exists():                      # a checkpoint written with another friction (the old 0.01) keeps its own: resuming must not change the dynamics
        with np.load(source, allow_pickle=False) as z0: old = json.loads(str(z0['metadata'])).get('friction')
        if not bussi and old is not None and abs(float(old)-friction) > 1e-12:
            if getattr(a, 'legacy_friction', False) or getattr(a, 'friction_fs', None) is not None:
                raise ValueError(f'{source}: written with friction {old:.6g}, requested {friction:.6g}; resume without --friction-fs/--legacy-friction to keep its own')
            print(f'WARNING {label}: resuming a checkpoint written with the old friction {old:.6g} (damping ~{1/(float(old)*units.fs):.0f} fs); keeping it for exact continuation', flush=True)
            friction = float(old)
    metadata = dict(version=1, ase=ase.__version__, numpy=np.__version__,
                    model_fp=model_fingerprint(a.model), T=a.T, dt=a.dt,
                    equil_steps=eq, every=every, frames=nframes, seed=seed,
                    label=label, friction=friction, fixcm=True)
    if bussi: metadata.update(thermostat='bussi', taut_fs=float(getattr(a, 'friction_fs', None) or 100.0))      # key only present for bussi: Langevin checkpoints (incl. pre-fix ones) keep their metadata
    at = at0.copy(); at.calc = model.calc
    rng = np.random.default_rng(seed)
    positions, pressures, step = [], [], 0
    trace = []                                           # (time_fs, kinetic T, E_pot eV) every 10 steps, incl. equilibration; side file <prefix>_<label>_trace.npz (diagnostic only, not part of the checkpoint)
    trace_file = Path(a._checkpoint_prefix + f'_{label}_trace.npz'); trace_src = Path(a.resume_from + f'_{label}_trace.npz') if a.resume_from else None
    if source and source.exists():
        with np.load(source, allow_pickle=False) as z:
            if json.loads(str(z['metadata'])) != metadata:
                raise ValueError(f'{source}: checkpoint model/settings/software mismatch')
            if (not np.array_equal(z['numbers'], at.numbers)
                    or not np.array_equal(z['masses'], at.get_masses())
                    or not np.array_equal(z['cell'], at.cell.array)
                    or not np.array_equal(z['pbc'], at.pbc)
                    or not np.array_equal(z['initial_positions'], at0.positions)):
                raise ValueError(f'{source}: checkpoint cell/species/initial configuration mismatch')
            step = int(z['step'])
            p, mom, frames, pressure = (np.array(z[k]) for k in ('positions','momenta','frames','pressures'))
            expected = max(0, (step-eq)//every)
            if (not 0 <= step <= total or p.shape != at.positions.shape or mom.shape != p.shape
                    or frames.shape != (expected, len(at), 3) or pressure.shape != (expected,)
                    or not all(np.isfinite(x).all() for x in (p, mom, frames, pressure))):
                raise ValueError(f'{source}: invalid checkpoint arrays/step')
            at.set_positions(p); at.set_momenta(mom)
            rng.bit_generator.state = json.loads(str(z['rng_state']))
            positions, pressures = list(frames), list(pressure)
        print(f'RESUME {label}: step {step}/{total}, {len(positions)}/{nframes} saved frames; no velocity reset or repeated equilibration', flush=True)
    else:
        # A missing ensemble checkpoint means this ensemble never started. The
        # launcher requires at least one checkpoint/cache across a resumed job.
        MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=rng)
    if source and source.exists() and trace_src is not None and trace_src.exists():
        try:
            with np.load(trace_src) as zt: trace = [tuple(r) for r in zt['trace'] if r[0] <= step * a.dt + 1e-9]
        except Exception: trace = []
    dyn = make_dynamics(at, a, rng, friction=friction)
    dyn.nsteps = step
    requested = [None]
    previous = {}
    last_save = time.monotonic()
    last_observed = step

    def save():
        nonlocal last_save
        atomic_npz(destination, metadata=json.dumps(metadata, sort_keys=True),
                   positions=at.positions, momenta=at.get_momenta(), numbers=at.numbers,
                   masses=at.get_masses(), cell=at.cell.array, pbc=at.pbc,
                   initial_positions=at0.positions, step=dyn.nsteps,
                   frames=np.asarray(positions).reshape(-1,len(at),3), pressures=np.asarray(pressures),
                   rng_state=json.dumps(rng.bit_generator.state))
        try: atomic_npz(trace_file, trace=np.asarray(trace, float).reshape(-1, 3), columns='time_fs,T_kin_K,Epot_eV')
        except Exception: pass
        last_save = time.monotonic()
        print(f"CHECKPOINT {label}: step {dyn.nsteps}/{total}, {len(positions)}/{nframes} frames -> {destination}", flush=True)

    def observe():
        nonlocal last_observed
        current = dyn.nsteps
        if current != last_observed and current % 10 == 0:
            try: trace.append((current * a.dt, 2 * at.get_kinetic_energy() / (3 * len(at) * units.kB), float(at.get_potential_energy())))
            except Exception: pass
        if current != last_observed:
            if current > eq and (current-eq) % every == 0:
                positions.append(at.positions.copy())
                pressures.append(-at.get_stress()[:3].mean()*EV_A3_GPA
                                 + len(at)*units.kB*a.T/at.get_volume()*EV_A3_GPA)
            last_observed = current
        progress = getattr(a, '_progress', None)
        if progress and (current % 20 == 0 or current == total):
            sampling = current >= eq
            progress.update(f'MD {label} '+('sampling' if sampling else 'equilibration'),
                            current-eq if sampling else current,
                            nframes*every if sampling else eq, 'steps')
        if requested[0] is not None or time.monotonic()-last_save >= interval or current in (eq,total):
            save()
        if requested[0] is not None:
            raise SystemExit(128+requested[0])

    def stop(signum, frame):
        requested[0] = signum

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[sig] = signal.signal(sig, stop)
    try:
        save()  # Also preserve a resumed source before doing any more work.
        dyn.attach(observe, interval=1)
        dyn.run(total-step)
        save()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return np.asarray(positions), np.asarray(pressures)
