"""Throughput benchmark of the TDI lambda engine on fixed production systems (no files written). Driven by mlp_pipeline/bench_tdi.sh; results in docs/PERFORMANCE.md.
usage (repo root, inside a GPU job, ~/grace-venv/bin/python):
  python mlp_pipeline/tdi/bench.py engine H   --steps 150        one lambda process (the per-lambda engine): prints one RESULT line (steps/s of this process)
  python mlp_pipeline/tdi/bench.py batch  H   --nrep 8           the batched engine: 8 lambda values in this process, one model call per step (tdi/batched.py)
  python mlp_pipeline/tdi/bench.py dense  insH                   default (segment_sum) vs dense ('uniform', GRACE_MODE=uniform) tensorpotential engine: ms per call, differences
  python mlp_pipeline/tdi/bench.py calls  H   --sizes 1,2,4,8,16 model calls alone: one batched call of B structures vs B single TPCalculator calls (task: does batching pay?)
systems: h2o = fcc H2O, 108 O (324 atoms, Einstein springs, 400 GPa 3000 K); H = pure-H fluid, 250 atoms (500 GPa 6000 K);
         ins = insertion of 1 H into fcc H2O (325 atoms, two model calls per step: host + ghost, host alone; 500 GPa 6000 K); insH = 1 H into the 250-atom pure-H fluid (251 atoms)
Volumes from --from-dft (default inventory_binned.csv, as the production runs) or --vpa. The lambda value only sets the potential mixture (cost is the same).
"""
import argparse, json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

SYSTEMS = {"h2o": ["--phase", "fcc", "--P", "400", "--T", "3000", "--k", "22.5"],
           "H": ["--phase", "H", "--P", "500", "--T", "6000"],
           "ins": ["--phase", "insert", "--nghost", "1", "--P", "500", "--T", "6000", "--bulk-gpa", "1000"],
           "insH": ["--phase", "insert", "--host-phase", "H", "--nghost", "1", "--P", "500", "--T", "6000", "--bulk-gpa", "1000"]}


def build(b):
    import io, contextlib
    from tdi import run_real
    argv = [b.model] + SYSTEMS[b.system] + (["--vpa", str(b.vpa), "--bulk-gpa", "1000"] if b.vpa else ["--from-dft", b.from_dft]) + ["--dt", str(b.dt), "--device", b.device, "--outdir", b.workdir]
    if b.cell_nrep is not None:
        argv += ["--nrep", str(b.cell_nrep)]
    for k, v in (("--P", b.P), ("--T", b.T)):
        if v is not None:
            argv += [k, str(v)]
    a = run_real.make_parser().parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):   # no plan banner
        c = run_real.setup(a)
    kE = a.k or 0.0
    return run_real, a, c, kE


def comps_single(run_real, c, kE):
    return {"full": c["full"], "E": run_real.E_component(c, kE)}


def result(**d):
    d.update(host=os.uname().nodename, pid=os.getpid(), mps=bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY")), t=time.strftime("%Y-%m-%d %H:%M:%S"))
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            memory = tf.config.experimental.get_memory_info("GPU:0")
            d["tensorflow_peak_MiB"] = memory["peak"] / (1024*1024)
        except (ValueError, RuntimeError):
            pass
    print("RESULT " + json.dumps(d), flush=True)


def mode_engine(b):
    run_real, a, c, kE = build(b)
    r = run_real.LambdaRun(c["at0"], comps_single(run_real, c, kE), b.lam, a.T, c["iO"], a.dt, a.tau, 1, a.every, 0)
    r.at.get_forces(); r.start(); r.advance(b.warm)   # compile + warm-up outside the timing
    if b.barrier:   # several processes: start the timed part together (file barrier written by bench_tdi.sh once every process printed READY)
        print("READY", flush=True)
        while not os.path.exists(b.barrier):
            time.sleep(0.2)
    t = time.monotonic(); r.advance(b.steps); end = time.monotonic(); el = end - t
    result(mode="engine", system=b.system, natoms=len(c["at0"]), steps=b.steps, seconds=el, steps_per_s=b.steps / el, ms_per_step=1000 * el / b.steps,
           timed_started=t, timed_ended=end, lambda_steps=b.steps)


def wait_for_benchmark(b):
    if b.barrier:
        print("READY", flush=True)
        while not os.path.exists(b.barrier):
            time.sleep(.2)


def mode_pwmd(b):
    """The direct ASE Langevin MD used by pwidom: one full-model force call/step."""
    from importlib import import_module
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from pwidom import build_host, Model
    phase = "H" if b.system == "H" else "fcc"
    args = argparse.Namespace(phase=phase, vpa=b.vpa, nrep=b.cell_nrep or (5 if phase == "H" else 3),
                              hper=2., natoms=0, seed=3)
    at = build_host(args)
    at.calc = Model(b.model,b.device,1).calc
    rng = np.random.default_rng(3)
    MaxwellBoltzmannDistribution(at, temperature_K=b.T or 3000, rng=rng)
    md = Langevin(at,b.dt*units.fs,temperature_K=b.T or 3000,friction=.01,rng=rng)
    at.get_forces(); md.run(b.warm); wait_for_benchmark(b)
    start = time.monotonic(); md.run(b.steps); end = time.monotonic()
    result(mode="pwmd",system=b.system,natoms=len(at),steps=b.steps,seconds=end-start,
           work_units=b.steps,unit="MD steps",timed_started=start,timed_ended=end)


def mode_pwsp(b):
    """The pwidom energy evaluator on independent host/host+H structures."""
    from pwidom import build_host, Model
    from tdi.run_real import add_ghosts
    phase = "H" if b.system == "H" else "fcc"
    args = argparse.Namespace(phase=phase, vpa=b.vpa, nrep=b.cell_nrep or (5 if phase == "H" else 3),
                              hper=2., natoms=0, seed=3)
    host = build_host(args); full = add_ghosts(host,1,10)
    rng = np.random.default_rng(5); structures = []
    for i in range(b.nrep):
        at = (host if i%2==0 else full).copy()
        at.positions += rng.normal(0,.01,at.positions.shape); structures.append(at)
    model = Model(b.model,b.device,b.nrep)
    # Model.energies self-checks batched against single calls on its first call.
    def evaluate():
        # In particular B=1 must not benchmark ASE's cached result on the same
        # geometry. Reset calculator results while retaining the loaded model.
        model.calc.reset()
        return model.energies(structures)
    for _ in range(max(1,b.warm)): evaluate()
    wait_for_benchmark(b)
    start = time.monotonic()
    for _ in range(b.steps): evaluate()
    end = time.monotonic()
    result(mode="pwsp",system=b.system,natoms=len(host),batch=b.nrep,seconds=end-start,
           work_units=b.nrep*b.steps,unit="structures",timed_started=start,timed_ended=end,
           effective_max_atoms=getattr(model.gb,"max_atoms",0))


def mode_batch(b):
    from tdi import batched
    run_real, a, c, kE = build(b)
    ev = batched.make_evaluator(c, b.max_atoms); bt = batched.Batcher(ev); n = b.nrep
    if isinstance(ev, batched.GraceBatch):
        structs = [c["at0"].copy()]
        if c.get("host_idx") is not None:
            structs.append(c["at0"][c["host_idx"]])
        print(ev.selfcheck(structs), flush=True)
    lams = np.linspace(0.05, 1.0, n); runs = []
    for k in range(n):
        sm = batched.SlotModel(bt, k, c.get("host_idx")); bt.register(k)
        E = sm.host_part if c.get("host_idx") is not None else run_real.E_component(c, kE)
        runs.append(run_real.LambdaRun(c["at0"], {"full": sm.full, "E": E}, float(lams[k]), a.T, c["iO"], a.dt, a.tau, 1 + k, a.every, 0))
    gate = threading.Barrier(n + 1); done = threading.Barrier(n + 1)
    go = threading.Event(); errors = []

    def work(k):
        r = runs[k]
        try:
            r.at.get_forces(); r.start(); r.advance(b.warm); gate.wait(); go.wait(); r.advance(b.steps); done.wait()
        except BaseException as ex:
            errors.append(repr(ex)); gate.abort(); done.abort(); go.set()
        finally:
            bt.unregister(k)
    th = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(n)]
    for x in th:
        x.start()
    try:
        gate.wait()
        if b.barrier:
            print("READY", flush=True)
            while not os.path.exists(b.barrier):
                time.sleep(.2)
        t = time.monotonic(); c0, m0 = bt.ncalls, bt.t_model; go.set(); done.wait(); end = time.monotonic(); el = end - t
    except threading.BrokenBarrierError:
        raise RuntimeError("batched benchmark worker failed: " + "; ".join(errors))
    calls = bt.ncalls - c0
    result(mode="batch", system=b.system, natoms=len(c["at0"]), nrep=n, steps=b.steps, seconds=el, steps_per_s=n * b.steps / el, ms_per_step=1000 * el / b.steps,
           calls=calls, ms_per_call=1000 * (bt.t_model - m0) / max(calls, 1), model_frac=(bt.t_model - m0) / el, extract_s=getattr(ev, "t_extract", float("nan")),
           timed_started=t, timed_ended=end, lambda_steps=n*b.steps, effective_max_atoms=getattr(ev, "max_atoms", 0))


def mode_calls(b):
    from tdi import batched
    run_real, a, c, kE = build(b)
    ev = batched.make_evaluator(c, b.max_atoms); calc = c["calc"]; rng = np.random.default_rng(0)
    base = [c["at0"]] + ([c["at0"][c["host_idx"]]] if c.get("host_idx") is not None else [])
    if isinstance(ev, batched.GraceBatch):
        print(ev.selfcheck([x.copy() for x in base]), flush=True)
    for B in [int(x) for x in b.sizes.split(",")]:
        S = []
        for i in range(B):
            x = base[i % len(base)].copy(); x.positions += rng.normal(0, 0.03, x.positions.shape); S.append(x)
        for x in S[:2]:   # warm-up (compile) single
            calc.reset(); y = x.copy(); y.calc = calc; y.get_forces()
        t = time.time()
        for _ in range(b.repeat):
            for x in S:
                calc.reset(); y = x.copy(); y.calc = calc; y.get_forces()   # reset: never the calculator's cached result
        ts = (time.time() - t) / b.repeat
        ev([0], [(S, False)])   # warm-up (compile) batched shape
        e0 = getattr(ev, "t_extract", 0.0); t = time.time()
        for _ in range(b.repeat):
            ev([0], [(S, False)])
        tb = (time.time() - t) / b.repeat; te = (getattr(ev, "t_extract", 0.0) - e0) / b.repeat
        result(mode="calls", system=b.system, natoms=len(c["at0"]), B=B, single_ms_per_structure=1000 * ts / B, batched_ms_per_structure=1000 * tb / B, batched_call_ms=1000 * tb,
               extract_ms_per_call=1000 * te, speedup=ts / tb)


def mode_dense(b):
    """tensorpotential engines: the default segment_sum ('diverse') against the dense reshape ('uniform', GRACE_MODE=uniform) on the same structures: ms per call and the
    largest differences (round-off is expected: different summation order). Tells whether the SavedModel has the dense signature at all."""
    from importlib import import_module
    run_real, a, c, kE = build(b)
    spec = import_module("04_benchmark").get_calc_spec; os.environ.pop("GRACE_MODE", None)
    seg = spec(b.model, b.device); os.environ["GRACE_MODE"] = "uniform"; den = spec(b.model, b.device); os.environ.pop("GRACE_MODE", None)
    has = bool(getattr(den, "dense_reshape", False))
    print(f"SavedModel signatures of the dense calculator: dense engine {'ACTIVE' if has else 'NOT available (fell back to segment_sum: the model must be re-exported with tensorpotential >= 0.6 for it)'}", flush=True)
    rng = np.random.default_rng(0); base = [c["at0"]] + ([c["at0"][c["host_idx"]]] if c.get("host_idx") is not None else [])
    S = []
    for i in range(8):
        x = base[i % len(base)].copy(); x.calc = None; x.set_constraint(); x.positions += rng.normal(0, 0.03, x.positions.shape); S.append(x)
    out, t = {}, {}
    for name, calc in (("segment", seg), ("dense", den)):
        for x in S[:2]:
            calc.reset(); y = x.copy(); y.calc = calc; y.get_forces()   # compile
        r = []; t0 = time.time()
        for _ in range(b.repeat):
            r = []
            for x in S:
                calc.reset(); y = x.copy(); y.calc = calc; r.append((y.get_potential_energy() / len(y), y.get_forces(), y.get_stress()))
        t[name] = (time.time() - t0) / (b.repeat * len(S)); out[name] = r
    dE = max(abs(p[0] - q[0]) for p, q in zip(out["segment"], out["dense"])); dF = max(float(np.abs(p[1] - q[1]).max()) for p, q in zip(out["segment"], out["dense"]))
    dS = max(float(np.abs(p[2] - q[2]).max()) for p, q in zip(out["segment"], out["dense"]))
    result(mode="dense", system=b.system, natoms=len(c["at0"]), dense_active=has, segment_ms_per_call=1000 * t["segment"], dense_ms_per_call=1000 * t["dense"],
           speedup=t["segment"] / t["dense"], max_dE_eV_atom=dE, max_dF_eV_A=dF, max_dstress=dS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["engine", "batch", "calls", "dense", "pwmd", "pwsp"]); ap.add_argument("system", choices=list(SYSTEMS))
    ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--from-dft", default="inventory_binned.csv"); ap.add_argument("--vpa", type=float, default=None)
    ap.add_argument("--P", type=float, default=None); ap.add_argument("--T", type=float, default=None); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--warm", type=int, default=20); ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--barrier", default=None)
    ap.add_argument("--nrep", type=int, default=8); ap.add_argument("--max-atoms", type=int, default=0); ap.add_argument("--sizes", default="1,2,4,8,16"); ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--cell-nrep", type=int, default=None, help="cell replication override (local toy benchmark checks)")
    ap.add_argument("--workdir", default="runs/bench_tmp", help="isolated benchmark setup directory")
    b = ap.parse_args()
    {"engine": mode_engine, "batch": mode_batch, "calls": mode_calls, "dense": mode_dense,
     "pwmd":mode_pwmd,"pwsp":mode_pwsp}[b.mode](b)
