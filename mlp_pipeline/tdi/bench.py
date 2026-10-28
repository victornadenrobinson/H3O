"""Throughput benchmark of the TDI lambda engine on fixed production systems (no files written). Driven by mlp_pipeline/bench_tdi.sh; results in docs/PERFORMANCE.md.
usage (repo root, inside a GPU job, ~/grace-venv/bin/python):
  python mlp_pipeline/tdi/bench.py engine H   --steps 150        one lambda process (the per-lambda engine): prints one RESULT line (steps/s of this process)
  python mlp_pipeline/tdi/bench.py batch  H   --nrep 8           the batched engine: 8 lambda values in this process, one model call per step (tdi/batched.py)
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
    argv = [b.model] + SYSTEMS[b.system] + (["--vpa", str(b.vpa), "--bulk-gpa", "1000"] if b.vpa else ["--from-dft", b.from_dft]) + ["--dt", str(b.dt), "--device", b.device, "--outdir", "runs/bench_tmp"]
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
    print("RESULT " + json.dumps(d), flush=True)


def mode_engine(b):
    run_real, a, c, kE = build(b)
    r = run_real.LambdaRun(c["at0"], comps_single(run_real, c, kE), b.lam, a.T, c["iO"], a.dt, a.tau, 1, a.every, 0)
    r.at.get_forces(); r.start(); r.advance(b.warm)   # compile + warm-up outside the timing
    if b.barrier:   # several processes: start the timed part together (file barrier written by bench_tdi.sh once every process printed READY)
        print("READY", flush=True)
        while not os.path.exists(b.barrier):
            time.sleep(0.2)
    t = time.time(); r.advance(b.steps); el = time.time() - t
    result(mode="engine", system=b.system, natoms=len(c["at0"]), steps=b.steps, seconds=el, steps_per_s=b.steps / el, ms_per_step=1000 * el / b.steps)


def mode_batch(b):
    from tdi import batched
    run_real, a, c, kE = build(b)
    ev = batched.make_evaluator(c, b.max_atoms); bt = batched.Batcher(ev); n = b.nrep
    lams = np.linspace(0.05, 1.0, n); runs = []
    for k in range(n):
        sm = batched.SlotModel(bt, k, c.get("host_idx")); bt.register(k)
        E = sm.host_part if c.get("host_idx") is not None else run_real.E_component(c, kE)
        runs.append(run_real.LambdaRun(c["at0"], {"full": sm.full, "E": E}, float(lams[k]), a.T, c["iO"], a.dt, a.tau, 1 + k, a.every, 0))
    gate = threading.Barrier(n + 1); done = threading.Barrier(n + 1)

    def work(k):
        r = runs[k]
        try:
            r.at.get_forces(); r.start(); r.advance(b.warm); gate.wait(); r.advance(b.steps); done.wait()
        finally:
            bt.unregister(k)
    th = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(n)]
    for x in th:
        x.start()
    gate.wait(); t = time.time(); c0, m0 = bt.ncalls, bt.t_model; done.wait(); el = time.time() - t
    calls = bt.ncalls - c0
    result(mode="batch", system=b.system, natoms=len(c["at0"]), nrep=n, steps=b.steps, seconds=el, steps_per_s=n * b.steps / el, ms_per_step=1000 * el / b.steps,
           calls=calls, ms_per_call=1000 * (bt.t_model - m0) / max(calls, 1), model_frac=(bt.t_model - m0) / el, extract_s=getattr(ev, "t_extract", float("nan")))


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["engine", "batch", "calls"]); ap.add_argument("system", choices=list(SYSTEMS))
    ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--from-dft", default="inventory_binned.csv"); ap.add_argument("--vpa", type=float, default=None)
    ap.add_argument("--P", type=float, default=None); ap.add_argument("--T", type=float, default=None); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--warm", type=int, default=20); ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--barrier", default=None)
    ap.add_argument("--nrep", type=int, default=8); ap.add_argument("--max-atoms", type=int, default=0); ap.add_argument("--sizes", default="1,2,4,8,16"); ap.add_argument("--repeat", type=int, default=5)
    b = ap.parse_args()
    {"engine": mode_engine, "batch": mode_batch, "calls": mode_calls}[b.mode](b)
