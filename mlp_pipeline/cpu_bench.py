"""CPU throughput of the MLP for our TDI cells: how many MD steps per second does a whole CPU node give, compared with a GPU (A100: 11.5 steps/s for an insertion
run, 22.7 GRACE calls/s for 250 atoms, NPAR=4, docs/PERFORMANCE.md)? Runs P processes x T threads at the same time (each: model load, 3 warm-up calls, then --calls
timed energy + forces calls on its own copy of the cell) for every layout in --layouts, and prints s/call per process and the node total in calls (= MD steps of a
1-call-per-step run) per second. TDI lambda values are independent processes, so the node total is what a TDI job would get.
VERBOSE on purpose: a header with the node / CPU / software, then per layout: the start, process 0's own lines (model load time, warm-up, every timed call with its
running mean and ETA), a heartbeat every --heartbeat s (processes finished, elapsed), and a summary line. Results are also appended to --results (jsonl) with --label,
so runs in different queues (Gold vs Free) or nodes can be compared:  python3 mlp_pipeline/cpu_bench.py --table runs/cpu_bench.jsonl
usage (on a CPU node, repo root):  ~/grace-venv/bin/python -u mlp_pipeline/cpu_bench.py [--model grace-ft:runs/grace/ft_m2] [--cell H2O|H|insert] [--layouts 1x40 4x10 8x5 ...]
  Young (SGE): qsub -P Gold -A UKCP_ED_P mlp_pipeline/young_cpu_bench.sge     (see that file)
"""
import argparse, glob, json, os, platform, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_REF = dict(insert=11.5, H=22.7, H2O=22.7 * 250 / 324)   # A100 totals (steps/s) at NPAR=4: insertion measured (2 calls/step); single-call cells scaled from 250-atom pure H


def child(a):
    t_start = time.time(); say = (lambda *x: print(f"      [proc 0] {time.time() - t_start:7.1f} s", *x, flush=True)) if a.rank == 0 else (lambda *x: None)
    try:   # thread count before the model is built (TensorFlow reads it once)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(a.threads); tf.config.threading.set_inter_op_parallelism_threads(1)
        say(f"TensorFlow {tf.__version__}: {a.threads} intra-op threads")
    except Exception as e:
        say(f"TensorFlow thread setting skipped ({type(e).__name__})")
    sys.path.insert(0, HERE)
    from importlib import import_module
    md = import_module("17_eos_md")
    at = md.build("H", 1.3, 5) if a.cell == "H" else md.build("fcc", 2.2, 3, a.seed, 2.0)   # pure H 250 atoms / fcc H2O 108 O = 324 atoms
    if a.cell == "insert": at = at + at[:1]; at.positions[-1] += 0.9                          # 325 atoms, like a host + one inserted H (2 calls per MD step in a real insertion)
    say(f"cell {a.cell}: {len(at)} atoms; loading model {a.model}")
    at.calc = import_module("04_benchmark").get_calc_spec(a.model, "cpu"); say("model loaded")
    for k in range(3):
        t = time.time(); at.positions += 0.001; at.get_forces(); say(f"warm-up call {k + 1}/3: {time.time() - t:.2f} s")
    t0 = time.time(); ts = []
    for k in range(a.calls):
        t = time.time(); at.positions += 0.001; at.get_potential_energy(); at.get_forces(); ts.append(time.time() - t)
        m = sum(ts) / len(ts); say(f"call {k + 1:3d}/{a.calls}: {ts[-1]:.3f} s (mean {m:.3f} s, ETA {m * (a.calls - k - 1):.0f} s)")
    json.dump(dict(s_per_call=(time.time() - t0) / a.calls, natoms=len(at), load_s=t0 - t_start, min_s=min(ts), max_s=max(ts)), open(a.out, "w"))


def table(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{'label':22s} {'host':16s} {'cpus':>5s} {'cell':6s} {'layout':>7s} {'s/call':>7s} {'steps/s':>8s} {'x A100':>7s} {'load s':>7s}")
    for r in rows:
        print(f"{r['label']:22s} {r['host'][:16]:16s} {r['cpus']:5d} {r['cell']:6s} {r['layout']:>7s} {r['s_per_call']:7.3f} {r['node_steps_s']:8.2f} {r['x_a100']:7.2f} {r['load_s']:7.0f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--cell", default="H2O", choices=["H2O", "H", "insert"])
    ap.add_argument("--layouts", nargs="+", default=None, help="PxT (processes x threads); default from the core count: 1xN, N/8 x 8, N/4 x 4, N/2 x 2, N x 1")
    ap.add_argument("--calls", type=int, default=20); ap.add_argument("--heartbeat", type=float, default=30.0); ap.add_argument("--label", default=os.environ.get("BENCH_LABEL", ""))
    ap.add_argument("--results", default="runs/cpu_bench.jsonl"); ap.add_argument("--table", default=None, help="print the table of a results file and exit")
    ap.add_argument("--child", action="store_true"); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--out", default=""); a = ap.parse_args()
    if a.table: return table(a.table)
    if a.child: return child(a)
    ncpu = int(os.environ.get("NSLOTS") or 0) or len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    lays = a.layouts or [f"1x{ncpu}"] + [f"{ncpu // t}x{t}" for t in (8, 4, 2) if ncpu // t >= 2] + [f"{ncpu}x1"]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
    print(f"== cpu_bench on {platform.node()} ({cpu}); {ncpu} CPUs usable (NSLOTS {os.environ.get('NSLOTS', '-')}, logical {os.cpu_count()}); label '{a.label}'", flush=True)
    print(f"   model {a.model}; cell {a.cell}; {a.calls} timed calls per process; layouts {' '.join(lays)}; GPU reference {GPU_REF[a.cell]:.1f} steps/s (A100, NPAR=4)", flush=True)
    print(f"   job {os.environ.get('JOB_ID') or os.environ.get('SLURM_JOB_ID') or '-'}  queue {os.environ.get('QUEUE', '-')}  python {sys.version.split()[0]}  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    tmp = tempfile.mkdtemp(prefix="cpubench_"); t_all = time.time(); summary = []
    for li, lay in enumerate(lays):
        P, T = map(int, lay.split("x"))
        if P * T > ncpu: print(f"\n-- {lay}: skipped ({P * T} > {ncpu} CPUs)", flush=True); continue
        print(f"\n-- layout {li + 1}/{len(lays)}: {P} processes x {T} threads ({time.strftime('%H:%M:%S')}); process 0 reports every call:", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(T), MKL_NUM_THREADS=str(T), TF_NUM_INTRAOP_THREADS=str(T), TF_NUM_INTEROP_THREADS="1",
                   TF_CPP_MIN_LOG_LEVEL="2")
        t0 = time.time(); outs = [os.path.join(tmp, f"{lay}_{i}.json") for i in range(P)]; errs = [os.path.join(tmp, f"{lay}_{i}.err") for i in range(P)]
        ps = [subprocess.Popen([sys.executable, "-u", __file__, "--child", "--model", a.model, "--cell", a.cell, "--threads", str(T), "--calls", str(a.calls),
                                "--seed", str(i), "--rank", str(i), "--out", outs[i]], stdout=None if i == 0 else subprocess.DEVNULL, stderr=open(errs[i], "w"), env=env)
              for i in range(P)]
        last = time.time()
        while any(p.poll() is None for p in ps):
            time.sleep(1)
            if time.time() - last >= a.heartbeat:
                last = time.time(); done = sum(p.poll() is not None for p in ps)
                print(f"   .. {lay}: {done}/{P} processes finished, {time.time() - t0:.0f} s elapsed (whole benchmark {(time.time() - t_all) / 60:.1f} min)", flush=True)
        res = []
        for i, p in enumerate(ps):
            if os.path.exists(outs[i]): res.append(json.load(open(outs[i])))
            else:
                e = [l for l in open(errs[i]).read().splitlines() if l.strip()]
                print(f"   process {i} FAILED (exit {p.returncode}): {e[-1] if e else 'no output'}", flush=True)
        if not res: continue
        s = sum(r["s_per_call"] for r in res) / len(res); tot = len(res) / s
        if a.cell == "insert": tot /= 2   # a real insertion step needs 2 calls (host and host + ghost)
        x = tot / GPU_REF[a.cell]
        row = dict(label=a.label, host=platform.node(), cpus=ncpu, cpu=cpu, cell=a.cell, layout=lay, natoms=res[0]["natoms"], s_per_call=s, node_steps_s=tot, x_a100=x,
                   load_s=sum(r["load_s"] for r in res) / len(res), spread=f"{min(r['min_s'] for r in res):.3f}-{max(r['max_s'] for r in res):.3f}", wall_s=time.time() - t0,
                   job=os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID") or "", date=time.strftime("%Y-%m-%d %H:%M"))
        summary.append(row)
        print(f"== {lay}: {len(res)} processes x {T} threads, {row['natoms']} atoms: {s:.3f} s/call each (single calls {row['spread']} s), model load {row['load_s']:.0f} s "
              f"-> node total {tot:.2f} MD steps/s = {x:.2f} x one A100   [{row['wall_s']:.0f} s]", flush=True)
        os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
        with open(a.results, "a") as fh: fh.write(json.dumps(row) + "\n")
    if summary:
        best = max(summary, key=lambda r: r["node_steps_s"])
        print(f"\n== summary ({(time.time() - t_all) / 60:.1f} min): best layout {best['layout']}: {best['node_steps_s']:.2f} MD steps/s = {best['x_a100']:.2f} x one A100; "
              f"an insertion task (50,000 steps) would take ~{50000 / best['node_steps_s'] / 3600 * (1 if a.cell == 'insert' else 2):.1f} h on this node; appended to {a.results}", flush=True)


if __name__ == "__main__":
    main()
