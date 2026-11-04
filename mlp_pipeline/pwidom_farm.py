#!/usr/bin/env python3
"""Farm many independent MD replicas, pool their frames, then run ONE single-point stage per host (no GPU-memory clash between hosts' single-point stages, MD replicas are cheap and can share a GPU).
Everything is taken from the manifest of an earlier launcher run (its job commands hold every host argument), so the host, volume, model, proxy, K, sel, batch are exactly those of that run unless you override them.
  1. farm : python3 mlp_pipeline/pwidom_farm.py farm TEMPLATE_RUN --seeds 31 32 33 34 --out runs/farm_500GPa_5000K [--per-gpu 3] [--gpus 0,1,2,3] [--hosts H2O,pureH] [--md-args "--ps 2.0 --equil-ps 0.5"] [--dry-run]
            MD-only (--frames-only): one process per (host, seed), at most --per-gpu concurrent per GPU, frames to OUT/farm/<host>_s<seed>/. Resumable: finished frame files are skipped.
  2. pool : python3 mlp_pipeline/pwidom_farm.py pool OUT [--hosts ...] [--exclude-seeds N ...]
            merges all replicas of each host into OUT/pooled/<tag>_frames.npz (checks identical species / cell / volume / T / model fingerprint; keeps per-trajectory lengths so the melt test and block errors stay per trajectory)
            and writes OUT/pooled/manifest.json so pwidom_hull.py / plot scripts read it like any run.
  0. status: python3 mlp_pipeline/pwidom_farm.py status OUT [--watch 20]     live table: stage, bar, elapsed/ETA, GPU per run (farm and sp stage)
  3. sp   : python3 mlp_pipeline/pwidom_farm.py sp TEMPLATE_RUN OUT [--gpus 0,1,2,3] [--sp-args "--sel 24 --K 3000 --batch 4"] [--dry-run]
            the single-point stage on the pooled frames, ONE process per GPU at a time (queued when hosts > GPUs), results in OUT/pooled/.
Do not pool frames made with the old thermostat or another model: the pool refuses a different model fingerprint, V or T, but it cannot know the friction; use only runs made after the thermostat fix."""
import argparse, glob, json, os, queue, shutil, subprocess, sys, threading, time
import numpy as np

PY = sys.executable


def template_jobs(run, hosts=None):
    m = json.load(open(os.path.join(run, "manifest.json"))); jobs = m["jobs"]
    if hosts: jobs = [j for j in jobs if j["name"] in hosts]
    if not jobs: sys.exit(f"{run}: no matching hosts in the manifest")
    return m, jobs


def set_val(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd: cmd[cmd.index(flag) + 1] = str(value)
    else: cmd += [flag, str(value)]
    return cmd


def drop_flag(cmd, flag):                      # boolean flag
    return [c for c in cmd if c != flag]


def run_pool(tasks, gpus, per_gpu, dry, out=None, dash=15, plain=False, pattern="*_s[0-9]*.json", plan=None):
    """tasks: list of (name, cmd, logfile, outcheck); runs them with at most per_gpu concurrent processes per GPU"""
    q = queue.Queue(); [q.put(t) for t in tasks]; lock = threading.Lock(); fails = []; board = Dashboard(out, dash, plain, pattern, plan, q.qsize, len(gpus) * per_gpu) if (out and dash and not dry) else None
    def say(msg):                      # with the dashboard up, events go to the log (a print would break the in-place redraw)
        if board is None: print(msg, flush=True)
        else:
            with open(os.path.join(out, "events.log"), "a") as ef: ef.write(time.strftime("%H:%M:%S ") + msg + "\n")
    def worker(g):
        while True:
            try: name, cmd, log, done = q.get_nowait()
            except queue.Empty: return
            if done and os.path.exists(done): say(f"[gpu {g}] skip {name} (finished)"); continue
            say(f"[gpu {g}] start {name}"); t0 = time.time()
            if dry: print("   DRY:", " ".join(cmd)); continue
            os.makedirs(os.path.dirname(log), exist_ok=True)
            prog = os.path.join(os.path.dirname(os.path.dirname(log)), "progress", os.path.basename(log)[:-4] + ".json"); os.makedirs(os.path.dirname(prog), exist_ok=True)
            open(prog + ".gpu", "w").write(str(g))
            env = dict(os.environ, PW_PROGRESS_FILE=prog, CUDA_VISIBLE_DEVICES=str(g), TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"); env.pop("PW_GPU_LOCK", None)
            with open(log, "a") as lf:
                proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT); open(prog + ".pid", "w").write(str(proc.pid)); rc = proc.wait()
            with lock:
                say(f"[gpu {g}] {'done' if rc == 0 else 'FAILED rc=%d' % rc} {name} ({(time.time() - t0) / 60:.1f} min) log {log}")
                if rc: fails.append(name)
    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(per_gpu)]
    if board: os.makedirs(os.path.join(out, "progress"), exist_ok=True); board.start()
    try: [t.start() for t in ths]; [t.join() for t in ths]
    except KeyboardInterrupt: print("\ninterrupted: running replicas keep their checkpoints; rerun the same command to continue", flush=True); os._exit(130)
    finally:
        if board: board.finish()
    if fails: print("FAILED:", fails, "(see events.log and the run logs)"); sys.exit(1)


def cmd_farm(a):
    m, jobs = template_jobs(a.template, a.hosts.split(",") if a.hosts else None); gpus = [int(x) for x in a.gpus.split(",")]; tasks = []
    extra = a.md_args.split() if a.md_args else []
    for s in a.seeds:
        for j in jobs:
            out = os.path.join(a.out, "farm", f"{j['name']}_s{s}"); cmd = list(j["command"])
            cmd = set_val(cmd, "--seed", s); cmd = set_val(cmd, "--out", out); cmd = set_val(cmd, "--checkpoint-seconds", a.checkpoint_seconds)
            if "--frames-only" not in cmd: cmd.append("--frames-only")
            cmd += extra
            tag = cmd[cmd.index("--tag") + 1]; tasks.append((f"{j['name']} seed {s}", cmd, os.path.join(a.out, "logs", f"{j['name']}_s{s}.log"), os.path.join(out, tag + "_frames.npz")))
    print(f"{len(tasks)} MD replicas ({len(jobs)} hosts x {len(a.seeds)} seeds) on GPUs {gpus}, {a.per_gpu} per GPU", flush=True)
    c = tasks[0][1]; val = lambda f, d: float(c[len(c) - 1 - c[::-1].index(f) + 1]) if f in c else d       # last occurrence wins (argparse)
    dt, ps, eq, ev = val("--dt", 0.5), val("--ps", 4.0), val("--equil-ps", 0.5), val("--every-fs", 20.0); nfr = int(ps * 1000 / ev); every = max(1, int(round(ev / dt)))
    plan = [("MD host equilibration", int(eq * 1000 / dt)), ("MD host sampling", nfr * every), ("MD host + 1 H equilibration", int(eq * 1000 / dt)), ("MD host + 1 H sampling", nfr * every)]
    run_pool(tasks, gpus, a.per_gpu, a.dry_run, out=a.out, dash=a.dash, plain=a.plain, plan=plan)


def cmd_pool(a):
    hosts = a.hosts.split(",") if a.hosts else sorted({os.path.basename(d).rsplit("_s", 1)[0] for d in glob.glob(os.path.join(a.out, "farm", "*_s*"))})
    pooled = os.path.join(a.out, "pooled"); os.makedirs(pooled, exist_ok=True); jobs = []; P = T = model = None
    for h in hosts:
        files = sorted(f for d in glob.glob(os.path.join(a.out, "farm", f"{h}_s*")) for f in glob.glob(os.path.join(d, "*_frames.npz")) if int(os.path.basename(d).rsplit("_s", 1)[1]) not in set(a.exclude_seeds))
        if not files: print(f"{h}: no frame files, skipped"); continue
        zs = [np.load(f, allow_pickle=False) for f in files]; z0 = zs[0]
        for f, z in zip(files, zs):
            for k in ("numbers", "cell"):
                if not np.array_equal(z[k], z0[k]): sys.exit(f"{f}: different {k} from {files[0]}")
            if abs(float(z["V"]) - float(z0["V"])) > 1e-6 * float(z0["V"]) or float(z["T"]) != float(z0["T"]) or str(z["model_fp"]) != str(z0["model_fp"]): sys.exit(f"{f}: other V / T / model fingerprint than {files[0]}")
        out = os.path.join(pooled, os.path.basename(files[0]))
        np.savez(out, host=np.concatenate([z["host"] for z in zs]), full=np.concatenate([z["full"] for z in zs]), P_host=np.concatenate([z["P_host"] for z in zs]), P_full=np.concatenate([z["P_full"] for z in zs]),
                 seg_host=np.array([len(z["host"]) for z in zs]), seg_full=np.array([len(z["full"]) for z in zs]), V=z0["V"], T=z0["T"], natoms=z0["natoms"], numbers=z0["numbers"], cell=z0["cell"], model=z0["model"], model_fp=z0["model_fp"],
                 segment_seed=0, segments=len(zs), pooled_from=np.array([os.path.abspath(f) for f in files]))
        print(f"{h}: pooled {len(files)} replicas -> {sum(len(z['host']) for z in zs)} host + {sum(len(z['full']) for z in zs)} host+1H frames  {out}")
        jobs.append(h)
    json.dump(dict(pooled_hosts=jobs), open(os.path.join(pooled, "pooled_hosts.json"), "w"))


def bar_txt(f, n=16):
    k = 0 if f is None else int(round(max(0.0, min(1.0, f)) * n)); return "#" * k + "." * (n - k)


def smi(query, fields):
    try: out = subprocess.check_output(["nvidia-smi", f"--query-{query}=" + ",".join(fields), "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception: return []
    return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def mem_gb():
    try:
        m = {l.split(":")[0]: float(l.split()[1]) for l in open("/proc/meminfo")}; return (m["MemTotal"] - m["MemAvailable"]) / 1048576, m["MemTotal"] / 1048576
    except Exception: return None, None


def render(out, prev, pattern="*_s[0-9]*.json", plan=None, queued=0, slots=0):
    """dashboard lines for the runs in OUT/progress: per-GPU utilisation / memory / throughput with crowding warnings, then one line per run (stage, bar, speed, GPU memory of that process, elapsed, ETA).
    speed = progress units per second since the previous call (recent), else the stage average. prev: dict kept by the caller between calls."""
    now = time.time(); runs = []; L = []
    for pf in sorted(glob.glob(os.path.join(out, "progress", "*.json"))):
        name = os.path.basename(pf)[:-5]; gpu = open(pf + ".gpu").read().strip() if os.path.exists(pf + ".gpu") else "?"; pid = open(pf + ".pid").read().strip() if os.path.exists(pf + ".pid") else ""
        try: r = json.load(open(pf))
        except Exception: r = None
        runs.append((name, gpu, pid, r))
    apps = {x[0]: x[1] for x in smi("compute-apps", ["pid", "used_memory"])}; gpus = {x[0]: x for x in smi("gpu", ["index", "utilization.gpu", "memory.used", "memory.total"])}
    rows = []; per_gpu = {}; run_etas = []; speeds = []
    for name, gpu, pid, r in runs:
        if r is None: rows.append((name, gpu, "starting", None, "", apps.get(pid, "") if pid else "", "", "")); continue
        el = r.get("elapsed_seconds", 0); done = r.get("done", 0); unit = r.get("unit", ""); stale = now - r.get("updated", 0)
        key = (name, r["stage"]); sp = None
        if key in prev and done > prev[key][1] and now > prev[key][0]: sp = (done - prev[key][1]) / (now - prev[key][0])
        elif el > 20 and done > 0: sp = done / el
        prev[key] = (now, done); eta = r.get("eta_seconds"); running = r["stage"] != "complete" and stale < 300
        if sp and running: per_gpu.setdefault(gpu, []).append(sp)
        reta = None                                    # whole-run ETA: rest of this stage + the later stages of the plan, at this run's stage-average speed
        if plan and r["stage"] in [x[0] for x in plan] and el > 20 and done > 0 and r["stage"] != "complete":
            i = [x[0] for x in plan].index(r["stage"]); rem = max(0, r.get("total", 0) - done) + sum(x[1] for x in plan[i + 1:]); reta = rem / (done / el); run_etas.append(reta)
        if plan and sp and r["stage"] in [x[0] for x in plan] and running: speeds.append(sp)
        rows.append((name, gpu, r["stage"] + ("  (stale %dm)" % (stale / 60) if stale > 300 and r["stage"] != "complete" else ""), r.get("fraction"), ("%.2f %s/s" % (sp, unit[:4])) if sp else "-", apps.get(pid, "") if pid else "", "%d:%02d" % divmod(int(el // 60), 60), ("-" if eta is None else "%d:%02d" % divmod(int(eta // 60), 60)) + ("/" + ("%d:%02d" % divmod(int(reta // 60), 60)) if reta else "")))
    done_n = len(glob.glob(os.path.join(out, "farm", "*", "*_frames.npz")))
    L.append(f"{out}: {done_n} frame files finished, {len(runs)} runs   ({time.strftime('%H:%M:%S')})")
    used, tot = mem_gb(); load = os.getloadavg()[0]; ncpu = os.cpu_count() or 1
    L.append(f"node: load {load:.0f}/{ncpu} cores" + (f", RAM {used:.0f}/{tot:.0f} GB" if used else ""))
    warn = []
    for g in sorted(gpus):
        _, ut, mu, mt = gpus[g]; ut, mu, mt = float(ut), float(mu) / 1024, float(mt) / 1024; nrun = sum(1 for x in runs if x[1] == g and x[3] and x[3]["stage"] != "complete"); tp = per_gpu.get(g, [])
        msg = f"GPU{g}: util {ut:3.0f}%  mem {mu:5.1f}/{mt:.0f} GiB  {nrun} run(s)" + (f"  total {sum(tp):.1f} steps/s ({sum(tp) / len(tp):.1f} each)" if tp else ""); note = ""
        if ut >= 95 and nrun >= 2: note = "  << SATURATED: more runs here will not add throughput"; warn.append(g)
        elif ut < 60 and nrun >= 1: note = "  (headroom: could take another run)"
        if mu / mt > 0.85: note += "  << MEMORY high: single-point stages could run out"
        L.append(msg + note)
    if warn: L.append("warning: GPU(s) " + ",".join(warn) + " saturated; if per-run steps/s is a fraction of a lone run, lower --per-gpu")
    if plan and run_etas:
        full = sum(x[1] for x in plan); sp_mean = sum(speeds) / len(speeds) if speeds else None
        tail = (queued / max(1, slots)) * (full / sp_mean) if (queued and sp_mean) else 0.0; tot = max(run_etas) + tail
        L.append(f"whole farm: {queued} queued, about {tot / 60:.0f} min to go (running runs finish in ~{max(run_etas) / 60:.0f} min" + (f", then ~{tail / 60:.0f} min for the queue" if tail else "") + "; assumes the current speeds)")
    L.append(f"{'run':<14}{'GPU':<4}{'stage':<30}{'progress':<22}{'speed':<16}{'GPU mem':<9}{'elapsed':<9}{'ETA stage/run':<14}")
    for n, g, st, f, sp, gm, el, eta in rows: L.append(f"{n:<14}{g:<4}{st:<30}{bar_txt(f)} {('%3.0f%%' % (100 * f)) if f is not None else '  - '}   {sp:<16}{(gm + ' MiB') if gm else '-':<9}{el:<9}{eta:<14}")
    return L


class Dashboard:
    """redraws render() every `interval` seconds in a background thread: in place by cursor movement on a terminal (scrollback is kept), plain blocks every minute when stdout is not a terminal"""
    def __init__(self, out, interval, plain=False, pattern="*_s[0-9]*.json", plan=None, queued=None, slots=0):
        self.pattern, self.plan, self.queued, self.slots = pattern, plan, queued, slots; self.out, self.interval, self.prev, self.n, self.stop = out, interval, {}, 0, threading.Event(); self.tty = sys.stdout.isatty() and not plain; self.t = None
    def draw(self, final=False):
        L = render(self.out, self.prev, self.pattern, self.plan, self.queued() if self.queued else 0, self.slots)
        if self.tty:
            if self.n: sys.stdout.write("\033[%dA" % self.n)
            cols = max(60, shutil.get_terminal_size((200, 24)).columns - 1); sys.stdout.write("".join(l[:cols] + "\033[K\n" for l in L) + ("\033[J" if True else "")); sys.stdout.flush(); self.n = len(L)
        elif final or time.time() - getattr(self, "last", 0) > 60:
            print("\n".join(L), flush=True); self.last = time.time()
    def start(self):
        def loop():
            while not self.stop.wait(self.interval): self.draw()
        self.t = threading.Thread(target=loop, daemon=True); self.t.start()
    def finish(self):
        self.stop.set(); self.t and self.t.join(); self.draw(final=True)


def cmd_status(a):
    prev = {}
    while True:
        if a.watch and sys.stdout.isatty(): d = Dashboard(a.out, a.watch, pattern=a.pattern); d.prev = prev; d.draw(); time.sleep(a.watch); continue      # in-place redraw (no screen clear, scrollback stays)
        print("\n".join(render(a.out, prev, a.pattern)))
        if not a.watch: return
        time.sleep(a.watch)


def cmd_sp(a):
    m, jobs = template_jobs(a.template, a.hosts.split(",") if a.hosts else None); pooled = os.path.join(a.out, "pooled"); gpus = [int(x) for x in a.gpus.split(",")]; tasks = []; mjobs = []; extra = a.sp_args.split() if a.sp_args else []
    for j in jobs:
        cmd = drop_flag(set_val(set_val(list(j["command"]), "--out", pooled), "--seed", a.seed), "--frames-only")
        cmd += extra; tag = cmd[cmd.index("--tag") + 1]; lab = cmd[cmd.index("--label") + 1] if "--label" in cmd else "candidate"; K = cmd[cmd.index("--K") + 1]
        res = os.path.join(pooled, f"{tag}_{lab}_K{K}.json"); mjobs.append(dict(name=j["name"], result=res, gpu="-", command=cmd))
        tasks.append((j["name"], cmd, os.path.join(a.out, "logs", f"sp_{j['name']}.log"), res))
    run_pool(tasks, gpus, 1, a.dry_run, out=a.out, dash=a.dash, plain=a.plain, pattern="sp_*.json")
    ref = m.get("reference")
    json.dump(dict(model=m.get("model"), P=m.get("P"), T=m.get("T"), jobs=mjobs, status="complete", pooled_from=os.path.abspath(a.out), **({"reference": ref} if ref else {})), open(os.path.join(pooled, "manifest.json"), "w"), indent=1)
    print("wrote", os.path.join(pooled, "manifest.json"), "-> python3 mlp_pipeline/pwidom_hull.py", pooled)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("farm"); f.add_argument("template"); f.add_argument("--seeds", type=int, nargs="+", required=True); f.add_argument("--out", required=True); f.add_argument("--gpus", default="0,1,2,3"); f.add_argument("--per-gpu", type=int, default=2)
    f.add_argument("--hosts", default=None); f.add_argument("--md-args", default=""); f.add_argument("--checkpoint-seconds", type=int, default=300); f.add_argument("--dry-run", action="store_true"); f.add_argument("--dash", type=int, default=15, help="dashboard refresh seconds (0 = none)"); f.add_argument("--plain", action="store_true", help="no in-place redraw: a status block every minute")
    p = sub.add_parser("pool"); p.add_argument("out"); p.add_argument("--hosts", default=None); p.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    s = sub.add_parser("sp"); s.add_argument("template"); s.add_argument("out"); s.add_argument("--gpus", default="0,1,2,3"); s.add_argument("--hosts", default=None); s.add_argument("--sp-args", default=""); s.add_argument("--seed", type=int, default=3); s.add_argument("--dry-run", action="store_true"); s.add_argument("--dash", type=int, default=15); s.add_argument("--plain", action="store_true")
    t = sub.add_parser("status"); t.add_argument("out"); t.add_argument("--watch", type=int, default=0, help="refresh every N seconds"); t.add_argument("--pattern", default="*_s[0-9]*.json", help="progress files to show (sp stage: sp_*.json)")
    a = ap.parse_args(); {"farm": cmd_farm, "pool": cmd_pool, "sp": cmd_sp, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
