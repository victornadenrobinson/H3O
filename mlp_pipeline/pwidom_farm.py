#!/usr/bin/env python3
"""Farm many independent MD replicas, pool their frames, then run ONE single-point stage per host (no GPU-memory clash between hosts' single-point stages, MD replicas are cheap and can share a GPU).
Everything is taken from the manifest of an earlier launcher run (its job commands hold every host argument), so the host, volume, model, proxy, K, sel, batch are exactly those of that run unless you override them.
  1. farm : python3 mlp_pipeline/pwidom_farm.py farm TEMPLATE_RUN --seeds 31 32 33 34 --out runs/farm_500GPa_5000K [--per-gpu 3] [--gpus 0,1,2,3] [--hosts H2O,pureH] [--md-args "--ps 2.0 --equil-ps 0.5"] [--dry-run]
            MD-only (--frames-only): one process per (host, seed), at most --per-gpu concurrent per GPU, frames to OUT/farm/<host>_s<seed>/. Resumable: finished frame files are skipped.
  2. pool : python3 mlp_pipeline/pwidom_farm.py pool OUT [--hosts ...] [--exclude-seeds N ...]
            merges all replicas of each host into OUT/pooled/<tag>_frames.npz (checks identical species / cell / volume / T / model fingerprint; keeps per-trajectory lengths so the melt test and block errors stay per trajectory)
            and writes OUT/pooled/manifest.json so pwidom_hull.py / plot scripts read it like any run.
  3. sp   : python3 mlp_pipeline/pwidom_farm.py sp TEMPLATE_RUN OUT [--gpus 0,1,2,3] [--sp-args "--sel 24 --K 3000 --batch 4"] [--dry-run]
            the single-point stage on the pooled frames, ONE process per GPU at a time (queued when hosts > GPUs), results in OUT/pooled/.
Do not pool frames made with the old thermostat or another model: the pool refuses a different model fingerprint, V or T, but it cannot know the friction; use only runs made after the thermostat fix."""
import argparse, glob, json, os, queue, subprocess, sys, threading, time
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


def run_pool(tasks, gpus, per_gpu, dry):
    """tasks: list of (name, cmd, logfile, outcheck); runs them with at most per_gpu concurrent processes per GPU"""
    q = queue.Queue(); [q.put(t) for t in tasks]; lock = threading.Lock(); fails = []
    def worker(g):
        while True:
            try: name, cmd, log, done = q.get_nowait()
            except queue.Empty: return
            if done and os.path.exists(done): print(f"[gpu {g}] skip {name} (finished)", flush=True); continue
            print(f"[gpu {g}] start {name}", flush=True); t0 = time.time()
            if dry: print("   DRY:", " ".join(cmd)); continue
            os.makedirs(os.path.dirname(log), exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"); env.pop("PW_GPU_LOCK", None)
            with open(log, "a") as lf: rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            with lock:
                print(f"[gpu {g}] {'done' if rc == 0 else 'FAILED rc=%d' % rc} {name} ({(time.time() - t0) / 60:.1f} min) log {log}", flush=True)
                if rc: fails.append(name)
    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(per_gpu)]
    [t.start() for t in ths]; [t.join() for t in ths]
    if fails: print("FAILED:", fails); sys.exit(1)


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
    run_pool(tasks, gpus, a.per_gpu, a.dry_run)


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


def cmd_sp(a):
    m, jobs = template_jobs(a.template, a.hosts.split(",") if a.hosts else None); pooled = os.path.join(a.out, "pooled"); gpus = [int(x) for x in a.gpus.split(",")]; tasks = []; mjobs = []; extra = a.sp_args.split() if a.sp_args else []
    for j in jobs:
        cmd = drop_flag(set_val(set_val(list(j["command"]), "--out", pooled), "--seed", a.seed), "--frames-only")
        cmd += extra; tag = cmd[cmd.index("--tag") + 1]; lab = cmd[cmd.index("--label") + 1] if "--label" in cmd else "candidate"; K = cmd[cmd.index("--K") + 1]
        res = os.path.join(pooled, f"{tag}_{lab}_K{K}.json"); mjobs.append(dict(name=j["name"], result=res, gpu="-", command=cmd))
        tasks.append((j["name"], cmd, os.path.join(a.out, "logs", f"sp_{j['name']}.log"), res))
    run_pool(tasks, gpus, 1, a.dry_run)
    ref = m.get("reference")
    json.dump(dict(model=m.get("model"), P=m.get("P"), T=m.get("T"), jobs=mjobs, status="complete", pooled_from=os.path.abspath(a.out), **({"reference": ref} if ref else {})), open(os.path.join(pooled, "manifest.json"), "w"), indent=1)
    print("wrote", os.path.join(pooled, "manifest.json"), "-> python3 mlp_pipeline/pwidom_hull.py", pooled)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("farm"); f.add_argument("template"); f.add_argument("--seeds", type=int, nargs="+", required=True); f.add_argument("--out", required=True); f.add_argument("--gpus", default="0,1,2,3"); f.add_argument("--per-gpu", type=int, default=2)
    f.add_argument("--hosts", default=None); f.add_argument("--md-args", default=""); f.add_argument("--checkpoint-seconds", type=int, default=300); f.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pool"); p.add_argument("out"); p.add_argument("--hosts", default=None); p.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    s = sub.add_parser("sp"); s.add_argument("template"); s.add_argument("out"); s.add_argument("--gpus", default="0,1,2,3"); s.add_argument("--hosts", default=None); s.add_argument("--sp-args", default=""); s.add_argument("--seed", type=int, default=3); s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); {"farm": cmd_farm, "pool": cmd_pool, "sp": cmd_sp}[a.cmd](a)


if __name__ == "__main__":
    main()
