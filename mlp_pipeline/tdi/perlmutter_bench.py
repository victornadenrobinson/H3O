"""Bounded four-GPU TDI throughput screening, then measured whole-node validation.

No production trajectories/results are written. Warm-up/compilation is excluded
from rates, but included in the wall cap. No MPS daemon/settings are changed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from perlmutter_pwidom import duration, save, write_origin


def telemetry():
    try:
        text = subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        result = {}
        for row in csv.reader(text.splitlines(), skipinitialspace=True):
            index, uuid, used, total, util = (x.strip() for x in row)
            data = dict(memory_MiB=float(used), total_MiB=float(total), utilization=float(util))
            result[index] = result[uuid] = data
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def stop(workers):
    for w in workers:
        try:
            os.killpg(w["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+3
    while any(w["process"].poll() is None for w in workers) and time.monotonic() < deadline:
        time.sleep(.1)
    for w in workers:
        try:
            os.killpg(w["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        w["process"].wait()


def timed_summary(results):
    if not results or any(r["timed_ended"] <= r["timed_started"] for r in results):
        raise ValueError("Missing/invalid measured timing intervals")
    span = max(r["timed_ended"] for r in results)-min(r["timed_started"] for r in results)
    steps = sum(r.get("work_units",r.get("lambda_steps",0)) for r in results)
    return dict(lambda_steps=steps, measured_seconds=span, lambda_steps_s=steps/span,
                work_units=steps,work_units_s=steps/span,unit=results[0].get("unit","lambda-steps"))


def memory_projection(mode, size, profile, safety):
    """Conservative disconnected-graph scaling from a one-worker calibration."""
    if mode in ("engine","pwmd"):
        return size * profile["process_MiB"] * safety
    # Lambda insertion batches contain a full system AND its host per replica.
    graphs = size * (profile["graphs_per_lambda"] if mode == "batch" else 1)
    if profile.get("tensor_MiB"):
        return (profile["context_MiB"] + graphs*profile["tensor_MiB"]) * safety
    return graphs*profile["process_MiB"]*safety


def phase(a, name, layouts, out, common, deadline, interrupted):
    print(f"\n== {name}: " + ", ".join(f"GPU {g}: {mode} {n}" for g, mode, n in layouts), flush=True)
    barrier = out / f"{name}.go"
    cases = []; all_workers = []; started = time.monotonic(); released = False
    try:
        available = {} if a.cpu else telemetry()
        for gpu, mode, size in layouts:
            case = dict(phase=name, gpu=gpu, mode=mode, size=size, workers=[], samples=[], error=None)
            cases.append(case)
            if not a.cpu and name != "memory_calibration":
                profile = getattr(a,"_memory_profile",None)
                metric = available.get(gpu)
                if profile is None or metric is None:
                    case.update(error="memory calibration/telemetry missing; not admitted",skipped=True)
                    continue
                projected = memory_projection(mode,size,profile,a.memory_safety)
                reserve = max(a.reserve_gib*1024,.2*metric["total_MiB"])
                if projected+reserve+metric["memory_MiB"] > metric["total_MiB"]:
                    case.update(error=f"projected {projected/1024:.1f} GiB + reserve {reserve/1024:.1f} GiB exceeds free VRAM",skipped=True)
                    continue
            for rank in range(size if mode in ("engine","pwmd") else 1):
                log = out / f"{name}_gpu{gpu}_{mode}{size}_{rank}.log"
                cmd = [sys.executable, "-u", "mlp_pipeline/tdi/bench.py", mode, a.system,
                       *common, "--barrier", str(barrier)]
                if mode in ("batch","pwsp"): cmd += ["--nrep", str(size)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if a.cpu else gpu,
                    OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS="1",
                    TF_NUM_INTRAOP_THREADS=str(a.threads), TF_NUM_INTEROP_THREADS="1",
                    H3O_NL_WORKERS="0", TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2")
                env.pop("PW_GPU_LOCK", None)
                with log.open("x") as fh:
                    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
                w = dict(process=p, log=log, command=cmd)
                case["workers"].append(w); all_workers.append(w)
        last = sample = -float("inf")
        while any(w["process"].poll() is None for w in all_workers):
            now = time.monotonic()
            if interrupted() or now >= deadline or now-started >= a.stage_seconds:
                for c in cases:
                    if any(w["process"].poll() is None for w in c["workers"]):
                        c["error"] = "interrupted/wall cap or stage timeout"
                break
            for c in cases:
                if c["error"]:
                    continue
                if any(w["process"].poll() not in (None, 0) for w in c["workers"]):
                    c["error"] = "worker failed (see log; OOM/self-check failures are not valid measurements)"
                    stop(c["workers"])
            live = [c for c in cases if not c["error"]]
            ready = sum("\nREADY\n" in ("\n"+w["log"].read_text(errors="replace"))
                        for c in live for w in c["workers"])
            expected = sum(len(c["workers"]) for c in live)
            if not released and live and ready == expected:
                barrier.touch(); released = True
                print(f"All {ready} surviving workers warmed up; starting synchronized timing (load {now-started:.1f}s)", flush=True)
            if now-sample >= 5:
                metrics = {} if a.cpu else telemetry()
                for c in cases:
                    if c["gpu"] in metrics:
                        c["samples"].append(metrics[c["gpu"]])
                sample = now
            if now-last >= 15:
                print(f"  elapsed {duration(now-started)}; benchmark cap remaining {duration(deadline-now)}; "
                      f"{'timing' if released else 'warm-up'}; ready {ready}/{expected}", flush=True)
                for c in cases:
                    if c["samples"]:
                        m = c["samples"][-1]
                        active = sum(w["process"].poll() is None for w in c["workers"])
                        print(f"    GPU {c['gpu']}: {m['utilization']:.0f}% util, "
                              f"{m['memory_MiB']/1024:.1f}/{m['total_MiB']/1024:.1f} GiB; "
                              f"size {c['size']} ({c['mode']}), {active} live processes", flush=True)
                last = now
            time.sleep(.2)
    finally:
        stop(all_workers)
    rows = []
    for c in cases:
        results = []
        for w in c["workers"]:
            lines = w["log"].read_text(errors="replace").splitlines()
            found = []
            for line in lines:
                if line.startswith("RESULT "):
                    try:
                        found.append(json.loads(line[7:]))
                    except ValueError:
                        c["error"] = "truncated/invalid result (excluded from ranking)"
            if len(found) == 1 and w["process"].returncode == 0:
                results.append(found[0])
            else:
                c["error"] = c["error"] or f"missing result/failed worker: {w['log']}"
        row = dict(phase=name, gpu=c["gpu"], mode=c["mode"], size=c["size"],
                   status="skipped" if c.get("skipped") else "failed" if c["error"] else "complete", error=c["error"],
                   workers=results, logs=[str(w["log"]) for w in c["workers"]],
                   worker_pids=[w["process"].pid for w in c["workers"]],
                   configured_size=c["size"],
                   sampled_peak_MiB=max((s["memory_MiB"] for s in c["samples"]), default=None))
        if not c["error"]:
            row.update(timed_summary(results))
            print(f"RESULT GPU {c['gpu']} {c['mode']} {c['size']}: {row['work_units_s']:.2f} {row['unit']}/s; "
                  f"sampled peak {row['sampled_peak_MiB']} MiB", flush=True)
        else:
            print(f"{row['status'].upper()} GPU {c['gpu']} {c['mode']} {c['size']}: {c['error']}", flush=True)
        rows.append(row)
        with (out / "results.jsonl").open("a") as fh: fh.write(json.dumps(row)+"\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflow",choices=["pwidom","tdi"],default="pwidom",
                    help="pwidom: direct MD + energy batches; tdi: independent/batched lambda engines")
    ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2")
    ap.add_argument("--state-json", default="envs/pwidom_500GPa_3000K.json")
    ap.add_argument("--system", choices=["ins", "h2o", "H", "insH"], default=None)
    ap.add_argument("--processes", nargs="+", type=int, default=[1,2,4,8])
    ap.add_argument("--replicas", nargs="+", type=int, default=[2,4,8,16])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--stage-seconds", type=float, default=180)
    ap.add_argument("--reserve-gib",type=float,default=12,help="minimum VRAM reserve (also reserve at least 20% of device memory)")
    ap.add_argument("--memory-safety",type=float,default=1.5,help="multiplier on calibrated VRAM projections")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="tiny toy workflow check only")
    a = ap.parse_args()
    if not all(math.isfinite(v) for v in (a.minutes,a.stage_seconds,a.reserve_gib,a.memory_safety)):
        raise ValueError("Time limits and VRAM guard settings must be finite")
    a.system = a.system or ("h2o" if a.workflow == "pwidom" else "ins")
    if a.workflow == "pwidom" and a.system not in ("h2o","H"):
        raise ValueError("pwidom benchmark system must be h2o or H (no lambda insertion path)")
    if (a.steps <= 0 or a.warm < 0 or a.threads <= 0 or a.minutes <= 0 or a.stage_seconds <= 0
            or a.reserve_gib < 0 or a.memory_safety < 1
            or not a.processes or not a.replicas or max(len(a.processes),len(a.replicas)) > 4
            or min(a.processes+a.replicas) < 1):
        raise ValueError("Need 1-4 positive sizes per sweep, positive limits/steps/threads and nonnegative warm-up")
    if not a.cpu and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run inside the existing four-GPU Slurm allocation")
    if a.cpu and a.model != "toyice":
        raise ValueError("--cpu requires --model toyice")
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError("An MPS environment is active; this benchmark measures an unchanged non-MPS baseline")
    data = json.loads(Path(a.state_json).read_text())
    key = "H" if a.system in ("H","insH") else "ice"
    vpa = float(data[key]["vpa"])
    if not math.isfinite(vpa) or not vpa > 0:
        raise ValueError("Invalid benchmark volume")
    if a.model.startswith("grace-ft:"):
        path = Path(a.model.split(":",1)[1]).expanduser().resolve()
        exports = sorted(path.rglob("saved_model.pb"))
        if len(exports) != 1: raise ValueError("Specify one unambiguous GRACE SavedModel export")
        a.model = "grace-ft:"+str(exports[0].parent)
    if a.cpu:
        gpus = ["cpu0","cpu1","cpu2","cpu3"]
    else:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        metrics = telemetry()
        for gpu in gpus:
            if gpu not in metrics:
                hit = next((key for key in metrics if key.startswith("GPU-") and key.startswith(gpu)), None)
                if hit:
                    metrics[gpu] = metrics[hit]
        if len(gpus) != 4 or len(set(gpus)) != 4 or any(g not in metrics for g in gpus):
            raise ValueError("Need four visible GPUs and working nvidia-smi telemetry")
        if any(metrics[g]["memory_MiB"] > 512 for g in gpus):
            raise ValueError("GPUs have allocated memory: finish other calculations before benchmarking")
    out = Path(a.out or time.strftime("runs/bench_%Y%m%d_%H%M%S"))
    if out.exists(): raise ValueError(f"Output exists: {out}; choose a fresh directory")
    cpus = len(os.sched_getaffinity(0))
    print(f"Benchmark: {a.system}, {data['P']:g} GPa / {data['T']:g} K, vpa={vpa:g}; {a.model}")
    print(f"4 GPUs; {cpus} available CPU threads; TF threads/process {a.threads}; neighbor worker pool disabled")
    print(f"Workflow {a.workflow}: processes/GPU {a.processes}; batch sizes/GPU {a.replicas}; whole-node validation follows")
    print(f"{a.warm} warm-up + {a.steps} timed MD steps; {a.minutes:g}-minute global cap; outputs {out}; no MPS changes")
    if a.cpu: print("TOY WORKFLOW TEST: rates are not GPU performance measurements")
    if a.dry_run: return
    out.mkdir(parents=True, exist_ok=False)
    save(out/"manifest.json", dict(settings=vars(a), state=data, cpus=cpus,
         job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
         git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
    write_origin(out, toy=a.cpu)
    common = ["--model",a.model,"--vpa",str(vpa),"--P",str(data["P"]),"--T",str(data["T"]),
              "--steps",str(a.steps),"--warm",str(a.warm),"--device","cpu" if a.cpu else "cuda",
              "--workdir",str(out/"setup")]
    if a.cpu: common += ["--cell-nrep","1"]
    interrupted = [False]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    deadline = time.monotonic()+a.minutes*60
    rows = []
    if not a.cpu:
        mode = "pwmd" if a.workflow == "pwidom" else "engine"
        calibration = phase(a,"memory_calibration",[(g,mode,1) for g in gpus],out,common,deadline,lambda:interrupted[0])
        good = [r for r in calibration if r["status"] == "complete"]
        if not good:
            raise ValueError("No valid memory calibration; larger layouts were not launched")
        process_peak = max(r["sampled_peak_MiB"] or 0 for r in good)
        tensor_peak = max(w.get("tensorflow_peak_MiB",0) for r in good for w in r["workers"])
        process_peak = max(process_peak,tensor_peak+2048)
        a._memory_profile = dict(process_MiB=process_peak,tensor_MiB=tensor_peak,
            context_MiB=max(2048,process_peak-tensor_peak),
            graphs_per_lambda=2 if a.system in ("ins","insH") else 1)
        save(out/"memory_calibration.json",a._memory_profile)
        print(f"VRAM guard: baseline {process_peak/1024:.1f} GiB/worker; safety x{a.memory_safety:g}; reserve >=20% or {a.reserve_gib:g} GiB. Unsafe layouts are skipped.",flush=True)
    phases = (("process_screen","engine",a.processes),("batch_screen","batch",a.replicas)) if a.workflow == "tdi" else (
        ("md_screen","pwmd",a.processes),("energy_screen","pwsp",a.replicas))
    for name, mode, sizes in phases:
        if interrupted[0] or time.monotonic() >= deadline: break
        rows += phase(a,name,[(g,mode,n) for g,n in zip(gpus,sizes)],out,common,deadline,lambda:interrupted[0])
    valid = [r for r in rows if r["status"] == "complete"]
    winner = max(valid,key=lambda r:r["lambda_steps_s"]) if valid and a.workflow == "tdi" else None
    winners = {}
    if a.workflow == "pwidom":
        for mode in ("pwmd","pwsp"):
            candidates = [r for r in valid if r["mode"] == mode]
            if candidates: winners[mode] = max(candidates,key=lambda r:r["work_units_s"])
    elif winner:
        winners["tdi"] = winner
    validation = []
    totals = {}
    for key, best in winners.items():
        if interrupted[0] or time.monotonic() >= deadline: break
        name = "node_validation" if a.workflow == "tdi" else key+"_node_validation"
        group = phase(a,name,[(g,best["mode"],best["size"]) for g in gpus],out,common,deadline,lambda:interrupted[0])
        validation += group
        if len(group)==4 and all(r["status"] == "complete" for r in group):
            totals[key] = timed_summary([w for r in group for w in r["workers"]])
            print(f"MEASURED FOUR-GPU {key}: {totals[key]['work_units_s']:.2f} {totals[key]['unit']}/s; size {best['size']}/GPU",flush=True)
    required = 1 if a.workflow == "tdi" else 2
    summary = dict(best_observed_screen=winner, validation=validation,
                   workflow=a.workflow,best_by_mode=winners,whole_node_by_mode=totals,
                   complete_node_validation=len(totals)==required,
                   interpretation="Short hardware benchmark, not thermodynamic sampling; screening configurations share node CPU resources")
    if summary["complete_node_validation"] and a.workflow == "tdi":
        total = timed_summary([w for r in validation for w in r["workers"]])
        summary["whole_node"] = total
        print(f"\nMEASURED FOUR-GPU TOTAL: {total['lambda_steps_s']:.2f} lambda-steps/s; {winner['mode']} {winner['size']}/GPU", flush=True)
    save(out/"summary.json",summary)
    print(f"Saved {out}/summary.json and results.jsonl; failed/incomplete configurations are excluded from ranking.")
    if not summary["complete_node_validation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as ex:
        raise SystemExit(str(ex))
