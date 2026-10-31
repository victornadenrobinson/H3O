"""Four independent pwidom hosts in an existing allocation; no scheduler submission.

The shell entry points are perlmutter_pwidom.sh and perlmutter_pwidom_status.sh.
Scientific estimators and volume interpolation remain in the existing modules.
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pwidom_scan as scan
from pwidom_progress import duration


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def write_origin(out, toy=False, parent=None):
    """Small human-readable tag; paths stay under runs/ for existing analysis."""
    lines = ["Origin: local CPU toy check" if toy else "Origin: NERSC Perlmutter",
             f"Host: {os.uname().nodename}", f"Slurm allocation: {os.environ.get('SLURM_JOB_ID', 'none')}",
             f"Created: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
             "Inputs, exact commands and Git revision: manifest.json"]
    if parent:
        lines.append(f"Additional sampling from: {parent}")
    (out / "RUN_ORIGIN.txt").write_text("\n".join(lines)+"\n")


class HostLogs:
    """Stream every emitted child line with its host name; retain the full files."""

    def __init__(self, jobs):
        self.jobs = jobs
        self.files = {}
        self.pending = {}

    def drain(self, final=False):
        for job in self.jobs:
            name = job["name"]
            if name not in self.files:
                try:
                    self.files[name] = open(job["log"], errors="replace")
                except FileNotFoundError:
                    continue
            text = self.pending.pop(name, "") + self.files[name].read()
            lines = text.splitlines(keepends=True)
            for line in lines:
                if line.endswith(("\n", "\r")) or final:
                    print(f"[{name}] {line.rstrip(chr(10) + chr(13))}", flush=True)
                else:
                    self.pending[name] = line

    def close(self):
        for file in self.files.values():
            file.close()


def gpu_metrics(jobs):
    ids = [job["gpu"] for job in jobs if job["gpu"]]
    if not ids:
        return {}
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--id=" + ",".join(ids),
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        metrics = {}
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            index, uuid, util, used, total = (value.strip() for value in row)
            text = f"GPU {index}: {util}% util, {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GiB"
            for gpu in ids:
                if gpu == index or uuid.startswith(gpu):
                    metrics[gpu] = text
        return metrics
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def progress_text(job):
    try:
        p = json.loads(Path(job["progress"]).read_text())
        fraction = p["fraction"]
        pct = f" {100 * fraction:.0f}% ({p['done']}/{p['total']} {p['unit']})" if fraction is not None else ""
        eta = "estimating" if p["eta_seconds"] is None else f"{p['eta_seconds'] / 60:.1f} min"
        age = max(0, time.time() - p["updated"])
        elapsed = duration(p.get("elapsed_seconds", 0))
        left = "estimating" if p["eta_seconds"] is None else duration(p["eta_seconds"])
        return f"{p['stage']}{pct}; elapsed {elapsed} / remaining {left} (stage); stage ETA {eta}; updated {age:.0f}s ago"
    except (OSError, ValueError, KeyError):
        # Attach to runs launched before progress sidecars were introduced.
        try:
            text = Path(job["log"]).read_text(errors="replace")
            matches = list(re.finditer(r"MD (host(?: \+ 1 H)?): (\d+)/(\d+) frames,[^\n]*?, ([\d.]+) min", text))
            if matches:
                last = matches[-1]
                stage, done, total, elapsed = last.groups()
                if "wrote " in text[last.end():] or "proxy " in text[last.end():]:
                    return "MD finished; later-stage ETA unavailable in this older log"
                done, total, elapsed = int(done), int(total), float(elapsed) * 60
                previous = [m for m in matches[:-1] if m[1] == stage]
                left = "estimating"
                if previous and done > int(previous[-1][2]):
                    remaining = (elapsed - float(previous[-1][4]) * 60) / (done - int(previous[-1][2])) * (total - done)
                    left = "~" + duration(remaining)
                return f"MD {stage}: {done}/{total} frames; elapsed {duration(elapsed)} / remaining {left} (this MD only; estimated from log)"
        except (OSError, KeyError):
            pass
        return "initializing; stage ETA estimating"


def short_time(seconds):
    text = duration(seconds)
    return text[3:] if text.startswith("00:") else text


def read_result(job):
    try:
        return json.loads(Path(job["result"]).read_text())
    except (OSError, ValueError):
        return None


def shared_reference(manifest):
    ref = manifest.get("reference")
    return (ref.get("snapshot") or json.loads(Path(ref["result"]).read_text())) if ref else None


def transfer_estimates(manifest):
    """Only completed values; never reinterpret old values as a live estimate."""
    results = {j["name"]: read_result(j) for j in manifest["jobs"]}
    if manifest.get("reference"):
        results["pureH"] = shared_reference(manifest)
    previous = {}
    if manifest.get("extends"):
        try:
            old = json.loads((Path(manifest["extends"]) / "manifest.json").read_text())
            previous = {j["name"]: read_result(j) for j in old["jobs"]}
            if old.get("reference"):
                previous["pureH"] = shared_reference(old)
        except (OSError, ValueError):
            pass
    estimates = {}
    for job in manifest["jobs"]:
        name = job["name"]
        if name == "pureH":
            estimates[name] = "H reference"; continue
        for source, label in ((results, "new"), (previous, "prior")):
            if source.get(name) and source.get("pureH"):
                value, error = scan.dg(source[name], source["pureH"])
                estimates[name] = f"{label} dg {value:+.0f} +/-{error:.0f} meV/H"
                break
        else:
            estimates[name] = "dg awaiting probes"
    return estimates


def dashboard(manifest, elapsed=None, exits=None, plain=False, clear=False):
    """One compact block; full output remains in each host's log."""
    metrics = gpu_metrics(manifest["jobs"])
    estimates = transfer_estimates(manifest)
    if elapsed is None:
        try:
            from datetime import datetime
            elapsed = max(0, time.time()-datetime.fromisoformat(manifest["started"]).timestamp())
        except (KeyError, ValueError):
            elapsed = 0
    remaining_cap = max(0, manifest["minutes"]*60-elapsed)
    from pwidom_dashboard import render
    rows = []
    for job in manifest["jobs"]:
        name = job["name"]
        code = (exits or {}).get(name, job.get("exit_code"))
        if code is not None:
            stage = "DONE" if code == 0 else f"FAILED({code})"
            fraction, timing = (1.0 if code == 0 else None), "--"
        else:
            try:
                p = json.loads(Path(job["progress"]).read_text())
                stage = p["stage"].replace("MD host + 1 H", "Host+H").replace("MD host", "Host").replace("equilibration", "eq").replace("sampling", "MD")
                fraction = p["fraction"]
                left = short_time(p["eta_seconds"]) if p["eta_seconds"] is not None else "--:--"
                timing = f"[{short_time(p.get('elapsed_seconds',0))}/{left} ETA]"
                age = max(0, time.time()-p["updated"])
                if age > 40:
                    stage += f" stale {age:.0f}s"
            except (OSError, ValueError, KeyError):
                stage, fraction, timing = "Initializing", None, "--"
                # Preserve useful frame-derived ETA for pre-sidecar runs.
                legacy = progress_text(job)
                match = re.search(r"MD (.*?): (\d+)/(\d+) frames; elapsed (\S+) / remaining (\S+)",legacy)
                if match:
                    stage = "MD "+match[1]
                    fraction = int(match[2])/int(match[3])
                    timing = f"[{match[4]}/{match[5]} ETA]"
        gpu = metrics.get(job["gpu"], "GPU n/a" if job["gpu"] else "CPU")
        gpu = gpu.replace("GPU ", "GPU").replace(": ", " ").replace(" util, ", " ")
        rows.append(dict(name=name,phase=stage,fraction=fraction,timing=timing,gpu=gpu,free_energy=estimates[name]))
    tty = sys.stdout.isatty()
    color = tty and not plain and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    if clear and tty and not plain:
        print("\033[H\033[2J",end="")
    text = render(f"Perlmutter pwidom | {manifest['P']:g} GPa / {manifest['T']:g} K | allocation {manifest.get('allocation','-')}",
                  f"Elapsed {short_time(elapsed)}   Wall cap left {short_time(remaining_cap)}   State: {manifest['status']}",
                  rows, width=shutil.get_terminal_size((140,40)).columns, color=color, plain=plain)
    print(text,flush=True)


def plan(a):
    parent = None
    if a.extra_h is not None and (a.extend_run or not a.state_json or not a.reference_run):
        raise ValueError("--extra-h requires --state-json and --reference-run; cannot combine with --extend-run")
    if a.reference_run and a.extra_h is None:
        raise ValueError("--reference-run requires --extra-h")
    if a.extend_run:
        parent = json.loads((Path(a.extend_run) / "manifest.json").read_text())
        if parent["status"] != "complete":
            raise ValueError("Extend only a completed run; check the original host processes first")
        if a.state_json or a.roots:
            raise ValueError("--extend-run inherits its state; omit --state-json/--roots")
        byname = {j["name"]: j for j in parent["jobs"]}
        custom_parent = parent.get("extra_h") is not None
        expected = [f"Hplus{k}" for k in parent["extra_h"]] if custom_parent else ["H2O", "H2p083O", "H2p25O", "pureH"]
        for name in expected:
            if name not in byname:
                raise ValueError(f"Parent run missing {name}")
        a.P, a.T, a.model = parent["P"], parent["T"], parent["model"]
        def option(name, key):
            cmd = byname[name]["command"]
            return cmd[cmd.index(key)+1]
        first = expected[0]
        a.ice_proxy = option(first, "--proxy")
        a.h_proxy = parent["reference"]["snapshot"]["proxy"] if custom_parent else option("pureH", "--proxy")
        for key in ("K", "sel", "ndel", "batch"):
            setattr(a, key, int(option(first, "--"+key)))
        explicit = dict(P=a.P, T=a.T)
        if custom_parent:
            explicit = parent["eos_inputs"]
            a.extra_h = parent["extra_h"]
            a.reference_run = parent["reference"]["run"]
        for key, name in (() if custom_parent else (("ice", "H2O"), ("h225", "H2p25O"), ("H", "pureH"))):
            size = "natoms" if key == "H" else "nrep"
            explicit[key] = dict(vpa=float(option(name,"--vpa")),
                                 B_GPa=float(option(name,"--bulk-gpa")),
                                 source=f"inherited from {a.extend_run}/manifest.json",
                                 **{size: int(option(name, "--"+size))})
        a._extension_eos = explicit
    if not all(math.isfinite(v) for v in (a.P, a.T, a.ps, a.equil_ps, a.minutes)):
        raise ValueError("P/T/time settings must be finite")
    if a.P <= 0 or a.T <= 0 or min(a.K, a.sel, a.ndel, a.batch) <= 0:
        raise ValueError("P/T/K/sel/ndel/batch must be positive")
    if a.minutes <= 0 or a.ps <= 0 or a.equil_ps < 0:
        raise ValueError("minutes/ps must be positive; equil-ps must be nonnegative")
    if a.ps < 0.1:
        raise ValueError("ps must be >= 0.1: the existing estimator needs five 20-fs frames")
    roots = a.roots or sorted(p for p in glob.glob("runs/tdi*") if Path(p).is_dir())
    explicit = getattr(a, "_extension_eos", None) or (json.loads(Path(a.state_json).read_text()) if a.state_json else None)
    if explicit and (explicit["P"] != a.P or explicit["T"] != a.T):
        raise ValueError("state-json P/T do not match the requested state")
    state = scan.hosts(roots).get((a.P, a.T), {}) if explicit is None else {}
    for key in ("ice", "h225", "H"):
        if explicit is not None:
            s = explicit[key]
        else:
            if key not in state:
                raise ValueError(f"Missing {key} stage0 at {a.P:g} GPa / {a.T:g} K in {roots}; use --state-json for EOS inputs without prior TDI")
            s = json.loads(Path(state[key]).read_text())
        if not all(math.isfinite(float(s[k])) and float(s[k]) > 0 for k in ("vpa", "B_GPa")):
            raise ValueError(f"Invalid volume/bulk modulus: {key}")
        if explicit is not None:
            if not s.get("source"):
                raise ValueError(f"state-json {key} needs a source description for its EOS inputs")
            size_key = "natoms" if key == "H" else "nrep"
            if not isinstance(s.get(size_key), int) or s[size_key] <= 0:
                raise ValueError(f"state-json {key} needs a positive integer {size_key}")
    model = a.model
    if model.startswith("grace-ft:"):
        p = Path(model.split(":", 1)[1]).expanduser().resolve()
        exports = sorted(p.rglob("saved_model.pb"))
        if len(exports) != 1:
            raise ValueError(f"Expected exactly one SavedModel under {p}, found {len(exports)}; specify its export directory")
        model = "grace-ft:" + str(exports[0].parent)
    elif model != "toyice":
        raise ValueError("Supported models: grace-ft:<export directory>, or toyice for local checks")
    for proxy in (a.ice_proxy, a.h_proxy):
        if proxy not in ("none", "toyice"):
            json.loads(Path(proxy).read_text())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if a.cpu:
        if model != "toyice":
            raise ValueError("--cpu is only for local toy workflow checks")
        gpus = [""] * 4
    else:
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Run inside the existing Slurm GPU allocation")
        if visible:
            gpus = visible.split(",")
        else:
            gpus = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            ).split()
        if len(gpus) != 4 or len(set(gpus)) != 4 or any(x in ("-1", "NoDevFiles") for x in gpus):
            raise ValueError(f"Expected four distinct visible GPUs, found {gpus}")
        if not a.dry_run:
            for module in ("tensorflow", "tensorpotential", "matscipy"):
                if importlib.util.find_spec(module) is None:
                    raise ValueError(f"Missing {module} in {sys.executable}")
    if explicit is None:
        jobs = [
            ("H2O", scan.pw_args(state["ice"], "ice"), a.ice_proxy),
            ("H2p083O", scan.interp_args(state["ice"], state["h225"], 2 + 9 / 108), a.ice_proxy),
            ("H2p25O", scan.pw_args(state["h225"], "fcc"), a.ice_proxy),
            ("pureH", scan.pw_args(state["H"], "H"), a.h_proxy),
        ]
    else:
        def args(key, hp=2):
            s = explicit[key]
            return ["--phase", "H" if key == "H" else "fcc",
                    "--natoms" if key == "H" else "--nrep",
                    str(s["natoms" if key == "H" else "nrep"]),
                    "--hper", str(hp), "--vpa", str(s["vpa"]),
                    "--bulk-gpa", str(s["B_GPa"]), "--P", str(a.P), "--T", str(a.T)]
        ice, rich = explicit["ice"], explicit["h225"]
        if ice["nrep"] != rich["nrep"]:
            raise ValueError("state-json ice/h225 must have the same nrep")
        hp = 2 + 9 / 108
        mid = args("ice", hp)
        # Same per-O interpolation and ice bulk-modulus convention as scan.interp_args.
        mid[mid.index("--vpa") + 1] = str((3 * ice["vpa"] + (hp - 2) / .25 *
                                          (3.25 * rich["vpa"] - 3 * ice["vpa"])) / (hp + 1))
        jobs = [("H2O", args("ice"), a.ice_proxy), ("H2p083O", mid, a.ice_proxy),
                ("H2p25O", args("h225", 2.25), a.ice_proxy),
                ("pureH", args("H"), a.h_proxy)]
        if a.extra_h is not None:
            counts = a.extra_h
            oxygen = 4 * ice["nrep"]**3
            if len(counts) != 4 or len(set(counts)) != 4 or any(k <= 0 or k/oxygen > .25 for k in counts):
                raise ValueError("--extra-h needs four distinct positive integer counts within the H2.25O anchor")
            jobs = []
            for count in counts:
                hp = 2 + count/oxygen
                cmdargs = args("ice", hp)
                cmdargs[cmdargs.index("--vpa")+1] = str((3*ice["vpa"] + (hp-2)/.25 *
                    (3.25*rich["vpa"]-3*ice["vpa"]))/(hp+1))
                jobs.append((f"Hplus{count}", cmdargs, a.ice_proxy))
    reference = None
    if a.reference_run:
        refdir = Path(a.reference_run).resolve()
        refmanifest = json.loads((refdir/"manifest.json").read_text())
        if refmanifest.get("status") != "complete":
            raise ValueError("Shared reference run must be complete")
        refs = [j for j in refmanifest["jobs"] if j["name"] == "pureH"]
        if len(refs) != 1:
            raise ValueError("Reference run must contain one pureH result")
        resultpath = Path(refs[0]["result"]).resolve()
        ref = json.loads(resultpath.read_text())
        from pwidom import model_fingerprint
        if (ref.get("T") != a.T or ref.get("P_target") != a.P or ref.get("phase") != "H"
                or ref.get("model_fp") != model_fingerprint(model)):
            raise ValueError("Reference P/T/phase/model fingerprint do not match")
        frames = refdir / f"{ref['tag']}_frames.npz"
        if not frames.exists() or scan.min_hh(str(frames)) < .25:
            raise ValueError("Reference frame archive missing or pure H collapsed")
        reference = dict(run=str(refdir), result=str(resultpath), frames=str(frames),
                         snapshot=ref)
    # All validation and command construction precedes output creation or child startup.
    out = Path(a.out or time.strftime("runs/pwidom_pm_%Y%m%d_%H%M%S"))
    if out.exists():
        raise ValueError(f"Output already exists: {out}; choose a fresh --out (no overwrite/resume)")
    records = []
    from pwidom import build_host
    for gpu, (name, args, proxy) in zip(gpus, jobs):
        options = dict(zip(args[::2], args[1::2]))
        host = build_host(argparse.Namespace(
            phase=options["--phase"], natoms=int(options.get("--natoms", 0)),
            nrep=int(options.get("--nrep", 3)), vpa=float(options["--vpa"]),
            hper=float(options.get("--hper", 2)), seed=3))
        tag = f"pw_{name}_{a.P:g}GPa_{a.T:g}K"
        args = [x for key, value in zip(args[::2], args[1::2])
                if key != "--model" for x in (key, value)]
        cmd = [sys.executable, "-u", "mlp_pipeline/tdi/pwidom.py", "run", *args,
               "--model", model, "--device", "cpu" if a.cpu else "cuda",
               "--ps", str(a.ps), "--equil-ps", str(a.equil_ps),
               "--K", str(a.K), "--sel", str(a.sel), "--ndel", str(a.ndel),
               "--batch", str(a.batch), "--proxy", proxy, "--out", str(out),
               "--tag", tag, "--label", "candidate"]
        if a.candidate_dir:
            cmd += ["--candidate-dir", a.candidate_dir, "--candidate-limit", str(a.candidate_limit)]
        if parent is not None:
            oldcmd = byname[name]["command"]
            oldout = oldcmd[oldcmd.index("--out")+1]
            oldtag = oldcmd[oldcmd.index("--tag")+1]
            source = Path(oldout) / f"{oldtag}_frames.npz"
            from pwidom import model_fingerprint
            import numpy as np
            with np.load(source) as z:
                if not all(k in z.files for k in ("host", "full", "P_host", "P_full", "model_fp", "numbers", "T", "V")):
                    raise ValueError(f"Incomplete extension archive: {source}")
                if str(z["model_fp"]) != model_fingerprint(model):
                    raise ValueError(f"Model fingerprint differs from extension source: {source}")
                if (float(z["T"]) != a.T or not np.array_equal(z["numbers"], host.numbers)
                        or abs(float(z["V"]) - host.get_volume()) > 1e-6 * host.get_volume()):
                    raise ValueError(f"T/species/volume differs from extension source: {source}")
            cmd += ["--extend-from", str(source.resolve())]
        records.append(dict(name=name, gpu=gpu, atoms=len(host), command=cmd,
                            log=str(out / f"{name}.log"),
                            progress=str(out / f"{name}_progress.json"),
                            result=str(out / f"{tag}_candidate_K{a.K}.json")))
    return out, dict(model=model, stage0=state, eos_inputs=explicit,
                     state_json=a.state_json, extends=a.extend_run, reference=reference,
                     extra_h=a.extra_h, P=a.P, T=a.T, ps=a.ps,
                     equil_ps=a.equil_ps, minutes=a.minutes, jobs=records,
                     allocation=os.environ.get("SLURM_JOB_ID"),
                     git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())


def run(a):
    if a.inspect_eos:
        inspect_eos(a)
        return
    if a.save_eos or a.include_h25:
        raise ValueError("--save-eos/--include-h25 require --inspect-eos CSV")
    out, manifest = plan(a)
    print(f"Four-host scan: {a.P:g} GPa / {a.T:g} K; {a.ps:g} ps + {a.equil_ps:g} ps equil per MD; wall cap {a.minutes:g} min")
    print(f"Model: {manifest['model']}\nOutput: {out}")
    print("Candidate proxies; short-run errors are statistical only.")
    if a.T == 3000:
        print("3000 K is outside the supplied fit-label temperatures: this is an extrapolation check.")
    print("Intermediate volume uses the existing per-O interpolation; no refitting or scheduler submission.")
    if manifest.get("reference"):
        print(f"Reusing completed pure-H reference: {manifest['reference']['result']}; shared uncertainty across compositions.")
    if a.extend_run:
        print(f"EXTENDING {a.extend_run}: add {a.ps:g} ps per ensemble, re-equilibrate {a.equil_ps:g} ps with fresh thermal velocities; retain old frames. Recompute insertion estimates on combined frames.")
    for r in manifest["jobs"]:
        print(f"{r['name']}: GPU {r['gpu'] or 'CPU'}, {r['atoms']} host atoms")
        if a.dry_run or a.verbose:
            print("  " + shlex.join(r["command"]))
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    manifest["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["status"] = "running"
    save(out / "manifest.json", manifest)
    write_origin(out, toy=a.cpu, parent=a.extend_run)
    children = []
    stopped = None
    logs = HostLogs(manifest["jobs"])

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = f"signal {signum}"

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    started = time.monotonic()
    try:
        for r in manifest["jobs"]:
            if stopped:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=r["gpu"],
                       TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       PW_PROGRESS_FILE=os.path.abspath(r["progress"]))
            # Each GPU is independent; do not inherit another launcher's serialization lock.
            env.pop("PW_GPU_LOCK", None)
            with open(r["log"], "x") as log:
                log.write(shlex.join(r["command"]) + "\n"); log.flush()
                p = subprocess.Popen(r["command"], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            r["pid"] = p.pid
            children.append((r, p))
            save(out / "manifest.json", manifest)
        last = -30.0
        while any(p.poll() is None for _, p in children):
            elapsed = time.monotonic() - started
            if a.verbose and not a.quiet:
                logs.drain()
            if stopped or elapsed >= a.minutes * 60:
                stopped = stopped or "wall-time cap"
                break
            if elapsed - last >= a.interval:
                dashboard(manifest, elapsed, {r['name']:p.poll() for r,p in children}, plain=a.plain, clear=not a.verbose)
                last = elapsed
            time.sleep(0.5)
    finally:
        for _, p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while any(p.poll() is None for _, p in children) and time.monotonic() < deadline:
            time.sleep(0.1)
        for r, p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            r["exit_code"] = p.wait()
        complete = len(children) == 4 and all(
            p.returncode == 0 and Path(r["result"]).is_file() for r, p in children)
        manifest["status"] = "complete" if complete else (stopped or "failed")
        save(out / "manifest.json", manifest)
        if a.verbose and not a.quiet:
            logs.drain(final=True)
        logs.close()
    print(f"{manifest['status']}: {out}; unfinished MD is not checkpointed")
    status(argparse.Namespace(out=str(out), gpu=False))
    if not complete:
        raise SystemExit(124 if stopped == "wall-time cap" else 1)


def inspect_eos(a):
    """Read-only EOS report, requiring the exact temperature bin (no T extrapolation)."""
    from importlib import import_module
    import numpy as np
    df = import_module("01_bin_inventory").load_inventory(a.inspect_eos)
    eos = import_module("17_eos_md")
    pressures = [a.P * .97, a.P, a.P * 1.03]
    inputs = dict(P=a.P, T=a.T)
    compositions = [("ice", "h2o"), ("h225", "h2.25o"), ("H", "h")]
    if a.include_h25:
        compositions.append(("h25", "h2.5o"))
    for key, comp in compositions:
        d = df[(df.comp_snap == comp) & (df.T_bin == a.T)]
        if key != "H":
            d = d[d.Path.str.contains("fcc", case=False) &
                  ~d.Path.str.contains("liquid|solid|melt", case=False)]
        print(f"{key}: {len(d)} runs in exact {a.T:g} K bin after existing EOS phase filters")
        if len(d) < (5 if key == "H" else 20):
            print("  Insufficient exact-bin coverage; no volume/B recommended.")
            continue
        if key == "H":
            v = [eos.h_vpa(a.inspect_eos, a.T, p) for p in pressures]
        else:
            v = eos.dft_vpa(a.inspect_eos, "fcc", a.T, pressures,
                            hper={"ice":2,"h225":2.25,"h25":2.5}[key], max_extrap=0)
        if not np.isfinite(v).all() or v[2] >= v[0]:
            print("  EOS does not cover the slope interval; no volume/B recommended.")
            continue
        bulk = -.5 * (v[0] + v[2]) * (pressures[2] - pressures[0]) / (v[2] - v[0])
        print(f"  {a.P:g} GPa: vpa={v[1]:.8g} A^3/atom; B_GPa={bulk:.8g} (same EOS finite-difference convention)")
        inputs[key] = dict(vpa=float(v[1]), B_GPa=float(bulk),
            source=f"{Path(a.inspect_eos).resolve()}; {len(d)} exact-{a.T:g}K runs; existing ln P(V) fit and P +/-3% finite difference; generated {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            **({"natoms":250} if key == "H" else {"nrep":3}))
    if a.save_eos:
        if not all(k in inputs for k,_ in compositions):
            raise ValueError("Incomplete EOS coverage: no configuration saved")
        path = Path(a.save_eos)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as file:
            file.write(json.dumps(inputs, indent=2) + "\n")
        print(f"Saved EOS inputs to {path}; no MD or GPU calls. Crystal size nrep=3 (108 O), pure H 250 atoms.")
    else:
        print("Read-only report: no MD, GPU calls or output files. Review EOS coverage/phase before choosing run inputs.")


def status(a):
    out = Path(a.out)
    if getattr(a, "follow", False):
        manifest = json.loads((out / "manifest.json").read_text())
        logs = HostLogs(manifest["jobs"])
        print(f"Following {out}; Ctrl-C stops this viewer only.", flush=True)
        last = -float("inf")
        try:
            while True:
                logs.drain()
                if time.monotonic() - last >= 30:
                    for job in manifest["jobs"]:
                        print(f"[{job['name']}] {progress_text(job)}", flush=True)
                    last = time.monotonic()
                if json.loads((out / "manifest.json").read_text())["status"] != "running":
                    logs.drain(final=True)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            return
        finally:
            logs.close()
    m = json.loads((out / "manifest.json").read_text())
    print(f"{out}: {m['status']}; allocation {m['allocation']}; {m['P']:g} GPa / {m['T']:g} K")
    results = {}
    for r in m["jobs"]:
        print(f"\n{r['name']}: GPU {r['gpu'] or 'CPU'}, PID {r.get('pid', '-')}, exit {r.get('exit_code', '-')}")
        if m["status"] == "running":
            print("  " + progress_text(r))
        if Path(r["result"]).exists():
            w = results[r["name"]] = json.loads(Path(r["result"]).read_text())
            print(f"  Widom {w['mu_ex_widom_meV']:.1f}; deletion {w['mu_ex_delete_meV']:.1f}; BAR {w['mu_ex_bar_meV']:.1f} meV")
            print(f"  gap(ins/del) {w['proxy_gap_std_kT_ins']:.2f}/{w['proxy_gap_std_kT_del']:.2f} kT; Kish {w['kish_frac_ins']:.3f}; melted={w['melted']}")
        else:
            print("  Result missing; recent log:")
            log = Path(r["log"])
            if log.exists():
                print("\n".join(log.read_text(errors="replace").splitlines()[-4:]))
    if m.get("reference"):
        results["pureH"] = shared_reference(m)
        print(f"\nShared pure-H reference: {m['reference']['result']}")
    if "pureH" in results:
        h = results["pureH"]
        frames = Path(m["reference"]["frames"]) if m.get("reference") else out / f"{h['tag']}_frames.npz"
        hh = scan.min_hh(str(frames))
        print(f"\nPure-H minimum sampled H-H separation: {hh:.3f} A" + ("; COLLAPSE" if hh < 0.25 else ""))
        for name, w in results.items():
            if name == "pureH":
                continue
            value, err = scan.dg(w, h)
            measured, _ = scan.dg(w, h, "measured")
            flags = []
            if w["melted"] or scan.o_growing(w): flags.append("HOST MELTED")
            if hh < 0.25: flags.append("PURE-H COLLAPSE")
            if abs(w["mu_ex_delete_meV"] - w["mu_ex_widom_meV"]) > max(200, 3 * err): flags.append("UNCONVERGED")
            if w["kish_frac_ins"] < 0.05: flags.append("LOW KISH")
            print(f"{name}: dg {value:.1f} +- {err:.1f} meV (scan robust P correction); measured-P dg {measured:.1f}; {', '.join(flags) or 'no scan flags (not proof of convergence)'}")
    if a.gpu:
        subprocess.run(["nvidia-smi"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--P", type=float, default=500)
    r.add_argument("--T", type=float, default=3000)
    r.add_argument("--ps", type=float, default=0.4)
    r.add_argument("--equil-ps", type=float, default=0.2)
    r.add_argument("--minutes", type=float, default=10)
    r.add_argument("--K", type=int, default=2000)
    r.add_argument("--sel", type=int, default=8)
    r.add_argument("--ndel", type=int, default=4)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--roots", nargs="+")
    r.add_argument("--state-json", help="explicit EOS inputs instead of prior stage0 files; see README")
    r.add_argument("--extend-run", help="completed run directory: preserve inputs and append a re-equilibrated sampling segment in new output")
    r.add_argument("--extra-h", nargs=4, type=int, help="four extra-H counts in the ice cell; requires explicit EOS and completed shared H reference")
    r.add_argument("--reference-run", help="completed four-host scan whose pureH result is reused with --extra-h")
    r.add_argument("--inspect-eos", metavar="CSV", help="read-only exact-temperature-bin DFT EOS report; no GPU/run")
    r.add_argument("--save-eos", metavar="JSON", help="with --inspect-eos: save all three EOS inputs to a fresh file; no MD")
    r.add_argument("--include-h25",action="store_true",help="with --inspect-eos: also require/report H2.5O coverage for scans beyond 27 extra H")
    r.add_argument("--out")
    r.add_argument("--model", default="grace-ft:runs/grace/ft_m2")
    r.add_argument("--ice-proxy", default="runs/pwidom_fit/proxy_ice.json")
    r.add_argument("--h-proxy", default="runs/pwidom_fit/proxy_H.json")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--cpu", action="store_true", help="local toy checks only")
    r.add_argument("--quiet", action="store_true", help="show summaries only; full host output remains in logs")
    r.add_argument("--verbose", action="store_true", help="also stream every host log line; default is compact dashboard")
    r.add_argument("--interval", type=float, default=30, help="dashboard refresh seconds")
    r.add_argument("--plain", action="store_true", help="ASCII dashboard without color/screen clearing")
    r.add_argument("--candidate-dir", help="optional shared unlabeled extxyz queue for later DFT review")
    r.add_argument("--candidate-limit", type=int, default=10, help="maximum diagnostic pairs per host")
    s = sub.add_parser("status")
    s.add_argument("out", help="exact output directory printed by the launcher")
    s.add_argument("--gpu", action="store_true", help="also run nvidia-smi on this machine")
    s.add_argument("--follow", action="store_true", help="stream full host logs until completion; Ctrl-C stops viewer only")
    s.add_argument("--interval", type=float, default=30, help="dashboard refresh seconds")
    s.add_argument("--plain", action="store_true", help="ASCII dashboard without color/screen clearing")
    a = ap.parse_args()
    if a.interval <= 0:
        ap.error("--interval must be positive")
    try:
        (run if a.action == "run" else status)(a)
    except (ValueError, OSError, KeyError) as ex:
        raise SystemExit(str(ex))


if __name__ == "__main__":
    main()
