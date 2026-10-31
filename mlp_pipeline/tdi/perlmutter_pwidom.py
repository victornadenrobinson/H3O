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


def plan(a):
    parent = None
    if a.extend_run:
        parent = json.loads((Path(a.extend_run) / "manifest.json").read_text())
        if parent["status"] != "complete":
            raise ValueError("Extend only a completed run; check the original host processes first")
        if a.state_json or a.roots:
            raise ValueError("--extend-run inherits its state; omit --state-json/--roots")
        byname = {j["name"]: j for j in parent["jobs"]}
        for name in ("H2O", "H2p083O", "H2p25O", "pureH"):
            if name not in byname:
                raise ValueError(f"Parent run missing {name}")
        a.P, a.T, a.model = parent["P"], parent["T"], parent["model"]
        def option(name, key):
            cmd = byname[name]["command"]
            return cmd[cmd.index(key)+1]
        a.ice_proxy, a.h_proxy = option("H2O", "--proxy"), option("pureH", "--proxy")
        for key in ("K", "sel", "ndel", "batch"):
            setattr(a, key, int(option("H2O", "--"+key)))
        explicit = dict(P=a.P, T=a.T)
        for key, name in (("ice", "H2O"), ("h225", "H2p25O"), ("H", "pureH")):
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
                     state_json=a.state_json, extends=a.extend_run, P=a.P, T=a.T, ps=a.ps,
                     equil_ps=a.equil_ps, minutes=a.minutes, jobs=records,
                     allocation=os.environ.get("SLURM_JOB_ID"),
                     git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())


def run(a):
    if a.inspect_eos:
        inspect_eos(a)
        return
    if a.save_eos:
        raise ValueError("--save-eos requires --inspect-eos CSV")
    out, manifest = plan(a)
    print(f"Four-host scan: {a.P:g} GPa / {a.T:g} K; {a.ps:g} ps + {a.equil_ps:g} ps equil per MD; wall cap {a.minutes:g} min")
    print(f"Model: {manifest['model']}\nOutput: {out}")
    print("Candidate proxies; short-run errors are statistical only.")
    if a.T == 3000:
        print("3000 K is outside the supplied fit-label temperatures: this is an extrapolation check.")
    print("Intermediate volume uses the existing per-O interpolation; no refitting or scheduler submission.")
    if a.extend_run:
        print(f"EXTENDING {a.extend_run}: add {a.ps:g} ps per ensemble, re-equilibrate {a.equil_ps:g} ps with fresh thermal velocities; retain old frames. Recompute insertion estimates on combined frames.")
    for r in manifest["jobs"]:
        print(f"{r['name']}: GPU {r['gpu'] or 'CPU'}, {r['atoms']} host atoms\n  {shlex.join(r['command'])}")
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    manifest["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["status"] = "running"
    save(out / "manifest.json", manifest)
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
            if not a.quiet:
                logs.drain()
            if stopped or elapsed >= a.minutes * 60:
                stopped = stopped or "wall-time cap"
                break
            if elapsed - last >= 30:
                print(f"elapsed {duration(elapsed)} / cap {duration(a.minutes * 60)}; cap remaining {duration(a.minutes * 60 - elapsed)} | " + " | ".join(
                    f"{r['name']}: " + ("running" if p.poll() is None else f"exit {p.returncode}") for r, p in children))
                metrics = gpu_metrics(manifest["jobs"])
                for r, p in children:
                    print(f"  {r['name']} | {metrics.get(r['gpu'], 'GPU telemetry unavailable' if r['gpu'] else 'CPU')} | "
                          + (progress_text(r) if p.poll() is None else f"exit {p.returncode}"), flush=True)
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
        if not a.quiet:
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
    for key, comp in (("ice", "h2o"), ("h225", "h2.25o"), ("H", "h")):
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
                            hper=2 if key == "ice" else 2.25, max_extrap=0)
        if not np.isfinite(v).all() or v[2] >= v[0]:
            print("  EOS does not cover the slope interval; no volume/B recommended.")
            continue
        bulk = -.5 * (v[0] + v[2]) * (pressures[2] - pressures[0]) / (v[2] - v[0])
        print(f"  {a.P:g} GPa: vpa={v[1]:.8g} A^3/atom; B_GPa={bulk:.8g} (same EOS finite-difference convention)")
        inputs[key] = dict(vpa=float(v[1]), B_GPa=float(bulk),
            source=f"{Path(a.inspect_eos).resolve()}; {len(d)} exact-{a.T:g}K runs; existing ln P(V) fit and P +/-3% finite difference; generated {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            **({"natoms":250} if key == "H" else {"nrep":3}))
    if a.save_eos:
        if not all(k in inputs for k in ("ice", "h225", "H")):
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
    if "pureH" in results:
        h = results["pureH"]
        frames = out / f"{h['tag']}_frames.npz"
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
    r.add_argument("--inspect-eos", metavar="CSV", help="read-only exact-temperature-bin DFT EOS report; no GPU/run")
    r.add_argument("--save-eos", metavar="JSON", help="with --inspect-eos: save all three EOS inputs to a fresh file; no MD")
    r.add_argument("--out")
    r.add_argument("--model", default="grace-ft:runs/grace/ft_m2")
    r.add_argument("--ice-proxy", default="runs/pwidom_fit/proxy_ice.json")
    r.add_argument("--h-proxy", default="runs/pwidom_fit/proxy_H.json")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--cpu", action="store_true", help="local toy checks only")
    r.add_argument("--quiet", action="store_true", help="show summaries only; full host output remains in logs")
    r.add_argument("--candidate-dir", help="optional shared unlabeled extxyz queue for later DFT review")
    r.add_argument("--candidate-limit", type=int, default=10, help="maximum diagnostic pairs per host")
    s = sub.add_parser("status")
    s.add_argument("out", help="exact output directory printed by the launcher")
    s.add_argument("--gpu", action="store_true", help="also run nvidia-smi on this machine")
    s.add_argument("--follow", action="store_true", help="stream full host logs until completion; Ctrl-C stops viewer only")
    a = ap.parse_args()
    try:
        (run if a.action == "run" else status)(a)
    except (ValueError, OSError, KeyError) as ex:
        raise SystemExit(str(ex))


if __name__ == "__main__":
    main()
