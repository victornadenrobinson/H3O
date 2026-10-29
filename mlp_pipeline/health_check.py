"""Integrity check of everything in runs/ (login node, no GPU, read-only except the generated runs/finish_jobs.sh): what crashed, ran out of GPU memory or time,
and what is unfinished, with the command that fixes each.
  A. error scan of recent logs (Slurm *.out, run_on_gpu, pwidom, TDI node logs, last --days): out of memory, Python tracebacks, time limit / cancelled / killed,
     CUDA / cuDNN start-up failures. Informational: a crash that was later retried and finished shows up here AND as complete below; the sections B-E decide.
  B. TDI runs (runs/tdi*): tdi/finish_runs.py: lambda values missing or short, and runs/finish_jobs.sh with the exact commands that finish them.
  C. pwidom scans (runs/pwidom*): hosts that have MD frames but no result (rerun the same scan command: frames are reused).
  D. stability_npt (runs/stability_npt): blocks done vs requested, resumable (checkpoint) or not.
  E. h3o_static (runs/h3o_static): pressures with / without a result table.
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/health_check.py [--days 3] [--no-tdi]
"""
import argparse, glob, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = [("GPU OOM", re.compile(r"out of memory|ResourceExhausted|OOM when allocating|CUDA_ERROR_OUT_OF_MEMORY", re.I)),
            ("CUDA/cuDNN start", re.compile(r"cudaSetDevice|DNN library initialization failed|failed call to cuInit|CUDA_ERROR_(?!OUT)", re.I)),
            ("time limit", re.compile(r"DUE TO TIME LIMIT|TIME LIMIT")), ("cancelled/killed", re.compile(r"CANCELLED|Killed|oom-kill|Out Of Memory \(host\)", re.I)),
            ("traceback", re.compile(r"^Traceback \(most recent call last\)")), ("python error", re.compile(r"^\w+(Error|Exception): ")),
            ("refused", re.compile(r"refusing|refused|cannot be continued|not overwriting", re.I))]
NOISE = re.compile(r"_handle_getgr|I0000|W0000|cuda_executor\.cc.*(numa|NUMA)")


def scan_logs(days):
    t0 = time.time() - days * 86400
    files = sorted(set(sum((glob.glob(p) for p in ("runs/slurm_*.out", "runs/ongpu/*.log", "runs/pwidom*/*.log", "runs/tdi*/*.log", "runs/stability_npt/*/*.log",
                                                   "runs/h3o_static*.log", "runs/*.log")), [])))
    files = [f for f in files if os.path.getmtime(f) >= t0]
    print(f"== A. error scan: {len(files)} logs modified in the last {days:g} days")
    hits = []
    for f in files:
        found = {}
        try:
            for line in open(f, errors="replace"):
                if NOISE.search(line): continue
                for name, rx in PATTERNS:
                    if rx.search(line): found[name] = (found.get(name, (0, ""))[0] + 1, line.strip()[:160])
        except OSError:
            continue
        if found: hits.append((f, found))
    if not hits: print("   no error lines"); return []
    for f, found in hits:
        print(f"   {f}  ({time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f)))})")
        for name, (n, last) in found.items():
            print(f"      {name:16s} x{n:<3d} last: {last}")
    print("   (a retried / restarted run that finished later still shows its old errors here: check B-E for what is actually incomplete)")
    return hits


def stability():
    ds = sorted(glob.glob("runs/stability_npt/*/stability.json"))
    print(f"\n== D. stability_npt: {len(ds)} runs")
    todo = []
    for f in ds:
        try: j = json.load(open(f))
        except Exception as e: print(f"   {f}: unreadable ({e})"); continue
        rows = j.get("blocks", []); want = j.get("args", {}).get("ps", float("nan")); done = rows[-1]["t_ps"] if rows else 0.0
        ck = os.path.exists(os.path.join(os.path.dirname(f), "checkpoint.npz"))
        state = "complete" if done >= want - 1e-6 else ("INCOMPLETE, resumable (rerun the same command)" if ck else "INCOMPLETE, no checkpoint (older version: --report shows what exists)")
        print(f"   {j.get('tag', f):40s} {done:5.1f} / {want:g} ps   {state}")
        if done < want - 1e-6: todo.append(f)
    return todo


def h3o_static():
    d = "runs/h3o_static"
    if not os.path.isdir(d): return []
    ps = sorted(glob.glob(os.path.join(d, "P*")))
    print(f"\n== E. h3o_static: {len(ps)} pressure folders")
    todo = []
    for p in ps:
        ok = os.path.exists(os.path.join(p, "h3o_static.csv")); nx = len(glob.glob(os.path.join(p, "*_relaxed.extxyz")))
        print(f"   {p:28s} {'done' if ok else 'NO RESULT TABLE'}   {nx} relaxed structures saved" + ("" if ok else "  (rerun h3o_static for this P: finished relaxations are reused)"))
        if not ok: todo.append(p)
    return todo


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=float, default=3.0); ap.add_argument("--no-tdi", action="store_true"); a = ap.parse_args()
    if not os.path.isdir("mlp_pipeline"): sys.exit("run from the repo root")
    print(f"health check {time.strftime('%Y-%m-%d %H:%M')} in {os.getcwd()}\n")
    hits = scan_logs(a.days)
    tdi_dirs = sorted(d for d in glob.glob("runs/tdi*") if os.path.isdir(d) and not d.endswith("tdi_test"))
    if not a.no_tdi and tdi_dirs:
        print(f"\n== B. TDI runs ({', '.join(tdi_dirs)}): tdi/finish_runs.py (writes runs/finish_jobs.sh; nothing is submitted)", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "tdi", "finish_runs.py"), *tdi_dirs, "--skip-empty", "-o", "runs/finish_jobs.sh"])
    print("\n== C. pwidom scans", flush=True)
    if glob.glob("runs/pwidom*"):
        subprocess.run([sys.executable, os.path.join(HERE, "tdi", "pwidom_scan.py"), "--status", "runs/pwidom*"])
    else:
        print("   none")
    st = stability(); hs = h3o_static()
    print("\n== what to do")
    print("   B: look at the table above; finish TDI runs with:  bash runs/finish_jobs.sh   (or the lines in it inside an interactive GPU job)")
    print("   C: MISSING pwidom hosts: rerun the scan command that made that folder, with the same --out, --ps and --K")
    if st: print(f"   D: {len(st)} incomplete stability runs: rerun young_stability_npt.sbatch / run_on_gpu.sh with the same task (resumable ones continue)")
    if hs: print(f"   E: {len(hs)} h3o_static pressures without a table: rerun h3o_static for them (finished relaxations are reused)")
    if hits: print(f"   A: {len(hits)} logs with error lines: only act on those whose run is listed as incomplete in B-E")


if __name__ == "__main__":
    main()
