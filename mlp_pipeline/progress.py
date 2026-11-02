"""Progress of all running snapshot-generation jobs (A grid, batch1, B+/E/H command file, G, F). Read-only; run anywhere that sees the files (login node is fine).
  python3 mlp_pipeline/progress.py [--leg ~/leg] [--cmdfile rest_gpu.sh] [--stale 20]
Job state from the logs: done = log has "wrote N frames", RUN = log touched in the last --stale minutes, dead = older log without "wrote" (restart the launcher), todo = no log.
% for insertion jobs = host/n steps seen in the log / expected (2 hosts x number of n values; approximate), frames = Lattice= lines in the output file."""
import argparse, glob, os, re, time

ap = argparse.ArgumentParser(); ap.add_argument("--leg", default=os.path.expanduser("~/leg")); ap.add_argument("--cmdfile", nargs="*", default=None, help="command files to report (default: every one of rest_gpu.sh / e2_gpu.sh / h_gpu.sh that exists)"); ap.add_argument("--stale", type=float, default=20); ap.add_argument("--hosts", type=int, default=2)
a = ap.parse_args(); now = time.time()


def txt(p):
    try: return open(p, errors="ignore").read()
    except OSError: return ""


def nframes(p):
    try: return sum(1 for l in open(p, errors="ignore") if "Lattice=" in l)
    except OSError: return 0


def state(log, done_re=r"^wrote \d+ frames to", extra_done=None):
    if not os.path.exists(log): return "todo"
    t = txt(log)
    if re.search(done_re, t, re.M) or (extra_done and extra_done in t): return "done"
    return "RUN" if now - os.path.getmtime(log) < a.stale * 60 else "dead"


def bar(frac, w=20): n = int(round(frac * w)); return "#" * n + "." * (w - n)


def insertion(name, d, expect=None):
    logs = sorted(glob.glob(os.path.join(d, "ins_*.log")))
    if not logs: print(f"{name}: no logs in {d}"); return 0, 0
    allj = expect or len(logs); tot = 0.0; cnt = {}; fr = 0; rows = []
    for lg in logs:
        s = state(lg); t = txt(lg); steps = len(re.findall(r"^\s+host \d+ n=", t, re.M)); nl = 11 if "5000K_500GPa" in lg else 6
        zero = s == "done" and re.search(r"^wrote 0 frames", t, re.M) is not None
        frac = 1.0 if s == "done" else min(steps / (a.hosts * nl), 0.99) if s == "RUN" else 0.0
        if zero: s = "done(0)"
        cnt[s] = cnt.get(s, 0) + 1; tot += frac; f = nframes(lg[:-4] + ".extxyz"); fr += f; rows.append((s, frac, f, os.path.basename(lg)[:-4]))
    print(f"\n{name}: {len(logs)} of {allj} jobs have a log;  {'  '.join(f'{k} {v}' for k, v in sorted(cnt.items()))};  {fr} frames;  ~{100 * tot / allj:.0f}% (hosts-less 0-frame jobs count as finished)")
    for s, frac, f, n in rows:
        if s in ("RUN", "dead"): print(f"   {s:5s} {bar(frac)} {frac * 100:3.0f}%  {f:4d} fr  {n}")
    return tot, allj


insertion("A  insertion grid (ins_grid)", os.path.join(a.leg, "ins_grid"), expect=40)
insertion("   batch1 (older 5-state run)", os.path.join(a.leg, "ins_batch1"))
cmdfiles = a.cmdfile or [f for f in ("rest_gpu.sh", "e2_gpu.sh", "h_gpu.sh") if os.path.exists(f)]
kinds = {"B+ chains": "--chain", "E low-P MD": "17_eos_md", "H Huang supercells": "build_h3o_supercells"}; running = 0; unfinished = 0
for cf in cmdfiles:
    lines = [l for l in open(cf).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    print(f"\n{cf} ({len(lines)} commands; logs in {cf}.logs/):")
    for name, key in kinds.items():
        idx = [i for i, l in enumerate(lines) if key in l]; cnt = {}
        if not idx: continue
        for i in idx:
            lg = f"{cf}.logs/{i}.log"; s = state(lg, done_re=r"^DONE$")
            if s == "done" and "Traceback" in txt(lg): s = "FAILED"            # older run_lines_gpus.sh wrote DONE even after a crash
            if re.search(r"^FAILED rc=", txt(lg), re.M): s = "FAILED"
            cnt[s] = cnt.get(s, 0) + 1
        nd = cnt.get("done", 0); running += cnt.get("RUN", 0); unfinished += len(idx) - nd
        print(f"   {name:20s} {bar(nd / max(len(idx), 1))} {100 * nd / max(len(idx), 1):3.0f}%  {nd}/{len(idx)} done  " + "  ".join(f"{k} {v}" for k, v in sorted(cnt.items()) if k != "done"))
if not cmdfiles: print("\n(no command files here: run from the repo root, or pass --cmdfile)")
print("\nframes on disk:  B+ chains", sum(nframes(f) for f in glob.glob(os.path.join(a.leg, "chain/*.extxyz"))), "  H supercells", sum(nframes(f) for f in glob.glob(os.path.join(a.leg, "h3o_sc/*.extxyz"))),
      "  E pass-1 trajectories", len(glob.glob("runs/eos_lowp/*/traj.extxyz")), "  E pass-2 trajectories", len(glob.glob("runs/eos_lowp2/*/traj.extxyz")), "  C2 widom", nframes(os.path.join(a.leg, "widom_c2.extxyz")))
print("G  pure-H 729 MD:", nframes(os.path.join(a.leg, "H729_md.extxyz")) + nframes(os.path.join(a.leg, "H729_md_s2.extxyz")), "frames    F toy cells:", sum(nframes(f) for f in glob.glob(os.path.join(a.leg, "toy/*.extxyz"))), "frames")

import subprocess
try: nproc = int(subprocess.run("pgrep -fc 'build_insertion_[s]et|17_eos_[m]d|build_h3o_[s]upercells|run_lines_[g]pus|finish_[g]eneration'", shell=True, capture_output=True, text=True).stdout.strip() or 0)
except Exception: nproc = -1
print(f"\nthis node: {nproc} generation processes running (other jobs/nodes are not visible from here).")
print("VERDICT:", "all generation finished: run the assemble step" if running == 0 and unfinished == 0 and nproc <= 0 else f"not finished: {running} command lines marked RUN, {unfinished} not yet done, {nproc} processes here")
