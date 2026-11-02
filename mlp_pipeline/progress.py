"""Progress of all running snapshot-generation jobs (A grid, batch1, B+/E/H command file, G, F). Read-only; run anywhere that sees the files (login node is fine).
  python3 mlp_pipeline/progress.py [--leg ~/leg] [--cmdfile rest_gpu.sh] [--stale 20]
Job state from the logs: done = log has "wrote N frames", RUN = log touched in the last --stale minutes, dead = older log without "wrote" (restart the launcher), todo = no log.
% for insertion jobs = host/n steps seen in the log / expected (2 hosts x number of n values; approximate), frames = Lattice= lines in the output file."""
import argparse, glob, os, re, time

ap = argparse.ArgumentParser(); ap.add_argument("--leg", default=os.path.expanduser("~/leg")); ap.add_argument("--cmdfile", default="rest_gpu.sh"); ap.add_argument("--stale", type=float, default=20); ap.add_argument("--hosts", type=int, default=2)
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
if os.path.exists(a.cmdfile):
    lines = [l for l in open(a.cmdfile).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    kinds = {"B+ chains": "--chain", "E low-P MD": "17_eos_md", "H Huang supercells": "build_h3o_supercells"}
    print(f"\nrest sets ({a.cmdfile}, {len(lines)} commands; logs in {a.cmdfile}.logs/):")
    for name, key in kinds.items():
        idx = [i for i, l in enumerate(lines) if key in l]; cnt = {}
        for i in idx:
            s = state(f"{a.cmdfile}.logs/{i}.log", done_re=r"^DONE$"); cnt[s] = cnt.get(s, 0) + 1
        nd = cnt.get("done", 0); print(f"   {name:20s} {bar(nd / max(len(idx), 1))} {100 * nd / max(len(idx), 1):3.0f}%  {nd}/{len(idx)} done  " + "  ".join(f"{k} {v}" for k, v in sorted(cnt.items()) if k != "done"))
    for nm, pat in (("B+ chain frames", "chain/*.extxyz"), ("H supercell frames", "h3o_sc/*.extxyz")):
        print(f"   {nm}: {sum(nframes(f) for f in glob.glob(os.path.join(a.leg, pat)))}")
else: print(f"\n(no {a.cmdfile} here: run from the repo root, or pass --cmdfile)")
print("\nG  pure-H 729 MD:", nframes(os.path.join(a.leg, "H729_md.extxyz")), "frames    F toy cells:", sum(nframes(f) for f in glob.glob(os.path.join(a.leg, "toy/*.extxyz"))), "frames")
