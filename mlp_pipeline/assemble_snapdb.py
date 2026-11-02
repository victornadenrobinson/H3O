"""Collect every generated snapshot set into one database, then verify and plot it (no DFT, no GPU; run on a login node).
  python mlp_pipeline/assemble_snapdb.py [--leg ~/leg] [--root .] [--out snapdb] [--dry]
Looks for the standard outputs, concatenates each set's files (reading with ASE, so a truncated frame is skipped, not copied), tags the low-P MD trajectories, and calls
snapdb.py build -> verify -> plot. Missing sets are reported, not fatal. Caps and priorities follow docs/DFT_PLAN.md (prio 1 = label first). Edit SETS to change them.
Set -> source:  A ins_grid/*  A_b1 ins_batch1/*  A_part partial/*  B chain/*  C sel_pwidom  D_melt sel_meltcomp  D_eos sel_eos  E runs/eos_lowp/*/traj.extxyz  F toy/{h2o,h2.25o,h2.5o,h3o}  F_H toy/H729  G H729_md  H h3o_sc/*
"""
import argparse, glob, hashlib, os, re, subprocess, sys, time
import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--leg", default=os.path.expanduser("~/leg")); ap.add_argument("--root", default="."); ap.add_argument("--out", default="snapdb"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--skip-db", nargs="*", default=[], help="earlier database dirs whose chunked frames are left out (passed to snapdb build)"); ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1), help="processes for the collapsed-contact check"); ap.add_argument("--e-per-traj", type=int, default=8, help="low-P MD trajectories: frames kept per trajectory file (they are 5000+ highly correlated frames, 180 are used)"); a = ap.parse_args()
L = lambda *p: os.path.join(a.leg, *p); R = lambda *p: os.path.join(a.root, *p)
# name: (file globs, cap (0 = all), prio)
# C 440 = C 200 + C3 240 (the end-state D statistics come from the same pwidom caches); C2 has no cap because host and site geometries of a frame must stay together;
# D_melt 144 = D + D2 (4000/6000 K fill)
SETS = {
    "B": ([L("chain", "*.extxyz")], 0, 1), "C": ([R("sel_pwidom.extxyz")], 440, 1), "C2": ([L("widom_c2.extxyz")], 0, 1),
    "A": ([L("ins_grid", "ins_*.extxyz")], 520, 2), "A_b1": ([L("ins_batch1", "ins_*.extxyz")], 150, 2), "A_part": ([L("partial", "ins_*.extxyz")], 100, 3),
    "E": ([R("runs", "eos_lowp", "*", "traj.extxyz"), R("runs", "eos_lowp2", "*", "traj.extxyz")], 180, 3), "H": ([L("h3o_sc", "*.extxyz")], 220, 3),
    "D_melt": ([R("sel_meltcomp.extxyz")], 144, 4), "D_eos": ([R("sel_eos.extxyz")], 100, 4), "G": ([L("H729_md.extxyz"), L("H729_md_s2.extxyz")], 30, 4),
    "F": ([L("toy", n + ".extxyz") for n in ("h2o", "h2.25o", "h2.5o", "h3o")], 220, 5), "F_H": ([L("toy", "H729.extxyz")], 32, 5)}


FLOOR = {"HH": 0.35, "OH": 0.50, "OO": 1.0}          # same floors as snapdb verify: DFT must not meet collapsed geometries


def bad_contact(f):
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", f, 2.2); z = f.numbers
    for name, (p_, q_) in {"HH": (1, 1), "OH": (8, 1), "OO": (8, 8)}.items():
        m = ((z[i] == p_) & (z[j] == q_)) | ((z[i] == q_) & (z[j] == p_))
        if m.any() and d[m].min() < FLOOR[name]: return name
    return None


def _contact_arrays(t):
    from ase import Atoms
    return bad_contact(Atoms(numbers=t[0], positions=t[1], cell=t[2], pbc=True))


def check_contacts(frames):
    """list of 'HH' / 'OH' / 'OO' / None per frame, computed in parallel (the neighbour lists dominate the run time on big sets)"""
    from concurrent.futures import ProcessPoolExecutor
    work = [(f.numbers, f.positions, np.asarray(f.cell)) for f in frames]
    if a.jobs <= 1 or len(work) < 50: return [_contact_arrays(t) for t in work]
    with ProcessPoolExecutor(a.jobs) as ex: return list(ex.map(_contact_arrays, work, chunksize=32))


def tag_lowp(f, path):
    d = os.path.basename(os.path.dirname(path)); T = re.search(r"(\d+)K", d); nO = int((f.numbers == 8).sum()); nH = int((f.numbers == 1).sum())
    f.info.update(kind="eos_md", run_id=d, source=path, split="valid" if int(hashlib.md5(d.encode()).hexdigest(), 16) % 3 == 0 else "train", nO=nO, nH=nH, T_K=float(T.group(1)) if T else np.nan,
                  vpa=float(f.get_volume() / len(f)), config_type=f"lowP_{nH}H{nO}O_{int(float(T.group(1))) if T else 0}K", method="md"); return f


def load_file(job):
    """read one source file (in a worker process); low-P trajectories: every k-th frame only, ASE skips the others without parsing them"""
    p, name = job
    try: fr = read(p, index="::6" if name == "E" else ":")
    except Exception as e: return p, None, str(e)[:60]
    if name == "E" and len(fr) > a.e_per_traj: fr = [fr[i] for i in np.linspace(0, len(fr) - 1, a.e_per_traj).round().astype(int)]
    out = []
    for f in fr:
        if name == "E": f = tag_lowp(f, p)
        f.calc = None; f.info["_src"] = os.path.basename(os.path.dirname(p)) if name == "E" else os.path.basename(p); out.append(f)
    return p, out, ""


work = a.out + "_src"; os.makedirs(work, exist_ok=True); specs = []; report = []
for name, (globs, cap, prio) in SETS.items():
    files = sorted({p for g in globs for p in glob.glob(g)}); frames = []; bad = 0; dropped = {}
    print(f"reading {name}: {len(files)} files ...", flush=True)
    t_read = time.time()
    if a.jobs > 1 and len(files) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.jobs, len(files))) as ex: res = list(ex.map(load_file, [(p, name) for p in files]))
    else: res = [load_file((p, name)) for p in files]
    for p, fr, err in res:
        if fr is None: bad += 1; print(f"  skipped unreadable {p}: {err}"); continue
        frames.extend(fr)
    print(f"  {name}: read {len(frames)} frames from {len(files)} files in {time.time() - t_read:.0f} s", flush=True)
    t0 = time.time(); why_all = check_contacts(frames); keep = []
    for f, why in zip(frames, why_all):
        src = f.info.pop("_src")
        if why: k = f"{why} {src}"; dropped[k] = dropped.get(k, 0) + 1
        else: keep.append(f)
    frames = keep; print(f"  {name}: {len(frames)} frames kept, contact check {time.time() - t0:.0f} s", flush=True)
    report.append((name, len(files), len(frames), bad, cap, prio))
    if dropped: print(f"  {name}: dropped {sum(dropped.values())} frames with collapsed contacts: " + "; ".join(f"{k} x{v}" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])[:6]), flush=True)
    if frames and not a.dry: out = os.path.join(work, name + ".extxyz"); write(out, frames); specs.append(f"{name}={out}:{cap or ''}:{prio}")
print(f"{'set':8s} {'files':>5s} {'frames':>7s} {'unreadable':>10s} {'cap':>5s} {'prio':>4s}")
for r in report: print(f"{r[0]:8s} {r[1]:5d} {r[2]:7d} {r[3]:10d} {r[4]:5d} {r[5]:4d}" + ("   MISSING" if r[2] == 0 else ""))
if a.dry or not specs: sys.exit(0)
snap = [sys.executable, os.path.join(HERE, "snapdb.py")]
for cmd in (["build", "--out", a.out, "--sets", *specs, "--seed", str(a.seed)] + (["--skip-db", *a.skip_db] if a.skip_db else []), ["verify", a.out], ["plot", a.out]):
    print("\n$ snapdb.py", cmd[0]); r = subprocess.run(snap + cmd)
    if r.returncode and cmd[0] != "verify": sys.exit(r.returncode)
