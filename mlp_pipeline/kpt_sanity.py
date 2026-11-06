#!/usr/bin/env python3
"""READ-ONLY sanity check of the k-point relabelling folders before / while they run (login node, repo root, plain python3 is enough).
  python3 mlp_pipeline/kpt_sanity.py runs/kpt222mp runs/kpt222mp2 --ref runs/kpt222/frames.extxyz [--old runs/kpt222]
Checks, per folder: every point's KPOINTS (one mesh per folder, which one), INCAR (KPAR absent or 1, NCORE x KPAR divides 32, ISYM, ISMEAR, SIGMA = kB * T_K of its frame),
points prepared / finished (DONE.json), job files and the folder they run; across folders: every reference frame (kpt_src id) labelled exactly once, no duplicates, nothing missing;
the Slurm queue: which dgchk jobs run which folder (from each job's script path), so jobs of an old folder show up. --old: finished points of the old Gamma-centred run (kept for comparison)."""
import argparse, collections, csv, glob, json, os, re, subprocess, sys

KB = 8.617333e-5
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ref", default="runs/kpt222/frames.extxyz"); ap.add_argument("--old", default="runs/kpt222"); ap.add_argument("--expect-mesh", default="", help="e.g. Monkhorst: fail if a folder uses another mesh type"); a = ap.parse_args()
bad = []


def incar(path):
    d = {}
    for l in open(path):
        l = re.split(r"[#!]", l)[0]
        for part in l.split(";"):
            if "=" in part: k, v = part.split("=", 1); d[k.strip().upper()] = v.strip()
    return d


def outcar_timing(pd_):
    """(elapsed s, natoms, nkpts, cores) of a finished point: OUTCAR via DONE.json's attempt (worker layout) or directly; None if missing"""
    oc = os.path.join(pd_, "OUTCAR"); dj = os.path.join(pd_, "DONE.json")
    if os.path.exists(dj):
        att = json.load(open(dj)).get("attempt")
        if att: oc = os.path.join(pd_, att, "OUTCAR")
    if not os.path.exists(oc): return None
    with open(oc, "rb") as fh:
        head = fh.read(400000).decode("latin-1"); fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 20000)); tail = fh.read().decode("latin-1")
    el = re.search(r"Elapsed time \(sec\):\s*([\d.]+)", tail); na = re.search(r"NIONS =\s*(\d+)", head); nk = re.search(r"NKPTS =\s*(\d+)", head)
    co = re.search(r"running\s+on\s+(\d+)\s+total cores", head) or re.search(r"running\s+(\d+)\s+mpi-ranks", head)
    if not el: return None
    return float(el.group(1)), int(na.group(1)) if na else 0, int(nk.group(1)) if nk else 0, int(co.group(1)) if co else 0


def ref_ids(path):
    ids = []
    for l in open(path):
        m = re.search(r"kpt_src=(\S+)", l)
        if m: ids.append(m.group(1).strip('"'))
    return ids


ref = ref_ids(a.ref) if os.path.exists(a.ref) else []
print(f"reference selection {a.ref}: {len(ref)} frames ({len(set(ref))} distinct kpt_src ids)")
seen = collections.defaultdict(list)
for D in a.dirs:
    pts = sorted(glob.glob(os.path.join(D, "points_*")))
    meshes, kpars, ncores, isyms, sig_bad, n, ndone = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), 0, 0, 0
    times = []
    for p in pts:
        meta = {r["id"]: r for r in csv.DictReader(open(os.path.join(p, "frames_meta.csv")))} if os.path.exists(os.path.join(p, "frames_meta.csv")) else {}
        for pd_ in sorted(glob.glob(os.path.join(p, "p[0-9]*"))):
            if not os.path.isdir(pd_): continue
            n += 1; r = meta.get(os.path.basename(pd_), {})
            kp = open(os.path.join(pd_, "KPOINTS")).read().splitlines(); meshes[f"{kp[2].strip()} {' '.join(kp[3].split())}"] += 1
            inc = incar(os.path.join(pd_, "INCAR")); kpar = int(inc.get("KPAR", "1")); nc = int(inc.get("NCORE", "1")); kpars[kpar] += 1; ncores[nc] += 1; isyms[inc.get("ISYM", "?")] += 1
            if 32 % (kpar * nc): bad.append(f"{pd_}: NCORE {nc} x KPAR {kpar} does not divide 32 ranks")
            T = float(r.get("T_K", "nan")) if r.get("T_K") not in (None, "") else float("nan"); s = float(inc.get("SIGMA", "nan"))
            if T == T and abs(s - KB * max(T, 1000.0)) > 1e-4: sig_bad += 1
            if os.path.exists(os.path.join(pd_, "DONE.json")):
                ndone += 1; tm = outcar_timing(pd_)
                if tm: times.append(tm)
            if r.get("kpt_src"): seen[r["kpt_src"]].append(D)
    jobs = sorted(glob.glob(os.path.join(D, "job_*.sbatch")))
    print(f"\n{D}: {len(pts)} point folders, {n} points, {ndone} finished (DONE.json), {len(jobs)} job files")
    print(f"  KPOINTS : " + ", ".join(f"{k} x{v}" for k, v in meshes.items()))
    print(f"  INCAR   : KPAR {dict(kpars)}, NCORE {dict(ncores)}, ISYM {dict(isyms)}; SIGMA != kB*max(T_K,1000 K): {sig_bad}")
    if a.expect_mesh and any(a.expect_mesh.lower() not in k.lower() for k in meshes): bad.append(f"{D}: k-mesh {dict(meshes)} is not the expected '{a.expect_mesh}'")
    if times:
        bins = collections.defaultdict(list)
        for el, na, nk, co in times: bins["<= 260" if na <= 260 else "261-450" if na <= 450 else "451-600" if na <= 600 else "> 600"].append((el, nk, co))
        tot = sum(t[0] * (t[3] or 32) for t in times) / 3600 / 128
        print("  timing  : " + "; ".join(f"{k} atoms: {len(v)} pts, mean {sum(x[0] for x in v) / len(v) / 60:.1f} min, median {sorted(x[0] for x in v)[len(v) // 2] / 60:.1f} min ({v[0][1]} k, {v[0][2]} ranks)"
                                         for k, v in sorted(bins.items())) + f"; all {len(times)} pts: mean {sum(t[0] for t in times) / len(times) / 60:.1f} min/point, total {tot:.1f} CPU node-h")
    if len(meshes) > 1: bad.append(f"{D}: more than one k-mesh in one folder: {dict(meshes)}")
    if any(k > 1 for k in kpars): bad.append(f"{D}: KPAR > 1 present ({dict(kpars)}); the plan was KPAR = 1")
    if sig_bad: bad.append(f"{D}: {sig_bad} points with SIGMA not matching their frame's T_K")
    for j in jobs:
        t = open(j).read()
        if os.path.join(D, "points_") not in t: bad.append(f"{j}: does not run {D}/points_*")
ids = [i for i, ds in seen.items()]
dup = {i: ds for i, ds in seen.items() if len(ds) > 1}; miss = sorted(set(ref) - set(seen)); extra = sorted(set(seen) - set(ref))
print(f"\nacross {a.dirs}: {len(ids)} distinct frames labelled; duplicates {len(dup)}; missing vs reference {len(miss)}; not in reference {len(extra)}")
if dup: bad.append(f"{len(dup)} frames prepared in more than one folder, e.g. {list(dup.items())[:3]}")
if miss: bad.append(f"{len(miss)} reference frames not prepared anywhere, e.g. {miss[:5]}")
if a.old and os.path.isdir(a.old):
    od = glob.glob(os.path.join(a.old, "points_*", "p[0-9]*", "DONE.json")); print(f"\nold Gamma-centred run {a.old}: {len(od)} finished points kept (same-frame comparison with the MP labels)")
try:
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20).stdout.split("\n")   # python 3.6 safe (login nodes)
    rows = [l.split() for l in q if l.strip() and l.split()[1].startswith("dgchk")]
    per = collections.Counter()
    for jid, name, st in rows:
        sc = subprocess.run(["scontrol", "show", "job", jid], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20).stdout
        m = re.search(r"Command=(\S+)", sc); cmd = m.group(1) if m else "?"
        d = next((D for D in a.dirs + [a.old] if f"/{D.rstrip('/')}/job_" in "/" + os.path.relpath(cmd) or os.path.relpath(cmd).startswith(D.rstrip("/") + "/")), "OTHER: " + cmd)
        per[(d, st)] += 1
        if d == a.old or d.startswith("OTHER"): bad.append(f"job {jid} {name} ({st}) runs {cmd}: not one of the new folders")
    print("\nqueue (dgchk jobs by folder): " + ", ".join(f"{d} {st} {n}" for (d, st), n in sorted(per.items())) if per else "\nqueue: no dgchk jobs")
except Exception as e:
    print(f"\n(queue not checked: {e})")
print("\n" + ("ALL CHECKS PASSED" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad)))
sys.exit(1 if bad else 0)
