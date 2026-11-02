#!/usr/bin/env python3
"""Master status board for the VASP single-point campaign (snapshot database -> chunks -> OUTCARs). Read-only, standard library only (plain python3 is fine on Perlmutter).
  python3 mlp_pipeline/dft_status.py [--db snapdb] [--watch 30] [--no-color] [--stale 10]
Shows: Slurm jobs of the campaign, a progress bar per set and overall, per-chunk table (points done / converged / running / failed, s per point, ETA),
health warnings (crashed or killed points, SCF not converged, slow points against the cost model), and where everything lives (database, run directories, slurm logs, VASP executable).
Point states from the files: queued = no OUTCAR yet; running = OUTCAR touched in the last --stale minutes; ok = finished and EDIFF reached; unconv = finished, EDIFF never reached (NELM hit);
dead = OUTCAR cut off and not touched for --stale minutes (killed / out of time / crashed); chunked = chunk exists but its points are not prepared yet."""
import argparse, csv, glob, json, os, re, shutil, socket, subprocess, sys, time
from datetime import datetime

ap = argparse.ArgumentParser(); ap.add_argument("--db", nargs="+", default=["snapdb"], help="one or more database dirs (e.g. snapdb snapdb_v2)"); ap.add_argument("--watch", type=float, default=0); ap.add_argument("--no-color", action="store_true")
ap.add_argument("--stale", type=float, default=10.0, help="minutes without OUTCAR growth before a point counts as dead"); a = ap.parse_args()
COL = sys.stdout.isatty() and not a.no_color
def c(code, s): return f"\033[{code}m{s}\033[0m" if COL else s
G, Y, R, B, D, W = (lambda s: c("32", s)), (lambda s: c("33", s)), (lambda s: c("31", s)), (lambda s: c("36", s)), (lambda s: c("2", s)), (lambda s: c("1", s))
ORDER = ["ok", "unconv", "dead", "running", "queued", "chunked", "pending"]


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60).stdout
    except subprocess.TimeoutExpired: return None
    except Exception: return ""


def tail(path, n=400_000):
    try:
        with open(path, "rb") as f: f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - n)); return f.read().decode("utf8", "ignore"), sz
    except OSError: return None, 0


def find_outcar(d):
    """the worker keeps each run in <point>/attempts/<stamp>/ (DONE.json names the successful one); older layouts have <point>/OUTCAR"""
    dj = os.path.join(d, "DONE.json")
    if os.path.exists(dj):
        try:
            att = json.load(open(dj)).get("attempt")
            if att and os.path.exists(os.path.join(d, att, "OUTCAR")): return os.path.join(d, att, "OUTCAR")
        except Exception: pass
    cand = glob.glob(os.path.join(d, "attempts", "*", "OUTCAR")) + glob.glob(os.path.join(d, "OUTCAR"))
    return max(cand, key=os.path.getmtime) if cand else None


def point_state(d, now):
    oc = find_outcar(d); dj = os.path.join(d, "DONE.json"); lock = os.path.join(d, ".run.lock")
    if os.path.exists(dj):
        try: j = json.load(open(dj))
        except Exception: j = {}
        el = j.get("wall_seconds") or j.get("elapsed_seconds"); conv = True
        if oc: t, _ = tail(oc); conv = bool(t and "aborting loop because EDIFF is reached" in t)
        return ("ok" if conv else "unconv"), (float(el) if el else None), ""
    fail = glob.glob(os.path.join(d, "FAIL*"))
    if fail: return "dead", None, os.path.basename(fail[0])
    ref = oc or (lock if os.path.exists(lock) else None)
    if ref is None: return "queued", None, ""
    age = (now - os.path.getmtime(ref)) / 60; err = ""
    if oc is None and age >= a.stale: return "dead", None, "no OUTCAR"
    for f in glob.glob(os.path.join(d, "out*")) + glob.glob(os.path.join(d, "attempts", "*", "REPORT")):
        tt, _ = tail(f, 6000)
        if tt and re.search(r"ERROR|Killed|oom|forrtl|segmentation|internal error", tt): err = os.path.basename(f)
    return ("running" if age < a.stale and not err else "dead"), None, err


def bar(counts, total, w=34):
    if total <= 0: return D("·" * w)
    cols = {"ok": G, "unconv": Y, "dead": R, "running": B, "queued": D, "chunked": D, "pending": D}; ch = {"ok": "█", "unconv": "█", "dead": "█", "running": "▓", "queued": "▒", "chunked": "░", "pending": "·"}
    out = ""; used = 0
    for k in ORDER:
        n = round(counts.get(k, 0) / total * w) if k != "pending" else 0
        n = min(n, w - used); out += cols[k](ch[k] * n); used += n
    return out + D("·" * (w - used))


def fmt_t(s): return "–" if s is None else (f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 100 else f"{s:.0f} s")


def board(db, header=True):
    now = time.time(); W_ = max(shutil.get_terminal_size((110, 30)).columns, 100) if sys.stdout.isatty() else 110
    head = f"Perlmutter DFT single points (VASP, Gamma)   {datetime.now().strftime('%a %d %b %Y %H:%M:%S')}   user {os.environ.get('USER', '?')}   from {socket.gethostname()}"
    if header: print("╔" + "═" * (W_ - 2) + "╗"); print("║ " + W(head.ljust(W_ - 4)) + " ║"); print("╚" + "═" * (W_ - 2) + "╝")
    print(("\n" if header else "\n\n") + W(f"━━ database {os.path.abspath(db)} ━━"))
    ixp = os.path.join(db, "index.csv")
    if not os.path.exists(ixp): print(R(f"no {ixp}: run from the repo root or pass --db")); return
    ix = list(csv.DictReader(open(ixp))); sets = {}
    for r in ix: sets.setdefault(r["set"], {"prio": r.get("prio", "?"), "n": 0})["n"] += 1
    chunk_of = {}
    for r in ix:
        try: chunk_of[r["db_id"]] = int(float(r["chunk"]))
        except (ValueError, TypeError): pass
    # ---- gather point states
    per_set = {s: {} for s in sets}; chunks = []; times = []; dead = []; unconv = []; seen_ids = set(); slow = []; fin_ids = set(); set_times = {}
    for cd in sorted(glob.glob(os.path.join(db, "chunks", "chunk_*"))):
        name = os.path.basename(cd); pts = os.path.join(cd, "points"); mf = os.path.join(pts, "frames_meta.csv"); st = {}; ct = []
        if os.path.exists(mf):
            for r in csv.DictReader(open(mf)):
                d = os.path.join(pts, r["id"]); s, el, err = point_state(d, now); st[s] = st.get(s, 0) + 1; seen_ids.add(r.get("db_id")); setn = {"False": "F", "True": "T"}.get(r.get("db_set", "?"), r.get("db_set", "?"))      # extxyz reads the set name F as the boolean False
                per_set.setdefault(setn, {}); per_set[setn][s] = per_set[setn].get(s, 0) + 1
                if el: ct.append(el); times.append((int(r.get("natoms", 0) or 0), el)); set_times.setdefault(setn, []).append((int(r.get("natoms", 0) or 0), el))
                if s in ("ok", "unconv"): fin_ids.add(r.get("db_id"))
                if s == "dead": dead.append((name, r["id"], err))
                if s == "unconv": unconv.append((name, r["id"]))
        else:
            ids = [i for i, k in chunk_of.items() if f"chunk_{k:03d}" == name]
            for r in ix:
                if r["db_id"] in ids: per_set[r["set"]]["chunked"] = per_set[r["set"]].get("chunked", 0) + 1; seen_ids.add(r["db_id"])
            st["chunked"] = len(ids)
        chunks.append((name, cd, st, ct))
    for r in ix:
        if r["db_id"] not in seen_ids: per_set[r["set"]]["pending"] = per_set[r["set"]].get("pending", 0) + 1
    # ---- slurm
    print(W("\nSlurm jobs of the campaign"))
    q = ""; qerr = ""
    try:
        rr = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%j|%T|%M|%l|%D|%R|%Z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60); q = rr.stdout; qerr = rr.stderr.strip()
    except subprocess.TimeoutExpired: print("   " + R("squeue timed out (Slurm busy): the job list below is unknown, not empty"))
    except FileNotFoundError: qerr = "squeue not found"
    rows = [l.split("|") for l in q.splitlines() if l.count("|") >= 7]
    mine = [r for r in rows if re.match(r"^([a-z]\d{3}|snap|smoke|vasp)", r[1])]
    if qerr and not rows: print("   " + R(f"squeue said: {qerr[:100]}"))
    if rows and not mine: print("   " + Y(f"{len(rows)} jobs of yours are queued, none named like a chunk (c000 / p001 / q000 ...)"))
    if not mine: print("   " + D("none queued or running" + ("" if rows or shutil.which("squeue") else " (squeue not available here)")))
    for r in mine:
        col = G if r[2] == "RUNNING" else Y; extra = ""
        if r[2] == "PENDING":
            st_ = (sh(f"squeue -h --start -j {r[0]} -o %S") or "").strip(); extra = f"  est. start {st_}" if st_ and "N/A" not in st_ else ""
        print(f"   {r[0]:>9s} {r[1]:8s} {col(r[2]):>9s}  {r[3]:>9s} of {r[4]:>8s}  {r[5]} node  {r[6]}{extra}")
    # ---- bars
    DESC = {"B": "paired n->n+1 H insertion chains on the same cell (no MD): insertion-energy test", "C": "PWidom host / host+H frames at 400-600 GPa, 4000-6000 K: end-state D statistics",
            "C2": "PWidom reverse test: host and host+H at proxy-chosen sites (E_ins directly)", "A": "H2O + n H insertion, short GRACE-MD, T x P grid 3000-7000 K / 300-600 GPa",
            "A_b1": "same as A, older batch-1 run (5 states, more hosts)", "A_part": "same as A, partial files of jobs that were killed", "E": "low-P GRACE-MD, 50-200 GPa fill (H2O..H2.5O, 3000-7000 K)",
            "H": "Huang H3O-phase supercells: static / rattle / strain / defects / GRACE-MD", "D_eos": "EOS-scan frames, fcc/bcc, 3000-7000 K", "D_melt": "melt-composition trajectories, 4000-6000 K, H2O..H3O",
            "G": "pure H, 729 atoms, GRACE-MD (Gamma-point needs the big cell)", "F": "toy-model cells: shakes, voids, vacancies, insertions, compressions, close contacts", "F_H": "pure-H 729-atom toy cells (same perturbations)"}
    allt = [(n, t) for v in set_times.values() for n, t in v if n]; ratio = sorted(t / (105.0 * (n / 330.0) ** 2) for n, t in allt)[len(allt) // 2] if allt else 1.0
    by_set_atoms = {}
    for r in ix: by_set_atoms.setdefault(r["set"], []).append((r["db_id"], int(float(r["natoms"] or 0))))
    nodes_run = sum(int(r[5]) for r in mine if r[2] == "RUNNING") if mine else 0
    print(W("\nProgress by set") + D("   █ ok  █ unconverged  █ dead  ▓ running  ▒ queued  ░ chunked  · not chunked"))
    tot = {}; ntot = 0; left_h_all = 0.0
    for s in sorted(sets, key=lambda s: (int(sets[s]["prio"]) if str(sets[s]["prio"]).isdigit() else 9, s)):
        cn = per_set.get(s, {}); n = sets[s]["n"]; ntot += n
        for k, v in cn.items(): tot[k] = tot.get(k, 0) + v
        okn = cn.get("ok", 0); bad = cn.get("dead", 0) + cn.get("unconv", 0)
        print(f"   {s:8s} p{sets[s]['prio']} {bar(cn, n)} {okn:4d}/{n:<4d} {100 * okn / max(n, 1):3.0f}%" + (R(f"  {bad} bad") if bad else "") + (B(f"  {cn['running']} running") if cn.get("running") else ""))
        ts = set_times.get(s, []); avg = sum(t for _, t in ts) / len(ts) if ts else None
        left = [na for i_, na in by_set_atoms.get(s, []) if i_ not in fin_ids]
        est = sum((avg if avg and abs(na - sum(n_ for n_, _ in ts) / len(ts)) < 40 else 105.0 * (na / 330.0) ** 2 * ratio) for na in left) if left else 0.0
        left_h = est / 4 / 3600; left_h_all += left_h
        print("            " + D(f"{DESC.get(s, '')}  |  " + (f"avg {avg:.0f} s/pt over {len(ts)} pts" if avg else "no timing yet") + f"  |  ~{(sum(na for _, na in by_set_atoms.get(s, [])) / max(len(by_set_atoms.get(s, [])), 1)):.0f} atoms  |  {len(left)} left ~{left_h:.2f} node-h"))
    okn = tot.get("ok", 0); print(f"   {'OVERALL':11s} {bar(tot, ntot)} {okn:4d}/{ntot:<4d} {100 * okn / max(ntot, 1):3.0f}%")
    eta = left_h_all / max(nodes_run, 1)
    print("   " + B("ETA ") + f"{left_h_all:.1f} node-hours left (4 lanes per node, {ratio:.2f}x the cost model where no timing exists); " + (f"{nodes_run} node(s) running now -> ~{eta:.1f} h if the rest start at once" if nodes_run else "no node running now"))
    # ---- chunks
    print(W("\nChunks"))
    print("   " + D(f"{'chunk':14s}{'pts':>5s}{'ok':>5s}{'unconv':>7s}{'dead':>6s}{'run':>5s}{'queued':>7s}{'s/pt':>8s}   directory"))
    def cell(v, w, fn=None, dim0=True):
        t = str(v).rjust(w); return (fn(t) if fn and v not in (0, "0") else D(t) if dim0 and v in (0, "0") else t)
    for name, cd, st, ct in chunks:
        n = sum(st.values()); mean = sum(ct) / len(ct) if ct else None
        print("   " + f"{name:14s}" + cell(n, 5) + cell(st.get("ok", 0), 5, G) + cell(st.get("unconv", 0), 7, Y) + cell(st.get("dead", 0), 6, R) + cell(st.get("running", 0), 5, B)
              + cell(st.get("queued", 0) + st.get("chunked", 0), 7) + cell(f"{mean:.0f}" if mean else "–", 8, None, False) + "   " + os.path.abspath(cd))
    # ---- health
    print(W("\nHealth"))
    msgs = []
    if times:
        ts = sorted(t for _, t in times); med = ts[len(ts) // 2]; per_atom = [(n, t) for n, t in times if n]
        ratio = [t / (105.0 * (n / 330.0) ** 2) for n, t in per_atom]; rr = sorted(ratio)[len(ratio) // 2] if ratio else None
        msgs.append(G("✓ ") + f"{len(ts)} points timed: median {med:.0f} s, max {max(ts):.0f} s, mean {sum(ts) / len(ts):.0f} s" + (f"; {rr:.2f}x the cost model (105 s x (N/330)^2 on 32 ranks)" if rr else ""))
        if rr and rr > 1.5: msgs.append(Y("! ") + "points are slower than the cost model: chunks will not fit the 50-min lanes; use fewer points per chunk (snapdb.py chunk --budget-min lower)")
        running = tot.get("running", 0); queued = tot.get("queued", 0) + tot.get("chunked", 0)
        if med: msgs.append(B("→ ") + f"ETA for the {running + tot.get('queued', 0)} points already prepared: ~{fmt_t((running + tot.get('queued', 0)) * sum(ts) / len(ts) / 4)} on one node (4 lanes); all {tot.get('pending', 0) + queued} unfinished: ~{(tot.get('pending', 0) + queued + running) * sum(ts) / len(ts) / 4 / 3600:.1f} node-hours")
    else: msgs.append(D("· no finished point to time yet"))
    if dead: msgs.append(R("✗ ") + f"{len(dead)} dead points (OUTCAR stopped growing > {a.stale:.0f} min, no timing block): " + ", ".join(f"{c_}/{p_}" + (f" [{e_}]" if e_ else "") for c_, p_, e_ in dead[:6]) + (" ..." if len(dead) > 6 else ""))
    if unconv: msgs.append(Y("! ") + f"{len(unconv)} points finished without EDIFF reached (SCF hit NELM): " + ", ".join(f"{c_}/{p_}" for c_, p_ in unconv[:6]))
    if not dead and not unconv and times: msgs.append(G("✓ ") + "no dead or unconverged points")
    if not any(st.get("running") or st.get("ok") for _, _, st, _ in chunks) and mine: msgs.append(Y("! ") + "a job is in the queue or running but no point has produced an OUTCAR yet")
    for m in msgs: print("   " + m)
    # ---- directories
    print(W("\nWhere things are"))
    print(f"   database       {os.path.abspath(db)}   (db.extxyz, index.csv, coverage.png)")
    tmpl = ""
    for cd in glob.glob(os.path.join(db, "chunks", "chunk_*", "points", "manifest.json")):
        try: tmpl = json.load(open(cd)).get("template", ""); break
        except Exception: pass
    if tmpl: print(f"   VASP template  {tmpl}   (INCAR / KPOINTS / POTCAR, never copied into the repo)")
    outs = sorted(glob.glob("slurm-*.out"), key=os.path.getmtime, reverse=True)[:3]
    for o in outs: print(f"   slurm log      {os.path.abspath(o)}")
    exe = ""
    for o in outs:
        t, _ = tail(o, 20000)
        m = re.search(r"srun .* (\S*vasp_\w+)", t or "")
        if m: exe = m.group(1); break
    if exe: print(f"   VASP binary    {exe}")
    lab = os.path.join(db, "labelled.extxyz")
    print(f"   labelled out   {os.path.abspath(lab)}" + ("" if os.path.exists(lab) else D("   (not written: python snapdb.py collect snapdb)")))


def show():
    for i, d in enumerate(a.db): board(d, i == 0)


if a.watch:
    try:
        while True: sys.stdout.write("\033[2J\033[H" if sys.stdout.isatty() else ""); show(); sys.stdout.flush(); time.sleep(a.watch)
    except KeyboardInterrupt: pass
else: show()
