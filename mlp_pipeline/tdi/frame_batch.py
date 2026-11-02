#!/usr/bin/env python
"""Score a model on the lambda = 1 DFT frames of every campaign in a survey, over many (P, T, composition) states.

Uses frame_eval.py (one process per window of the run, many in parallel) and then tabulates, per (P, T):
  * per system: mean and sd of D = E_DFT - E_model per cell, per atom, relative force error, linear trend of D;
  * per ice+H system with a matching ice host (same lattice) and a pure-H run at the same (P, T): the composition
    error c per added H, mean-only and with the Gaussian variance term var(D)/(2kT), plus beta*sd as a warning.

  python frame_batch.py SURVEY.csv ROOT --model grace-ft:... --outdir ~/leg/fb [--threads 8 --windows 8 --frames 5]
  python frame_batch.py SURVEY.csv ROOT --outdir ~/leg/fb --table-only      # re-tabulate what is already computed
  --count  inventories the lambda = 1 DFT frames (steps and ps) per campaign and per (P, T, composition); needs no model.
  --dry-run lists the jobs. Existing window CSVs are reused. Paths with bad/fail/broken/crash in the name are skipped
  unless --all is given. P comes from the last '<n>gpa' in the campaign path; T from the survey's INCAR column.
SURVEY.csv is the output of `survey ROOT --csv` (fe_stages.py / the legacy analysis tool).
"""
import argparse, concurrent.futures as cf, csv, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = 8.617333e-5                                         # eV/K
BAD = re.compile(r"bad|fail|broken|crash", re.I)


def parse_comp(s):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"([A-Z][a-z]?)(\d+)", s)}


def read_survey(path, root, allow_bad):
    rows = []
    for r in csv.DictReader(open(os.path.expanduser(path))):
        c = parse_comp(r["comp"]); Ps = re.findall(r"(\d+)gpa", r["path"], re.I)
        if "H" not in c or not Ps or set(c) - {"H", "O"}: continue
        if BAD.search(r["path"]) and not allow_bad: continue
        try: T = float(r["T"].split(",")[0])
        except ValueError: continue
        nO, nH = c.get("O", 0), c["H"]; lat = "bcc" if "bcc" in r["path"] else "fcc" if "fcc" in r["path"] else "-"
        tag = re.sub(r"[^A-Za-z0-9.+-]+", "_", r["path"])
        rows.append(dict(path=r["path"], dir=os.path.join(os.path.expanduser(root), r["path"]), tag=tag, P=float(Ps[-1]), T=T, N=nO + nH, nO=nO, nH=nH,
                         lat=lat if nO else "H", x=(nH / nO if nO else float("inf"))))
    return rows


def windows(a):
    return [(a.lo + i * (1 - a.lo) / a.windows, a.lo + (i + 1) * (1 - a.lo) / a.windows) for i in range(a.windows)]


def run_jobs(rows, a):
    jobs = []
    for r in rows:
        for i, (s, e) in enumerate(windows(a)):
            out = os.path.join(a.outdir, f"{r['tag']}_w{i}.csv")
            if os.path.exists(out) and os.path.getsize(out) > 0: continue
            jobs.append((r, i, out, [sys.executable, a.eval_script, "eval", r["dir"], "--model", a.model, "--n", str(a.frames), "--start", f"{s:.6f}", "--end", f"{e:.6f}", "--out", out]))
    print(f"{len(rows)} systems, {len(jobs)} window jobs to run ({a.workers} at a time, {a.threads} threads each)", flush=True)
    if a.dry_run:
        for r, i, out, cmd in jobs: print(r["path"], f"window {i}")
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), TF_NUM_INTRAOP_THREADS=str(a.threads), TF_NUM_INTEROP_THREADS="1")
    def one(j):
        r, i, out, cmd = j
        with open(out.replace(".csv", ".log"), "w") as lg: rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env).returncode
        return r["path"], i, rc
    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for p, i, rc in ex.map(one, jobs):
            done += 1
            if rc: print(f"  FAILED ({rc}): {p} window {i}  (see its .log)", flush=True)
            elif done % 8 == 0 or done == len(jobs): print(f"  {done}/{len(jobs)} windows done", flush=True)


def lam1_dir(d):
    q = os.path.join(d, "lambda_1.000")
    return q if os.path.isdir(q) else d


def count_one(r):
    import glob
    ld = lam1_dir(r["dir"]); files = sorted(glob.glob(os.path.join(ld, "out*", "OUTCAR*")) + glob.glob(os.path.join(ld, "OUTCAR*")))
    by = {}
    for f in files: by.setdefault(re.sub(r"\.gz$", "", f), f)                       # one copy of each OUTCAR (plain or gz)
    n = 0
    for f in by.values():
        out = subprocess.run(f"zcat -f '{f}' | grep -ac 'free  energy   TOTEN'", shell=True, capture_output=True, text=True).stdout.strip()
        n += int(out or 0)
    potim = 0.0
    try:
        for line in open(os.path.join(ld, "INCAR")):
            m = re.match(r"\s*POTIM\s*=\s*([0-9.eE+-]+)", line)
            if m: potim = float(m.group(1))
    except OSError: pass
    return dict(r, files=len(by), frames=n, ps=n * potim / 1000.0)


def count_report(rows, a):
    with cf.ThreadPoolExecutor(a.workers) as ex: C = list(ex.map(count_one, rows))
    C.sort(key=lambda c: (c["P"], c["T"], c["lat"], c["x"], c["path"]))
    print(f"\nlambda = 1 (pure DFT) frames per campaign: every ionic step counted, ps = steps x POTIM")
    print(f"{'P':>4s} {'T':>5s} {'lat':>3s} {'H/O':>5s} {'N':>4s} {'files':>5s} {'frames':>8s} {'ps':>7s}  path")
    tot = {}
    for c in C:
        x = f"{c['x']:5.2f}" if np.isfinite(c["x"]) else "  H  "
        print(f"{c['P']:4.0f} {c['T']:5.0f} {c['lat']:>3s} {x} {c['N']:4d} {c['files']:5d} {c['frames']:8d} {c['ps']:7.1f}  {c['path']}")
        k = (c["P"], c["T"], c["lat"], round(c["x"], 3) if np.isfinite(c["x"]) else -1); t = tot.setdefault(k, [0, 0.0, 0]); t[0] += c["frames"]; t[1] += c["ps"]; t[2] += 1
    print(f"\nSummed per state (P, T, lattice, H/O); campaigns of the same state can overlap or repeat each other, so these are upper bounds on independent data")
    for k in sorted(tot): print(f"  P={k[0]:.0f} T={k[1]:.0f} {k[2]:>3s} H/O={'H' if k[3] < 0 else k[3]}: {tot[k][2]} campaigns, {tot[k][0]} frames, {tot[k][1]:.1f} ps")


def merge(r, a):
    """Merge a system's window CSVs in window order. Returns the D array and the force columns, or None."""
    D, F, Fr, got = [], [], [], 0
    for i in range(a.windows):
        f = os.path.join(a.outdir, f"{r['tag']}_w{i}.csv")
        if not (os.path.exists(f) and os.path.getsize(f) > 0): continue
        R = np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=1))
        if R.size == 0: continue
        D += list(R[:, 3]); F += list(R[:, 4]); Fr += list(R[:, 5]); got += 1
    if not D: return None
    return np.array(D), np.array(F), np.array(Fr), got


def stats(r, a):
    m = merge(r, a)
    if m is None: return None
    D, F, Fr, got = m; n = len(D); kT = KB * r["T"]; t = np.arange(n)
    mean, sd = D.mean(), D.std(ddof=1) if n > 1 else float("nan")
    slope = np.polyfit(t, D, 1)[0] if n > 2 else float("nan")
    slope_err = sd / np.sqrt(n * (n * n - 1) / 12) if n > 2 else float("nan")
    return dict(r, n=n, windows=got, mean=mean, sd=sd, sem=sd / np.sqrt(n), trend=slope * 1e3, trend_sig=slope / slope_err if n > 2 else float("nan"),
                fr=float(F.mean() / Fr.mean()), kT=kT, cum=mean + sd ** 2 / (2 * kT), bsd=sd / kT)


def composition_errors(S):
    out = []
    for h in S:
        if h["nO"] == 0 or h["x"] <= 2.0001: continue
        for ice in S:
            if ice["nO"] != h["nO"] or abs(ice["x"] - 2.0) > 1e-9 or ice["P"] != h["P"] or ice["T"] != h["T"] or ice["lat"] != h["lat"]: continue
            for pure in S:
                if pure["nO"] != 0 or pure["P"] != h["P"] or pure["T"] != h["T"]: continue
                nadd = h["nH"] - ice["nH"]; w = nadd / pure["N"]
                c = (h["mean"] - ice["mean"] - w * pure["mean"]) / nadd
                ce = np.sqrt(h["sem"] ** 2 + ice["sem"] ** 2 + (w * pure["sem"]) ** 2) / nadd
                cF = (h["cum"] - ice["cum"] - w * pure["cum"]) / nadd
                out.append(dict(P=h["P"], T=h["T"], lat=h["lat"], x=h["x"], nadd=nadd, hpath=h["path"], ipath=ice["path"], ppath=pure["path"], c=c * 1e3, ce=ce * 1e3, cF=cF * 1e3,
                                bsd=h["bsd"], per_atom_shift=c * 1e3 * (nadd / h["nO"]) / (h["N"] / h["nO"])))
    return out


def report(rows, a):
    S = [s for s in (stats(r, a) for r in rows) if s]
    if not S: print("no results yet"); return
    S.sort(key=lambda s: (s["P"], s["T"], s["lat"], s["x"]))
    print(f"\nPer system (D = E_DFT - E_model per cell; sd is the scatter of D; trend sig = slope / its standard error)")
    print(f"{'P':>4s} {'T':>5s} {'lat':>3s} {'H/O':>5s} {'N':>4s} {'n':>3s} {'mean D':>8s} {'meV/at':>7s} {'sd':>6s} {'sd/kT':>6s} {'F-err':>6s} {'trend':>8s} {'sig':>4s}  path")
    for s in S:
        x = f"{s['x']:5.2f}" if np.isfinite(s["x"]) else "  H  "
        print(f"{s['P']:4.0f} {s['T']:5.0f} {s['lat']:>3s} {x} {s['N']:4d} {s['n']:3d} {s['mean']:8.3f} {s['mean'] / s['N'] * 1e3:7.1f} {s['sd']:6.2f} {s['bsd']:6.1f} {s['fr'] * 100:5.1f}% {s['trend']:+8.1f} {s['trend_sig']:+4.1f}  {s['path']}")
    C = composition_errors(S)
    print(f"\nComposition error per added H, meV: c = mean-only, c_F = with var/(2kT).  sd/kT > ~3 means c_F (Gaussian) is not reliable.")
    print(f"{'P':>4s} {'T':>5s} {'lat':>3s} {'H/O':>5s} {'n_add':>5s} {'c':>8s} {'+-':>5s} {'c_F':>8s} {'sd/kT':>6s} {'hull shift':>10s}  H-rich system")
    for c in sorted(C, key=lambda c: (c["P"], c["T"], c["lat"], c["x"])):
        print(f"{c['P']:4.0f} {c['T']:5.0f} {c['lat']:>3s} {c['x']:5.2f} {c['nadd']:5d} {c['c']:+8.0f} {c['ce']:5.0f} {c['cF']:+8.0f} {c['bsd']:6.1f} {c['per_atom_shift']:+8.1f}/at  {c['hpath']}")
    print("hull shift = c * n_add / N: the per-atom shift of the mixing free energy implied by the mean-only c (meV/atom).")
    if a.summary:
        with open(a.summary, "w", newline="") as f:
            keys = ["P", "T", "lat", "x", "N", "n", "mean", "sd", "sem", "trend", "trend_sig", "fr", "path"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(S)
        print(f"wrote {a.summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("survey"); p.add_argument("root"); p.add_argument("--model"); p.add_argument("--outdir", required=True)
    p.add_argument("--threads", type=int, default=8); p.add_argument("--workers", type=int, default=0, help="parallel processes (default 128 // threads)")
    p.add_argument("--windows", type=int, default=8); p.add_argument("--frames", type=int, default=5, help="frames per window"); p.add_argument("--lo", type=float, default=0.2, help="discard this fraction of every run")
    p.add_argument("--all", action="store_true", help="also run paths named bad/fail/broken/crash"); p.add_argument("--only", help="regex on the campaign path")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--table-only", action="store_true"); p.add_argument("--count", action="store_true", help="only count the lambda = 1 DFT frames available per campaign (no model needed)")
    p.add_argument("--eval-script", default=os.path.join(HERE, "frame_eval.py")); p.add_argument("--summary")
    a = p.parse_args(); a.outdir = os.path.expanduser(a.outdir); os.makedirs(a.outdir, exist_ok=True)
    a.workers = a.workers or max(1, 128 // a.threads)
    rows = read_survey(a.survey, a.root, a.all)
    if a.only: rows = [r for r in rows if re.search(a.only, r["path"])]
    if a.count:
        count_report(rows, a); return
    if not a.table_only:
        if not a.model and not a.dry_run: sys.exit("--model is required unless --table-only / --dry-run")
        run_jobs(rows, a)
    if not a.dry_run: report(rows, a)


if __name__ == "__main__":
    main()
