#!/usr/bin/env python3
"""ONE report of everything that ran: Slurm jobs of the last N hours grouped by kind (with the failures and the tail of their logs), the DFT point jobs (how many points are done), every PWidom result (pooled per host, model
name, flags), the pure-H finite-size table, and what is still unscored. Login node, CPU only, about a minute. Writes runs/morning_review_<date>.txt as well as printing.
   python3 mlp_pipeline/morning_review.py [--hours 48] [--root .]          (use the grace env python: it needs numpy)
Missing directories or a missing Slurm are reported, not fatal."""
import argparse, collections, datetime, glob, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tdi"))
ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=int, default=48); ap.add_argument("--root", default="."); a = ap.parse_args(); os.chdir(a.root)
OUT = []


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); OUT.append(s)


def sh(cmd, timeout=90):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e: return ""


GROUPS = [(r"^dgchk", "DFT point jobs (dgcheck / reweight)"), (r"^pw_ref|^pw_xh|^pw_h2o", "PWidom campaign (shared GPU)"), (r"^fs_N", "finite-size pure-H"), (r"^ft_|^ft[AB]$", "fine-tunes"), (r"^score", "scoring"),
          (r"^kpt_", "k-point tests"), (r"^interactive|^sys/dashboard", "interactive"), (r"^dft-pwidom", "DFT PWidom (48 h)")]
grp = lambda name: next((g for r, g in GROUPS if re.search(r, name)), name.split("_")[0])
say(f"=== MORNING REVIEW {datetime.datetime.now():%Y-%m-%d %H:%M}   (jobs of the last {a.hours} h) ===")
# ---- 1. Slurm
say("\n[1] SLURM")
raw = sh(f"sacct -u $USER -X -S now-{a.hours}hours -P -n -o JobID,JobName%60,State,Elapsed,ExitCode"); rows = [l.split("|") for l in raw.strip().splitlines() if l.count("|") >= 4]
if not rows: say("  sacct returned nothing (not on a Slurm node, or no jobs in the window)")
by = collections.defaultdict(list)
for r in rows: by[grp(r[1])].append(r)
for g, rs in sorted(by.items()):
    cnt = collections.Counter(r[2].split()[0] for r in rs); say(f"  {g:40s} {len(rs):3d} jobs: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
bad = [r for r in rows if r[2].split()[0] in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED") and not r[3].startswith("00:00:0")]
if bad:
    say("  non-completed jobs that ran for a while (job, name, state, elapsed, exit):")
    for r in bad[:25]:
        say(f"    {r[0]:>10s} {r[1][:34]:34s} {r[2][:18]:18s} {r[3]:>9s} {r[4]}")
        for lf in glob.glob(f"logs_pw/*{r[0]}*") + glob.glob(f"slurm-{r[0]}*.out"):
            tail = [l.strip() for l in open(lf, errors="replace").read().splitlines() if l.strip()][-2:]; say(f"        {lf}: " + " | ".join(t[:110] for t in tail))
q = sh("squeue -u $USER -h -o '%j|%T'").strip().splitlines(); qc = collections.Counter((grp(l.split('|')[0]), l.split('|')[1]) for l in q if '|' in l)
say("  queue now: " + (", ".join(f"{g} {s} {n}" for (g, s), n in sorted(qc.items())) if qc else "empty"))
# ---- 2. DFT point jobs
say("\n[2] DFT POINT JOBS (points finished / prepared; a point is done when it has DONE.json)")
sets = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob("runs/*/points_*/dirs.txt")})      # every folder with prepared points, whatever its name
for d in sets:
    tot = done = 0; shas = set()
    for dl in glob.glob(f"{d}/points_*/dirs.txt"):
        for l in open(dl).read().split():
            tot += 1; done += os.path.exists(os.path.join(l, "DONE.json"))
    jobs = len(glob.glob(f"{d}/job_*.sbatch"))
    say(f"  {d:28s} {done:5d} / {tot:5d} points  ({100 * done / max(tot, 1):.0f} %)   {jobs} job scripts" + ("" if glob.glob(f"{d}/points_*/.frames_sha") else "   (no .frames_sha: prepared before the stale-input check)"))
# ---- 3. PWidom results
say("\n[3] PWIDOM RESULTS (dg meV per added H, statistical errors only; replicates of the same host pooled)")
try:
    import pwidom_hull as ph
except Exception as e: ph = None; say(f"  cannot import pwidom_hull ({e}): run with the grace env python")


def mol_frac(npz, nfr=5, bond=1.0, ratio=1.6):
    """fraction of H atoms that sit in an isolated pair: nearest neighbour closer than `bond` A AND the second neighbour more than `ratio` x farther (an H2 molecule has r2/r1 of ~2.5; atomic compressed fluid ~1.1-1.3).
    ~0 = atomic fluid, ~1 = molecular. The single-H insertion of PWidom is only meaningful where this is small. Minimum image, a few frames."""
    import numpy as np
    z = np.load(npz); X = z["host"]; cell = np.asarray(z["cell"]) if "cell" in z.files else np.eye(3) * float(z["V"]) ** (1 / 3); inv = np.linalg.inv(cell); fr = []
    for f in np.linspace(0, len(X) - 1, min(nfr, len(X))).astype(int):
        d = X[f][:, None, :] - X[f][None, :, :]; d -= np.round(d @ inv) @ cell; r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf); rs = np.sort(r, axis=1)
        fr.append(float(((rs[:, 0] < bond) & (rs[:, 1] > ratio * rs[:, 0])).mean()))
    return float(np.mean(fr))


def show(label, dirs):
    if ph is None: return
    runs, models, errs = [], set(), 0
    for d in dirs:
        if not os.path.isfile(os.path.join(d, "manifest.json")): continue
        try: m, rws = ph.load_run(d); runs.append(rws); models.add(re.sub(r".*ft_([^/]+)/seed.*", r"\1", str(m.get("model", "?"))))
        except (SystemExit, Exception): errs += 1
    if not runs: say(f"  {label}: no finished run" + (f" ({errs} not readable yet)" if errs else "")); return
    dP = collections.defaultdict(list)                      # P_host - P_target per host over all replicate runs: a bad EOS (e.g. a T-mixed one) shows here
    for d in dirs:
        try:
            for j in json.load(open(os.path.join(d, "manifest.json")))["jobs"]:
                w = json.load(open(j["result"])); dP[j["name"]].append(w["P_host_GPa"] - w["P_target"])
        except Exception: pass
    rws = ph.pool_rows(runs) if len(runs) > 1 else runs[0]
    mf = ""
    for d in dirs:                                    # molecular fraction of the pure-H reference of this state (own frames, or the shared reference run)
        cand = glob.glob(os.path.join(d, "pw_pureH_*_frames.npz"))
        try:
            ref = json.load(open(os.path.join(d, "manifest.json"))).get("reference") or {}
            if ref.get("frames"): cand = cand or [ref["frames"]]
        except Exception: pass
        if cand:
            try: m = mol_frac(cand[0]); mf = f"   pure H molecular fraction {m:.2f}" + ("  <-- molecular: single-H insertion / dg not meaningful" if m > 0.5 else ""); break
            except Exception: pass
    say(f"  {label}   model {','.join(sorted(models))}   {len(runs)} run dir(s)" + (f", {errs} unreadable" if errs else "") + mf)
    for r in sorted(rws, key=lambda r: r["n"]): say(f"      {r['name']:9s} n {r['n']:.3f}   dg {r['dg']:+6.0f} +- {r['err']:4.0f}   {' '.join(r['flags'])}" + (f"   ({r['k']} runs, spread {r['spread']:.0f})" if r.get("k", 1) > 1 else "") + (f"   P_host-P {sum(dP[r['name']]) / len(dP[r['name']]):+.0f} GPa" + ("  <-- >15 GPa: EOS too crude" if abs(sum(dP[r['name']]) / len(dP[r['name']])) > 15 else "") if dP.get(r["name"]) else ""))


show("farm end members, pooled", ["runs/farm_endmem_500GPa_5000K/pooled"])
show("end members 20-frame runs", ["runs/pwidom_endmem_new_500GPa_5000K", "runs/pwidom_endmem_rep2_500GPa_5000K"])
show("pw_rep1 (Hplus 6/12/18/27)", sorted(glob.glob("runs/pw_rep1/xh*_s??")))
show("pw_prod (Hplus 1/2/3/6)", sorted(glob.glob("runs/pw_prod/xh*_s??")))
for sd in sorted(glob.glob("runs/pw_states/*")): show(f"state {os.path.basename(sd)}", sorted(glob.glob(f"{sd}/xh*_s??")))
# ---- 4. finite size
say("\n[4] PURE-H FINITE SIZE  G_H(N) = mu_P - fs  (should not depend on N)")
for mode in ("robust", "measured"):
    o = sh(f"{sys.executable} {HERE}/pwidom_finite_size.py 'runs/fs_pureH/N*/pw_pureH_*_candidate_K*.json' --pressure-mode {mode} 2>&1 | grep -A12 'G_H per size'")
    say(f"  [{mode}]" + (("\n" + "\n".join("    " + l for l in o.strip().splitlines())) if o.strip() else " no results yet"))
# ---- 5. scoring / models
say("\n[5] FINE-TUNES")
fts = [os.path.basename(d)[3:] for d in glob.glob("runs/grace/ft_*") if os.path.exists(d + "/seed/1/final_model/saved_model.pb")]
uns = [n for n in fts if not (os.path.exists(f"runs/D_{n}.csv") and os.path.exists(f"runs/Dmd_{n}.csv"))]
say(f"  {len(fts)} finished models; unscored on valid_n / DFT-MD: {len(uns)}" + (f" ({', '.join(sorted(uns)[:10])})" if uns else ""))
pc = len(glob.glob("runs/dgcheck/models/D_*.csv")); say(f"  scored on the sealed pair set: {pc}")
say("\n[6] CHECKLIST (what each finished piece unlocks): see the plan; the commands for the DFT-correction chain are in docs / the last message.")
os.makedirs("runs", exist_ok=True); fn = f"runs/morning_review_{datetime.datetime.now():%Y%m%d_%H%M}.txt"; open(fn, "w").write("\n".join(OUT) + "\n"); print(f"\n-> {fn}")
