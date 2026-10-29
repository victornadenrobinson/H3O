"""Find TDI runs that are not finished (lambda values missing, e.g. a process that died of GPU out-of-memory, or shorter than the run's target) and write the
commands that finish them with EXACTLY the settings recorded in each run's <tag>_stage0.json (any other setting would be refused by run_real.py: samples
are never mixed). Only the missing / short lambda values are launched (ONLYNODES); finished ones are not touched.
usage (repo root, login node, plain python3 with numpy):
  python3 mlp_pipeline/tdi/finish_runs.py runs/tdi runs/tdi_fb runs/tdi_prep [--only-tags 'H_*,bcc_*'] [--exclude 'insert_*'] [--exclude-from FILE] [--skip-empty] [-o runs/finish_jobs.sh]
  bash runs/finish_jobs.sh          # submits one GPU job per run, then moves them to the top of your queue (scontrol top)
The table says per run: lambda values missing / short, the remaining MD steps, the estimated wall time; runs with a progress file written in the last
3 min are RUNNING and skipped; runs with no node log at all were never launched (check squeue first: they may still be queued).
--local: plain bash commands instead of sbatch (for an interactive GPU node, or the toy model on CPU).
"""
import argparse, fnmatch, glob, json, os, shlex, sys, time
import numpy as np

KNOBS = ("MODEL P T HPER NGHOST HOSTPHASE NATOMS MINRUNS INIT RULE NGAUSS VPA NREP BULK PS PS1 NODES POWER DT CHUNK EQUIL OUTDIR SEED K HOT HOTT "
         "PREPHOLD PREPFREE ONLYNODES NPAR FRESH TAG ONLY0 UQ ENGINE MEMPROC TOYICE TOYICE_FIT ALLOW_NOGPU").split()
# run_real.py defaults for settings that older stage0 jsons do not record (they predate the option, so the run used its default behaviour)
DEFAULTS = dict(hper=2.0, nghost=1, host_phase="fcc", natoms=0, min_runs=20, rule="gl", ngauss=0, every=2, seed=0, hot_ps=0.0, hot_T=12000.0, init_from=None,
                prep_hold_ps=1.0, prep_free_ps=1.0, k=None, bulk_gpa=None, vpa=None, from_dft=None, nrep=None, ps1=3.0, equil_ps=1.0)
ESSENTIAL = ("model", "phase", "P", "T", "ps", "dt", "chunk_ps", "nodes", "power")   # never guessed: a run without them is listed, not resubmitted
SPS = [(1, 0.16), (2, 0.36), (8, 0.75), (10, 0.88)]   # s/step per process vs processes on one A100 (251 atoms; CURRENT_ISSUES 2c)


def s_per_step(c):
    return float(np.interp(c, [x for x, _ in SPS], [y for _, y in SPS]))


def tag_fn():
    """make_tag of run_real.py, taken from its source (run_real imports ASE, which the login-node python may not have)"""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_real.py")).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "make_tag")
    ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), "run_real.py", "exec"), ns); return ns["make_tag"]


def same_tag(A, tag, e):
    """does the launcher, given env e, rebuild exactly this tag? (a different tag would start a NEW run instead of finishing this one)"""
    mk = tag_fn(); a = argparse.Namespace(**{**A, "tag": None, "nrep": int(e["NREP"]) if e["NREP"] else None, "natoms": int(e["NATOMS"] or 0),
                                             "nghost": int(e["NGHOST"]), "hper": float(e["HPER"]), "host_phase": e["HOSTPHASE"] or "fcc"})
    return mk(a)


def args_of(st):
    A = {**DEFAULTS, **st["args"]}; miss = [k for k in ESSENTIAL if k not in A]
    if miss: raise KeyError(f"stage0 json has no {', '.join(miss)}: resubmit by hand")
    return A


def env_of(st, d, tag):
    """launcher environment that reproduces this run (every knob set: empty = unset, so nothing leaks in from the submitting shell)"""
    A = args_of(st); e = {k: "" for k in KNOBS}
    e.update(MODEL=A["model"], P=f"{A['P']:g}", T=f"{A['T']:g}", HPER=f"{A['hper']:g}", NGHOST=str(A["nghost"]), PS=f"{A['ps']:g}", PS1=f"{A['ps1']:g}",
             NODES=str(A["nodes"]), POWER=f"{A['power']:g}", DT=f"{A['dt']:g}", CHUNK=f"{A['chunk_ps']:g}", EQUIL=f"{A['equil_ps']:g}", OUTDIR=d, SEED=str(A.get("seed", 0)))
    if A["phase"] == "insert" and A.get("host_phase", "fcc") != "fcc": e["HOSTPHASE"] = A["host_phase"]
    if A.get("natoms"): e["NATOMS"] = str(A["natoms"])
    if A.get("min_runs") is not None: e["MINRUNS"] = str(A["min_runs"])
    if (A.get("rule") or "gl") != "gl": e.update(RULE=A["rule"], NGAUSS=str(A.get("ngauss") or ""))
    if A.get("vpa") and not A.get("from_dft"):
        e.update(VPA=f"{A['vpa']!r}", NREP=str(A.get("nrep") or 2))
    elif A.get("nrep"):
        e["NREP"] = str(A["nrep"])
    if A.get("bulk_gpa"): e["BULK"] = f"{A['bulk_gpa']!r}"
    if A.get("k"): e["K"] = f"{A['k']!r}"
    if A.get("hot_ps"): e.update(HOT=f"{A['hot_ps']:g}", HOTT=f"{A.get('hot_T', 12000):g}")
    init = A.get("init_from")
    if init and init.endswith(f"{tag}_prep.extxyz"):   # PREPHOLD run: the launcher re-adds the prep frame itself
        e.update(PREPHOLD=f"{A.get('prep_hold_ps', 1):g}", PREPFREE=f"{A.get('prep_free_ps', 1):g}")
    elif init:
        e["INIT"] = init
    return e


def scan(d, tag):
    st = json.load(open(os.path.join(d, f"{tag}_stage0.json"))); A = args_of(st)
    from tdi import analyze
    lam = analyze.rule_nodes(dict(nodes=A["nodes"], power=A["power"], rule=A.get("rule") or "gl", ngauss=A.get("ngauss") or 0))[0]
    per_ps = int(round(1000 / A["dt"])); target = int(round(A["ps"] * per_ps / A["every"]))   # samples per lambda at the target length
    now, running, todo, steps_left, never = time.time(), [], [], 0, True
    for k in range(len(lam)):
        if os.path.exists(os.path.join(d, f"{tag}_node{k}.log")): never = False
        pf = os.path.join(d, f"{tag}_node{k}_progress.json")
        try:
            p = json.load(open(pf))
            if p.get("state", "running") == "running" and now - p["t"] < 180: running.append(k); continue
        except (OSError, ValueError, KeyError):
            pass
        f = os.path.join(d, f"{tag}_node{k}.npy")
        n = len(np.load(f)) if os.path.exists(f) else 0
        if n < target:
            todo.append(k); steps_left += (target - n) * A["every"] + (0 if n else int(round(A["equil_ps"] * per_ps)))
    return st, todo, running, steps_left, never


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--only-tags", default=None, help="comma list of glob patterns")
    ap.add_argument("--exclude", default=None, help="comma list of tag glob patterns to leave out")
    ap.add_argument("--exclude-from", default=None, help="file with one <dir>/<tag> (or tag) per line to leave out, e.g. the pending tasks of an extension array")
    ap.add_argument("--skip-empty", action="store_true", help="leave out runs with no samples at all (never launched)"); ap.add_argument("-o", default="runs/finish_jobs.sh")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--time-cap-h", type=float, default=12.0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pats = a.only_tags.split(",") if a.only_tags else None
    excl = a.exclude.split(",") if a.exclude else []
    if a.exclude_from:
        excl += [os.path.basename(l.strip()) for l in open(a.exclude_from) if l.strip()]
    lines, rows = [], []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
            tag = os.path.basename(f)[:-len("_stage0.json")]
            if pats and not any(fnmatch.fnmatch(tag, p) for p in pats): continue
            if any(fnmatch.fnmatch(tag, p) for p in excl): print(f"{d + '/' + tag:52s} excluded"); continue
            try:
                st, todo, running, left, never = scan(d, tag)
            except Exception as ex:   # noqa: BLE001 (a broken json must not stop the others)
                print(f"{d}/{tag}: cannot read ({ex})"); continue
            if running: print(f"{d + '/' + tag:52s} RUNNING now (lambda {running}): skipped"); continue
            if not todo: continue
            if never and a.skip_empty: print(f"{d + '/' + tag:52s} never launched: skipped (--skip-empty)"); continue
            A = args_of(st); npar = min(len(todo), 8); natoms = st.get("N_atoms_total") or st.get("N") or 251
            ins = 2 if A["phase"] == "insert" else 1   # insertion: two model calls per step
            hours = (left / len(todo)) * s_per_step(npar) * max(1.0, natoms / 251) * ins / 3600 * int(np.ceil(len(todo) / npar)) + 0.3
            hours = min(a.time_cap_h, max(1.0, 1.5 * hours))
            try:
                e = env_of(st, d, tag)
            except KeyError as ex:
                print(f"{d + '/' + tag:52s} lambda {todo} to run, but {ex}"); continue
            if same_tag(A, tag, e) != tag and e["NREP"]:   # older jsons store nrep after setup() filled in the default: the original launch had no NREP
                e["NREP"] = ""
            if same_tag(A, tag, e) != tag:
                print(f"{d + '/' + tag:52s} lambda {todo} to run, but the rebuilt settings give tag {same_tag(A, tag, e)}: resubmit by hand"); continue
            e.update(ONLYNODES=" ".join(map(str, todo)), NPAR=str(npar))
            if a.local:   # interactive / toy use: keep the caller's toy settings
                for k in ("TOYICE", "TOYICE_FIT", "ALLOW_NOGPU"): e.pop(k)
            cmd = "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in e.items()) + f" bash mlp_pipeline/young_tdi_par.sh {A['phase']}"
            note = "never launched (check squeue)" if never else ""
            rows.append((f"{d}/{tag}", todo, left, hours, note))
            if a.local:
                lines.append(f"{cmd} > {d}/{tag}_finish.log 2>&1")
            else:
                hh = int(np.ceil(hours))
                lines.append(f"J=$(sbatch --parsable --partition=gpu --account=allusers --qos=freegpu --gres=gpu:1 --cpus-per-task=8 --mem=60G --time={hh:02d}:00:00 "
                             f"--job-name=fin_{tag[:20]} --output=runs/slurm_%x_%j.out --wrap {shlex.quote('cd ' + os.getcwd() + ' && ' + cmd)}) && echo \"$J {d}/{tag}\" && IDS=\"$IDS $J\"")
    print(f"\n{'run':52s} {'lambda to run':24s} {'MD steps left':>13s} {'est. h':>7s}")
    for r, todo, left, h, note in rows:
        print(f"{r:52s} {str(todo):24s} {left:13d} {h:7.1f}  {note}")
    if not rows: print("nothing to finish"); return
    with open(a.o, "w") as f:
        f.write("#!/bin/bash\n# written by mlp_pipeline/tdi/finish_runs.py " + time.strftime("%Y-%m-%d %H:%M") + "\ncd " + os.getcwd() + "\nIDS=\"\"\n" + "\n".join(lines) + "\n")
        if not a.local: f.write('[ -n "$IDS" ] && scontrol top $IDS && echo "moved to the top of your queue:$IDS"\n')
    print(f"\nwrote {a.o} ({len(rows)} runs): check it, then  bash {a.o}")


if __name__ == "__main__":
    main()
