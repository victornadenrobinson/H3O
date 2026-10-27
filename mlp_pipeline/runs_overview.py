"""Overview of EVERYTHING in runs/ (not only TDI): what each result directory holds and how far it got, and why Slurm tasks failed.
usage (repo root, plain python3, no ASE/pandas needed):  python3 mlp_pipeline/runs_overview.py [--runs runs] [--active-min 45]
1. per directory under runs/: TDI directories (any *_stage0.json): tags, how many have every lambda file (complete = number of <tag>_node<k>.npy equals the json's lambda count; the file count says NOTHING about samples per lambda: use summarize.py), stale-json warnings
   (more node files than declared), tags active in the last N minutes; CSV directories (eos_scan, eos_map, melt, melt_comp ...): csv files, rows, and RESULT lines in logs; everything else: file count, newest file age.
2. Slurm outputs (runs/slurm_<jobname>_<jobid>_<task>.out): per job name how many tasks finished ('== done' printed), failed with a recognised reason (counted by reason, example tasks), hit the time limit, or neither (running, cancelled before the end, or a script that prints no '== done')."""
import argparse, collections, glob, json, os, re, time

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--active-min", type=float, default=45); a = ap.parse_args()
now = time.time(); R = a.runs
print(f"== directories under {R}/")
for d in sorted(x for x in glob.glob(os.path.join(R, "*")) if os.path.isdir(x)):
    name = os.path.basename(d); files = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f)]
    if not files: print(f"{name:18s} empty"); continue
    newest = max(os.path.getmtime(f) for f in files); age = (now - newest) / 3600
    js = sorted(glob.glob(os.path.join(d, "*_stage0.json")))
    if js:
        complete = partial = stale = active = 0; warn = []
        for f in js:
            tag = os.path.basename(f)[:-len("_stage0.json")]
            try: A = json.load(open(f)).get("args", {}); nodes = int(A["nodes"])
            except Exception: warn.append(f"{tag}: unreadable json"); continue
            have = len(glob.glob(os.path.join(d, f"{tag}_node[0-9]*.npy")))
            if have > nodes: stale += 1; warn.append(f"{tag}: {have} node files, json declares {nodes}")
            elif have == nodes: complete += 1
            else: partial += 1
            if any(now - os.path.getmtime(x) < a.active_min * 60 for x in glob.glob(os.path.join(d, f"{tag}_node*"))): active += 1
        print(f"{name:18s} TDI: {len(js)} tags ({complete} with all lambda files, {partial} partial, {stale} STALE-JSON), {active} active in the last {a.active_min:g} min; newest file {age:.1f} h ago")
        for w in warn[:8]: print(f"{'':18s}   ! {w}")
        continue
    csvs = [f for f in files if f.endswith(".csv") and "thermo" not in os.path.basename(f)]; logs = [f for f in files if f.endswith(".log")]
    rows = sum(max(0, sum(1 for _ in open(f, errors="ignore")) - 1) for f in csvs)
    res = sum(1 for f in logs if "RESULT" in open(f, errors="ignore").read(200000)) if logs else 0
    print(f"{name:18s} {len(files)} files, {len(csvs)} result csv ({rows} rows), {len(logs)} logs ({res} with a RESULT line); newest file {age:.1f} h ago")

PAT = [("hull EOS covers only 5000-9000 K", r"hull EOS only covers"), ("composition not in the hull EOS sheet", r"no hull EOS column"), ("volume outside the EOS data (give --vpa)", r"outside the DFT EOS data|give --vpa"),
       ("volume clipped (older code: result at the wrong pressure)", r"volume clipped"), ("no GPU visible", r"NO GPU visible"), ("lambda LOCKED by another job", r"LOCKED:"), ("refused: settings differ from the saved files", r"refus"),
       ("unstable dynamics", r"UNSTABLE"), ("out of memory", r"out of memory|OOM|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|Killed"), ("python traceback", r"Traceback"), ("time limit", r"DUE TO TIME LIMIT")]
print(f"\n== Slurm outputs ({R}/slurm_<name>_<jobid>_<task>.out)")
by = collections.defaultdict(list)
for f in glob.glob(os.path.join(R, "slurm_*.out")):
    m = re.match(r"slurm_(.+?)_(\d+)(?:_(\d+))?\.out$", os.path.basename(f))
    if m: by[m.group(1)].append((m.group(2), m.group(3) or "-", f))
for name in sorted(by):
    st = collections.Counter(); ex = collections.defaultdict(list)
    for job, task, f in sorted(by[name]):
        t = open(f, errors="ignore").read(); verdict = None
        for lab, pat in PAT:
            if re.search(pat, t): verdict = lab; break
        if verdict is None: verdict = "done" if re.search(r"== done|\bexit 0\b", t) else "no end marker (running, cancelled, or a script that prints none)"
        st[verdict] += 1; ex[verdict].append(f"{job}_{task}")
    print(f"{name}: {len(by[name])} outputs")
    for lab, n in st.most_common(): print(f"    {n:4d}  {lab}   e.g. {', '.join(ex[lab][:4])}")
print("\nNote: a reason is the FIRST pattern found in the file (a 'volume clipped' warning does not by itself mean the task failed); jobs still running show 'no end marker'.")
