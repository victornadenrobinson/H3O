#!/bin/bash
# EOS scans with a live dashboard and a job queue: every (composition, volume) is one 17_eos_md.py process; up to PER_GPU processes per GPU run at once and a new one starts as soon as
# a slot frees (most expensive first, on the least-loaded GPU), so several compositions fill the GPUs together and the slow high-pressure volumes start first. Interactive GPU node, repo root.
#   bash mlp_pipeline/eos_live.sh                                   H2O liquid, 7000 K, 16 volumes (~10-1000 GPa)
#   COMP="h2o h3o h" T=7000 bash mlp_pipeline/eos_live.sh             several compositions in one queue
#   COMP = h2o | h2.25 | h3o | h5o | h7o | h  (the compositions of data-h3o-convex-hull.csv; DFT EOS at 5000/7000/9000 K)
# env: MODEL (grace-ft:runs/grace/ft_<run>; default the F10 control, else F200), T (7000), NV (16 volumes per composition), VMAX / VMIN (A^3/atom; default per composition),
#      PS / EQ (production / equilibration ps; 0.75 / 0.25, pure H 0.5 / 0.2), PM_T / PM_PS (premelt 9000 K / 0.25 ps, mixtures), NREP (bcc x3 or x4, default the one closest to 384 atoms),
#      NATOMS (pure H, 729), PER_GPU (4), REFRESH (10 s), PLOT (braille | none).
# Cells: H2O 384, H2.25O 416, H3O 512, H5O 324, H7O 432 atoms (liquid branch: bcc start, premelt); pure H 729 (bcc start that melts). Default volume range per composition: ideal per-atom
# mixing of the H2O range (10 -> 1.75 A^3/atom) and the pure-H range (8 -> 0.9).
# Dashboard (in place, every REFRESH s): per composition done / running / queued; one row per running job (progress, T, P +- error, P_DFT at the same volume, dP %, ms/step, ETA, GPU);
# a dense braille V(P) plot of every composition against its DFT EOS. At the end, per composition: table (P, P_DFT, dP %, T, D_O, D_H, O lattice, closest contacts, flags FROZE / COLLAPSE /
# CLOSE-HH / T-off / P-drift); all compositions on one figure: python3 mlp_pipeline/plot_eos_all.py --T <T>.
# Outputs: runs/eos_live/<model>_<comp>_<T>K_<N>at/ (logs, 17_eos_md folders, <tag>.csv). Volumes already in <tag>.csv are skipped. Ctrl-C kills the running MD processes.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
if [ -z "${MODEL:-}" ]; then for r in mfin_f_lr1e-2_e50 mfin_f_lr1e-2_e50_F200; do [ -e runs/grace/ft_$r/seed/1/final_model/saved_model.pb ] && { MODEL=grace-ft:runs/grace/ft_$r; break; }; done; fi
[ -n "${MODEL:-}" ] || { echo "set MODEL=grace-ft:runs/grace/ft_<run>"; exit 1; }
NG=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "$NG" -ge 1 ] || { echo "no GPU visible"; exit 1; }
exec $PY - "$MODEL" "$NG" <<'EOF'
import ast, csv, math, os, re, shutil, signal, subprocess, sys, time
import numpy as np
MODEL, NG = sys.argv[1], int(sys.argv[2]); E = os.environ.get
T = int(float(E("T", "7000"))); NV = int(E("NV", "16")); PER = int(E("PER_GPU", "4")); REF = float(E("REFRESH", "10")); PM_T = E("PM_T", "9000")
COMPS = E("COMP", "h2o").split(); HPER = {"h": 0.0, "h2o": 2.0, "h2.25": 2.25, "h3o": 3.0, "h5o": 5.0, "h7o": 7.0}
NAME = {"h2o": "H2O", "h": "H", "h2.25": "H2.25O", "h3o": "H3O", "h5o": "H5O", "h7o": "H7O"}
bad = [c for c in COMPS if c not in HPER]
if bad: sys.exit(f"COMP must be among {list(HPER)}; got {bad}")
mname = os.path.basename(MODEL.split(":", 1)[-1].rstrip("/"))

def dft_eos(comp):
    try: rows = list(csv.reader(open("data-h3o-convex-hull.csv")))
    except OSError: return []
    h = rows[0]
    try: iP, iV = h.index(f"P {T}K {comp}"), h.index(f"V {T}K {comp} norm")
    except ValueError: return []
    out = []
    for r in rows[1:]:
        try: out.append((float(r[iP]), float(r[iV])))
        except (ValueError, IndexError): pass
    return sorted(out)

def p_dft(comp, vfu):
    pts = sorted(DFT[comp], key=lambda x: x[1])
    for (P1, V1), (P2, V2) in zip(pts, pts[1:]):
        if V1 <= vfu <= V2 and V2 > V1: return math.exp(math.log(P1) + (math.log(P2) - math.log(P1)) * (vfu - V1) / (V2 - V1))
    return float("nan")

# ---- composition settings and the job list
C, jobs, DFT = {}, [], {}
for c in COMPS:
    h = HPER[c]; DFT[c] = dft_eos(c)
    if c == "h":
        nat = int(E("NATOMS", "729")); fu = 1.0; vmax, vmin = float(E("VMAX", "8")), float(E("VMIN", "0.9")); ps, eq, pm = float(E("PS", "0.5")), float(E("EQ", "0.2")), 0.0
        sysargs = ["--phase", "H", "--branch", "fluid", "--natoms", str(nat)]; desc = f"pure H fluid, {nat} atoms"
    else:
        nrep = int(E("NREP")) if E("NREP") else min((3, 4), key=lambda n: abs(2 * n ** 3 * (1 + h) - 384))
        nat = int(2 * nrep ** 3 + round(h * 2 * nrep ** 3)); fu = 1.0 + h
        vmax = float(E("VMAX")) if E("VMAX") else round((3 * 10 + (h - 2) * 8) / (1 + h), 3); vmin = float(E("VMIN")) if E("VMIN") else round((3 * 1.75 + (h - 2) * 0.9) / (1 + h), 3)
        ps, eq, pm = float(E("PS", "0.75")), float(E("EQ", "0.25")), float(E("PM_PS", "0.25"))
        sysargs = ["--phase", "bcc", "--branch", "liquid", "--hper", f"{h:g}", "--nrep", str(nrep), "--premelt-T", PM_T, "--premelt-ps", str(pm)]; desc = f"{NAME[c]} liquid, {nat} atoms (premelt {PM_T} K {pm} ps)"
    tag = f"{mname}_{c}_{T}K_{nat}at"; D = os.path.join("runs", "eos_live", tag); os.makedirs(D, exist_ok=True)
    C[c] = dict(fu=fu, nat=nat, ps=ps, eq=eq, pm=pm, tag=tag, D=D, mpa=1.008 if c == "h" else (15.999 + 1.008 * h) / (1 + h))
    done_v = set()
    if os.path.exists(os.path.join(D, tag + ".csv")):
        for r in csv.DictReader(open(os.path.join(D, tag + ".csv"))):
            try: done_v.add(round(float(r["vpa"]), 4))
            except (KeyError, ValueError): pass
    print(f"{NAME[c]:7s} {desc} | {NV} volumes {vmax} -> {vmin} A^3/atom | equil {eq} + prod {ps} ps | DFT EOS {len(DFT[c])} points | {len(done_v)} volumes already done -> {D}")
    for k, v in enumerate(np.geomspace(vmax, vmin, NV)):
        v = round(float(v), 4)
        if v in done_v: continue
        steps = (pm + eq + ps) * 2000
        jobs.append(dict(comp=c, k=k, vpa=v, log=os.path.join(D, f"log_{k}.txt"), cost=nat * steps * (0.5 + 4.0 / v), state="queued", pid=None, gpu=None, res=None,
                         cmd=[sys.executable, "mlp_pipeline/17_eos_md.py", MODEL] + sysargs + ["--T", str(T), "--vpa", f"{v}", "--equil-ps", str(eq), "--ps", str(ps), "--report-fs", "50", "--tag", tag, "--outdir", D]))
if not jobs: sys.exit("nothing to run (all volumes done)")
jobs.sort(key=lambda j: -j["cost"])
print(f"{len(jobs)} jobs queued on {NG} GPU x {PER} slots, most expensive first; model {MODEL}", flush=True)
procs = {}

def stop(*_):
    for p in procs.values():
        try: p.kill()
        except Exception: pass
    print("\ninterrupted: MD processes killed"); sys.exit(1)
signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

def launch():
    load = [0.0] * NG; cnt = [0] * NG
    for j in jobs:
        if j["state"] == "running": load[j["gpu"]] += j["cost"]; cnt[j["gpu"]] += 1
    for j in jobs:
        if j["state"] != "queued": continue
        free = [g for g in range(NG) if cnt[g] < PER]
        if not free: break
        g = min(free, key=lambda g: (load[g], cnt[g]))
        p = subprocess.Popen(j["cmd"], stdout=open(j["log"], "w"), stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
        procs[p.pid] = p; j.update(state="running", pid=p.pid, gpu=g); load[g] += j["cost"]; cnt[g] += 1

STEP = re.compile(r"step (\d+)/(\d+)\s+\[(.*?), (\d+) pts\]\s+T (\d+) K\s+P ([-\d.]+) \+- ([-\w.]+) GPa\s+\(now T (\d+) P ([-\d.]+)\)\s+(\d+) ms/step")
def update(j):
    """parse the job's log: stage, progress fraction, T, P, error, ms/step, ETA; result dict and final state when the process has ended"""
    c = C[j["comp"]]; txt = open(j["log"], errors="ignore").read() if os.path.exists(j["log"]) else ""
    st = dict(stage="loading", frac=0.0, T="", P="", Pe="", ms="", eta=""); tot = c["pm"] + c["eq"] + c["ps"]
    if "premelt" in txt: st["stage"] = "premelt"
    if "premelt done" in txt: st.update(stage="equil", frac=c["pm"] / tot)
    m = STEP.findall(txt)
    if m:
        s, n, ph, npts, Tt, P, Pe, Tn, Pn, ms = m[-1]; s, n = int(s), int(n)
        st.update(stage="prod" if ph.startswith("prod") else "equil", frac=(c["pm"] + (c["eq"] + c["ps"]) * s / n) / tot, T=Tt, P=P, Pe=Pe, ms=ms, eta=f"{(n - s) * int(ms) / 1000:.0f}s")
    p = procs.get(j["pid"]); ended = p is not None and p.poll() is not None
    if "RESULT" in txt:
        r = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", txt.split("RESULT", 1)[1].strip().splitlines()[0])   # numpy 2 prints np.float64(...)
        try: j["res"] = ast.literal_eval(re.sub(r"\bnan\b", "None", r))
        except Exception: pass
        st.update(stage="DONE", frac=1.0)
        if ended: j["state"] = "done"
    elif ended:
        err = [l for l in txt.splitlines() if "Error" in l]; j["state"] = "failed"; st["stage"] = "FAILED: " + (err[-1][:70] if err else f"exit {p.returncode} (see {j['log']})")
    j["st"] = st

def gpus():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return "  ".join(f"GPU{a.strip()} {b.strip():>3s}% {int(cc) / 1024:.0f}/{int(d) / 1024:.0f}G" for a, b, cc, d in (l.split(",") for l in o.strip().splitlines()))
    except Exception: return "(nvidia-smi unavailable)"

VIS = re.compile(r"\x1b\[[0-9;]*m")
def fit(line, w):
    """cut to w visible characters (colour codes do not count): a wrapped line breaks the in-place refresh"""
    if len(VIS.sub("", line)) <= w: return line
    out, n, i = "", 0, 0
    while i < len(line) and n < w:
        mm = VIS.match(line, i)
        if mm: out += mm.group(0); i = mm.end(); continue
        out += line[i]; n += 1; i += 1
    return out + "\x1b[0m"

CI = [1, 2, 3, 4, 5, 6]                                                    # termplot palette slots per composition; DFT lines grey (11)
def plot(rows=18):
    try: from termplot import Canvas, frame, fg, PAL
    except Exception: return []
    pts = []
    for i, c in enumerate(COMPS):
        for P, V in DFT[c]: pts.append((math.log10(P), V, i, "dft"))
    for j in jobs:
        ci = COMPS.index(j["comp"]); fu = C[j["comp"]]["fu"]
        if j["res"]: pts.append((math.log10(max(j["res"]["P_GPa"], 1e-3)), fu * j["vpa"], ci, "done"))
        elif j.get("st", {}).get("P"):
            try: pts.append((math.log10(max(float(j["st"]["P"]), 1e-3)), fu * j["vpa"], ci, "run"))
            except ValueError: pass
    if not pts: return []
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; xlo, xhi = min(xs) - 0.05, max(xs) + 0.05; ylo, yhi = min(ys) * 0.95, max(ys) * 1.03
    cv = Canvas(72, rows); X = lambda x: (x - xlo) / (xhi - xlo) * (cv.w - 1); Y = lambda y: (1 - (y - ylo) / (yhi - ylo)) * (cv.h - 1)
    for i in range(len(COMPS)):
        d = sorted((x, y) for x, y, ci, k in pts if ci == i and k == "dft")
        for (x0, y0), (x1, y1) in zip(d, d[1:]): cv.line(X(x0), Y(y0), X(x1), Y(y1), 11)
    for x, y, ci, k in pts:
        if k == "dft": continue
        dd = np.array([-1, 0, 1]) if k == "done" else np.array([0]); gx, gy = np.meshgrid(dd, dd); cv.dots(X(x) + gx.ravel(), Y(y) + gy.ravel(), CI[ci % 6])
    leg = "  ".join(fg(PAL[CI[i % 6]], "● " + NAME[c]) for i, c in enumerate(COMPS)) + "  " + fg(PAL[11], "— DFT") + "   (block = done, dot = running)"
    return frame(cv, (xlo, xhi), (ylo, yhi), "log10 P (GPa)   [1 = 10, 2 = 100, 3 = 1000 GPa]", "V per formula unit (A^3)", f"EOS at {T} K: model vs DFT", legend=leg).split("\n")

t0 = time.time(); prev = 0
while True:
    launch()
    for j in jobs:
        if j["state"] == "running": update(j)                              # update() also sets done / failed when the process has ended
    sz = shutil.get_terminal_size((160, 50)); tw, th = max(40, sz.columns - 1), sz.lines
    out = [f"elapsed {time.time() - t0:5.0f} s   {gpus()}"]
    for c in COMPS:
        js = [j for j in jobs if j["comp"] == c]; n = lambda s: sum(1 for j in js if j["state"] == s)
        out.append(f"  {NAME[c]:7s} done {n('done'):2d}  running {n('running'):2d}  queued {n('queued'):2d}  failed {n('failed'):2d}   -> {C[c]['D']}")
    out.append(f"{'comp':7s} {'vpa':>6s} {'gpu':>3s}  {'progress':22s} {'stage':8s} {'T K':>6s} {'P GPa':>8s} {'+-':>5s} {'P_DFT':>7s} {'dP%':>6s} {'ms/st':>5s} {'ETA':>6s}")
    for j in [j for j in jobs if j["state"] == "running"] + [j for j in jobs if j["state"] == "failed"]:
        st = j.get("st") or dict(stage="loading", frac=0, T="", P="", Pe="", ms="", eta=""); c = j["comp"]; pd_ = p_dft(c, C[c]["fu"] * j["vpa"])
        try: pn = float(st["P"])
        except ValueError: pn = float("nan")
        dps = f"{100 * (pn - pd_) / pd_:+6.1f}" if pd_ == pd_ and pn == pn else f"{'':6s}"; bar = "#" * int(20 * st["frac"]) + "." * (20 - int(20 * st["frac"]))
        out.append(f"{NAME[c]:7s} {j['vpa']:6.3f} {j['gpu']:>3d}  [{bar}] {st['stage'][:8]:8s} {st['T']:>6s} {st['P']:>8s} {st['Pe']:>5s} {(f'{pd_:7.1f}' if pd_ == pd_ else f'{chr(45):>7s}')} {dps} {st['ms']:>5s} {st['eta']:>6s}"
                   + (f"  {st['stage']}" if st["stage"].startswith("FAILED") else ""))
    room = th - 2 - len(out) - 7                                           # rows left for the plot canvas (title, legend, axis label, x axis and its labels take ~7 lines)
    if E("PLOT", "braille") == "braille" and room >= 6: out += [""] + plot(min(18, room))
    elif E("PLOT", "braille") == "braille": out.append(f"(plot hidden: terminal too short by {6 - room} lines; enlarge the window or use NV / fewer compositions)")
    out = [fit(l, tw) for l in out][: max(10, th - 2)]                     # never wider or taller than the terminal
    if prev: sys.stdout.write(f"\x1b[{prev}A")
    sys.stdout.write("\n".join(l + "\x1b[K" for l in out) + "\n\x1b[J"); sys.stdout.flush(); prev = len(out)
    if all(j["state"] in ("done", "failed") for j in jobs): break
    time.sleep(REF)

# ---- final tables with flags, throughput
print(f"\nall {len(jobs)} jobs finished in {time.time() - t0:.0f} s")
def num(r, k):
    """value or nan (None = nan in the RESULT line; 0.0 is a real value, not missing)"""
    v = r.get(k); return float("nan") if v is None else float(v)
def flags(r, comp):
    f = []
    if comp != "h" and (str(r.get("O_struct") or "") in ("bcc", "fcc", "hcp") or (num(r, "D_O_cm2s") == num(r, "D_O_cm2s") and num(r, "D_O_cm2s") < 2e-5)): f.append("FROZE")
    if (r.get("dmin_HH") or 9) < 0.25: f.append("COLLAPSE")
    elif num(r, "dmin_HH") < 0.4 or (r.get("frac_HH_below_0p5") or 0) > 0.2: f.append("CLOSE-HH")
    if abs(r["T_mean"] - T) > 0.05 * T: f.append("T-off")
    if r.get("P_block_sd") and abs(r.get("P_drift_GPa") or 0) > 3 * r["P_block_sd"] and abs(r.get("P_drift_GPa") or 0) > 0.01 * abs(r["P_GPa"]): f.append("P-drift")
    return ",".join(f) or "ok"
for c in COMPS:
    rows = sorted([j for j in jobs if j["comp"] == c and j["res"]], key=lambda j: -j["vpa"]); fu = C[c]["fu"]
    print(f"\n{NAME[c]} at {T} K ({C[c]['nat']} atoms)   table: {os.path.join(C[c]['D'], C[c]['tag'] + '.csv')}")
    print(f"{'vpa':>6s} {'V/fu':>6s} {'rho':>6s} {'P GPa':>8s} {'+-':>5s} {'P_DFT':>7s} {'dP%':>6s} {'T K':>6s} {'D_O':>9s} {'D_H':>9s} {'O latt':>8s} {'dHH':>5s} {'dOH':>5s} {'HH<0.5':>6s} {'ms/st':>5s}  flags")
    for j in rows:
        r = j["res"]; pd_ = p_dft(c, fu * j["vpa"]); rho = C[c]["mpa"] / j["vpa"] / 0.60221
        print(f"{j['vpa']:6.3f} {fu * j['vpa']:6.2f} {rho:6.3f} {r['P_GPa']:8.1f} {r['P_sem']:5.1f} {pd_:7.1f} {100 * (r['P_GPa'] - pd_) / pd_:+6.1f} {r['T_mean']:6.0f} "
              f"{num(r, 'D_O_cm2s'):9.1e} {num(r, 'D_H_cm2s'):9.1e} {str(r.get('O_struct') or ''):>8s} {num(r, 'dmin_HH'):5.2f} {num(r, 'dmin_OH'):5.2f} {100 * num(r, 'frac_HH_below_0p5'):5.1f}% {r['ms_per_step']:5.0f}  {flags(r, c)}")
    for j in [j for j in jobs if j["comp"] == c and j["state"] == "failed"]: print(f"  FAILED vpa {j['vpa']}: {j['st']['stage']}  (log {j['log']})")
print("\nflags: FROZE (O lattice label bcc/fcc/hcp or D_O < 2e-5 cm^2/s: not a liquid point; liquids here have D_O >= ~2e-4), COLLAPSE (H-H pair < 0.25 A: GRACE short-range hole, invalid), CLOSE-HH (closest H-H < 0.4 A, or > 20 % of frames with an H-H pair < 0.5 A; a few frames at 0.45-0.5 A are normal thermal approaches at 7000 K),"
      " T-off (mean T > 5 % off target), P-drift (not equilibrated); D in cm^2/s, contacts in A, HH<0.5 = % of sampled frames with an H-H pair closer than 0.5 A")
by = {}
for j in jobs:
    if j["res"]: by.setdefault(j["gpu"], []).append(C[j["comp"]]["nat"] * 1000.0 / j["res"]["ms_per_step"])
print("MD speed per run (atom-steps/s, mean over the runs on each GPU, with PER_GPU sharing it): " + ", ".join(f"GPU{g}: {sum(v) / len(v) / 1000:.1f}k x {PER}" for g, v in sorted(by.items())))
print(f"all compositions on one figure: python3 mlp_pipeline/plot_eos_all.py --T {T} --model {mname}")
EOF
