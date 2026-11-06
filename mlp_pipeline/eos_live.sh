#!/bin/bash
# Quick liquid EOS scan with a live dashboard: one 17_eos_md.py process per volume, PER_GPU processes per GPU, all at once. Interactive GPU node, repo root.
#   bash mlp_pipeline/eos_live.sh                                  H2O liquid, 7000 K, 16 volumes 10 -> 1.75 A^3/atom (~10-1000 GPa), 384 atoms (bcc x4 = 128 H2O), 0.25 ps premelt + 0.25 equil + 0.75 ps
#   MODEL=grace-ft:runs/grace/ft_<run> T=7000 NV=16 VMAX=10 VMIN=1.75 PS=0.75 EQ=0.25 PER_GPU=4 NREP=4 bash mlp_pipeline/eos_live.sh
# COMP=h: pure-H fluid instead (729 atoms, no premelt, 8 -> 0.9 A^3/atom, equil 0.2 + prod 0.5 ps by default): COMP=h bash mlp_pipeline/eos_live.sh
# Reference: the DFT H2O (or pure-H) EOS at the same T from data-h3o-convex-hull.csv (columns "P <T>K h2o", "V <T>K h2o norm" = V per H2O), shown as + in the live plot, interpolated in the final table.
# Dashboard (refreshed in place every REFRESH s, no screen clear): per volume a progress bar, stage, running T and P (+- block error), ms/step, ETA; per GPU utilisation and memory;
# below it a plain-text V(P) plot (V per H2O vs log P) that fills in as the runs progress (* running average, o finished liquid, x finished but the O sublattice froze, + DFT).
# At the end also <tag>.png (model vs DFT, matplotlib).
# At the end: P(V) table with T, D_O / D_H (cm^2/s; D_O ~ 1e-5 = liquid, ~0 = O lattice survived), O-sublattice label, and the measured MD throughput per GPU.
# Outputs: runs/eos_live/<tag>/  (logs, 17_eos_md folders, <tag>.csv). A volume that is already in <tag>.csv is not rerun.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
if [ -z "${MODEL:-}" ]; then for r in mfin_f_lr1e-2_e50 mfin_f_lr1e-2_e50_F200; do [ -e runs/grace/ft_$r/seed/1/final_model/saved_model.pb ] && { MODEL=grace-ft:runs/grace/ft_$r; break; }; done; fi
[ -n "${MODEL:-}" ] || { echo "set MODEL=grace-ft:runs/grace/ft_<run>"; exit 1; }
COMP=${COMP:-h2o}; T=${T:-7000}; NV=${NV:-16}; PM_T=${PM_T:-9000}; NREP=${NREP:-4}
if [ "$COMP" = h ]; then VMAX=${VMAX:-8}; VMIN=${VMIN:-0.9}; PS=${PS:-0.5}; EQ=${EQ:-0.2}; PM_PS=0; NAT=${NATOMS:-729}; FU=1
  SYS="--phase H --branch fluid --natoms $NAT"; DESC="pure H fluid, $NAT atoms (bcc start that melts)"
else VMAX=${VMAX:-10}; VMIN=${VMIN:-1.75}; PS=${PS:-0.75}; EQ=${EQ:-0.25}; PM_PS=${PM_PS:-0.25}; NAT=$((6 * NREP ** 3)); FU=3
  SYS="--phase bcc --branch liquid --nrep $NREP --premelt-T $PM_T --premelt-ps $PM_PS"; DESC="H2O liquid, $NAT atoms (bcc x$NREP start, premelt $PM_T K ${PM_PS} ps)"; fi
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || { echo "no GPU visible"; exit 1; }; PER=${PER_GPU:-4}
TAG=${TAG:-$(basename ${MODEL#*:})_${COMP}_${T}K_${NAT}at}; D=runs/eos_live/$TAG; mkdir -p $D
VPA=$($PY -c "import numpy as np; print(' '.join(f'{v:.4f}' for v in np.geomspace($VMAX, $VMIN, $NV)))")
echo "model $MODEL | $DESC | $T K | $NV volumes $VMAX -> $VMIN A^3/atom | equil $EQ ps, prod $PS ps, dt 0.5 fs | $NG GPU x $PER per GPU"
pids=(); k=0; slot=0
for v in $VPA; do
  if [ -e $D/$TAG.csv ] && grep -q ",$v," $D/$TAG.csv 2>/dev/null; then echo "  vpa $v already done"; k=$((k + 1)); continue; fi
  g=$((slot % NG)); slot=$((slot + 1))
  CUDA_VISIBLE_DEVICES=$g nohup $PY mlp_pipeline/17_eos_md.py $MODEL $SYS --T $T --vpa $v \
      --equil-ps $EQ --ps $PS --report-fs 50 --tag $TAG --outdir $D > $D/log_$k.txt 2>&1 &
  pids+=($!); echo "$k $g $v $!" >> $D/runs.txt; k=$((k + 1))
  [ $((slot % (NG * PER))) -eq 0 ] && [ $slot -lt $NV ] && echo "  (more volumes than GPU slots: the rest start now too, PER_GPU is a soft limit here)"
done
trap 'echo; echo "interrupted: killing the MD processes"; kill ${pids[*]} 2>/dev/null; exit 1' INT
$PY - "$D" "$TAG" "${REFRESH:-10}" "$EQ" "$PS" "$PM_PS" "$NG" "$T" "$COMP" "$FU" ${pids[*]} <<'EOF'
import ast, csv, math, os, re, subprocess, sys, time
D, TAG, REF, EQ, PS, PM, NG, TT, COMP, FU = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(float(sys.argv[8])), sys.argv[9], int(sys.argv[10]); pids = [int(x) for x in sys.argv[11:]]
NAME = {"h2o": "H2O", "h": "H"}.get(COMP, COMP); MPA = {"h2o": 18.015 / 3, "h": 1.008}.get(COMP, 1.0)      # formula name, molar mass per atom
def dft_eos(T):
    """(P GPa, V per formula unit A^3) of the DFT EOS of COMP at T from data-h3o-convex-hull.csv, sorted by P; [] if that T is not in the file"""
    try: rows = list(csv.reader(open("data-h3o-convex-hull.csv")))
    except OSError: return []
    h = rows[0]
    try: iP, iV = h.index(f"P {T}K {COMP}"), h.index(f"V {T}K {COMP} norm")
    except ValueError: return []
    out = []
    for r in rows[1:]:
        try: out.append((float(r[iP]), float(r[iV])))
        except (ValueError, IndexError): pass
    return sorted(out)
DFT = dft_eos(TT)
def p_dft(vfu):
    """DFT pressure at V per formula unit by linear interpolation of log P in V (inside the DFT range only)"""
    pts = sorted(DFT, key=lambda x: x[1])
    for (P1, V1), (P2, V2) in zip(pts, pts[1:]):
        if V1 <= vfu <= V2 and V2 > V1: return math.exp(math.log(P1) + (math.log(P2) - math.log(P1)) * (vfu - V1) / (V2 - V1))
    return float("nan")
runs = [l.split() for l in open(os.path.join(D, "runs.txt")) if l.strip()]
runs = {int(r[0]): dict(gpu=r[1], vpa=float(r[2]), pid=int(r[3])) for r in runs if int(r[3]) in pids}
STEP = re.compile(r"step (\d+)/(\d+)\s+\[(.*?), (\d+) pts\]\s+T (\d+) K\s+P ([-\d.]+) \+- ([-\w.]+) GPa\s+\(now T (\d+) P ([-\d.]+)\)\s+(\d+) ms/step")
t0 = time.time(); lines_prev = 0; res = {}
def alive(p):
    try: os.kill(p, 0); return True
    except OSError: return False
def state(k):
    f = os.path.join(D, f"log_{k}.txt"); txt = open(f, errors="ignore").read() if os.path.exists(f) else ""
    st = dict(stage="starting (TF/model load)", frac=0.0, T="", P="", Pe="", ms="", eta="")
    if "premelt" in txt: st["stage"] = "premelt"
    if "premelt done" in txt: st["stage"] = "equil"; st["frac"] = PM / (PM + EQ + PS)
    m = STEP.findall(txt)
    if m:
        s, n, ph, npts, T, P, Pe, Tn, Pn, ms = m[-1]; s, n = int(s), int(n)
        st.update(stage="prod" if ph.startswith("prod") else "equil", frac=(PM + (EQ + PS) * s / n) / (PM + EQ + PS), T=T, P=P, Pe=Pe, ms=ms,
                  eta=f"{(n - s) * int(ms) / 1000:.0f}s")
    if "RESULT" in txt:
        st["stage"] = "DONE"; st["frac"] = 1.0
        r = txt.split("RESULT", 1)[1].strip().splitlines()[0]
        r = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", r)            # numpy 2 prints np.float64(0.1)
        try: res[k] = ast.literal_eval(re.sub(r"\bnan\b", "None", r))
        except Exception: pass
    elif ("Traceback" in txt or "Error" in txt) and not alive(runs[k]["pid"]): st["stage"] = "FAILED: " + [l for l in txt.splitlines() if "Error" in l][-1][:60]
    elif not alive(runs[k]["pid"]) and "RESULT" not in txt: st["stage"] = "DIED (see log)"
    return st
def pv_plot(pts, W=72, H=20, plo=5.0, phi=2000.0):
    """plain-text V(P) plot: x = P (log, plo..phi GPa), y = V per H2O (linear); pts = (V_fu, P, mark): o finished liquid, x finished but frozen, * running, + DFT"""
    allv = [v for v, P, m in pts] + [FU * r["vpa"] for r in runs.values()]; ylo, yhi = min(allv) * 0.95, max(allv) * 1.02
    g = [[" "] * W for _ in range(H)]
    for v, P, m in sorted(pts, key=lambda q: q[2] != "+"):          # DFT first, model marks drawn over it
        if P is None or P != P: continue
        c = int(round((math.log10(min(max(P, plo), phi)) - math.log10(plo)) / (math.log10(phi) - math.log10(plo)) * (W - 1)))
        r = H - 1 - int(round((v - ylo) / (yhi - ylo) * (H - 1)))
        if 0 <= r < H and 0 <= c < W: g[r][c] = m
    out = [f"V per {NAME} (A^3) vs P (GPa, log):  o model liquid   x model, O froze   * model running   + DFT (data-h3o-convex-hull.csv, same T)"]
    for r in range(H):
        yv = yhi - (yhi - ylo) * r / (H - 1); out.append(f"{yv:6.1f} |" + "".join(g[r]) if r % 4 == 0 else "       |" + "".join(g[r]))
    out.append(" " * 7 + "+" + "-" * W)
    lab = [" "] * (W + 6)
    for t in (10, 30, 100, 300, 1000):
        c = int(round((math.log10(t) - math.log10(plo)) / (math.log10(phi) - math.log10(plo)) * (W - 1))); lab[c:c + len(str(t))] = list(str(t))
    out.append(" " * 8 + "".join(lab) + " GPa")
    return out
def gpus():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return "  ".join(f"GPU{a.strip()} {b.strip():>3s}% {int(c) / 1024:.1f}/{int(d) / 1024:.0f}G" for a, b, c, d in (l.split(",") for l in o.strip().splitlines()))
    except Exception: return "(nvidia-smi unavailable)"
while True:
    out = [f"{TAG}   elapsed {time.time() - t0:5.0f} s   {gpus()}",
           f"{'k':>2s} {'gpu':>3s} {'vpa':>6s}  {'progress':22s} {'stage':10s} {'T K':>6s} {'P GPa':>8s} {'+-':>5s} {'ms/st':>5s} {'ETA':>6s}"]
    done = 0; pts = [(V, P, "+") for P, V in DFT]
    for k in sorted(runs):
        st = state(k)
        if k in res: r = res[k]; pts.append((FU * r["vpa"], r["P_GPa"], "x" if COMP != "h" and (r.get("D_O_cm2s") or 0) < 2e-6 else "o"))
        elif st["P"]: pts.append((FU * runs[k]["vpa"], float(st["P"]), "*"))
        done += st["stage"] in ("DONE",) or st["stage"].startswith(("FAILED", "DIED"))
        bar = "#" * int(20 * st["frac"]) + "." * (20 - int(20 * st["frac"]))
        out.append(f"{k:2d} {runs[k]['gpu']:>3s} {runs[k]['vpa']:6.3f}  [{bar}] {st['stage'][:10]:10s} {st['T']:>6s} {st['P']:>8s} {st['Pe']:>5s} {st['ms']:>5s} {st['eta']:>6s}"
                   + (f"   {st['stage'][10:]}" if len(st["stage"]) > 10 else ""))
    out += [""] + pv_plot(pts)
    if lines_prev: sys.stdout.write(f"\x1b[{lines_prev}A")
    sys.stdout.write("\n".join(l + "\x1b[K" for l in out) + "\n"); sys.stdout.flush(); lines_prev = len(out)
    if done == len(runs): break
    time.sleep(REF)
print(f"\nall {len(runs)} runs finished in {time.time() - t0:.0f} s\n")
rows = sorted(res.values(), key=lambda r: -r["vpa"])
print(f"{'vpa':>6s} {'V/fu':>6s} {'rho g/cc':>8s} {'P GPa':>8s} {'+-':>5s} {'P_DFT':>7s} {'dP %':>6s} {'T K':>6s} {'D_O cm2/s':>10s} {'D_H cm2/s':>10s} {'O lattice':>10s} {'dL':>5s} {'ms/step':>7s}")
for r in rows:
    rho = MPA / r["vpa"] / 0.60221; pd_ = p_dft(FU * r["vpa"])
    print(f"{r['vpa']:6.3f} {FU * r['vpa']:6.2f} {rho:8.3f} {r['P_GPa']:8.1f} {r['P_sem']:5.1f} {pd_:7.1f} {100 * (r['P_GPa'] - pd_) / pd_:+6.1f} {r['T_mean']:6.0f} "
          f"{r.get('D_O_cm2s') or float('nan'):10.2e} {r.get('D_H_cm2s') or float('nan'):10.2e} {str(r.get('O_struct', '')):>10s} {r.get('delta_L') or float('nan'):5.2f} {r['ms_per_step']:7.0f}")
print(f"P_DFT: DFT {NAME} EOS at {TT} K (data-h3o-convex-hull.csv: {len(DFT)} points, {min(p for p, v in DFT):.0f}-{max(p for p, v in DFT):.0f} GPa) interpolated at the same V; nan outside its range" if DFT else f"(no DFT {NAME} EOS at {TT} K in data-h3o-convex-hull.csv)")
if COMP != "h": print("D_O ~1e-5 cm2/s or more: liquid; D_O ~0 with a lattice label: the liquid froze into a solid (O_struct / dL Lindemann ratio) at this volume -> not a liquid EOS point")
by = {}
for k, r in res.items(): by.setdefault(runs[k]["gpu"], []).append(1000.0 / r["ms_per_step"])
print("MD throughput (steps/s, sum over the processes that shared a GPU): " + ", ".join(f"GPU{g}: {sum(v):.1f} ({len(v)} procs, {sum(v) / len(v):.1f} each)" for g, v in sorted(by.items())))
print("for comparison (2026-10-31, 324 atoms, ft_m2): 1/2/4/8 procs per GPU = 5.8/11.5/21/20 steps/s per GPU")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if DFT: ax.plot([p for p, v in DFT], [v for p, v in DFT], "s", color="#e34948", mec="k", label=f"DFT {NAME} {TT} K")
    liq = [r for r in rows if COMP == "h" or (r.get("D_O_cm2s") or 0) >= 2e-6]; fro = [r for r in rows if COMP != "h" and (r.get("D_O_cm2s") or 0) < 2e-6]
    ax.errorbar([r["P_GPa"] for r in liq], [FU * r["vpa"] for r in liq], xerr=[r["P_sem"] for r in liq], fmt="o-", color="#2a78d6", label=f"{TAG.split('_h2o')[0]} (liquid)")
    if fro: ax.plot([r["P_GPa"] for r in fro], [FU * r["vpa"] for r in fro], "x", color="#2a78d6", ms=9, label="model, O froze")
    ax.set_xscale("log"); ax.set_xlabel("P (GPa)"); ax.set_ylabel(f"V per {NAME} (A^3)"); ax.set_title(f"{NAME} fluid EOS at {TT} K", loc="left"); ax.grid(True, color="#eee"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(D, TAG + ".png"), dpi=140); print("figure:", os.path.join(D, TAG + ".png"))
except Exception as e: print("(no figure:", e, ")")
print(f"table: {os.path.join(D, TAG + '.csv')}   per-volume logs: {D}/log_<k>.txt")
EOF
