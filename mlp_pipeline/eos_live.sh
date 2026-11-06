#!/bin/bash
# Quick liquid EOS scan with a live dashboard: one 17_eos_md.py process per volume, PER_GPU processes per GPU, all at once. Interactive GPU node, repo root.
#   bash mlp_pipeline/eos_live.sh                                  H2O liquid, 7000 K, 16 volumes 6.0 -> 1.75 A^3/atom (~10-1000 GPa), 162 atoms, 0.25 ps premelt + 0.25 equil + 0.75 ps
#   MODEL=grace-ft:runs/grace/ft_<run> T=7000 NV=16 VMAX=6.0 VMIN=1.75 PS=0.75 EQ=0.25 PER_GPU=4 NREP=3 bash mlp_pipeline/eos_live.sh
# Dashboard (refreshed in place every REFRESH s, no screen clear): per volume a progress bar, stage, running T and P (+- block error), ms/step, ETA; per GPU utilisation and memory.
# At the end: P(V) table with T, D_O / D_H (cm^2/s; D_O ~ 1e-5 = liquid, ~0 = O lattice survived), O-sublattice label, and the measured MD throughput per GPU.
# Outputs: runs/eos_live/<tag>/  (logs, 17_eos_md folders, <tag>.csv). A volume that is already in <tag>.csv is not rerun.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
if [ -z "${MODEL:-}" ]; then for r in mfin_f_lr1e-2_e50 mfin_f_lr1e-2_e50_F200; do [ -e runs/grace/ft_$r/seed/1/final_model/saved_model.pb ] && { MODEL=grace-ft:runs/grace/ft_$r; break; }; done; fi
[ -n "${MODEL:-}" ] || { echo "set MODEL=grace-ft:runs/grace/ft_<run>"; exit 1; }
T=${T:-7000}; NV=${NV:-16}; VMAX=${VMAX:-6.0}; VMIN=${VMIN:-1.75}; PS=${PS:-0.75}; EQ=${EQ:-0.25}; PM_T=${PM_T:-9000}; PM_PS=${PM_PS:-0.25}; NREP=${NREP:-3}
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || { echo "no GPU visible"; exit 1; }; PER=${PER_GPU:-4}
TAG=${TAG:-$(basename ${MODEL#*:})_h2o_liq_${T}K}; D=runs/eos_live/$TAG; mkdir -p $D
VPA=$($PY -c "import numpy as np; print(' '.join(f'{v:.4f}' for v in np.geomspace($VMAX, $VMIN, $NV)))")
echo "model $MODEL | H2O liquid $T K | $NV volumes $VMAX -> $VMIN A^3/atom | bcc x$NREP start ($((2 * NREP ** 3)) O) | premelt $PM_T K ${PM_PS} ps, equil $EQ ps, prod $PS ps, dt 0.5 fs | $NG GPU x $PER per GPU"
pids=(); k=0; slot=0
for v in $VPA; do
  if [ -e $D/$TAG.csv ] && grep -q ",$v," $D/$TAG.csv 2>/dev/null; then echo "  vpa $v already done"; k=$((k + 1)); continue; fi
  g=$((slot % NG)); slot=$((slot + 1))
  CUDA_VISIBLE_DEVICES=$g nohup $PY mlp_pipeline/17_eos_md.py $MODEL --phase bcc --branch liquid --T $T --nrep $NREP --vpa $v --premelt-T $PM_T --premelt-ps $PM_PS \
      --equil-ps $EQ --ps $PS --report-fs 50 --tag $TAG --outdir $D > $D/log_$k.txt 2>&1 &
  pids+=($!); echo "$k $g $v $!" >> $D/runs.txt; k=$((k + 1))
  [ $((slot % (NG * PER))) -eq 0 ] && [ $slot -lt $NV ] && echo "  (more volumes than GPU slots: the rest start now too, PER_GPU is a soft limit here)"
done
trap 'echo; echo "interrupted: killing the MD processes"; kill ${pids[*]} 2>/dev/null; exit 1' INT
$PY - "$D" "$TAG" "${REFRESH:-10}" "$EQ" "$PS" "$PM_PS" "$NG" ${pids[*]} <<'EOF'
import ast, os, re, subprocess, sys, time
D, TAG, REF, EQ, PS, PM, NG = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]); pids = [int(x) for x in sys.argv[8:]]
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
        try: res[k] = ast.literal_eval(re.sub(r"\bnan\b", "None", r))
        except Exception: pass
    elif ("Traceback" in txt or "Error" in txt) and not alive(runs[k]["pid"]): st["stage"] = "FAILED: " + [l for l in txt.splitlines() if "Error" in l][-1][:60]
    elif not alive(runs[k]["pid"]) and "RESULT" not in txt: st["stage"] = "DIED (see log)"
    return st
def gpus():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return "  ".join(f"GPU{a.strip()} {b.strip():>3s}% {int(c) / 1024:.1f}/{int(d) / 1024:.0f}G" for a, b, c, d in (l.split(",") for l in o.strip().splitlines()))
    except Exception: return "(nvidia-smi unavailable)"
while True:
    out = [f"{TAG}   elapsed {time.time() - t0:5.0f} s   {gpus()}",
           f"{'k':>2s} {'gpu':>3s} {'vpa':>6s}  {'progress':22s} {'stage':10s} {'T K':>6s} {'P GPa':>8s} {'+-':>5s} {'ms/st':>5s} {'ETA':>6s}"]
    done = 0
    for k in sorted(runs):
        st = state(k); done += st["stage"] in ("DONE",) or st["stage"].startswith(("FAILED", "DIED"))
        bar = "#" * int(20 * st["frac"]) + "." * (20 - int(20 * st["frac"]))
        out.append(f"{k:2d} {runs[k]['gpu']:>3s} {runs[k]['vpa']:6.3f}  [{bar}] {st['stage'][:10]:10s} {st['T']:>6s} {st['P']:>8s} {st['Pe']:>5s} {st['ms']:>5s} {st['eta']:>6s}"
                   + (f"   {st['stage'][10:]}" if len(st["stage"]) > 10 else ""))
    if lines_prev: sys.stdout.write(f"\x1b[{lines_prev}A")
    sys.stdout.write("\n".join(l + "\x1b[K" for l in out) + "\n"); sys.stdout.flush(); lines_prev = len(out)
    if done == len(runs): break
    time.sleep(REF)
print(f"\nall {len(runs)} runs finished in {time.time() - t0:.0f} s\n")
rows = sorted(res.values(), key=lambda r: -r["vpa"])
print(f"{'vpa':>6s} {'rho g/cc':>8s} {'P GPa':>8s} {'+-':>5s} {'T K':>6s} {'D_O cm2/s':>10s} {'D_H cm2/s':>10s} {'O lattice':>10s} {'dL':>5s} {'ms/step':>7s}")
for r in rows:
    rho = 18.015 / 3 / r["vpa"] / 0.60221
    print(f"{r['vpa']:6.3f} {rho:8.3f} {r['P_GPa']:8.1f} {r['P_sem']:5.1f} {r['T_mean']:6.0f} {r.get('D_O_cm2s') or float('nan'):10.2e} {r.get('D_H_cm2s') or float('nan'):10.2e} "
          f"{str(r.get('O_struct', '')):>10s} {r.get('delta_L') or float('nan'):5.2f} {r['ms_per_step']:7.0f}")
print("D_O ~1e-5 cm2/s or more: liquid; D_O ~0 with a lattice label: the liquid froze into a solid (O_struct / dL Lindemann ratio) at this volume -> not a liquid EOS point")
by = {}
for k, r in res.items(): by.setdefault(runs[k]["gpu"], []).append(1000.0 / r["ms_per_step"])
print("MD throughput (steps/s, sum over the processes that shared a GPU): " + ", ".join(f"GPU{g}: {sum(v):.1f} ({len(v)} procs, {sum(v) / len(v):.1f} each)" for g, v in sorted(by.items())))
print("for comparison (2026-10-31, 324 atoms, ft_m2): 1/2/4/8 procs per GPU = 5.8/11.5/21/20 steps/s per GPU")
print(f"table: {os.path.join(D, TAG + '.csv')}   per-volume logs: {D}/log_<k>.txt")
EOF
