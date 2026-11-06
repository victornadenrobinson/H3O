#!/bin/bash
# TDI throughput on Perlmutter: a DIFFERENT number of lambda processes on each GPU, all at the same time (one interactive 4-GPU node, ~5-10 min), with the per-lambda engine of
# run_real.py (tdi/bench.py engine: one LambdaRun per process, the real MD step incl. the Einstein/reference terms; the lambda value only changes the mixture, not the cost).
#   bash mlp_pipeline/tdi_bench_layout.sh                              fcc H2O crystal (108 O, 324 atoms), layout "2 4 6 8" processes on GPU0..3
#   SYSTEM=H LAYOUT="1 2 4 6" bash mlp_pipeline/tdi_bench_layout.sh     pure-H fluid (250 atoms); SYSTEM = h2o (fcc crystal) | H | ins (1 H into fcc H2O) | insH
# env: MODEL (grace-ft:runs/grace/ft_<run>; default the F10 control), SYSTEM (h2o), LAYOUT ("2 4 6 8": processes per GPU, one entry per GPU), STEPS (200 timed steps per process), WARM (20),
#      VPA (volume per atom; default per system: h2o 2.2 ~ 400 GPa 3000 K, H 1.3, ins 2.2) or FROM_DFT=<inventory csv> to take it from the DFT EOS, PY.
# Output: live progress (processes ready / timed / done per GPU, GPU memory and utilisation), then a table per GPU: processes, steps/s per process, steps/s per GPU, peak GPU memory, and
# the wall time of a full TDI (NODES lambda values x (EQUIL + PS) ps at dt 0.5 fs) if all lambda values of one TDI run share one GPU with that many processes at a time.
# Logs: runs/bench_layout/<time>/gpu<g>_p<i>.log (one RESULT json line each).
set -u; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
if [ -z "${MODEL:-}" ]; then for r in mfin_f_lr1e-2_e50 mfin_f_lr1e-2_e50_F200; do [ -e runs/grace/ft_$r/seed/1/final_model/saved_model.pb ] && { MODEL=grace-ft:runs/grace/ft_$r; break; }; done; fi
SYSTEM=${SYSTEM:-h2o}; LAYOUT=(${LAYOUT:-2 4 6 8}); STEPS=${STEPS:-200}; WARM=${WARM:-20}
case $SYSTEM in h2o|ins) V=${VPA:-2.2};; H|insH) V=${VPA:-1.3};; *) echo "SYSTEM = h2o | H | ins | insH"; exit 1;; esac
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge ${#LAYOUT[@]} ] || { echo "LAYOUT has ${#LAYOUT[@]} entries but only $NG GPUs are visible"; exit 1; }
D=runs/bench_layout/$(date +%Y%m%d_%H%M%S); mkdir -p $D; B=$D/GO; VOL=${FROM_DFT:+--from-dft $FROM_DFT}; VOL=${VOL:---vpa $V}
echo "TDI layout benchmark: model ${MODEL:-?}, system $SYSTEM ($VOL), layout ${LAYOUT[*]} processes on GPU 0..$((${#LAYOUT[@]} - 1)), $STEPS timed steps each -> $D"
pids=(); n=0
for g in "${!LAYOUT[@]}"; do
  for ((i = 0; i < ${LAYOUT[$g]}; i++)); do
    lam=$(python3 -c "print(round(($i + 0.5) / ${LAYOUT[$g]}, 3))")
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 nohup $PY mlp_pipeline/tdi/bench.py engine $SYSTEM --model "$MODEL" $VOL \
      --steps $STEPS --warm $WARM --lam $lam --barrier $B --workdir $D/work_g${g}_p$i > $D/gpu${g}_p$i.log 2>&1 &
    pids+=($!); n=$((n + 1))
  done
done
trap 'kill ${pids[*]} 2>/dev/null; echo; echo interrupted; exit 1' INT
prev=0; t0=$(date +%s); go=0
while :; do
  ready=$(grep -l "^READY" $D/gpu*_p*.log 2>/dev/null | wc -l); done_=$(grep -l "^RESULT" $D/gpu*_p*.log 2>/dev/null | wc -l)
  dead=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null || dead=$((dead + 1)); done
  failed=$((dead - done_)); [ $failed -lt 0 ] && failed=0
  if [ $go = 0 ] && [ $((ready + failed)) -ge $n ]; then touch $B; go=1; tg=$(date +%s); fi           # every process compiled and warmed up: start the timed part together
  out="elapsed $(( $(date +%s) - t0 )) s   ready $ready/$n   timed-start $([ $go = 1 ] && echo yes || echo waiting)   done $done_/$n   failed $failed"
  smi=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '{printf "GPU%d %3d%% %5.1fG   ", $1, $2, $3/1024}')
  [ $prev -gt 0 ] && printf "\033[%dA" $prev; printf "%s\033[K\n%s\033[K\n" "$out" "$smi"; prev=2
  [ $dead -ge $n ] && break; sleep 3
done
for f in $D/gpu*_p*.log; do grep -q "^RESULT" $f || { echo "FAILED $f:"; grep -E "Error|error" $f | tail -n 2; }; done
$PY - "$D" "${LAYOUT[*]}" <<'EOF'
import glob, json, math, os, re, sys
D, lay = sys.argv[1], [int(x) for x in sys.argv[2].split()]
res = {}
for f in glob.glob(os.path.join(D, "gpu*_p*.log")):
    g = int(re.search(r"gpu(\d+)_p", f).group(1))
    for l in open(f):
        if l.startswith("RESULT"): res.setdefault(g, []).append(json.loads(l[7:]))
if not res: sys.exit("no results")
nat = next(iter(res.values()))[0]["natoms"]
print(f"\n{'GPU':>3s} {'procs':>5s} {'ok':>3s} {'steps/s per proc':>17s} {'steps/s per GPU':>16s} {'atom-steps/s':>13s} {'peak TF MiB/proc':>17s}")
rate = {}
for g in sorted(res):
    r = res[g]; per = sum(x["steps_per_s"] for x in r) / len(r); tot = sum(x["steps_per_s"] for x in r); rate[g] = (lay[g], per, tot)
    mem = max((x.get("tensorflow_peak_MiB") or 0) for x in r)
    print(f"{g:3d} {lay[g]:5d} {len(r):3d} {per:17.2f} {tot:16.2f} {tot * nat:13.0f} {mem:17.0f}")
print(f"\nfull TDI wall time if one run's lambda values share ONE GPU, NPAR = processes at a time (steps = (EQUIL + PS) ps / 0.5 fs per lambda; {nat} atoms):")
print(f"{'NPAR':>4s}   " + "   ".join(f"{n:2d} lam x {ps:g} ps" for n in (10, 17, 21) for ps in (2, 4)))
for g in sorted(rate):
    npar, per, tot = rate[g]
    row = []
    for n in (10, 17, 21):
        for ps in (2, 4):
            steps = (1 + ps) * 2000; waves = math.ceil(n / npar); row.append(f"{waves * steps / per / 3600:13.1f} h")
    print(f"{npar:4d}   " + "   ".join(row))
best = max(rate.values(), key=lambda x: x[2]); print(f"\nbest throughput: {best[0]} processes per GPU ({best[2]:.1f} steps/s per GPU, {best[1]:.2f} per process). "
      "Lambda values are independent, so one GPU at the best NPAR is the cheapest; a TDI with fewer lambda values than NPAR wastes slots (fill them with other runs).")
print("for comparison (Young, ft_m2): insH 4 procs 11.5 steps/s per GPU, pure H 250 22.7 steps/s per GPU, saturation at 4 processes")
EOF
