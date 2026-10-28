#!/bin/bash -l
# Throughput benchmark of the TDI lambda engine on ONE GPU (task 2c of docs/CURRENT_ISSUES.md; results go into docs/PERFORMANCE.md):
#   per-lambda engine: 1, 2, 4, 8 processes at once (as young_tdi_par.sh NPAR), total MD steps/s, without and with NVIDIA MPS;
#   batched engine (tdi/batched.py): NREPS lambda values in one process; model calls: one batched call of B structures vs B single calls.
# Fixed systems (tdi/bench.py): h2o = fcc H2O 108 O (324 atoms), H = pure H 250, insH = insertion of 1 H into pure H (251 atoms, 2 calls/step), ins = 1 H into fcc H2O (325 atoms).
# usage on Young, repo root, inside a GPU job (interactive: srun ... --pty bash, then):
#    SLURM_SUBMIT_DIR=$PWD bash mlp_pipeline/bench_tdi.sh                           (everything: about 1.5 h)
#    SYSTEMS=insH NPROCS="1 8" MPS="0 1" PARTS="engine" bash mlp_pipeline/bench_tdi.sh (a subset)
#  or as a batch job:  sbatch mlp_pipeline/bench_tdi.sh
# knobs: SYSTEMS ("h2o H insH"), NPROCS ("1 2 4 8"), MPS ("0 1"), NREPS ("4 8 16"), SIZES ("1,2,4,8,16"), STEPS (100 timed steps per process), PARTS ("engine batch calls"), PY
# output: runs/bench/bench_<time>.jsonl (one RESULT per process / run, one SUM line per engine configuration) and the markdown table at the end (tdi/bench_table.py)
#SBATCH --job-name=tdibench
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm_%x_%j.out
main() {
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
[ -d mlp_pipeline ] || { echo "run from the repo root"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible: run inside a GPU job"; exit 1; }
PY=${PY:-$HOME/grace-venv/bin/python}; SYSTEMS=${SYSTEMS:-"h2o H insH"}; NPROCS=${NPROCS:-"1 2 4 8"}; MPSL=${MPS:-"0 1"}; NREPS=${NREPS:-"4 8 16"}; SIZES=${SIZES:-"1,2,4,8,16"}
STEPS=${STEPS:-100}; PARTS=${PARTS:-"engine batch calls"}
OUT=runs/bench; mkdir -p $OUT; RES=$OUT/bench_$(date +%Y%m%d_%H%M%S).jsonl; TMPB=${TMPDIR:-/tmp}/tdibench_$$; mkdir -p $TMPB
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
echo "== TDI throughput benchmark on $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), $(nproc) CPUs, code $(git rev-parse --short HEAD 2>/dev/null) -> $RES"
mps_on() {
  export CUDA_MPS_PIPE_DIRECTORY=$TMPB/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPB/mps_log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && sleep 2 && echo "   MPS daemon started" && return 0
  echo "   MPS daemon did not start (nvidia-cuda-mps-control -d failed): MPS rows skipped"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; return 1
}
mps_off() { [ -n "$CUDA_MPS_PIPE_DIRECTORY" ] || return 0; echo quit | nvidia-cuda-mps-control; sleep 3; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }
trap 'mps_off; rm -rf $TMPB' EXIT
gpumem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc 0-9; }
engine() {   # $1 system, $2 number of processes, $3 mps 0/1: start them, release the timed part together once every process is warmed up
  local s=$1 n=$2 m=$3 i mem=0 B=$TMPB/go_$1_$2_$3; rm -f $B
  for i in $(seq $n); do $PY -u mlp_pipeline/tdi/bench.py engine $s --steps $STEPS --barrier $B > $TMPB/e_${s}_${n}_${m}_$i.log 2>&1 & done
  local t0=$(date +%s)
  while [ $(cat $TMPB/e_${s}_${n}_${m}_*.log 2>/dev/null | grep -c "^READY") -lt $n ]; do
    sleep 2; [ $(jobs -r | wc -l) -lt $n ] && break; [ $(( $(date +%s) - t0 )) -gt 900 ] && break
  done
  touch $B; sleep 20; mem=$(gpumem); wait
  grep -h "^RESULT" $TMPB/e_${s}_${n}_${m}_*.log | sed 's/^RESULT //' >> $RES
  local nres=$(grep -h "^RESULT" $TMPB/e_${s}_${n}_${m}_*.log | wc -l)
  [ $nres -lt $n ] && { echo "   WARNING: only $nres of $n processes finished (tail of a failed log:)"; for f in $TMPB/e_${s}_${n}_${m}_*.log; do grep -q "^RESULT" $f || { tail -n 3 $f; break; }; done; }
  grep -h "^RESULT" $TMPB/e_${s}_${n}_${m}_*.log | sed 's/^RESULT //' | python3 -c "
import sys, json; r=[json.loads(l) for l in sys.stdin]
if r:
    tot=sum(x['steps_per_s'] for x in r); ms=sum(x['ms_per_step'] for x in r)/len(r)
    d=dict(mode='engine_sum', system='$s', natoms=r[0]['natoms'], nproc=$n, finished=len(r), mps=bool($m), total_steps_per_s=tot, ms_per_step_each=ms, gpu_mem_MiB=int('${mem:-0}' or 0))
    print(json.dumps(d), file=open('$RES','a')); print(f'   engine {d[\"system\"]:5s} {$n} proc{\" MPS\" if $m else \"    \"}: {tot:6.2f} steps/s in total, {ms:6.0f} ms/step each, GPU memory {d[\"gpu_mem_MiB\"]} MiB')"
}
for s in $SYSTEMS; do
  case " $PARTS " in *" engine "*)
    for m in $MPSL; do
      if [ "$m" = 1 ]; then mps_on || continue; fi
      for n in $NPROCS; do engine $s $n $m; done
      [ "$m" = 1 ] && mps_off
    done;;
  esac
  case " $PARTS " in *" batch "*)
    for r in $NREPS; do
      $PY -u mlp_pipeline/tdi/bench.py batch $s --nrep $r --steps $STEPS ${BATCHMAX:+--max-atoms $BATCHMAX} > $TMPB/b_${s}_$r.log 2>&1
      grep -h "^RESULT" $TMPB/b_${s}_$r.log | sed 's/^RESULT //' >> $RES
      grep -h "^RESULT" $TMPB/b_${s}_$r.log | sed 's/^RESULT //' | python3 -c "
import sys, json
for l in sys.stdin:
    x=json.loads(l); print(f'   batch  {x[\"system\"]:5s} {x[\"nrep\"]:2d} lambda in 1 process: {x[\"steps_per_s\"]:6.2f} steps/s in total, {x[\"ms_per_step\"]:6.0f} ms/step, {x[\"ms_per_call\"]:6.0f} ms/model call, model {100*x[\"model_frac\"]:.0f}% of the time')" || true
      grep -q "^RESULT" $TMPB/b_${s}_$r.log || { echo "   batch $s nrep $r FAILED:"; grep -vE "^(W0000|I0000|E0000)" $TMPB/b_${s}_$r.log | tail -n 4; }
    done;;
  esac
  case " $PARTS " in *" calls "*)
    $PY -u mlp_pipeline/tdi/bench.py calls $s --sizes $SIZES ${BATCHMAX:+--max-atoms $BATCHMAX} > $TMPB/c_$s.log 2>&1
    grep -h "self-check" $TMPB/c_$s.log | sed 's/^/   /'
    grep -h "^RESULT" $TMPB/c_$s.log | sed 's/^RESULT //' >> $RES
    grep -q "^RESULT" $TMPB/c_$s.log || { echo "   calls $s FAILED:"; grep -vE "^(W0000|I0000|E0000)" $TMPB/c_$s.log | tail -n 4; };;
  esac
done
echo "== done $(date): $RES"; python3 mlp_pipeline/tdi/bench_table.py $RES
}
main "$@"; exit $?
