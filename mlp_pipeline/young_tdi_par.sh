#!/bin/bash
# TDI for one phase with the lambda nodes run as separate processes sharing ONE GPU (the A100 is only ~20-30% busy with one 384-atom MD).
# usage in an interactive GPU job (repo root on Young):
#    bash mlp_pipeline/young_tdi_par.sh bcc           # real run: 12 nodes, 4 ps each, 4 processes at a time
#    bash mlp_pipeline/young_tdi_par.sh fcc
#    quick test (a few minutes):  TAG=test NODES=3 PS=0.5 PS1=0.5 CHUNK=0.25 EQUIL=0.2 NPAR=3 bash mlp_pipeline/young_tdi_par.sh bcc
# env knobs: P (400) MODEL NODES (12) DT (0.5 fs) ONLY0 POWER (4) PS (4) PS1 (1.5) CHUNK (0.5) EQUIL (1) NPAR (4) TAG
# Logs: runs/tdi/<tag>_stage0.log, runs/tdi/<tag>_node<k>.log (every line is a full health report); watch the integral from any shell with
#    watch -n 120 python3 mlp_pipeline/tdi/collect.py runs/tdi <tag>
PH=${1:?bcc or fcc}; P=${P:-400}; MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; NODES=${NODES:-12}; POWER=${POWER:-4}; PS=${PS:-4}; PS1=${PS1:-1.5}
CHUNK=${CHUNK:-0.5}; EQUIL=${EQUIL:-1}; DT=${DT:-0.5}; NPAR=${NPAR:-4}; OUT=runs/tdi; [ -n "$TAG" ] && OUT=runs/tdi_test
TAGNAME=${PH}_${P}GPa_3000K
mkdir -p $OUT
export GRACE_UQ=1 TF_FORCE_GPU_ALLOW_GROWTH=true
PY=${PY:-$HOME/grace-venv/bin/python}
SRC="--from-dft inventory_binned.csv"; [ -n "$VPA" ] && SRC="--vpa $VPA --nrep ${NREP:-2}"   # VPA/NREP: harness test without the inventory
COMMON="$MODEL --phase $PH --P $P --T 3000 $SRC --ps $PS --ps1 $PS1 --nodes $NODES --power $POWER --dt $DT --chunk-ps $CHUNK --equil-ps $EQUIL --outdir $OUT"
NOISE="tensorflow|XLA|cuda|absl|computation_placer|_handle_getgr|E0000|W0000|I0000|ptxas|^$"
echo "== stage 0 (lambda = 1 run) $(date)"
$PY -u mlp_pipeline/tdi/run_real.py $COMMON --stage0 2>&1 | tee $OUT/${TAGNAME}_stage0.log | grep --line-buffered -vE "$NOISE"
[ -f $OUT/${TAGNAME}_stage0.json ] || { echo "stage 0 failed"; exit 1; }
[ -n "$ONLY0" ] && exit 0   # ONLY0=1: stop after the lambda=1 run (e.g. to compare DT=0.5 and DT=1.0)
echo "== nodes 0..$((NODES-1)), $NPAR at a time $(date)"
seq 0 $((NODES-1)) | xargs -P $NPAR -I{} bash -c "$PY -u mlp_pipeline/tdi/run_real.py $COMMON --node {} > $OUT/${TAGNAME}_node{}.log 2>&1" &
XP=$!
while kill -0 $XP 2>/dev/null; do
  sleep 120; echo "-- $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  python3 mlp_pipeline/tdi/collect.py $OUT $TAGNAME 2>&1 | grep -E "samples per node|t/node" | tail -3
done
echo "== done $(date)"; python3 mlp_pipeline/tdi/collect.py $OUT $TAGNAME
