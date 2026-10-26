#!/bin/bash
# TDI for one phase with the lambda nodes run as separate processes sharing ONE GPU (the A100 is only ~20-30% busy with one 384-atom MD).
# usage in an interactive GPU job (repo root on Young):
#    bash mlp_pipeline/young_tdi_par.sh bcc           # real run: 12 nodes, 4 ps each, 4 processes at a time
#    bash mlp_pipeline/young_tdi_par.sh fcc
#    T=5000 bash mlp_pipeline/young_tdi_par.sh H          # pure hydrogen fluid (ideal-gas reference, no O): G of H at the same P, T
#    quick test (a few minutes):  TAG=test NODES=3 PS=0.5 PS1=0.5 CHUNK=0.25 EQUIL=0.2 NPAR=3 bash mlp_pipeline/young_tdi_par.sh bcc
# K: Einstein constant (eV/A^2); default bcc 12.2, fcc 22.8 (measured at 400 GPa, 3000 K): stage 0 then skips its MD. K=0 re-measures it with a lambda=1 run (also checks MLP pressure and O structure).
# NREP=2 (fcc: 32 O, 96 atoms) / NREP=3 (bcc: 54 O, 162 atoms): small cells like Wilson 2013 for a finite-size check; tag gets _rep<NREP> (defaults: fcc 3 = 108 O, bcc 4 = 128 O)
# NATOMS=729: pure-H cell size (sc 9^3; default bcc 250)
# insert mode: NGHOST=1,2,3,.. protons added to fcc H2O (ghosts at lambda=0): mu_H in the crystal; tag ..._n<NGHOST>
# ONLYNODES="0 3": run just these nodes (e.g. after a GPU out-of-memory failure of one node); large cells (>400 atoms) need NPAR=8 or less on a 40 GB card
# env knobs: HPER (H per O, 2; HPER=2.25 for H2.25O: tag gets _h2.25o) T (3000) P (400) MODEL NODES (12) DT (0.5 fs) ONLY0 POWER (4) PS (4) PS1 (1.5) CHUNK (0.5) EQUIL (1) NPAR (4) TAG
# Logs: runs/tdi/<tag>_stage0.log, runs/tdi/<tag>_node<k>.log (every line is a full health report); watch the integral from any shell with
#    watch -n 120 python3 mlp_pipeline/tdi/collect.py runs/tdi <tag>
PH=${1:?bcc, fcc, liquid, insert or H}; P=${P:-400}; T=${T:-3000}; HPER=${HPER:-2}; NGHOST=${NGHOST:-1}; MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; NODES=${NODES:-12}; POWER=${POWER:-4}; PS=${PS:-4}; PS1=${PS1:-1.5}
CHUNK=${CHUNK:-0.5}; EQUIL=${EQUIL:-1}; DT=${DT:-0.5}; NPAR=${NPAR:-4}; OUT=runs/tdi; [ -n "$TAG" ] && OUT=runs/tdi_test
TAGNAME=${PH}_${P}GPa_${T}K; { [ "$HPER" != 2 ] || [ $PH = liquid ]; } && [ $PH != H ] && TAGNAME=${TAGNAME}_h${HPER}o; [ $PH = H ] && [ -n "$NATOMS" ] && TAGNAME=${TAGNAME}_N${NATOMS}; [ $PH = insert ] && TAGNAME=${TAGNAME}_n${NGHOST}; { [ $PH = bcc ] || [ $PH = fcc ]; } && [ -n "$NREP" ] && TAGNAME=${TAGNAME}_rep${NREP}
# default Einstein constants measured in earlier 3 ps lambda=1 runs at 400 GPa, 3000 K (bcc 12.182, fcc 22.842); K=0 runs the lambda=1 stage instead (re-measures k, pressure, structure)
# Einstein constant k (eV/A^2): scaled guess unless K is given (k only sets the efficiency, not the result): measured 12.2 (bcc) / 22.0-22.8 (fcc) at 400 GPa 3000 K, 12.8 (bcc) at 5000 K, 7.9 (bcc) at 200 GPa; k ~ P^0.6.
# K=0 measures it with a lambda=1 run instead (about 6 min; also prints MLP pressure and O structure).
if [ -z "$K" ] && [ "$PH" != H ]; then K=$(awk -v p=$P -v k4=$([ $PH = bcc ] && echo 12.5 || echo 22.5) 'BEGIN{printf "%.2f", k4*(p/400)^0.6}'); fi; [ "$K" = 0 ] && K=
mkdir -p $OUT
export TF_FORCE_GPU_ALLOW_GROWTH=true; [ -n "$UQ" ] && export GRACE_UQ=1   # UQ=1: per-atom extrapolation grade (cost not measured); use tdi/uq_check.py on saved snapshots instead
PY=${PY:-$HOME/grace-venv/bin/python}
SRC="--from-dft inventory_binned.csv ${NREP:+--nrep $NREP}"; [ -n "$VPA" ] && SRC="--vpa $VPA --nrep ${NREP:-2}"   # VPA/NREP: harness test without the inventory
COMMON="$MODEL --phase $PH --P $P --T $T --hper $HPER --nghost $NGHOST ${NATOMS:+--natoms $NATOMS} $SRC --ps $PS --ps1 $PS1 --nodes $NODES --power $POWER --dt $DT --chunk-ps $CHUNK --equil-ps $EQUIL --outdir $OUT${K:+ --k $K}"
NOISE="tensorflow|XLA|cuda|absl|computation_placer|_handle_getgr|E0000|W0000|I0000|ptxas|^$"
# restartable: rerunning the same command continues every node from its last finished chunk (state in $OUT/*_node<k>_state.npz); FRESH=1 starts over
if [ -n "$FRESH" ]; then rm -f $OUT/${TAGNAME}_*; FRESHFLAG=--fresh; fi
echo "== stage 0 (lambda = 1 run) $(date)"
[ -z "$K" ] && [ -f $OUT/${TAGNAME}_stage0.json ] && echo "stage 0 already done (measured k kept; FRESH=1 to redo)" || $PY -u mlp_pipeline/tdi/run_real.py $COMMON --stage0 2>&1 | tee $OUT/${TAGNAME}_stage0.log | grep --line-buffered -vE "$NOISE"
[ -f $OUT/${TAGNAME}_stage0.json ] || { echo "stage 0 failed"; exit 1; }
[ -n "$ONLY0" ] && exit 0   # ONLY0=1: stop after the lambda=1 run (e.g. to compare DT=0.5 and DT=1.0)
echo "== lambda 0..$((NODES-1)), $NPAR at a time $(date)"
# node order when NPAR < NODES: middle and both ends first, then the largest gaps (a coarse integral exists early; tdi/live.py PREVIEW uses whatever is there)
if [ -n "$ONLYNODES" ]; then echo $ONLYNODES | tr ' ' '\n'; else
python3 -c "
n=$NODES; o=[i for i in (n//2, n-1, 0) if 0<=i<n]
o=list(dict.fromkeys(o))
while len(o)<n:
    o.append(max((i for i in range(n) if i not in o), key=lambda i: min(abs(i-j) for j in o)))
print(*o, sep=chr(10))"
fi | xargs -P $NPAR -I{} bash -c "$PY -u mlp_pipeline/tdi/run_real.py $COMMON --node {} $FRESHFLAG >> $OUT/${TAGNAME}_node{}.log 2>&1" &
XP=$!
while kill -0 $XP 2>/dev/null; do
  sleep 120; echo "-- $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  for f in $OUT/${TAGNAME}_node*.log; do echo "   $(basename $f .log | sed "s/${TAGNAME}_node/lambda /"): $(grep -h ETA $f | tail -n1 | grep -o "[0-9]* ms/step.*")"; done
  python3 mlp_pipeline/tdi/live.py $OUT $TAGNAME --brief 2>&1 | grep -v "^$"
done
echo "== done $(date)"; python3 mlp_pipeline/tdi/collect.py $OUT $TAGNAME
