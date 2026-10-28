#!/bin/bash
# TDI for one phase with the lambda nodes run as separate processes sharing ONE GPU (the A100 is only ~20-30% busy with one 384-atom MD).
# usage in an interactive GPU job (repo root on Young):
#    bash mlp_pipeline/young_tdi_par.sh bcc           # real run: 12 nodes, 4 ps each, 4 processes at a time
#    bash mlp_pipeline/young_tdi_par.sh fcc
#    T=5000 bash mlp_pipeline/young_tdi_par.sh H          # pure hydrogen fluid (ideal-gas reference, no O): G of H at the same P, T
#    quick test (a few minutes):  TAG=test NODES=3 PS=0.5 PS1=0.5 CHUNK=0.25 EQUIL=0.2 NPAR=3 bash mlp_pipeline/young_tdi_par.sh bcc
# K: Einstein constant (eV/A^2); default bcc 12.2, fcc 22.8 (measured at 400 GPa, 3000 K): stage 0 then skips its MD. K=0 re-measures it with a lambda=1 run (also checks MLP pressure and O structure).
# NREP also sets the liquid cell (liquid: default 2 = 32 O fcc start cell, NREP=3 = 108 O; tag gets _rep<NREP>)
# NREP=2 (fcc: 32 O, 96 atoms) / NREP=3 (bcc: 54 O, 162 atoms): small cells like Wilson 2013 for a finite-size check; tag gets _rep<NREP> (defaults: fcc 3 = 108 O, bcc 4 = 128 O)
# MINRUNS=8: lower the minimum number of DFT runs per 1000 K bin used for the volume (EOS lookup; default 20; outside the bins the volume is extrapolated at most 2000 K with a WARNING)
# ADD LAMBDA VALUES to a finished Gauss-Legendre run without repeating it: Gauss-Kronrod keeps the old values and adds ngauss+1 new ones (all 2*ngauss+1 are then integrated, and |K - G| estimates the quadrature error).
#   e.g. an 8-value run -> 17:  ONLYNODES="8 9 10 11 12 13 14 15 16" RULE=gk NGAUSS=8 NODES=17 <same T P HPER NATOMS PS EQUIL CHUNK DT as the original run> bash mlp_pipeline/young_tdi_par.sh fcc   (same tag; the old node files are kept)
# WARM START: INIT=<extxyz> starts every lambda point from the last frame of that trajectory (e.g. INIT=runs/tdi/fcc_500GPa_6000K_node16_traj.extxyz, same system), and OUTDIR=runs/tdi_warm writes to a separate folder (summarize.py scans runs/tdi*; run it on that folder alone to keep warm and cold runs apart).
# HOT=0.5 (HOTT=12000): liquid only: heat every lambda to HOTT K for HOT ps before the equilibration, so the fcc start melts also below the melting temperature
# PREPHOLD=1 (PREPFREE=1): crystals: before the lambda runs, hold O on its sites for PREPHOLD ps while H thermalises, release O for PREPFREE ps (full MLP, free MD), and start EVERY lambda from that frame (O centre of mass put back on the lattice); log <tag>_prep.log shows the O rms displacement (melting check). New runs only: nodes that restart from a saved state ignore it.
# NATOMS=729: pure-H cell size (sc 9^3; default bcc 250)
# insert mode: NGHOST=1,2,3,.. protons added to fcc H2O (ghosts at lambda=0): mu_H in the crystal; tag ..._n<NGHOST>
# ONLYNODES="0 3": run just these nodes (e.g. after a GPU out-of-memory failure of one node); large cells (>400 atoms) need NPAR=8 or less on a 40 GB card
# env knobs: SEED (MD seed; not in the tag: use another OUTDIR) BULK (bulk modulus GPa, insert with VPA) HPER (H per O, 2; HPER=2.25 for H2.25O: tag gets _h2.25o) T (3000) P (400) MODEL NODES (12) DT (0.5 fs) ONLY0 POWER (4) PS (4) PS1 (1.5) CHUNK (0.5) EQUIL (1) NPAR (4) TAG
# Logs: runs/tdi/<tag>_stage0.log, runs/tdi/<tag>_node<k>.log (every line is a full health report); watch the integral from any shell with
#    watch -n 120 python3 mlp_pipeline/tdi/collect.py runs/tdi <tag>
# Design: the code is copied to runs/code/<time>_<pid> at launch and every process runs from that snapshot, and the body is one function parsed before it runs,
# so a git pull / edit during a run cannot change a running job (bash reads scripts incrementally; python processes started later would also pick up new code).
main() {
# refuse to run without a GPU (a login node would run TensorFlow on its CPUs for hours and slow everyone down): ALLOW_NOGPU=1 overrides, MODEL=lj (the harness test model) is allowed
case "${MODEL:-grace}" in lj*) ;; *) [ -n "$ALLOW_NOGPU" ] || nvidia-smi -L >/dev/null 2>&1 || { echo "NO GPU visible on $(hostname): this looks like a login node. Run it inside a GPU job (srun / sbatch), or set ALLOW_NOGPU=1 to force it."; exit 1; } ;; esac
SNAP=runs/code/$(date +%Y%m%d_%H%M%S)_$$; mkdir -p $SNAP && cp -r mlp_pipeline $SNAP/ && CODE=$SNAP/mlp_pipeline; export H3O_RUNS=$PWD/runs; export H3O_GIT=$(git rev-parse HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo "+local")
PH=${1:?bcc, fcc, liquid, insert or H}; P=${P:-400}; T=${T:-3000}; HPER=${HPER:-2}; NGHOST=${NGHOST:-1}; MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; NODES=${NODES:-12}; POWER=${POWER:-4}; PS=${PS:-4}; PS1=${PS1:-1.5}
CHUNK=${CHUNK:-0.5}; EQUIL=${EQUIL:-1}; DT=${DT:-0.5}; NPAR=${NPAR:-4}; OUT=runs/tdi; [ -n "$TAG" ] && OUT=runs/tdi_test; [ -n "$OUTDIR" ] && OUT=$OUTDIR
# default Einstein constants measured in earlier 3 ps lambda=1 runs at 400 GPa, 3000 K (bcc 12.182, fcc 22.842); K=0 runs the lambda=1 stage instead (re-measures k, pressure, structure)
# Einstein constant k (eV/A^2): scaled guess unless K is given (k only sets the efficiency, not the result): measured 12.2 (bcc) / 22.0-22.8 (fcc) at 400 GPa 3000 K, 12.8 (bcc) at 5000 K, 7.9 (bcc) at 200 GPa; k ~ P^0.6.
# K=0 measures it with a lambda=1 run instead (about 6 min; also prints MLP pressure and O structure).
if [ -z "$K" ] && [ "$PH" != H ]; then K=$(awk -v p=$P -v k4=$([ $PH = bcc ] && echo 12.5 || echo 22.5) 'BEGIN{printf "%.2f", k4*(p/400)^0.6}'); fi; [ "$K" = 0 ] && K=
mkdir -p $OUT
export TF_FORCE_GPU_ALLOW_GROWTH=true; [ -n "$UQ" ] && export GRACE_UQ=1   # UQ=1: per-atom extrapolation grade (cost not measured); use tdi/uq_check.py on saved snapshots instead
PY=${PY:-$HOME/grace-venv/bin/python}
SRC="--from-dft inventory_binned.csv ${NREP:+--nrep $NREP}"; [ -n "$VPA" ] && SRC="--vpa $VPA --nrep ${NREP:-2}"   # VPA/NREP: harness test without the inventory
COMMON="$MODEL ${BULK:+--bulk-gpa $BULK }--phase $PH --P $P --T $T --hper $HPER --nghost $NGHOST ${HOSTPHASE:+--host-phase $HOSTPHASE} ${NATOMS:+--natoms $NATOMS} ${MINRUNS:+--min-runs $MINRUNS} ${INIT:+--init-from $INIT} ${RULE:+--rule $RULE} ${NGAUSS:+--ngauss $NGAUSS} $SRC --ps $PS --ps1 $PS1 --nodes $NODES --power $POWER --dt $DT --chunk-ps $CHUNK --equil-ps $EQUIL --outdir $OUT${SEED:+ --seed $SEED}${K:+ --k $K}${HOT:+ --hot-ps $HOT --hot-T ${HOTT:-12000}} --prep-hold-ps ${PREPHOLD:-1} --prep-free-ps ${PREPFREE:-1}"
NOISE="tensorflow|XLA|cuda|absl|computation_placer|_handle_getgr|E0000|W0000|I0000|ptxas|^$"
TAGNAME=$($PY $CODE/tdi/run_real.py $COMMON --print-tag 2>/dev/null | tail -n1)   # the tag of the files run_real.py writes (one definition: make_tag)
[ -n "$TAGNAME" ] || { echo "cannot build the run tag ($PY $CODE/tdi/run_real.py ... --print-tag failed)"; exit 1; }
echo "== tag $TAGNAME in $OUT"
# restartable: rerunning the same command continues every node from its last finished chunk (state in $OUT/*_node<k>_state.npz); FRESH=1 starts over
if [ -n "$FRESH" ]; then
  if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$FRESH" != array ]; then echo "FRESH=$FRESH in a Slurm array task: refused (an exported FRESH would wipe every task's run); FRESH=array means it"; exit 1; fi
  rm -f $OUT/${TAGNAME}_stage0.* $OUT/${TAGNAME}_prep.* $OUT/tdi_${TAGNAME}.json $OUT/${TAGNAME}_node[0-9]*   # this tag only (a bare ${TAGNAME}_* also matches _h<x>o / _rep<n> / _N<n> runs)
  FRESHFLAG=--fresh
fi
echo "== stage 0 (lambda = 1 run) $(date)"
if [ -z "$K" ] && [ -f $OUT/${TAGNAME}_stage0.json ] && { [ $PH = bcc ] || [ $PH = fcc ] || [ $PH = hcp ] || [ $PH = fccl ]; }; then echo "stage 0 already done (measured k kept; FRESH=1 to redo)"
else
  $PY -u $CODE/tdi/run_real.py $COMMON --stage0 2>&1 | tee $OUT/${TAGNAME}_stage0.log | grep --line-buffered -vE "$NOISE"; RC=${PIPESTATUS[0]}
  [ $RC = 0 ] || { echo "stage 0 failed (exit $RC; see $OUT/${TAGNAME}_stage0.log): nodes not started"; exit 1; }   # never run the nodes on an older stage0.json
fi
[ -f $OUT/${TAGNAME}_stage0.json ] || { echo "stage 0 failed"; exit 1; }
[ -n "$ONLY0" ] && exit 0   # ONLY0=1: stop after the lambda=1 run (e.g. to compare DT=0.5 and DT=1.0)
if [ -n "$PREPHOLD" ] && { [ $PH = bcc ] || [ $PH = fcc ]; }; then   # PREPHOLD=<ps>: start every lambda from a configuration made by holding O, thermalising H, releasing O (PREPFREE ps, default 1)
  if [ -f $OUT/${TAGNAME}_prep.extxyz ]; then echo "== start configuration: $OUT/${TAGNAME}_prep.extxyz exists, used for every lambda not yet started (FRESH=1 to redo)"
  elif ls $OUT/${TAGNAME}_node*_state.npz >/dev/null 2>&1; then echo "prep requested but lambda states exist without a prep frame (a run started without PREPHOLD): not mixing start protocols; FRESH=1 or no PREPHOLD"; exit 1
  else
    echo "== start configuration: O held $PREPHOLD ps, then released ${PREPFREE:-1} ps $(date)"
    $PY -u $CODE/tdi/run_real.py $COMMON --prep 2>&1 | tee $OUT/${TAGNAME}_prep.log | grep --line-buffered -vE "$NOISE"; RC=${PIPESTATUS[0]}
    { [ $RC = 0 ] && [ -f $OUT/${TAGNAME}_prep.extxyz ]; } || { echo "prep failed (exit $RC)"; exit 1; }
  fi
  [ -z "$INIT" ] && COMMON="$COMMON --init-from $OUT/${TAGNAME}_prep.extxyz"   # also on a resubmit: nodes that never started begin from the same frame as the others
fi
# memory-aware parallelism: start only as many lambda processes as fit in the free GPU memory (one TensorFlow context each; ~4.6 GB measured for 250 H on an A100; MEMPROC MiB per process, override for big cells)
if [ "${MODEL%%:*}" != toyice ] && command -v nvidia-smi >/dev/null 2>&1; then   # the toy potential runs on CPU: no GPU memory cap
  GI=${CUDA_VISIBLE_DEVICES%%,*}; case "$GI" in ""|*[!0-9]*) GI=0;; esac
  FREE=$(nvidia-smi -i $GI --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc 0-9)
  if [ -n "$FREE" ]; then CAP=$(( FREE / ${MEMPROC:-4800} )); [ $CAP -lt 1 ] && CAP=1
    [ $CAP -lt $NPAR ] && { echo "NOTE: $FREE MiB free on GPU $GI: NPAR $NPAR -> $CAP (MEMPROC=${MEMPROC:-4800} MiB per lambda process)"; NPAR=$CAP; }
  fi
fi
echo "== lambda 0..$((NODES-1)), $NPAR at a time $(date)"
for k in $(seq 0 $((NODES-1))); do if [ -n "$ONLYNODES" ] && ! echo " $ONLYNODES " | grep -q " $k "; then continue; fi; echo "== launch $(date)" >> $OUT/${TAGNAME}_node$k.log; done   # tdi/eta.py: only lines after this count (a lambda still waiting for a slot shows "queued", not an old 100%)
# node order when NPAR < NODES: middle and both ends first, then the largest gaps (a coarse integral exists early; tdi/live.py PREVIEW uses whatever is there)
if [ -n "$ONLYNODES" ]; then echo $ONLYNODES | tr ' ' '\n'; else
python3 -c "
n=$NODES; o=[i for i in (n//2, n-1, 0) if 0<=i<n]
o=list(dict.fromkeys(o))
while len(o)<n:
    o.append(max((i for i in range(n) if i not in o), key=lambda i: min(abs(i-j) for j in o)))
print(*o, sep=chr(10))"
fi | xargs -P $NPAR -I{} bash -c "$PY -u $CODE/tdi/run_real.py $COMMON --node {} $FRESHFLAG >> $OUT/${TAGNAME}_node{}.log 2>&1" &
XP=$!
while kill -0 $XP 2>/dev/null; do
  sleep ${MONEVERY:-60}; echo "-- $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  python3 $CODE/tdi/eta.py $OUT $TAGNAME $NODES $NPAR "$ONLYNODES" 2>&1   # per lambda (this launch only) + whole-job ETA including the lambda values still queued
  python3 $CODE/tdi/live.py $OUT $TAGNAME --brief 2>&1 | grep -v "^$"
done
# second pass: lambda values whose process died of GPU out-of-memory at start-up (too many TensorFlow contexts on one card) are rerun, 2 at a time
OOM=$(for f in $OUT/${TAGNAME}_node*.log; do [ -f "$f" ] || continue; k=$(basename $f .log | sed "s/${TAGNAME}_node//"); case "$k" in *[!0-9]*|"") continue;; esac; tail -n 30 "$f" | grep -qiE "out of memory|ResourceExhausted" && echo $k; done | sort -n | tr '\n' ' ')
if [ -n "$OOM" ] && [ -z "$NO_OOM_RETRY" ]; then
  echo "== GPU out-of-memory at lambda $OOM: rerunning them 2 at a time $(date)"
  for k in $OOM; do echo "-- OOM retry $(date)" >> $OUT/${TAGNAME}_node$k.log; done
  echo $OOM | tr ' ' '\n' | grep -v "^$" | xargs -P 2 -I{} bash -c "$PY -u $CODE/tdi/run_real.py $COMMON --node {} >> $OUT/${TAGNAME}_node{}.log 2>&1"
fi
echo "== done $(date)"; python3 $CODE/tdi/collect.py $OUT $TAGNAME
}
main "$@"; exit $?   # on ONE line: bash reads a script as it runs, so after main returns it would read the next line at the old byte offset of a file that a `git pull` may have changed meanwhile
