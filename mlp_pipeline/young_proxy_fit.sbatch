#!/bin/bash -l
# Refit the pwidom insertion proxy (OH / HH pairs) to GRACE insertion energies (tdi/proxy_fit.py, docs/MU_METHODS.md) from the pwidom frames that ALREADY exist (no MD), then rerun pwidom
# on the production state with the old and the new proxy and print dg for both. One GPU, all stages in one job:
#   1. label: a few frames of every runs/pwidom*/pw_*_frames.npz (all compositions, P, T), GRACE dU of one added H at probe points   -> $FIT/*_probes.npz (resumable: existing files are skipped)
#   2. fit ice-like hosts and pure H separately; the production state is held OUT of the fit (HOLD_ICE / HOLD_H), so the proxy has never seen its frames  -> $FIT/proxy_{ice,H}.json / .png
#      (read the "held-out sources" lines and the per-source table first: old vs new std(dU - phi) in kT, Kish fraction)
#   3. pwidom on the production frames (runs/pwidom, reused) with the new and the old proxy, dg of both (PROD=0 skips)
# Cost: label is PER_SOURCE x (NSEL + NUNI + 1) single points per source, ~150 sources x 250 = ~40k structures (est. 15-60 min at the ~23 GRACE calls/s of docs/PERFORMANCE.md; PER_SOURCE=5 halves it); fit is seconds; step 3 ~10-20 min.
# usage (repo root on Young):  sbatch mlp_pipeline/young_proxy_fit.sbatch
#   or in an interactive GPU job (the #SBATCH lines are comments; label is resumable):  PROD=0 bash mlp_pipeline/young_proxy_fit.sbatch 2>&1 | tee runs/proxyfit.log
#   knobs: SRC (glob, default "runs/pwidom*/pw_*_frames.npz") PER_SOURCE (10) NSEL (16) NUNI (8) TEMPER (2) K (4000) BATCH (16) BASE (docs/data/toyfit_t4.json) FIT (runs/pwidom_fit) OUT (runs/pwidom)
#   ICE / HFL (stage0.json of the production insertion runs, only for PROD=1) HOLD_ICE (pwidom__pw_fcc_500GPa_6000K) HOLD_H (pwidom__pw_H_500GPa_6000K) MODEL (default: the model of $ICE) PROD (1) PY
#SBATCH --job-name=proxyfit
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_%x_%j.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PY=${PY:-~/grace-venv/bin/python}; PER_SOURCE=${PER_SOURCE:-10}; NSEL=${NSEL:-16}; NUNI=${NUNI:-8}; TEMPER=${TEMPER:-2}; K=${K:-4000}; BATCH=${BATCH:-16}
BASE=${BASE:-docs/data/toyfit_t4.json}; FIT=${FIT:-runs/pwidom_fit}; OUT=${OUT:-runs/pwidom}; PROD=${PROD:-1}; SRC=${SRC:-"runs/pwidom*/pw_*_frames.npz"}
ICE=${ICE:-runs/tdi/insert_500GPa_6000K_n1_stage0.json}; HFL=${HFL:-runs/tdi/insert_500GPa_6000K_n1_Hhost_stage0.json}
HOLD_ICE=${HOLD_ICE:-pwidom__pw_fcc_500GPa_6000K}; HOLD_H=${HOLD_H:-pwidom__pw_H_500GPa_6000K}
[ -f $BASE ] || { echo "missing $BASE"; exit 1; }
if [ "$PROD" = 1 ]; then for f in $ICE $HFL; do [ -f $f ] || { echo "missing $f (set ICE= / HFL= to an insertion run's stage0.json: ls runs/tdi*/insert_*_stage0.json; PROD=0 needs only MODEL=)"; exit 1; }; done; fi
if [ -z "$MODEL" ]; then [ -f $ICE ] || { echo "set MODEL=<model spec> (e.g. grace-ft:/path/to/model: the args.model field of the runs' stage0.json) or give ICE="; exit 1; }
  MODEL=$($PY -c "import json;print(json.load(open('$ICE'))['args'].get('model','toyice'))"); fi
FILES=$(ls $SRC 2>/dev/null); [ -n "$FILES" ] || { echo "no frames files match $SRC"; exit 1; }
mkdir -p $FIT $OUT; echo "== proxy refit on $(hostname) $(date) code $(git rev-parse --short HEAD): $(echo $FILES | wc -w) frames files, model $MODEL, base $BASE, $PER_SOURCE frames x ($NSEL selected + $NUNI uniform) probes per source, held out: $HOLD_ICE $HOLD_H"
export OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
pf() { $PY mlp_pipeline/tdi/proxy_fit.py "$@"; }
# 1. probe energies of the real model (resumable)
pf label $FILES --model $MODEL --batch $BATCH --proxy $BASE --per-source $PER_SOURCE --nsel $NSEL --nuni $NUNI --temper $TEMPER -o $FIT 2>&1 | tee $FIT/label.log
# 2. fits: ice-like hosts and pure H separately, the production state held out
echo "---- ice"; pf fit $FIT/*_probes.npz --base $BASE --skip pureH pw_H_ --hold $HOLD_ICE -o $FIT/proxy_ice --plot 2>&1 | tee $FIT/fit_ice.log
echo "---- pure H"; pf fit $FIT/*_probes.npz --base $BASE --only pureH pw_H_ --hold $HOLD_H -o $FIT/proxy_H --plot 2>&1 | tee $FIT/fit_H.log
[ "$PROD" = 1 ] || { echo "PROD=0: stopped after the fit; use --proxy $FIT/proxy_ice.json / proxy_H.json in pwidom.py"; exit 0; }
# 3. production frames (cached in $OUT; the MD runs here only if they are missing): new and old proxy on the same frames, then dg
pw() { $PY mlp_pipeline/tdi/pwidom.py run --out $OUT --batch $BATCH "$@"; }
declare -A LIKE=([ice]=$ICE [H]=$HFL)
for v in new old; do
  for h in ice H; do
    px=$BASE; [ $v = new ] && px=$FIT/proxy_$h.json
    pw --like ${LIKE[$h]} --proxy $px --K $K --label ${v}proxy > $OUT/${h}_${v}proxy.log 2>&1 &
  done; wait; tail -n 3 $OUT/ice_${v}proxy.log $OUT/H_${v}proxy.log
done
for v in new old; do
  w=$(ls -t $OUT/pw_fcc*_${v}proxy_K*.json | head -1); h=$(ls -t $OUT/pw_H*_${v}proxy_K*.json | head -1); echo "== ${v} proxy:"; $PY mlp_pipeline/tdi/pwidom.py dg $w $h
done
