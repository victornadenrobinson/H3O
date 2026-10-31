#!/bin/bash -l
# Refit the pwidom insertion proxy (OH / HH pairs) to GRACE insertion energies (tdi/proxy_fit.py, docs/MU_METHODS.md) for the ice host and the pure-H host of an insertion
# TDI run, then rerun pwidom with the refitted proxies and print dg next to the old proxy's. One GPU, all stages in one job:
#   1. fit frames: free NVT MD of both hosts with ANOTHER seed than production (in parallel)       -> $FIT/fit_{ice,H}_frames.npz
#   2. label probe points with GRACE (dU of one added H)                                           -> $FIT/{ice,H}_probes.npz
#   3. fit the OH / HH splines, held-out report (old vs new std(dU - phi), Kish)                    -> $FIT/proxy_{ice,H}.json / .png   (read the "held-out" lines first)
#   4. pwidom with the new proxies on the production frames in $OUT (reused if young_pwidom.sbatch ran, else the MD is run here, ~2 h) and dg of old vs new (PROD=0 skips)
# usage (repo root on Young):  sbatch mlp_pipeline/young_proxy_fit.sbatch
#   ICE=<insert stage0.json> HFL=<pure-H stage0.json> PS_FIT=2 NSEL=16 NUNI=8 TEMPER=2 PROD=1 sbatch mlp_pipeline/young_proxy_fit.sbatch
#   knobs: PS_FIT (2) NSEL (16) NUNI (8) TEMPER (2) K (2000) BATCH (16) BASE (docs/data/toyfit_t4.json) FIT (runs/pwidom_fit) OUT (runs/pwidom) PROD (1) PY (~/grace-venv/bin/python)
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
PY=${PY:-~/grace-venv/bin/python}; PS_FIT=${PS_FIT:-2}; NSEL=${NSEL:-16}; NUNI=${NUNI:-8}; TEMPER=${TEMPER:-2}; K=${K:-2000}; BATCH=${BATCH:-16}
BASE=${BASE:-docs/data/toyfit_t4.json}; FIT=${FIT:-runs/pwidom_fit}; OUT=${OUT:-runs/pwidom}; PROD=${PROD:-1}
ICE=${ICE:-runs/tdi/insert_500GPa_6000K_n1_stage0.json}; HFL=${HFL:-runs/tdi/insert_500GPa_6000K_n1_Hhost_stage0.json}
for f in $ICE $HFL $BASE; do [ -f $f ] || { echo "missing $f (set ICE= / HFL= to an insertion run's stage0.json: ls runs/tdi*/insert_*_stage0.json)"; exit 1; }; done
mkdir -p $FIT $OUT; echo "== proxy refit on $(hostname) $(date) code $(git rev-parse --short HEAD): ice $ICE, pure H $HFL, base $BASE, PS_FIT $PS_FIT, $NSEL selected + $NUNI uniform probes per frame"
export OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
pw() { $PY mlp_pipeline/tdi/pwidom.py run --out $1 --batch $BATCH "${@:2}"; }
pf() { $PY mlp_pipeline/tdi/proxy_fit.py "$@"; }
declare -A LIKE=([ice]=$ICE [H]=$HFL)
# 1. fit frames (seed 11, production uses the default seed 3)
for h in ice H; do pw $FIT --like ${LIKE[$h]} --frames-only --ps $PS_FIT --seed 11 --tag fit_$h > $FIT/${h}_md.log 2>&1 & done; wait; tail -n 2 $FIT/ice_md.log $FIT/H_md.log
# 2. probe energies of the real model
for h in ice H; do pf label $FIT/fit_${h}_frames.npz --like ${LIKE[$h]} --batch $BATCH --proxy $BASE --K $K --nsel $NSEL --nuni $NUNI --temper $TEMPER -o $FIT/${h}_probes.npz > $FIT/${h}_label.log 2>&1 & done; wait; tail -n 2 $FIT/ice_label.log $FIT/H_label.log
# 3. fit and held-out report
for h in ice H; do echo "---- $h"; pf fit $FIT/${h}_probes.npz --base $BASE -o $FIT/proxy_$h --plot 2>&1 | tee $FIT/${h}_fit.log | grep -E "probes|fit:|std|train|held-out|wrote"; done
[ "$PROD" = 1 ] || { echo "PROD=0: stopped after the fit; use --proxy $FIT/proxy_ice.json / proxy_H.json in pwidom.py"; exit 0; }
# 4. production frames in $OUT, old and new proxy on the same frames, dg
for h in ice H; do pw $OUT --like ${LIKE[$h]} --frames-only > $OUT/${h}_md.log 2>&1 & done; wait
for v in new old; do
  for h in ice H; do
    px=$BASE; [ $v = new ] && px=$FIT/proxy_$h.json
    pw $OUT --like ${LIKE[$h]} --proxy $px --K $K --label ${v}proxy > $OUT/${h}_${v}proxy.log 2>&1 &
  done; wait; tail -n 3 $OUT/ice_${v}proxy.log $OUT/H_${v}proxy.log
done
for v in new old; do
  w=$(ls -t $OUT/pw_fcc*_${v}proxy_K*.json | head -1); h=$(ls -t $OUT/pw_H*_${v}proxy_K*.json | head -1); echo "== ${v} proxy:"; $PY mlp_pipeline/tdi/pwidom.py dg $w $h
done
