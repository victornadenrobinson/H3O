#!/bin/bash -l
# Proxy-guided Widom / deletion + BAR (tdi/pwidom.py, docs/MU_METHODS.md R1): mu_H in the H2O crystal and in the pure-H fluid at one (P, T) WITHOUT a lambda path,
# on the same hosts and volumes as existing insertion TDI runs (--like <stage0.json>), then dg = mu_H(ice) - G_H to compare with the TDI value.
# One GPU: free NVT MD of host and host + 1 H for both hosts (4 processes share the GPU), then batched single points with the fitted pair proxy and, for
# comparison on the same frames, plain Widom (proxy none). New files only (runs/pwidom/); nothing in runs/tdi is touched.
# Cost (estimate): MD 4 x (0.5 + PS ps) at ~0.5 s/step with 4 processes on one GPU = ~80 min for PS=4; single points ~4 x 2000 = ~10-20 min.
# usage (repo root on Young):  sbatch mlp_pipeline/young_pwidom.sbatch                              (500 GPa 6000 K, the TDI runs in runs/tdi)
#   ICE=runs/tdi_long/insert_500GPa_6000K_n1_stage0.json HFL=runs/tdi/insert_500GPa_6000K_n1_Hhost_stage0.json sbatch mlp_pipeline/young_pwidom.sbatch
#   knobs: PS (4) K (4000) SEL (4) NDEL (4) BATCH (16) PROXY (docs/data/toyfit_t4.json) OUT (runs/pwidom) PY (~/grace-venv/bin/python)
#SBATCH --job-name=pwidom
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_%x_%j.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PY=${PY:-~/grace-venv/bin/python}; PS=${PS:-4}; K=${K:-4000}; SEL=${SEL:-4}; NDEL=${NDEL:-4}; BATCH=${BATCH:-16}; PROXY=${PROXY:-docs/data/toyfit_t4.json}; OUT=${OUT:-runs/pwidom}
ICE=${ICE:-runs/tdi/insert_500GPa_6000K_n1_stage0.json}; HFL=${HFL:-runs/tdi/insert_500GPa_6000K_n1_Hhost_stage0.json}
for f in $ICE $HFL; do [ -f $f ] || { echo "missing $f (set ICE= / HFL= to an insertion run's stage0.json: ls runs/tdi*/insert_*_stage0.json)"; exit 1; }; done
mkdir -p $OUT; echo "== pwidom on $(hostname) $(date) code $(git rev-parse --short HEAD): ice $ICE, pure H $HFL, PS $PS, K $K, proxy $PROXY"
export OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
pw() { $PY mlp_pipeline/tdi/pwidom.py run --ps $PS --out $OUT --batch $BATCH "$@"; }
pw --like $ICE --frames-only > $OUT/ice_md.log 2>&1 &
pw --like $HFL --frames-only > $OUT/H_md.log 2>&1 &
wait; tail -n 2 $OUT/ice_md.log $OUT/H_md.log
for px in $PROXY none; do
  pw --like $ICE --proxy $px --K $K --sel $SEL --ndel $NDEL > $OUT/ice_$(basename $px .json).log 2>&1 &
  pw --like $HFL --proxy $px --K $K --sel $SEL --ndel $NDEL > $OUT/H_$(basename $px .json).log 2>&1 &
  wait; tail -n 4 $OUT/ice_$(basename $px .json).log $OUT/H_$(basename $px .json).log
done
for px in $(basename $PROXY .json) none; do
  w=$(ls -t $OUT/pw_fcc*_${px}_K*.json | head -1); h=$(ls -t $OUT/pw_H*_${px}_K*.json | head -1)
  $PY mlp_pipeline/tdi/pwidom.py dg $w $h
done
echo "compare: dg of the insertion TDI at the same (P, T): ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py (insertion row minus the pure-H insertion row + fs)"
