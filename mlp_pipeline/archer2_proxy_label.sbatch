#!/bin/bash
# Proxy refit on ARCHER2 CPU nodes (no GPU): the GRACE insertion energies of tdi/proxy_fit.py `label` are independent single points, so one node runs NPROC independent
# shards (each with OMP threads), then the numpy fit. Needs, under /work (compute nodes do not see /home): the repo, a venv with tensorflow-cpu + tensorpotential, the cached
# pwidom frames (runs/pwidom*/pw_*_frames.npz) and the GRACE checkpoint (MODEL=grace-ft:<absolute path>).
# usage (login node, repo root):  ACCT=<your budget code> PY=<venv>/bin/python MODEL=grace-ft:$PWD/runs/grace/ft_m2 sbatch mlp_pipeline/archer2_proxy_label.sbatch
#   knobs: NPROC (16) THREADS (8) PER_SOURCE (5) SELECT ("500GPa_6000K 500GPa_5000K 600GPa_5000K 600GPa_6000K"; "" = all sources) NSEL (16) NUNI (8) FIT (runs/pwidom_fit)
#          BASE (docs/data/toyfit_t4.json) HOLD_ICE / HOLD_H (production state held out of the fit) SRC (glob)
# check the partition / qos / time names in the ARCHER2 docs; label is resumable (finished sources are skipped), so resubmit if it hits the time limit.
#SBATCH --job-name=proxylabel
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_%x_%j.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
NPROC=${NPROC:-16}; THREADS=${THREADS:-8}; PER_SOURCE=${PER_SOURCE:-5}; NSEL=${NSEL:-16}; NUNI=${NUNI:-8}; FIT=${FIT:-runs/pwidom_fit}; BASE=${BASE:-docs/data/toyfit_t4.json}
SELECT=${SELECT-"500GPa_6000K 500GPa_5000K 600GPa_5000K 600GPa_6000K"}; SRC=${SRC:-"runs/pwidom*/pw_*_frames.npz"}; PY=${PY:?"PY=<venv>/bin/python (a venv under /work)"}; MODEL=${MODEL:?"MODEL=grace-ft:<absolute path to the GRACE checkpoint>"}
HOLD_ICE=${HOLD_ICE:-pwidom__pw_fcc_500GPa_6000K}; HOLD_H=${HOLD_H:-pwidom__pw_H_500GPa_6000K}
FILES=$(ls $SRC 2>/dev/null); [ -n "$FILES" ] || { echo "no frames files match $SRC"; exit 1; }; mkdir -p $FIT
export OMP_NUM_THREADS=$THREADS TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=
echo "== proxy label on $(hostname) $(date): $(echo $FILES | wc -w) frames files, select '${SELECT}', $NPROC shards x $THREADS threads, per-source $PER_SOURCE, model $MODEL"
SEL=(); [ -n "$SELECT" ] && SEL=(--select $SELECT)
for i in $(seq 0 $((NPROC - 1))); do
  $PY mlp_pipeline/tdi/proxy_fit.py label $FILES --device cpu --batch 1 --model $MODEL --proxy $BASE --per-source $PER_SOURCE --nsel $NSEL --nuni $NUNI "${SEL[@]}" --shard $i/$NPROC -o $FIT > $FIT/label_$i.log 2>&1 &
done; wait; cat $FIT/label_*.log | grep -E "exists|probes" | wc -l | xargs echo "sources labelled or skipped:"
echo "---- ice"; $PY mlp_pipeline/tdi/proxy_fit.py fit $FIT/*_probes.npz --base $BASE --skip pureH pw_H_ --hold $HOLD_ICE -o $FIT/proxy_ice --plot 2>&1 | tee $FIT/fit_ice.log
echo "---- pure H"; $PY mlp_pipeline/tdi/proxy_fit.py fit $FIT/*_probes.npz --base $BASE --only pureH pw_H_ --hold $HOLD_H -o $FIT/proxy_H --plot 2>&1 | tee $FIT/fit_H.log
