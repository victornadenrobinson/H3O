#!/bin/bash -l
# GRACE fine-tune sweep on UCL Young as a Slurm job ARRAY: one array task = one configuration, each on its own A100, all at the same time.
#
# usage (from the repo root on Young, after `git pull`):
#     cd /lustre/scratch/mmm0037/H3O && sbatch mlp_pipeline/young_ft_sweep.sbatch                 # tasks 0-1 (A and B below)
#     sbatch --array=0-3 mlp_pipeline/young_ft_sweep.sbatch                                      # all four
#     sbatch --array=0-3%1 ...                                                                   # all four, one after the other
#     sbatch --mem=200G --time=10:00:00 mlp_pipeline/young_ft_sweep.sbatch                        # command-line options override the #SBATCH lines below
# Partition gpu, qos freegpu and account allusers are taken from the user's working example jobs (24 h limit there). --mem=160G is NOT from an example: GRACE's data
# conversion needed far more than 8 GB (an interactive job with the default memory was killed); if tasks stay pending, reduce it (e.g. 120G, which worked for the 747-frame set).
#
# task  name    data (train tier)            weights          energy weight   what it tests
#   0   ft_m4   m (747 frames)               frozen (as ft_m2) 1            A: only longer training
#   1   ft_m3   m                            ALL weights       1            B: only flexibility (unfrozen)
#   2   ft_l3   l_vm (every l run that is not in valid_m)  ALL weights  1   C: B with more data (same number of epochs)
#   3   ft_m5   m                            ALL weights       20           D: B with a 20x energy weight
# Every task is judged on the SAME held-out frames, valid_l_vm.extxyz = all l frames of the runs in valid_m (no run is in any training set).
# Training length is set in EPOCHS (default 21, about 7,800 updates for 747 frames at batch size 2), so the tiers get equal passes over their data.
#SBATCH --job-name=grace_sweep
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=runs/grace/slurm_%x_%A_%a.out

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PY=$HOME/grace-venv/bin/python
EPOCHS=${EPOCHS:-21}
BATCH=2
export GRACE_UQ=1

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) NAME=ft_m4; TIER=m;    EXTRA="" ;;
  1) NAME=ft_m3; TIER=m;    EXTRA="--naive" ;;
  2) NAME=ft_l3; TIER=l_vm; EXTRA="--naive" ;;
  3) NAME=ft_m5; TIER=m;    EXTRA="--naive --wE 20" ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 2 ;;
esac
echo "== $(date) task ${SLURM_ARRAY_TASK_ID}: $NAME tier $TIER extra '$EXTRA' on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p runs/grace
# the common validation set and the larger training tier (idempotent; flock keeps parallel tasks from writing the same file)
(
  flock 9
  [ -f runs/valid_l_vm.extxyz ] || $PY mlp_pipeline/03_split.py snapshots_l.extxyz --valid-from valid_m.extxyz
  for T in $TIER; do
    [ -f runs/grace/train_$T.pkl.gz ] && [ -f runs/grace/valid_$T.pkl.gz ] || bash mlp_pipeline/grace_prep_data.sh $T | tail -3
  done
) 9>runs/grace/.prep.lock

NTRAIN=$($PY -c "import pandas as pd; print(len(pd.read_pickle('runs/grace/train_${TIER}.pkl.gz')))")
UPD=$(( EPOCHS * NTRAIN / BATCH ))
echo "train frames $NTRAIN, $EPOCHS epochs -> $UPD updates"
$PY mlp_pipeline/grace_make_input.py runs/grace/input_${NAME}.yaml --train train_${TIER}.pkl.gz --test valid_${TIER}.pkl.gz --updates $UPD --batch $BATCH $EXTRA || exit 3
bash mlp_pipeline/grace_ft_run.sh runs/grace/input_${NAME}.yaml $NAME
RC=$?
LOG=runs/grace/$NAME/seed/1/log.txt
echo "== fine-tune exit code $RC; last evaluation lines (TRAIN(TEST)):"; grep "Iteration #" $LOG | tail -3 | cut -c1-400

# same held-out frames for every configuration, including the old models (run bench_grace_compare.sh afterwards for the plots)
if [ $RC -eq 0 ]; then
  $PY mlp_pipeline/04_benchmark.py grace-ft:runs/grace/$NAME valid_l_vm.extxyz --n 0 --min-box 8.5 --device cuda > runs/grace/$NAME/bench_valid_l_vm.txt 2>&1
  grep -E "^(overall|E_MAE|F_RMSE|S_MAE|P_err|N )" runs/grace/$NAME/bench_valid_l_vm.txt
fi
echo "== done $(date)"
