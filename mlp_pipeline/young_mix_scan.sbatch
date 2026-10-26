#!/bin/bash -l
# Batch version of young_mix_scan.sh (fcc H2O, fcc H2.25O, pure H at one (P, T), quick settings) so a dropped interactive session cannot kill it. One task per table entry "P_GPa T_K NH".
# Restartable: resubmit the same command; finished lambda values are skipped and unfinished ones continue (a changed dt, lambda or atom count is refused).
# Default table (3 tasks): 500 GPa at 5000, 6000 and 7000 K, NH = 27 (H2.25O). About 30 min each (three runs of about 9 min; estimate, not measured).
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_mix_scan.sbatch
#   own table:  JOBS="400 6500 27" sbatch --array=0-0 mlp_pipeline/young_mix_scan.sbatch        other settings: NODES=12 PS=1 sbatch ...
#SBATCH --job-name=mixscan
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
DEFAULT="500 5000 27;500 6000 27;500 7000 27"
IFS=';' read -ra TABLE <<< "${JOBS:-$DEFAULT}"
I=${SLURM_ARRAY_TASK_ID:-0}; read -r P T NH <<< "${TABLE[$I]}"
[ -n "$T" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: mixing scan at $P GPa, $T K, NH=${NH:-27} on $(hostname) $(date)"
P=$P T=$T NH=${NH:-27} bash mlp_pipeline/young_mix_scan.sh
