#!/bin/bash -l
#SBATCH --job-name=pw-dftcheck
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=pw-dftcheck-%j.out
# Set the correct account/queue explicitly at submission. No automatic submission.
# Sequential points or optional array: --array=0-(NPOINTS-1)%4; each task runs one point.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the H3O repository root}"
args=(run "${1:?Provide the prepared VASP check directory}" --ranks "${SLURM_NTASKS:-32}" --minutes "${DFT_MINUTES:-1380}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then args+=(--index "$SLURM_ARRAY_TASK_ID"); fi
bash mlp_pipeline/perlmutter_vasp_check.sh "${args[@]}"
