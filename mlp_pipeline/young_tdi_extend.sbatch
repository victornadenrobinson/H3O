#!/bin/bash -l
# Overnight batch: extend existing TDI runs in time (and start the missing dt 0.5 H2O references at 400 GPa / 5000 K). Array task i = i-th non-comment line of mlp_pipeline/extend_list.txt (edit that file to change the batch).
# Restartable: every lambda continues from its last finished chunk, so resubmitting the same command is safe. Do not submit it for a run that another job is working on right now (two jobs on the same state files would corrupt them):
#   check first   squeue -u $USER ; python3 mlp_pipeline/tdi/extend.py list --below 99      (ACTIVE = files changed in the last 30 min; this check does not see jobs that are queued but not started)
# Cost is an ESTIMATE, not measured: 17-lambda crystal runs 1 -> 3 ps about 1-1.5 h, 10-lambda 1.5 -> 3 ps about 30-60 min, liquids 2 -> 6 ps about 1 h, NEW 10-lambda 3 ps runs about 1.5 h.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch --array=0-12%6 mlp_pipeline/young_tdi_extend.sbatch        (array size = number of non-comment lines in extend_list.txt)
# then:  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py
#SBATCH --job-name=tdiext
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-12%6
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
. mlp_pipeline/array_env.sh   # drop run knobs exported by the submitting shell (they would apply to every task)
PY=${PY:-$HOME/grace-venv/bin/python}
LINE=$(grep -vE '^\s*(#|$)' mlp_pipeline/extend_list.txt | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in extend_list.txt"; exit 2; }
set -- $LINE; KIND=$1; shift
if [ "$KIND" = EXT ]; then CMD=$($PY mlp_pipeline/tdi/extend.py cmd "$1" "$2") || { echo "cannot read $1_stage0.json"; exit 2; }
elif [ "$KIND" = NEW ]; then CMD="env ${*:1:$#-1} bash mlp_pipeline/young_tdi_par.sh ${@: -1}"
else echo "unknown line type: $LINE"; exit 2; fi
echo "== task $SLURM_ARRAY_TASK_ID on $(hostname) $(date) code $(git rev-parse --short HEAD)"; echo "$LINE"; echo "$CMD"
export TF_FORCE_GPU_ALLOW_GROWTH=true
eval "$CMD"
echo "== task $SLURM_ARRAY_TASK_ID finished $(date)"
