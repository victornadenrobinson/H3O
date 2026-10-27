#!/bin/bash -l
# Extend every END MEMBER of the mixing / hull problem in time: fcc and bcc H2O crystals, liquid H2O and pure H (runs/tdi, dt 0.5 fs only; list from tdi/endmember_list.py) to PS ps per lambda (default 8),
# with the settings stored in each run's stage0.json (tdi/extend.py cmd: same dt, chunk, equil, k, lambda values, cell). Their statistical errors dominate every mixing / hull number
# (error per added H = N_partner * sigma_partner / n). One array task per tag; extending a finished run is a no-op. A run that is Gauss-only with the Kronrod lambda values pending (nodes 17) gets those too.
# Cost per task is an ESTIMATE, not measured: crystals 17 lambda values in two waves of at most 9 processes at about 0.4 s/step, 5-7 extra ps = 3 h or more; liquids and pure H less. --time=08:00:00.
# Restartable (resubmit the same command). Locks stop two jobs from writing one lambda value, so overlapping with a running job on the same tag is safe but wasteful; tags of running jobs: squeue.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && ~/grace-venv/bin/python mlp_pipeline/tdi/endmember_list.py | wc -l      (N tags)
#                      sbatch --array=0-$((N-1))%6 mlp_pipeline/young_tdi_endmembers.sbatch          longer or shorter: PS=12 sbatch ...     (default array 0-59%6: tasks beyond the list exit at once)
#SBATCH --job-name=tdiend
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-59%6
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
ARRAY_KEEP="PS"; . mlp_pipeline/array_env.sh
mapfile -t BASES < <(python3 mlp_pipeline/tdi/endmember_list.py runs/tdi)
I=${SLURM_ARRAY_TASK_ID:-0}; [ "$I" -lt "${#BASES[@]}" ] || { echo "task $I: only ${#BASES[@]} end-member runs: nothing to do"; exit 0; }
base=${BASES[$I]}; PSX=${PS:-8}; unset PS
echo "== task $I of ${#BASES[@]}: extend $base to $PSX ps per lambda on $(hostname) $(date) code $(git rev-parse --short HEAD)"
line=$(python3 mlp_pipeline/tdi/extend.py cmd $base $PSX) || { echo "extend.py cmd failed for $base"; exit 1; }
echo "$line"; bash -c "$line"
