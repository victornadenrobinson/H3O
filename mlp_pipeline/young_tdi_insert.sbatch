#!/bin/bash -l
# Ghost-proton insertion into fcc H2O (mu_H, per added H) as a Slurm array: task i adds n = entry i of NS protons (default 1..10, one GPU each).
# Restartable: resubmit the same command and unfinished lambda values continue. Cost per task is an estimate, not measured: 12 lambda values x (EQUIL + PS) ps; about 70 ms/step effective with 4 processes on the GPU
# (PS=4 -> roughly 2.5 h, PS=8 -> roughly 4.5 h). NPAR=10 processes were used for 324 atoms in young_tdi_scan.sbatch; throughput at NPAR=10 for insertion is untested.
# usage on Young:   cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit
#   sbatch --array=0-9 mlp_pipeline/young_tdi_insert.sbatch                  # n = 1..10, PS=4
#   NS="1" PS=8 sbatch --array=0-0 --time=06:00:00 mlp_pipeline/young_tdi_insert.sbatch   # longer run for n = 1 only
#   NS="2 3 4" sbatch --array=0-2 mlp_pipeline/young_tdi_insert.sbatch       # chosen n
#   T=5000 P=400 are defaults; override in the environment.
# results: runs/tdi/insert_<P>GPa_<T>K_n<n>_* ; table: python3 mlp_pipeline/tdi/collect.py runs/tdi insert_400GPa_5000K_n1
#SBATCH --job-name=tdiins
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
read -ra NSA <<< "${NS:-1 2 3 4 5 6 7 8 9 10}"
I=${SLURM_ARRAY_TASK_ID:-0}; N=${NSA[$I]}
[ -n "$N" ] || { echo "no n for array task $I"; exit 2; }
echo "== task $I: insert n=$N at ${P:-400} GPa, ${T:-5000} K on $(hostname) $(date)"
NGHOST=$N P=${P:-400} T=${T:-5000} NODES=${NODES:-12} NPAR=${NPAR:-10} DT=${DT:-0.5} EQUIL=${EQUIL:-1} PS=${PS:-4} CHUNK=0.5 bash mlp_pipeline/young_tdi_par.sh insert
