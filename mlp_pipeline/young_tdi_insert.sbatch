#!/bin/bash -l
# Ghost-proton insertion into fcc H2O (mu_H per added H) as a Slurm array: task i runs entry i of a table "P_GPa T_K n_added" (one GPU per task). Default table: n = 1, 2, 4 at four (P, T) points, 11 tasks (n = 1 at 400 GPa / 5000 K is left out: that tag already holds the unfinished 12-lambda run, whose lambda grid differs; continue it with its own command).
# Restartable: resubmit the same command and unfinished lambda values continue (state is checked: a changed dt, lambda or atom count is refused).
# Cost per task is an ESTIMATE, not measured: 10 lambda values x (0.5 + 2 ps) = 25 ps = 50,000 steps at about 0.07 s/step effective with 10 processes on one GPU (two MLP evaluations per step) = about 1 h;
# the earlier n=1 run (12 lambda values, 5 ps each, 6 processes) needed about 2 h. Expected statistical error for n = 1 at these settings is only a guess, roughly +-30-60 meV (n = 2, 4 smaller per added H).
# usage on Young (login node):   cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_insert.sbatch
#   own table:  JOBS="400 5000 1;400 6500 1" sbatch --array=0-1 mlp_pipeline/young_tdi_insert.sbatch
#   longer:     PS=4 NODES=12 sbatch --time=06:00:00 ...
# volumes: DFT EOS of fcc H2O at (P, T); above the DFT bins (about 5000 K) the volume is extrapolated (<= 2000 K, with a WARNING in the log); check the MLP pressure at the highest lambda.
# results: runs/tdi/insert_<P>GPa_<T>K_n<n>_* ; table with everything:  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py   (insertion rows: mu_H - G of pure H at the same P, T; needs a pure-H run there)
#SBATCH --job-name=tdiins
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
. mlp_pipeline/array_env.sh   # drop run knobs exported by the submitting shell (they would apply to every task)
DEFAULT="400 5000 2;400 5000 4;400 6500 1;400 6500 2;400 6500 4;500 5000 1;500 5000 2;500 5000 4;500 7000 1;500 7000 2;500 7000 4"
IFS=';' read -ra TABLE <<< "${JOBS:-$DEFAULT}"
I=${SLURM_ARRAY_TASK_ID:-0}; read -r P T N <<< "${TABLE[$I]}"
[ -n "$N" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: insert n=$N at $P GPa, $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
NGHOST=$N P=$P T=$T NODES=${NODES:-10} NPAR=${NPAR:-10} DT=${DT:-0.5} EQUIL=${EQUIL:-0.5} PS=${PS:-2} CHUNK=0.5 bash mlp_pipeline/young_tdi_par.sh insert
