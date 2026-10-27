#!/bin/bash -l
# Liquid convex hull at 400 GPa, 7000 and 9000 K: array task i = one composition and temperature (fluid H(2+n)O, all atoms ideal gas at lambda=0), plus pure H at both T.
# Restartable: resubmit the same command and unfinished nodes continue.
# Tasks 0-4: 7000 K (H2O, H2.25O, H3O, H5O fluids, pure H); tasks 5-9: 9000 K. 32-O fcc start cells (96-192 atoms; pure H 250 atoms), volumes from the hull EOS columns, lambda=0 = ideal gas of all atoms,
# 1 ps equilibration (the lattice start must melt: check the O 'u rms' in the node logs), dt 0.5 fs, 2 ps per node. About 20-30 min per task.
# usage on Young:  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase && sbatch mlp_pipeline/young_tdi_hull.sbatch        (subset: --array=0-4)
# then:  python3 mlp_pipeline/tdi/summarize.py runs/tdi   -> mixing free energy per atom vs x, in the same form as the hull sheet (needs the collect.py json of each run, written at the end of each task)
#SBATCH --job-name=tdihull
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
. mlp_pipeline/array_env.sh   # drop run knobs exported by the submitting shell (they would apply to every task)
DEFAULT="liquid 400 7000 2;liquid 400 7000 2.25;liquid 400 7000 3;liquid 400 7000 5;H 400 7000 2;liquid 400 9000 2;liquid 400 9000 2.25;liquid 400 9000 3;liquid 400 9000 5;H 400 9000 2"
IFS=';' read -ra TABLE <<< "${JOBS:-$DEFAULT}"
I=${SLURM_ARRAY_TASK_ID:-0}; read -r PH P T HP <<< "${TABLE[$I]}"
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: $PH at $P GPa, $T K, H per O $HP on $(hostname) $(date)"
P=$P T=$T HPER=$HP NODES=10 NPAR=10 DT=${DT:-0.5} EQUIL=${EQUIL:-1.0} PS=${PS:-2} CHUNK=0.25 PS1=1.5 bash mlp_pipeline/young_tdi_par.sh $PH
