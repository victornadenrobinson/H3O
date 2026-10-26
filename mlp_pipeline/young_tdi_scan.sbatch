#!/bin/bash -l
# Free-energy scan as a Slurm array: task i runs entry i of the job table (phase P_GPa T_K H_per_O). Each task = one GPU, about 40 min: a 6 min lambda=1 stage that measures k,
# then 10 lambda nodes x (0.4 + 3 ps) in parallel on the GPU. Restartable: resubmit the same command and unfinished nodes continue.
# default table (6 tasks, --array=0-5):  3000 K: bcc/fcc at 300 and 500 GPa;  5000 K: bcc/fcc at 400 GPa
# usage on Young:   cd /lustre/scratch/mmm0037/H3O && git pull && sbatch mlp_pipeline/young_tdi_scan.sbatch
#   own table:  JOBS="bcc 200 3000 2;fcc 200 3000 2;bcc 400 5000 2.25;fcc 400 5000 2.25" sbatch --array=0-3 mlp_pipeline/young_tdi_scan.sbatch
#   pure hydrogen entry (memory per process for 729 atoms not yet measured, so test first):  "H 400 5000 2"  (needs NATOMS=729 in the environment)
#   longer: PS=6 sbatch --time=02:00:00 ...
# results in runs/tdi/<phase>_<P>GPa_<T>K[_h<x>o]_* ; dashboard: ~/grace-venv/bin/python mlp_pipeline/tdi/live.py runs/tdi fcc_300GPa_3000K bcc_300GPa_3000K
# literature G_fcc - G_bcc, meV/atom at 3000 K (Cheng 2021 / Wilson 2013): 200 GPa -10.5 / -9.0, 300 GPa -17.6, 400 GPa -22.1, 500 GPa -26.5 / -24.7
#SBATCH --job-name=tdiscan
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
DEFAULT="bcc 300 3000 2;fcc 300 3000 2;bcc 500 3000 2;fcc 500 3000 2;bcc 400 5000 2;fcc 400 5000 2"
IFS=';' read -ra TABLE <<< "${JOBS:-$DEFAULT}"
I=${SLURM_ARRAY_TASK_ID:-0}; read -r PH P T HP <<< "${TABLE[$I]}"
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: $PH at $P GPa, $T K, H per O $HP on $(hostname) $(date)"
P=$P T=$T HPER=$HP NODES=10 NPAR=10 DT=1.0 EQUIL=0.4 PS=${PS:-3} CHUNK=0.25 PS1=1.5 bash mlp_pipeline/young_tdi_par.sh $PH
