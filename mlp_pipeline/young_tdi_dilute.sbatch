#!/bin/bash -l
# DIRECT crystal route at LOW H content (H2.02O, H2.05O, H2.10O: NH = 2, 5, 11 added H in the 108-O fcc cell), 10 ps per lambda, to compare with the insertion mu_H and the H2.25O row at the same (P, T).
# Settings match the converged 10 ps end members in runs/tdi_long (fcc H2O 9 lambda values, pure H 8): dt 0.5, equil 1.0, chunk 0.5, 9 lambda values. Results in runs/tdi_dil; summarize.py runs/tdi runs/tdi_long runs/tdi_dil pairs them with their H2O and pure-H partners.
# Statistical error per added H is about N_partner * sigma_partner / n: at 10 ps (sigma about 1 meV/atom for 324-351 atoms) roughly +-45 (n=11), +-100 (n=5), +-250 (n=2) meV per H: long runs of the partners matter more than these. Partners at 500 GPa / 6000 K exist (tdi_long);
# for 500 GPa / 5500 K and 400 GPa / 5000 K extend the end members first (young_tdi_endmembers.sbatch).
# Cost per task is an ESTIMATE: 9 processes at about 0.4 s/step, 22,000 steps = 2.5-3 h. Table "P T NH" per task. --time=08:00:00.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_dilute.sbatch        own table: JOBS="500 6000 5;400 5000 11" sbatch --array=0-1 ...
#SBATCH --job-name=tdidil
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-6%4
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
ARRAY_KEEP="PS"; . mlp_pipeline/array_env.sh
DEFAULT="500 6000 2;500 6000 5;500 6000 11;500 5500 5;500 5500 11;400 5000 5;400 5000 11"
IFS=';' read -ra TABLE <<< "${JOBS:-$DEFAULT}"
I=${SLURM_ARRAY_TASK_ID:-0}; read -r P T NH <<< "${TABLE[$I]}"
[ -n "$NH" ] || { echo "no table entry for array task $I"; exit 2; }
HP=$(python3 -c "print(round(2 + $NH / 108, 6))")
echo "== task $I: fcc H$HP O ($NH added H in 108 O) at $P GPa $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
P=$P T=$T HPER=$HP NODES=9 NPAR=9 DT=0.5 EQUIL=1.0 PS=${PS:-10} PS1=1.5 CHUNK=0.5 MINRUNS=8 OUTDIR=runs/tdi_dil bash mlp_pipeline/young_tdi_par.sh fcc
