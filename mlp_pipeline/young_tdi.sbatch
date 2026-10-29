#!/bin/bash -l
# Thermodynamic integration (Einstein O + ideal-gas H -> GRACE) at 400 GPa, 3000 K: array task 0 = bcc, 1 = fcc (one phase per job; both run at the same time).
# usage on Young:  cd /lustre/scratch/mmm0037/H3O && git pull && sbatch mlp_pipeline/young_tdi.sbatch
#   other state point / model:  P=500 MODEL=grace-ft:runs/grace/ft_m4 sbatch mlp_pipeline/young_tdi.sbatch      longer:  PS=8 sbatch --time=16:00:00 ...
# Output: the log prints G (meV/atom) versus simulation time per lambda node every 0.5 ps; json in runs/tdi/. When both are done:
#   python mlp_pipeline/tdi/compare.py runs/tdi/tdi_fcc_*.json runs/tdi/tdi_bcc_*.json
# Quick test (about 5 min, writes to runs/tdi_test, numbers not meaningful):  TAG=test PS=0.5 NODES=3 PS1=0.5 CHUNK=0.25 EQUIL=0.2 bash mlp_pipeline/young_tdi.sbatch
# Cost guess: 8 nodes x 4 ps = 64,000 steps x 0.17 s = about 3 h (+ 20 min for the lambda=1 run).
#SBATCH --job-name=tdi
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; P=${P:-400}; PS=${PS:-4}; NODES=${NODES:-8}; PS1=${PS1:-3}; CHUNK=${CHUNK:-0.5}; EQUIL=${EQUIL:-1}; TAG=${TAG:-}
case ${SLURM_ARRAY_TASK_ID:-0} in 0) PH=bcc ;; 1) PH=fcc ;; esac
export GRACE_UQ=1
$HOME/grace-venv/bin/python -u mlp_pipeline/tdi/run_real.py $MODEL --phase $PH --P $P --T 3000 --from-dft inventory_binned.csv --ps $PS --nodes $NODES --power 4 --ps1 $PS1 --chunk-ps $CHUNK --equil-ps $EQUIL ${TAG:+--tag $TAG --outdir runs/tdi_test}
