#!/bin/bash -l
# 20 ps NVT stability check of the O sublattice (does bcc stay bcc, fcc stay fcc?) at the TDI state points, with the fine-tuned GRACE model.
# usage on Young:  cd /lustre/scratch/mmm0037/H3O && git pull && sbatch mlp_pipeline/young_stability.sbatch
# tasks: 0 bcc 500 GPa, 1 fcc 500, 2 bcc 400, 3 fcc 400 (3000 K; volume from the DFT EOS; override the model with MODEL=grace-ft:runs/grace/ft_m4 sbatch ...)
# Result lines (RESULT {...}) contain O_struct, CN1, CN2, delta_L (Lindemann ratio of the O lattice) and msd_O_end.
#SBATCH --job-name=stab
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) PH=bcc; P=500 ;; 1) PH=fcc; P=500 ;; 2) PH=bcc; P=400 ;; 3) PH=fcc; P=400 ;;
esac
export GRACE_UQ=1
$HOME/grace-venv/bin/python mlp_pipeline/17_eos_md.py $MODEL --phase $PH --T 3000 --from-dft inventory_binned.csv --P $P --ps 20 --equil-ps 2 --tag stab_${PH}_${P}GPa
