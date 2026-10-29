#!/bin/bash
#SBATCH -A YOUR_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 12:00:00
#SBATCH -J mace_ft_h3o
# Flags are from memory: verify with `mace_run_train --help`. --E0s=estimated re-fits the foundation model E0s to the new reference energies (docs advise against "average" for fine-tuning).
module load python
TAG=${TAG:-m}                                   # which train/valid pair: train_${TAG}.extxyz
RUNS=${H3O_RUNS:-/pscratch/sd/v/victornr/H3O/runs}
mkdir -p $RUNS/mace_$TAG && cd $RUNS/mace_$TAG   # checkpoints/logs stay inside runs/
# conda activate <env with mace-torch>
mace_run_train \
  --name=mace_h3o_ft_$TAG \
  --foundation_model=medium \
  --train_file=$RUNS/train_$TAG.extxyz --valid_file=$RUNS/valid_$TAG.extxyz \
  --energy_key=energy --forces_key=forces --stress_key=stress \
  --loss=universal --energy_weight=1 --forces_weight=100 --stress_weight=100 \
  --E0s=estimated \
  --lr=0.0001 --ema --ema_decay=0.99 --amsgrad \
  --batch_size=4 --max_num_epochs=100 \
  --default_dtype=float64 --device=cuda --seed=1 --save_cpu
