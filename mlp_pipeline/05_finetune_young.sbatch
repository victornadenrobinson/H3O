#!/bin/bash -l
# Slurm job for UCL Young (A100). account/qos below are from the working interactive line (intergpu/allusers); a 12 h fine-tune probably
# needs a longer-running GPU partition: check `sinfo -o '%P %l %G'` and fill in the partition/qos before submitting.
#SBATCH --job-name=mace_ft_h3o
#SBATCH --partition=<long-gpu-partition>
#SBATCH --account=allusers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# usage: TAG=m sbatch mlp_pipeline/05_finetune_young.sbatch      (run from the repo root; TAG picks train_${TAG}/valid_${TAG})
# Flags are from memory: verify with `mace_run_train --help`. --E0s=estimated re-fits the foundation model E0s to the new reference energies (docs advise against "average" for fine-tuning).
source ~/work/envs/mace-env-ng.sh              # helper script in ~/work/envs (check what it loads); or: source ~/work/envs/mace-ng/bin/activate
module load cuda/12.6.2/gcc-12.3.0-avx2         # torch in mace-ng is the cu126 build
TAG=${TAG:-m}
RUNS=${H3O_RUNS:-$PWD/runs}
mkdir -p $RUNS/mace_$TAG && cd $RUNS/mace_$TAG
mace_run_train \
  --name=mace_h3o_ft_$TAG --foundation_model=medium \
  --train_file=$RUNS/train_$TAG.extxyz --valid_file=$RUNS/valid_$TAG.extxyz \
  --energy_key=energy --forces_key=forces --stress_key=stress \
  --loss=universal --energy_weight=1 --forces_weight=100 --stress_weight=100 \
  --E0s=estimated --lr=0.0001 --ema --ema_decay=0.99 --amsgrad \
  --batch_size=4 --max_num_epochs=100 --default_dtype=float64 --device=cuda --seed=1 --save_cpu
