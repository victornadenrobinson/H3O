#!/bin/bash -l
# Grid Engine job script (UCL Young). Directive syntax is from memory: check the Young docs for GPU request / project / queue names.
#$ -N mace_ft_h3o
#$ -l h_rt=12:00:00
#$ -l mem=40G
#$ -l gpu=1
#$ -pe smp 8
#$ -A <project_code>
#$ -wd /path/to/H3O
# usage: TAG=m qsub -v TAG=m mlp_pipeline/05_finetune_young.sh
source ~/venvs/mace/bin/activate
TAG=${TAG:-m}
RUNS=${H3O_RUNS:-$PWD/runs}
mkdir -p $RUNS/mace_$TAG && cd $RUNS/mace_$TAG
mace_run_train \
  --name=mace_h3o_ft_$TAG --foundation_model=medium \
  --train_file=$RUNS/train_$TAG.extxyz --valid_file=$RUNS/valid_$TAG.extxyz \
  --energy_key=energy --forces_key=forces --stress_key=stress \
  --loss=universal --energy_weight=1 --forces_weight=100 --stress_weight=100 \
  --E0s=average --lr=0.0001 --ema --ema_decay=0.99 --amsgrad \
  --batch_size=4 --max_num_epochs=100 --default_dtype=float64 --device=cuda --seed=1 --save_cpu
