#!/bin/bash -l
# One-snapshot smoke test per model on Young (interactive-GPU partition; values from the user's working srun line).
#SBATCH --job-name=smoke
#SBATCH --partition=intergpu
#SBATCH --qos=intergpu
#SBATCH --account=allusers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=runs/smoke_%x_%j.out
# usage, from the repo root:
#   MODEL=mace      PY=~/work/envs/mace-ng/bin/python        NAME=medium-omat-0 sbatch mlp_pipeline/smoke_test.sbatch
#   MODEL=mattersim PY=~/work/envs/mattersim-env/bin/python                     sbatch mlp_pipeline/smoke_test.sbatch
#   MODEL=grace     PY=~/grace-venv/bin/python               NAME=GRACE-2L-OMAT sbatch mlp_pipeline/smoke_test.sbatch
module load cuda/12.6.2/gcc-12.3.0-avx2 2>/dev/null   # torch envs are cu126; GRACE (TensorFlow) may need other libs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
$PY mlp_pipeline/smoke_test.py --model $MODEL ${NAME:+--name $NAME} --data ${DATA:-valid_m.extxyz}
