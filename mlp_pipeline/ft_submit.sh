#!/bin/bash
# Submit every fine-tune line of a command file as its OWN single-GPU shared-QOS job (-q shared -G 1 -c 32), skipping runs that already have a final model or a queued/running job named ft_<NAME>.
#   bash mlp_pipeline/ft_submit.sh ft_r1.sh            (login node, repo root; DRY=1 only prints)  [TIME=12:00:00]
# Host RAM: gracemaker builds all training batches in host memory first. The shared queue gives 32 cores (and ~1/4 of the node's RAM, ~55 GB) per GPU and refuses more memory per GPU,
# so the 4.3k-frame mfin tier (OOM-killed at 1 GPU) needs GPUS=2: -G 2 -c 64 and twice the RAM; the run itself uses only the first GPU (CUDA_VISIBLE_DEVICES=0). MEM adds --mem explicitly.
# Unfrozen ('naive') runs get an 80 GB GPU (-C 'gpu&hbm80g'); frozen runs take any A100 (batch 1 fits in 40 GB, ~2.3 updates/s).
set -u; cd "$(dirname "$0")/.."; F=${1:?command file}; T=${TIME:-12:00:00}; G=${GPUS:-1}; mkdir -p logs_ft
mapfile -t L < <(grep -v '^\s*#' "$F" | grep -v '^\s*$')
for line in "${L[@]}"; do
  name=$(echo "$line" | awk '{print $3}'); mode=$(echo "$line" | awk '{print $10}')
  [ -e "runs/grace/ft_$name/seed/1/final_model/saved_model.pb" ] && { echo "skip $name (model exists)"; continue; }
  squeue -u "$USER" -h -o %j | grep -qx "ft_$name" && { echo "skip $name (already queued/running)"; continue; }
  C="gpu"; [ "$mode" = naive ] && C="gpu&hbm80g"
  cmd="sbatch -A m3349_g -C \"$C\" -q shared -G $G -c $((32 * G)) ${MEM:+--mem=$MEM} -N 1 -t $T -J ft_$name -o logs_ft/$name.%j.out --wrap=\"cd $PWD && export PYTHONPATH=$PWD/mlp_pipeline && export CUDA_VISIBLE_DEVICES=0 && $line\""
  echo "$name  ($C, $G GPU, $((32 * G)) cores)"; [ "${DRY:-0}" = 1 ] || eval "$cmd"
done
