#!/bin/bash
#SBATCH -A m3349_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 12:00:00
#SBATCH -J ft4
# One GPU node = 4 A100s: run the 4 fine-tunes listed in $CMDS (a file made by ft_series_cmds.sh) at the same time, one per GPU. Resumable per line (run_lines_gpus.sh skips lines with a DONE log).
#   sbatch -J ftr1 --export=ALL,CMDS=ft_r1.sh mlp_pipeline/ft_node.sbatch
set -u; cd "${SLURM_SUBMIT_DIR:-.}"; export PYTHONPATH=$PWD/mlp_pipeline
bash mlp_pipeline/run_lines_gpus.sh "${CMDS:?CMDS=<command file>}" 1 4
