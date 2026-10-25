#!/bin/bash
# Everything worth doing in a GPU session, in priority order; each step logs to runs/gpu_session/ and a failure does not stop the rest.
# usage (inside the GPU job, repo root):  bash mlp_pipeline/gpu_session.sh [MIN_BOX=8.5] [N=60]
cd "$(dirname "$0")/.."
MB=${1:-8.5}; N=${2:-60}
P=$HOME/work/envs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load cuda/12.6.2/gcc-12.3.0-avx2 2>/dev/null
L=runs/gpu_session; mkdir -p $L
step() { name=$1; shift; echo "=== $(date +%H:%M:%S) $name"; "$@" > $L/$name.log 2>&1; echo "    exit $?  -> $L/$name.log"; grep -v -E "^(OK[0-9]|bashrc)" $L/$name.log | tail -n ${TAILN:-14}; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
# 1. how big a cell fits (decides whether MatterSim is usable at all)
TAILN=22 step mem_mattersim  $P/mattersim-env/bin/python mlp_pipeline/13_memory_scaling.py mattersim valid_m.extxyz --min-box $MB
TAILN=22 step mem_mace_omat  $P/mace-ng/bin/python       mlp_pipeline/13_memory_scaling.py mace:medium-omat-0 valid_m.extxyz --min-box $MB
# 2. zero-shot accuracy on production-like frames (stratified sample, big cells that do not fit are skipped and reported)
TAILN=26 step bench_mattersim   $P/mattersim-env/bin/python mlp_pipeline/04_benchmark.py mattersim valid_m.extxyz --n $N --min-box $MB --device cuda
TAILN=26 step bench_mace_omat   $P/mace-ng/bin/python       mlp_pipeline/04_benchmark.py mace:medium-omat-0 valid_m.extxyz --n $N --min-box $MB --device cuda
# 3. the larger MatterSim checkpoint (file name is a guess: see the log if it fails)
TAILN=26 step bench_mattersim5m $P/mattersim-env/bin/python mlp_pipeline/04_benchmark.py mattersim:mattersim-v1.0.0-5M.pth valid_m.extxyz --n $N --min-box $MB --device cuda
# 4. models that blew up on the first frame: smaller samples, to see if it is everywhere
TAILN=26 step bench_mace_mp0    $P/mace-ng/bin/python       mlp_pipeline/04_benchmark.py mace:medium valid_m.extxyz --n 20 --min-box $MB --device cuda
TAILN=26 step bench_grace       $HOME/grace-venv/bin/python mlp_pipeline/04_benchmark.py grace:GRACE-2L-OMAT valid_m.extxyz --n 20 --min-box $MB --device cuda
echo "=== done $(date +%H:%M:%S). Paste: tail -n 30 $L/*.log   (or send the five bench_* logs)"
