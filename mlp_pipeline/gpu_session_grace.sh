#!/bin/bash
# GRACE model comparison in a GPU session (names from the gracemaker docs; fp32 plain names). Logs: runs/gpu_session_grace/
# usage: bash mlp_pipeline/gpu_session_grace.sh [MIN_BOX=8.5] [N=30]
cd "$(dirname "$0")/.."
MB=${1:-8.5}; N=${2:-30}
G=$HOME/grace-venv/bin
export GRACE_UQ=1 TF_FORCE_GPU_ALLOW_GROWTH=true
L=runs/gpu_session_grace; mkdir -p $L
step() { name=$1; shift; echo "=== $(date +%H:%M:%S) $name"; "$@" > $L/$name.log 2>&1; echo "    exit $?  -> $L/$name.log"; grep -v -E "^(OK[0-9]|bashrc)|^I0000|^WARNING|tensorflow|xla|cuda_" $L/$name.log | tail -n ${TAILN:-22}; }
$G/grace_models list 2>&1 | head -40
for M in GRACE-2L-SMAX-OMAT-medium GRACE-2L-SMAX-OMAT-large GRACE-2L-OMAT-large-ft-E GRACE-1L-SMAX-OMAT-large GRACE-3L-OMAT-large; do
  TAILN=30 step "bench_$M" $G/python mlp_pipeline/04_benchmark.py grace:$M valid_m.extxyz --n $N --min-box $MB --device cuda
done
for M in GRACE-2L-SMAX-OMAT-medium GRACE-2L-SMAX-OMAT-large GRACE-1L-SMAX-OMAT-large; do
  TAILN=18 step "mem_$M" $G/python mlp_pipeline/13_memory_scaling.py grace:$M valid_m.extxyz --min-box $MB --cut 6
done
echo "=== done $(date +%H:%M:%S)"
