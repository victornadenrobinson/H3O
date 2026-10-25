#!/bin/bash
# Zero-shot benchmark of MACE candidates (names confirmed by list_models.py on mace-ng 0.3.16). MH_HEAD=<head> adds mh-1 on that head.
# usage: bash mlp_pipeline/gpu_session_models.sh [MIN_BOX=8.5] [N=30] [extra mace names...]
cd "$(dirname "$0")/.."
MB=${1:-8.5}; N=${2:-30}; shift 2 2>/dev/null
P=$HOME/work/envs
export MLP_DTYPE=float32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/gpu_session_models; mkdir -p $L
step() { name=$1; shift; echo "=== $(date +%H:%M:%S) $name"; "$@" > $L/$name.log 2>&1; echo "    exit $?  -> $L/$name.log"; grep -v -E "^(OK[0-9]|bashrc)|Warning|torch.load" $L/$name.log | tail -n ${TAILN:-24}; }
NAMES="medium-mpa-0 medium-0b3 large-0b2 medium-0b2 mace-matpes-pbe-0 small-omat-0"
NAMES="$NAMES $($P/mace-ng/bin/python mlp_pipeline/mace_heads.py mh-1 mh-0 --names 2>/dev/null | grep @ | tr '\n' ' ')"   # every head of mh-1 and mh-0
[ -n "$MH_HEAD" ] && NAMES="$NAMES mh-1@$MH_HEAD"
echo "models: $NAMES"
for M in $NAMES "$@"; do
  TAILN=26 step "bench_mace_$M" $P/mace-ng/bin/python mlp_pipeline/04_benchmark.py mace:$M valid_m.extxyz --n $N --min-box $MB --device cuda
done
echo "=== done $(date +%H:%M:%S)"
