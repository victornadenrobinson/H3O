#!/bin/bash
# Quick smoke test (DEVICE=cpu default, DEVICE=cuda on a GPU node) of all four model setups on a SMALL snapshot (<=120 atoms): checks env + API without waiting for a GPU.
# Run from the repo root on Young (login node is OK for ~a minute per model; threads are capped). usage: bash mlp_pipeline/smoke_cpu.sh [data.extxyz]
cd "$(dirname "$0")/.."
D=${1:-valid_m.extxyz}
DEV=${DEVICE:-cpu}   # DEVICE=cuda for a GPU node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
LOG=$(mktemp)
[ "$DEV" = cuda ] && { module load cuda/12.6.2/gcc-12.3.0-avx2 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; }
echo "device $DEV  host $(hostname)  cpus $(nproc)  $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/ \+/ /g')  threads $OMP_NUM_THREADS"
run() { echo "=================== $1"; shift; timeout 600 "$@" 2>&1 | grep -v -E "^(OK[0-9]|bashrc)" | tee -a "$LOG" | tail -n 16; }
run "MACE medium-omat-0" $HOME/work/envs/mace-ng/bin/python mlp_pipeline/smoke_test.py --model mace --name medium-omat-0 --data $D --max-atoms 120 --device $DEV
run "MACE medium (MP-0)" $HOME/work/envs/mace-ng/bin/python mlp_pipeline/smoke_test.py --model mace --name medium --data $D --max-atoms 120 --device $DEV
run "MatterSim" $HOME/work/envs/mattersim-env/bin/python mlp_pipeline/smoke_test.py --model mattersim --data $D --max-atoms 120 --device $DEV
run "GRACE" $HOME/grace-venv/bin/python mlp_pipeline/smoke_test.py --model grace --name GRACE-2L-OMAT --data $D --max-atoms 120 --device $DEV
echo; echo "=================== timing summary (steady = median of calls 3-6: energy+forces+stress)"
grep -h "^TIMING" "$LOG" | sed 's/^TIMING //' | (column -t || cat) 2>/dev/null
echo "(a failed model has no TIMING line; scroll up for its traceback)"
rm -f "$LOG"
