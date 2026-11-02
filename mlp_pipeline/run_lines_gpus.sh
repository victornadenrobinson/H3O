#!/usr/bin/env bash
# Run every line of a command file (e.g. low_p_pass1.sh from eos_low_p.py) spread over the GPUs of one node: PER_GPU processes per GPU, lines handed out round-robin.
# usage (GPU node, repo root, PYTHONPATH containing mlp_pipeline/):  bash mlp_pipeline/run_lines_gpus.sh low_p_pass1.sh [PER_GPU=2] [NGPU=4]
# Each line's output goes to <file>.logs/<line number>.log. Lines that already have a finished log ("DONE" marker) are skipped, so a restart resumes.
set -u
F=${1:?command file}; PER=${2:-2}; NG=${3:-4}; W=$((PER * NG)); mkdir -p "$F.logs"; export TF_FORCE_GPU_ALLOW_GROWTH=true
mapfile -t L < <(grep -v '^\s*#' "$F" | grep -v '^\s*$'); echo "${#L[@]} commands on $W workers"
for ((w = 0; w < W; w++)); do
  ( export CUDA_VISIBLE_DEVICES=$((w % NG)); for ((i = w; i < ${#L[@]}; i += W)); do
      lg="$F.logs/$i.log"; grep -q '^DONE$' "$lg" 2>/dev/null && continue
      { eval "${L[$i]}"; rc=$?; if [ $rc = 0 ]; then echo DONE; else echo "FAILED rc=$rc"; fi; } > "$lg" 2>&1   # DONE only on exit code 0, so a crashed line is rerun on restart
    done ) &
done
wait; echo "finished: $(grep -l '^DONE$' "$F".logs/*.log | wc -l) of ${#L[@]} logs marked DONE"
