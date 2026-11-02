#!/usr/bin/env bash
# Everything left before the DFT labelling, in order, unattended. Run inside a GPU job (tmux!), repo root:
#   export PY=$SCRATCH/grace-env/bin/python M=grace-ft:$PWD/runs/grace/ft_m2/seed/1/final_model PYTHONPATH=$PWD/mlp_pipeline
#   bash mlp_pipeline/finish_generation.sh 2>&1 | tee ~/leg/finish.log        (resumable: every stage skips finished work; STAGES="1 2 3 4" picks stages)
# 1. E pass 2: low-P MD at the corrected volumes (eos_low_p.py output -> runs/eos_lowp2), spread over the GPUs of this job
# 2. G top-up: second seed of the 729-atom pure-H MD (~/leg/H729_md_s2.extxyz), in parallel with stage 1
# 3. wait for the H (Huang supercell) runs from the other job: h_gpu.sh.logs all DONE, up to WAIT_MIN minutes
# 4. rebuild the snapshot database from everything, verify, plot  -> snapdb/
set -u
: "${PY:?}" "${M:?}"; STAGES=${STAGES:-"1 2 3 4"}; WAIT_MIN=${WAIT_MIN:-90}; PER_GPU=${PER_GPU:-2}; NGPU=${NGPU:-4}; LEG=${LEG:-$HOME/leg}; export PYTHONPATH=${PYTHONPATH:-$PWD/mlp_pipeline}
has() { [[ " $STAGES " == *" $1 "* ]]; }
if has 1; then echo "== stage 1: E pass 2 ($(date +%H:%M))"
  $PY mlp_pipeline/eos_low_p.py runs/eos_scan runs/eos_map runs/eos_lowp 2>/dev/null | grep '^python' | grep -v -- '--phase H\|--T 9000' | sed "s#^python3\? #$PY #; s#--outdir runs/eos_lowp#--outdir runs/eos_lowp2#" > e2_gpu.sh
  echo "$(wc -l < e2_gpu.sh) commands in e2_gpu.sh"; [ -s e2_gpu.sh ] && bash mlp_pipeline/run_lines_gpus.sh e2_gpu.sh $PER_GPU $NGPU > e2_gpu.run.log 2>&1 &
  E2=$!; fi
if has 2; then echo "== stage 2: G top-up ($(date +%H:%M))"
  if grep -qs "^wrote" $LEG/H729_md_s2.log 2>/dev/null; then echo "   already done"; else
    CUDA_VISIBLE_DEVICES=$((NGPU - 1)) TF_FORCE_GPU_ALLOW_GROWTH=true $PY -u ${SCRIPT:-$HOME/build_insertion_set.py} --hosts runs/train_l.extxyz --comp h --natoms 729 --n-list 0 --max-hosts 3 --model $M --equil-ps 0.3 --dt 1.0 --sample-ps 1.0 --save-every-ps 0.1 --seed 2 --out $LEG/H729_md_s2.extxyz > $LEG/H729_md_s2.log 2>&1 &
    G2=$!; fi; fi
[ -n "${E2:-}" ] && { wait $E2; echo "   stage 1 finished: $(grep -c '^DONE' e2_gpu.sh.logs/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s}') DONE, $(grep -l '^FAILED' e2_gpu.sh.logs/*.log 2>/dev/null | wc -l) FAILED of $(wc -l < e2_gpu.sh)"; }
[ -n "${G2:-}" ] && { wait $G2; tail -n 1 $LEG/H729_md_s2.log; }
if has 3; then echo "== stage 3: wait for H runs ($(date +%H:%M))"; t=0
  while [ -d h_gpu.sh.logs ] && [ "$(grep -l '^DONE' h_gpu.sh.logs/*.log 2>/dev/null | wc -l)" -lt "$(grep -c build_h3o h_gpu.sh 2>/dev/null || echo 8)" ] && [ $t -lt $WAIT_MIN ]; do sleep 60; t=$((t + 1)); done
  echo "   H logs DONE: $(grep -l '^DONE' h_gpu.sh.logs/*.log 2>/dev/null | wc -l); FAILED: $(grep -l '^FAILED' h_gpu.sh.logs/*.log 2>/dev/null | wc -l); H frames: $(cat $LEG/h3o_sc/*.extxyz 2>/dev/null | grep -c Lattice=)"; fi
if has 4; then echo "== stage 4: assemble, verify, plot ($(date +%H:%M))"; $PY mlp_pipeline/assemble_snapdb.py --leg $LEG --root . --out snapdb; fi
echo "== finished $(date +%H:%M): next  $PY mlp_pipeline/snapdb.py chunk snapdb --prio-max 2   (after reading snapdb/coverage.png and the verify block)"
