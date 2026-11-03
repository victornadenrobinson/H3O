#!/bin/bash
# Score every FINISHED fine-tune that has no results yet, using all GPUs of the allocation, then print the comparison tables.
#   per model:  runs/D_<name>.csv/.txt    on runs/valid_n.extxyz            (common held-out frames, A/B/C/... sets)
#               runs/Dmd_<name>.csv/.txt  on runs/dft_md_{400,500,600}.extxyz (the DFT-ensemble frames; composition test)
# Skips runs with a queued/running job named ft_<name> and runs without a final model. Models are dealt out round-robin to the GPUs, one process per GPU.
#   bash mlp_pipeline/eval_round.sh               (interactive GPU job, repo root; PY and TMOUT as usual)
#   ONLY="a b c" bash mlp_pipeline/eval_round.sh  restrict to these runs;   FORCE=1 re-score existing results
cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline
VALID=${VALID:-runs/valid_n.extxyz}; MD="runs/dft_md_400.extxyz runs/dft_md_500.extxyz runs/dft_md_600.extxyz"
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
busy=$(squeue -u "$USER" -h -o %j 2>/dev/null); todo=()
for d in runs/grace/ft_*; do
  n=${d#runs/grace/ft_}; [ -e "$d/seed/1/final_model/saved_model.pb" ] || continue
  echo "$busy" | grep -qx "ft_$n" && { echo "skip $n (job still queued/running)"; continue; }
  if [ -n "${ONLY:-}" ]; then case " $ONLY " in *" $n "*) ;; *) continue;; esac; fi
  if [ "${FORCE:-0}" != 1 ] && [ -e "runs/D_$n.csv" ] && [ -e "runs/Dmd_$n.csv" ]; then continue; fi
  todo+=("$n")
done
echo "${#todo[@]} models to score on $NG GPU(s): ${todo[*]}"
filt() { grep -v "tensorflow\|gpu_device\|tensorpotential\|cpu_feature\|rebuild\|absl\|ptxas\|XLA\|StreamExecutor\|cuDNN\|^$\|GRACE-D"; }
for ((g = 0; g < NG; g++)); do
  ( for ((i = g; i < ${#todo[@]}; i += NG)); do
      n=${todo[$i]}; m="grace-ft:runs/grace/ft_$n/seed/1/final_model"
      [ "${FORCE:-0}" = 1 ] || [ ! -e "runs/D_$n.csv" ] && { CUDA_VISIBLE_DEVICES=$g $PY mlp_pipeline/baseline_D.py $VALID --model "$m" --out "runs/D_$n.csv" 2>&1 | filt > "runs/D_$n.txt"; }
      [ "${FORCE:-0}" = 1 ] || [ ! -e "runs/Dmd_$n.csv" ] && { CUDA_VISIBLE_DEVICES=$g $PY mlp_pipeline/baseline_D.py $MD --model "$m" --out "runs/Dmd_$n.csv" 2>&1 | filt > "runs/Dmd_$n.txt"; }
      echo "[gpu $g] done $n"
    done ) &
done; wait
echo; $PY mlp_pipeline/compare_D.py runs/D_*.csv
echo; $PY mlp_pipeline/composition_check.py runs/Dmd_*.csv 2>&1 | grep -v "+   1 H" 
