#!/bin/bash
# Benchmark GRACE zero-shot and the fine-tuned models on the same validation frames, then plot them side by side. Run in a GPU job (Young), repo root.
# usage: bash mlp_pipeline/bench_grace_compare.sh [data.extxyz=valid_m.extxyz] [n_frames=0 (all)] [extra fine-tune dirs, e.g. runs/grace/ft_m2 ...]
cd "$(dirname "$0")/.."
D=${1:-valid_m.extxyz}; N=${2:-0}; shift 2 2>/dev/null
PY=$HOME/grace-venv/bin/python
MODELS=("grace:GRACE-2L-SMAX-OMAT-medium" "grace-ft:runs/grace/ft_s" "grace-ft:runs/grace/ft_m")
LABELS=("zero-shot" "FT s" "FT m")
for extra in "$@"; do MODELS+=("grace-ft:$extra"); LABELS+=("FT $(basename "$extra")"); done
FILES=()
B=$(basename "$D" .extxyz)
for m in "${MODELS[@]}"; do
  GRACE_UQ=1 $PY mlp_pipeline/04_benchmark.py "$m" "$D" --n "$N" --min-box 8.5 --device cuda 2>&1 | grep -E "^(overall|E_MAE|F_RMSE|S_MAE|P_err|N |SKIPPED)|Error|frames,"
  FILES+=("runs/bench_$(echo "$m" | sed 's#/#_#g; s#:#-#g' | cut -c1-60)_${B}.csv")
done
~/work/envs/mace-ng/bin/python mlp_pipeline/20_plot_benchmark.py "${FILES[@]}" --labels "${LABELS[@]}" --out runs/bench_compare_${B}.png
