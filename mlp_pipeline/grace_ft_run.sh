#!/bin/bash
# Run a gracemaker fine-tune in its own folder and log GPU memory. usage (GPU job, repo root): bash mlp_pipeline/grace_ft_run.sh runs/grace/input.yaml [run_name]
cd "$(dirname "$0")/.."
Y=$(readlink -f "$1"); R=${2:-ft_s}; D=$(dirname "$Y")/$R
mkdir -p "$D" && cp "$Y" "$D/input.yaml" && cd "$D" || exit 1
# data paths in input.yaml are relative to where gracemaker runs: make the pickles visible here
for f in ../*.pkl.gz; do [ -e "$f" ] && ln -sf "$f" . ; done
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > gpu_mem.csv &
SMI=$!
export TF_FORCE_GPU_ALLOW_GROWTH=true
$HOME/grace-venv/bin/gracemaker input.yaml -l log.txt 2>&1 | grep -v -E "^(I0000|W0000|E0000|WARNING|20[0-9][0-9]-[0-9-]+ [0-9:.]+: [EIW]|\[tensorpotential\])" | tail -n ${TAILN:-40}
kill $SMI 2>/dev/null
echo "peak GPU memory (MiB): $(awk -F, '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END{print m}' gpu_mem.csv)  (log: $D/log.txt)"
echo "next: export the model (gracemaker -s ...) and register it in ~/.cache/grace/models_registry.yaml to benchmark with grace:<name>"
