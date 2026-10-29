#!/bin/bash
# Run several sbatch array tasks side by side on ONE GPU inside an interactive allocation (e.g. Young's intergpu partition, 6 h limit), instead of waiting in the gpu queue.
# Each item is <sbatch script>[:<array task>]; the script is run with bash and SLURM_ARRAY_TASK_ID set (all our young_*.sbatch read their table entry from it), so environment
# settings (PS=, JOBS=, MODEL=, ...) given on this command line apply to every item. Processes share the GPU (TF_FORCE_GPU_ALLOW_GROWTH), starts are staggered, and a status
# block (elapsed time + last line of each log) is printed every STATUS minutes until all have finished.
# MIND THE GPU MEMORY: a single-process task (stability_npt, h3o_static per pressure, eos points) takes ~3-5 GB; a TDI task (young_tdi*.sbatch) itself starts up to ~9 lambda
# processes, so put at most one TDI task on a GPU. Interactive allocations end hard at their time limit: choose PS / tasks that fit (the logs say ms/step and ETA).
# usage (repo root, inside e.g.  srun -p intergpu --gres=gpu:1 -c 4 --mem=60G --time=6:00:00 --pty bash):
#   PS=20 bash mlp_pipeline/run_on_gpu.sh mlp_pipeline/young_stability_npt.sbatch:1 mlp_pipeline/young_stability_npt.sbatch:2 mlp_pipeline/young_stability_npt.sbatch:4
#   logs: runs/ongpu/<script>_<task>.log
[ -d mlp_pipeline ] || { echo "run from the repo root"; exit 2; }
[ $# -gt 0 ] || { sed -n 2,11p "$0"; exit 2; }
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p runs/ongpu; pids=(); logs=(); t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
for item in "$@"; do
  s=${item%%:*}; k=0; [[ $item == *:* ]] && k=${item##*:}
  [ -f "$s" ] || { echo "no such script: $s"; continue; }
  log=runs/ongpu/$(basename "$s" .sbatch)_$k.log
  SLURM_ARRAY_TASK_ID=$k SLURM_ARRAY_JOB_ID=ongpu$$ bash "$s" > "$log" 2>&1 &
  pids+=($!); logs+=("$log"); echo "== started $s task $k (pid $!) -> $log  ($(date +%H:%M:%S))"
  [ "$item" != "${@: -1}" ] && sleep ${STAGGER:-20}   # model load + XLA compile of several processes at once can exceed the GPU memory
done
while :; do
  sleep $(( ${STATUS:-10} * 60 )) & wait $!
  n=0; echo "== status after $(( ($(date +%s) - t0) / 60 )) min ($(date +%H:%M))"
  for i in "${!pids[@]}"; do
    if kill -0 ${pids[$i]} 2>/dev/null; then n=$((n + 1)); st=running; else st=finished; fi
    printf "   %-9s %-45s %s\n" $st "${logs[$i]}" "$(grep -v '^\s*$' "${logs[$i]}" | tail -n1 | cut -c1-150)"
  done
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/   GPU: /'
  [ $n -eq 0 ] && break
done
echo "== all finished after $(( ($(date +%s) - t0) / 60 )) min"; grep -h "VERDICT\|^RESULT\|Error\|Traceback" "${logs[@]}" 2>/dev/null | cut -c1-250
