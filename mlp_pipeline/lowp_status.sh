#!/bin/bash
# Status of a running lowp_snapshots / lowp_quick campaign (login node, repeatable; watch -n 20 bash mlp_pipeline/lowp_status.sh).  JOB=<slurm job id> adds the 4-GPU table.
cd "$(dirname "$0")/.."; OUT=${OUT:-runs/lowp3}; LOG=${LOG:-logs_pw/lowp3.log}; date +"== %H:%M:%S"
if [ -n "${JOB:-}" ]; then srun --overlap --jobid=$JOB --gres=none nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || srun --overlap --jobid=$JOB --gpus=4 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | head -6; fi
tot=$(cat "$OUT"/worker_*.txt 2>/dev/null | grep -c .); done=$(grep -c "^\[worker .* done" "$LOG" 2>/dev/null); fail=$(grep -c . "$OUT/failed.txt" 2>/dev/null)
echo "commands: $done / $tot finished, ${fail:-0} failed;  volume folders in runs/eos_lowp3: $(ls -d runs/eos_lowp3/*_v* 2>/dev/null | wc -l);  hydrate frames: $(ls runs/hydrates/*_relaxed.extxyz 2>/dev/null | wc -l) files"
echo "per worker (finished / assigned, last finished):"
for f in $(ls "$OUT"/worker_*.txt | sort -t_ -k2 -n); do w=$(basename $f .txt | sed 's/worker_//'); n=$(grep -c . $f); d=$(grep -c "^\[worker $w gpu" "$LOG"); last=$(grep "^\[worker $w gpu" "$LOG" | tail -1 | grep -o -- "--tag [^ ]*.*" | cut -c1-60)
  printf "  w%-2s gpu %s  %2s / %2s   %s\n" $w $((w % ${NG:-4})) $d $n "$last"; done
if [ -s "$OUT/failed.txt" ]; then echo "FAILED:"; cut -c1-170 "$OUT/failed.txt" | tail -8; fi
echo "recent MD progress (newest folders):"; ls -dt runs/eos_lowp3/*_v* 2>/dev/null | head -4 | while read d; do echo "  $(basename $d): $(ls $d | tr '\n' ' ' | cut -c1-70)"; done
