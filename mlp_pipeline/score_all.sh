#!/bin/bash
# Score every fine-tune that has a final model and no D_<name>.csv yet on ONE common held-out file (default runs/valid_n.extxyz), one GPU, in this shell's allocation.
# Writes runs/D_<name>.csv and runs/D_<name>.txt (the printed tables). Submit as a batch job so idle logout cannot kill it:
#   sbatch -A m3349_g -C gpu -q shared -G 1 -c 32 -t 02:00:00 -J score --wrap="cd $PWD && bash mlp_pipeline/score_all.sh"
cd "$(dirname "$0")/.."; VALID=${VALID:-runs/valid_n.extxyz}; PY=${PY:-$SCRATCH/grace-env/bin/python}; mkdir -p runs; n=0
for d in runs/grace/ft_*; do
  name=${d#runs/grace/ft_}; [ -e "$d/seed/1/final_model/saved_model.pb" ] || continue
  [ -e "runs/D_$name.csv" ] && [ "${FORCE:-0}" != 1 ] && { echo "have D_$name"; continue; }
  echo "scoring $name"; $PY mlp_pipeline/baseline_D.py "$VALID" --model "grace-ft:$d/seed/1/final_model" --out "runs/D_$name.csv" 2>&1 | grep -v "tensorflow\|gpu_device\|tensorpotential\|cpu_feature\|rebuild\|absl\|ptxas\|XLA\|StreamExecutor\|cuDNN\|^$\|GRACE-D" > "runs/D_$name.txt"; n=$((n+1))
done; echo "scored $n new models"
