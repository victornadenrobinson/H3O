#!/bin/bash
# Score EVERY finished fine-tune on the sealed pair set and rank them by the insertion-energy error c (pair_check.py, meV per added H). One process per GPU, models dealt out round-robin.
#   bash mlp_pipeline/score_pairs_all.sh                      (GPU shell, repo root)       env: SET (runs/dgcheck/labelled.extxyz), OUT (runs/dgcheck/models), PY, ONLY="a b c"
# Skips models that already have a non-empty CSV; a failure prints FAILED <model> and the reason is in $OUT/D_<model>.txt. Finishes with a table sorted by |c|.
cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=$PWD/mlp_pipeline
SET=${SET:-runs/dgcheck/labelled.extxyz}; OUT=${OUT:-runs/dgcheck/models}; mkdir -p "$OUT"; [ -s "$SET" ] || { echo "no labelled set $SET"; exit 1; }
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
models=(); for d in runs/grace/ft_*/; do n=${d#runs/grace/ft_}; n=${n%/}; [ -e "$d/seed/1/final_model/saved_model.pb" ] || continue
  if [ -n "${ONLY:-}" ]; then case " $ONLY " in *" $n "*) ;; *) continue;; esac; fi; models+=("$n"); done
echo "${#models[@]} models, $NG GPU(s), set $SET"
for ((g = 0; g < NG; g++)); do
  ( for ((i = g; i < ${#models[@]}; i += NG)); do
      n=${models[$i]}; [ -s "$OUT/D_$n.csv" ] && continue
      CUDA_VISIBLE_DEVICES=$g $PY mlp_pipeline/baseline_D.py "$SET" --model "grace-ft:runs/grace/ft_$n/seed/1/final_model" --out "$OUT/D_$n.csv" < /dev/null > "$OUT/D_$n.txt" 2>&1 \
        && echo "[gpu $g] done $n" || echo "[gpu $g] FAILED $n (see $OUT/D_$n.txt)"
    done ) &
done; wait
printf '\n%-28s %6s %9s %7s %9s   (dD per added H, meV; c = mean - var/2kT; pass |c| <= 50)\n' model pairs mean sd c
for f in "$OUT"/D_*.csv; do [ -s "$f" ] || continue; n=$(basename "$f" .csv); n=${n#D_}
  $PY mlp_pipeline/pair_check.py "$f" 2>/dev/null | awk -v n="$n" '$1=="ALL"{printf "%-28s %6s %9s %7s %9s\n", n, $2, $3, $4, $5}'; done | sort -k5 -g | awk '{a=$5; if (a<0) a=-a; print a, $0}' | sort -n | cut -d' ' -f2-
