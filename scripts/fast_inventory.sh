#!/usr/bin/env bash
# Multi-core streaming inventory of VASP OUTCARs -> mined_inventory_ground_truth.csv
# usage: fast_inventory.sh ROOT [OUT.csv] [WORKERS]
set -euo pipefail
ROOT=${1:?root dir}; OUT=${2:-mined_inventory_ground_truth.csv}; JOBS=${3:-32}

one() {
  f=$1
  if [[ $f == *.gz ]]; then cat_=(gzip -dc); else cat_=(cat); fi
  t=$("${cat_[@]}" "$f" 2>/dev/null | head -n 1000 | grep -m1 'TEBEG' | sed -E 's/.*TEBEG *= *([0-9.eE+-]+).*/\1/' || true)
  tail_=$("${cat_[@]}" "$f" 2>/dev/null | tail -n 2500 || true)
  p=$(grep 'external pressure' <<<"$tail_" | tail -1 | sed -E 's/.*external pressure *= *(-?[0-9.]+).*/\1/' || true)
  v=$(grep 'volume of cell' <<<"$tail_" | tail -1 | sed -E 's/.*volume of cell *: *([0-9.]+).*/\1/' || true)
  # VASP reports kB; 1 GPa = 10 kB
  [[ -n $p ]] && p=$(awk -v p="$p" 'BEGIN{printf "%.3f", p/10}')
  echo "$f,${t},${p},${v}"
}
export -f one

{ echo "Path,True_T_K,True_P_GPa,True_V_A3"
  find "$ROOT" -type f \( -name 'OUTCAR' -o -name 'OUTCAR.gz' \) -print0 \
    | xargs -0 -P "$JOBS" -I{} bash -c 'one "$1"' _ {}
} > "$OUT"
