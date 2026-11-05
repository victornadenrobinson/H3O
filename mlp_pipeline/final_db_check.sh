#!/bin/bash
# READ-ONLY inventory of everything that goes into the final training database. Login node, repo root, no python needed, nothing written.
#   bash mlp_pipeline/final_db_check.sh
# 1. DFT point sets (any runs/*/points_* or snapdb*/ chunk with frames_meta.csv): points finished / total, and whether a collected labelled.extxyz exists
# 2. every labelled / tier extxyz: frames, frames per db_set
# 3. GRACE pickles and fine-tunes (finished model or not)
# 4. dgchk / ft jobs in the queue
cd "$(dirname "$0")/.."
nfr() { grep -c 'Lattice=' "$1" 2>/dev/null || echo 0; }
sets() { grep -o 'db_set=[^ ]*' "$1" 2>/dev/null | sort | uniq -c | awk '{printf "%s:%s ", substr($2,8), $1}'; }
echo "== 1. DFT point sets (done = points with DONE.json; points_collect.py is the exact count)"
printf '%-34s %8s %8s   %s\n' "set" "done" "total" "collected file"
for root in $(ls -d runs/*/ snapdb*/ 2>/dev/null); do
  pts=$(ls -d ${root}points_* ${root}chunk_* 2>/dev/null | while read p; do [ -f "$p/frames_meta.csv" ] && echo "$p"; done)
  [ -n "$pts" ] || continue
  tot=0; dn=0
  for p in $pts; do
    t=$(($(wc -l < "$p/frames_meta.csv") - 1)); tot=$((tot + t))
    d=$(find "$p" -mindepth 2 -maxdepth 2 -name DONE.json 2>/dev/null | wc -l); dn=$((dn + d))
  done
  col=$(ls ${root}labelled*.extxyz 2>/dev/null | while read f; do echo -n "$(basename $f)($(nfr $f)) "; done)
  printf '%-34s %8d %8d   %s\n' "${root%/}" "$dn" "$tot" "${col:--}"
done
echo; echo "== 2. labelled and tier extxyz files (frames; frames per db_set)"
for f in $(ls snapdb*/labelled*.extxyz runs/*/labelled*.extxyz runs/train_*.extxyz runs/valid_*.extxyz runs/dft_md_*.extxyz 2>/dev/null); do
  printf '%-44s %6s  %s\n' "$f" "$(nfr $f)" "$(sets $f)"
done
echo; echo "== 3. GRACE pickles and fine-tunes"
ls runs/grace/*.pkl.gz 2>/dev/null | while read f; do printf '%-44s %s\n' "$f" "$(du -h "$f" | cut -f1)"; done
for d in runs/grace/ft_*; do [ -d "$d" ] || continue
  [ -e "$d/seed/1/final_model/saved_model.pb" ] && s=final || s=no-model
  printf '%-36s %-9s %s\n' "${d#runs/grace/}" "$s" "$(head -c 150 "$d/recipe.txt" 2>/dev/null)"
done
echo; echo "== 4. queue (dgchk / ft / score)"
squeue -u "$USER" -h -o '%j %T %M' 2>/dev/null | awk '$1 ~ /^(dgchk|ft|score)/' | sort | awk '{split($1,a,"_"); k=a[1]" "$2; n[k]++} END{for (k in n) print "  " k, n[k]}'
echo; echo "== scoring files present: $(ls runs/D_*.csv 2>/dev/null | wc -l) D_*.csv, $(ls runs/Dmd_*.csv 2>/dev/null | wc -l) Dmd_*.csv, $(ls runs/*pairs*.csv 2>/dev/null | wc -l) pair csv"
