#!/usr/bin/env bash
# Writes the GPU command file for every snapshot set that is not the main insertion grid (A): B+ chains, E low-P, H Huang supercells. Run it on Perlmutter from the repo root ($H), then
#   bash mlp_pipeline/make_rest_cmds.sh > rest_gpu.sh && bash mlp_pipeline/run_lines_gpus.sh rest_gpu.sh 2 4      (one 4-GPU node, 8 workers, resumable)
# needs: PY, M (model spec), HT / HV (host files, as for run_insertion_batch.sh), ~/build_insertion_set.py, low_p_pass1.sh (from young_pack.tgz). Outputs go to ~/leg/{chain,lowp,h3o_sc}/.
set -u
PY=${PY:-$SCRATCH/grace-env/bin/python}; M=${M:?model spec}; SC=${SC:-$HOME/build_insertion_set.py}; OUT=${OUT:-$HOME/leg}
HT=${HT:-runs/train_l.extxyz hosts_pool_train.extxyz hosts_pwidom_train.extxyz}; HV=${HV:-runs/valid_l.extxyz hosts_pool_valid.extxyz hosts_pwidom_valid.extxyz}
echo "# --- B+: n -> n+1 chains (raw insertions from one growing configuration, no MD), 3 T x 3 P; valid side at P=500 only"
for T in 3000 5000 7000; do for Pc in 400 500 600; do
  dv=$(awk "BEGIN{printf \"%.3f\", 1.26*(500/$Pc)^0.75}")
  for split in train valid; do [ $split = valid ] && [ $Pc != 500 ] && continue
    H=$HT; [ $split = valid ] && H=$HV
    echo "mkdir -p $OUT/chain; $PY -u $SC --hosts $H --comp h2o --T-bin $T --P $((Pc-50)) $((Pc+50)) --natoms 324 --n-list 1,2,3,4,5,6,7,8 --chain --copies 1 --max-hosts 2 --model $M --equil-ps 0 --dv-per-H $dv --seed $RANDOM --out $OUT/chain/chain_${T}K_${Pc}GPa_${split}.extxyz"
  done; done; done
echo "# --- E: low-P GRACE-MD at 50/100/200 GPa (pass 1; rerun eos_low_p.py with runs/eos_lowp added for pass 2)"
[ -s low_p_pass1.sh ] && grep -v '^\s*#' low_p_pass1.sh | grep -v '^\s*$' | sed "s#^python3\? #$PY #"
echo "# --- H: Huang H3O supercells, one process per pressure (450 GPa goes whole to validation)"
for p in 300 350 400 450 500 600 700 800; do
  echo "mkdir -p $OUT/h3o_sc; $PY -u mlp_pipeline/build_h3o_supercells.py --static-dir runs/h3o_static --model $M --P $p --T-list 1000 2500 4000 --valid-P 450 --out $OUT/h3o_sc/h3o_sc_P$p.extxyz"
done
