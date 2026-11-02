#!/usr/bin/env bash
# The 5000 K / 500 GPa centre state has 11 n-values per host and is the slowest grid job (~1 h for 2 hosts). This runs it as K shards, ONE host per process, so the whole state takes ~1/K of the time
# and uses K hosts per split. Hosts are taken from the Young pool/pwidom files first, then the old tiers (new hosts, not the ones batch1 / the grid already used).
# usage: kill the two running centre jobs first, then on every node of the job:  srun -N <nodes> --ntasks-per-node=1 bash mlp_pipeline/run_centre_shards.sh ~/leg/ins_grid [K=4]
#   each node takes the shards with index % SLURM_NNODES == SLURM_NODEID, one process per GPU. DRY=1 prints the commands. Output: OUT/ins_h2o_5000K_500GPa_<split>_s<i>.extxyz (+ .log, picked up by progress.py).
set -u
OUT=${1:?outdir}; K=${2:-4}; NN=${SLURM_NNODES:-1}; ND=${SLURM_NODEID:-0}; NGPU=${NGPU:-4}; M=${M:?model spec}; PY=${PY:-$SCRATCH/grace-env/bin/python}; SCRIPT=${SCRIPT:-$HOME/build_insertion_set.py}
HT=${HT:-hosts_pool_train.extxyz hosts_pwidom_train.extxyz runs/train_l.extxyz}; HV=${HV:-hosts_pool_valid.extxyz hosts_pwidom_valid.extxyz runs/valid_l.extxyz}
NL=${NLIST_CENTRE:-1,2,3,4,6,8,12,16,20,24,27}; mkdir -p "$OUT"; export TF_FORCE_GPU_ALLOW_GROWTH=true; idx=0
for split in train valid; do for ((s = 0; s < K; s++)); do
  if (( idx % NN == ND )); then gpu=$(( (idx / NN) % NGPU )); H=$HT; [ $split = valid ] && H=$HV; f="$OUT/ins_h2o_5000K_500GPa_${split}_s$s.extxyz"
    c="CUDA_VISIBLE_DEVICES=$gpu $PY -u $SCRIPT --hosts $H --comp h2o --T-bin 5000 --P 450 550 --natoms 324 --n-list $NL --copies 1 --max-hosts $K --shard $s/$K --model $M --equil-ps 0.2 --dt 1.0 --sample-ps 0.3 --save-every-ps 0.1 --dv-per-H 1.26 --seed $RANDOM --out $f"
    if [ "${DRY:-0}" = 1 ]; then echo "node $ND: $c"; elif grep -qs "^wrote .* frames to" "${f%.extxyz}.log"; then echo "skip $f (done)"; else eval "nohup env $c > ${f%.extxyz}.log 2>&1 &"; fi
  fi; idx=$((idx + 1)); done; done
wait
