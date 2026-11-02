#!/usr/bin/env bash
# Run build_insertion_set.py for several (T, P) states and both splits, spread over the GPUs of one node (default 2 processes per GPU, 4 GPUs).
# usage (GPU node, PYTHONPATH must contain mlp_pipeline/):  bash mlp_pipeline/run_insertion_batch.sh OUTDIR MODEL_SPEC [PER_GPU=2] [NGPU=4]
#   DRY=1 prints the commands without running them.  Edit STATES below: "T_bin:Pmin:Pmax:dv_per_H" (dv = H partial volume in A^3, about 1.26 at 500 GPa, an estimate; the stored P_grace_GPa shows whether it is right).
# Hosts are the H2O frames of the existing tiers (324-atom cells): train_l -> training set, valid_l -> validation set, so whole runs stay on one side.
set -u
OUT=${1:?outdir}; MODEL=${2:?model spec}; PER_GPU=${3:-2}; NGPU=${4:-4}; HOSTDIR=${HOSTDIR:-runs}
# default: full grid T 3000..7000 x P 300/400/500/600 (+-50 GPa windows), dv per H scaled from 1.26 A^3 at 500 GPa as (500/P)^0.75; GRID=0 keeps the old 5-state list. Fewer n values off the 5000 K / 500 GPa centre.
if [ "${GRID:-1}" = 1 ]; then STATES=""; for T in 3000 4000 5000 6000 7000; do for Pc in 300 400 500 600; do STATES="$STATES $T:$((Pc-50)):$((Pc+50)):auto"; done; done
else STATES="5000:450:550:1.26 3000:450:550:1.26 7000:450:550:1.26 5000:350:450:1.47 5000:550:650:1.11"; fi
STATES=${STATES_OVERRIDE:-$STATES}
NLIST=${NLIST:-1,2,4,8,16,27}; NLIST_CENTRE=${NLIST_CENTRE:-1,2,3,4,6,8,12,16,20,24,27}
HOSTS_TRAIN=${HOSTS_TRAIN:-$HOSTDIR/train_l.extxyz}; HOSTS_VALID=${HOSTS_VALID:-$HOSTDIR/valid_l.extxyz}   # e.g. host frames picked from the Young pool with young_select.py --as-hosts
PY=${PY:-${SCRATCH:-$HOME}/grace-env/bin/python}; SCRIPT=${SCRIPT:-$HOME/build_insertion_set.py}
mkdir -p "$OUT"; cmds=()
for st in $STATES; do IFS=: read -r T P0 P1 DV <<< "$st"; Pm=$(( (P0 + P1) / 2 ))
  [ "$DV" = auto ] && DV=$(python3 -c "print(round(1.26*(500/$Pm)**0.75,3))")
  NL=$NLIST; [ "$T" = 5000 ] && [ "$Pm" = 500 ] && NL=$NLIST_CENTRE
  for split in train valid; do
    HF=$HOSTS_TRAIN; [ $split = valid ] && HF=$HOSTS_VALID
    f="$OUT/ins_h2o_${T}K_${Pm}GPa_${split}.extxyz"
    cmds+=("$PY -u $SCRIPT --hosts $HF --comp h2o --T-bin $T --P $P0 $P1 --natoms 324 --n-list $NL --copies 1 --max-hosts ${MAXHOSTS:-12} --model $MODEL --equil-ps 0.2 --dt 1.0 --sample-ps 0.3 --save-every-ps 0.1 --dv-per-H $DV --seed $RANDOM --out $f > ${f%.extxyz}.log 2>&1")
  done
done
W=$(( PER_GPU * NGPU )); echo "${#cmds[@]} jobs on $W workers ($PER_GPU per GPU x $NGPU GPUs)"
[ "${DRY:-0}" = 1 ] && { printf '%s\n' "${cmds[@]}"; exit 0; }
export TF_FORCE_GPU_ALLOW_GROWTH=true
for ((w = 0; w < W; w++)); do
  ( export CUDA_VISIBLE_DEVICES=$((w % NGPU)); for ((i = w; i < ${#cmds[@]}; i += W)); do eval "${cmds[$i]}"; done ) &
done
wait; echo "done; frames per file:"; for f in "$OUT"/*.extxyz; do [ -e "$f" ] && echo "$(grep -c Lattice= "$f") $f"; done
