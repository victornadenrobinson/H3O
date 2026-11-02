#!/bin/bash
# One GRACE fine-tune in its own folder runs/grace/ft_<NAME>/ (input.yaml from grace_make_input.py, then gracemaker). Meant to be run on ONE GPU: CUDA_VISIBLE_DEVICES set by the caller.
#   bash mlp_pipeline/ft_one.sh NAME TRAIN_TAG VALID_TAG UPDATES wE wF wS [frozen|naive] [lr=0.001] [batch=2]
#   e.g. bash mlp_pipeline/ft_one.sh nEF n n 3000 10 10 1 frozen
# needs runs/grace/train_<TRAIN_TAG>.pkl.gz and valid_<VALID_TAG>.pkl.gz (grace_prep_pkl.sh). DRY=1 only writes input.yaml and prints the command.
cd "$(dirname "$0")/.."
NAME=${1:?name}; TT=${2:?train tag}; VT=${3:?valid tag}; UPD=${4:?updates}; WE=${5:?wE}; WF=${6:?wF}; WS=${7:?wS}; MODE=${8:-frozen}; LR=${9:-0.001}; BS=${10:-2}
GBIN=${GBIN:-$SCRATCH/grace-env/bin}; R=$(pwd)/runs/grace; D=$R/ft_$NAME; mkdir -p "$D"
[ -e "$R/train_$TT.pkl.gz" ] && [ -e "$R/valid_$VT.pkl.gz" ] || { echo "missing $R/train_$TT.pkl.gz or valid_$VT.pkl.gz"; exit 1; }
NAIVE=""; [ "$MODE" = naive ] && NAIVE="--naive"
"$GBIN/python" mlp_pipeline/grace_make_input.py "$D/input.yaml" --train "$R/train_$TT.pkl.gz" --test "$R/valid_$VT.pkl.gz" --updates "$UPD" --batch "$BS" --wE "$WE" --wF "$WF" --wS "$WS" --lr "$LR" $NAIVE || exit 1
echo "fine-tune $NAME: train $TT valid $VT updates $UPD weights E$WE F$WF S$WS $MODE lr $LR batch $BS  gpu ${CUDA_VISIBLE_DEVICES:-all}" | tee "$D/recipe.txt"
[ "${DRY:-0}" = 1 ] && { echo "(dry) cd $D && $GBIN/gracemaker input.yaml -l log.txt"; exit 0; }
cd "$D" && export TF_FORCE_GPU_ALLOW_GROWTH=true && "$GBIN/gracemaker" input.yaml -l log.txt > stdout.txt 2>&1; RC=$?
echo "gracemaker exit code $RC"; [ $RC -eq 0 ] && echo "model: $D/seed/1/final_model" || grep -v -E "^(I0000|W0000|E0000|WARNING|20[0-9][0-9]-[0-9-]+ [0-9:.]+: [EIW]|\[tensorpotential\])" stdout.txt | tail -n 15
exit $RC
