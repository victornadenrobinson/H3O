#!/bin/bash
# Continue fine-tuning an existing GRACE fine-tune on new (e.g. k-point-corrected) data: same architecture (-p <base>/seed/1/model.yaml), weights from the base run's checkpoint
# (-cn ...checkpoint.best_test_loss), fresh epoch/step counters (--reset-epoch-and-step), the base input.yaml with the data files, updates, learning rate and optional loss weights replaced.
#   bash mlp_pipeline/ft_continue.sh NAME BASE TRAIN_TAG VALID_TAG UPDATES [lr=0.001] [wE wF wS]        one GPU (CUDA_VISIBLE_DEVICES set by the caller / the job)
#   e.g. bash mlp_pipeline/ft_continue.sh mfinkc_cont mfin_f_lr1e-2_e50 mfinkc mfinkc 21450 0.001
# CHECK THE FIRST LOG LINES: the INITIAL test error must be close to the base model's final test error (~10 meV/atom for mfin_f_lr1e-2_e50), not the foundation model's (~150):
#   that shows the weights were loaded. CKPT=checkpoint (latest) instead of checkpoint.best_test_loss; DRY=1 writes input.yaml and prints the command.
cd "$(dirname "$0")/.."
NAME=${1:?name}; BASE=${2:?base run name (runs/grace/ft_<BASE>)}; TT=${3:?train tag}; VT=${4:?valid tag}; UPD=${5:?updates}; LR=${6:-0.001}; WE=${7:-}; WF=${8:-}; WS=${9:-}
GBIN=${GBIN:-$SCRATCH/grace-env/bin}; R=$(pwd)/runs/grace; B=$R/ft_$BASE; D=$R/ft_$NAME; CK=$B/seed/1/checkpoints/${CKPT:-checkpoint.best_test_loss}
[ -e $B/input.yaml ] && [ -e $B/seed/1/model.yaml ] && [ -e $CK.index ] || { echo "base run incomplete: need $B/input.yaml, $B/seed/1/model.yaml, $CK.index"; exit 1; }
[ -e $R/train_$TT.pkl.gz ] && [ -e $R/valid_$VT.pkl.gz ] || { echo "missing $R/train_$TT.pkl.gz or valid_$VT.pkl.gz (grace_prep_pkl.sh $TT)"; exit 1; }
mkdir -p $D
"$GBIN/python" - "$B/input.yaml" "$D/input.yaml" "$R/train_$TT.pkl.gz" "$R/valid_$VT.pkl.gz" "$UPD" "$LR" "$WE" "$WF" "$WS" <<'EOF' || exit 1
import sys, yaml
src, dst, tr, va, upd, lr, wE, wF, wS = sys.argv[1:10]
d = yaml.safe_load(open(src)); dd = d.setdefault("data", {}); f = d.setdefault("fit", {})
dd["filename"] = tr; dd["test_filename"] = va
f["target_total_updates"] = int(upd)
op = f.setdefault("opt_params", {}); op["learning_rate"] = float(lr)
sp = f.get("scheduler_params")
if isinstance(sp, dict): sp["minimum_learning_rate"] = float(lr) / 12
loss = f.get("loss", {})
for k, v in (("energy", wE), ("forces", wF), ("stress", wS)):
    if v and isinstance(loss.get(k), dict): loss[k]["weight"] = float(v)
yaml.safe_dump(d, open(dst, "w"), sort_keys=False, default_flow_style=False)
print("data:", dd["filename"], "|", dd["test_filename"]); print("fit: updates", f["target_total_updates"], "lr", op["learning_rate"], "| loss weights", {k: (loss.get(k) or {}).get("weight") for k in ("energy", "forces", "stress")})
print("potential section kept from the base input (the -p model.yaml + -cn checkpoint load the fine-tuned model):", {k: d.get("potential", {}).get(k) for k in ("finetune_foundation_model", "reduce_elements", "shift")})
EOF
echo "continue $NAME from $BASE ($CK): train $TT valid $VT updates $UPD lr $LR weights ${WE:-base} ${WF:-base} ${WS:-base}  gpu ${CUDA_VISIBLE_DEVICES:-all}" | tee $D/recipe.txt
CMD="$GBIN/gracemaker input.yaml -l log.txt -p $B/seed/1/model.yaml -cn $CK --reset-epoch-and-step"
[ "${DRY:-0}" = 1 ] && { echo "(dry) cd $D && $CMD"; exit 0; }
cd $D && export TF_FORCE_GPU_ALLOW_GROWTH=true && $CMD > stdout.txt 2>&1; RC=$?
echo "gracemaker exit code $RC"; [ $RC -eq 0 ] && echo "model: $D/seed/1/final_model" || grep -v -E "^(I0000|W0000|E0000|WARNING|20[0-9][0-9]-[0-9-]+ [0-9:.]+: [EIW]|\[tensorpotential\])" stdout.txt | tail -n 15
exit $RC
