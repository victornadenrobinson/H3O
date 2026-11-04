#!/bin/bash
# PWidom at several (P, T) states as shared single-GPU sbatch jobs, via pwidom_campaign.sh, with the best model. Per state: make the DFT EOS json if it is missing (read-only inspect-eos on the DFT inventory; a state without
# enough DFT coverage is skipped with a message), then REPLICAS replicates (different seeds, one shared pure-H reference) of the hosts Hplus3 / Hplus9 / Hplus15 / Hplus27 (n = 0.028 / 0.083 / 0.139 / 0.25 extra H per O).
#   bash mlp_pipeline/pwidom_states.sh                       dry run (prints the sbatch lines)
#   SUBMIT=1 bash mlp_pipeline/pwidom_states.sh              submit
# env: STATES ("600:5000 600:7000 700:5000 700:7000" = P:T), COUNTS ("3 9 15 27"), REPLICAS (2), MDPS (2.0), EQPS (0.5), HOURS (6), MODEL (campaign default: ft_mn_f_lr1e-2_e20), OUTBASE (runs/pw_states), INV (runs/inventory_binned.csv)
# Results: python3 mlp_pipeline/pwidom_hull.py runs/pw_states/<P>GPa_<T>K/xh*_s* --pool   (per state; xh dirs of all replicates pool per host)
cd "$(dirname "$0")/.."; STATES=${STATES:-"600:5000 600:7000 700:5000 700:7000"}; COUNTS=${COUNTS:-"3 9 15 27"}; REPLICAS=${REPLICAS:-2}; MDPS=${MDPS:-2.0}; EQPS=${EQPS:-0.5}; HOURS=${HOURS:-6}
OUTBASE=${OUTBASE:-runs/pw_states}; INV=${INV:-runs/inventory_binned.csv}; SUBMIT=${SUBMIT:-0}; mkdir -p logs_pw "$OUTBASE"
for st in $STATES; do
  P=${st%%:*}; T=${st##*:}; tag=${P}GPa_${T}K; EOS=runs/dft_eos_${tag}.json
  if [ ! -e "$EOS" ] && [ ! -e "${EOS%.json}_48h.json" ]; then
    if [ "$SUBMIT" = 1 ] || [ "${MAKE_EOS:-0}" = 1 ]; then
      bash mlp_pipeline/perlmutter_pwidom.sh --P "$P" --T "$T" --inspect-eos "$INV" --save-eos "$EOS" > "logs_pw/eos_${tag}.log" 2>&1 || { echo "== $tag: SKIPPED, no DFT EOS (not enough DFT coverage); see logs_pw/eos_${tag}.log"; rm -f "$EOS" 2>/dev/null; continue; }
    else echo "== $tag: no EOS json yet (dry run: it would be made with --inspect-eos from $INV)"; fi
  fi
  OUT=$OUTBASE/$tag; echo "== $tag: $REPLICAS replicates, hosts Hplus [$COUNTS], model ${MODEL:-default}"
  msg=$(env STATES=$P T=$T COUNTS="$COUNTS" MDPS=$MDPS EQPS=$EQPS HOURS=$HOURS SEED0=11 OUT=$OUT SUBMIT=$SUBMIT ${MODEL:+MODEL=$MODEL} bash mlp_pipeline/pwidom_campaign.sh 2>&1); echo "$msg" | sed 's/^/   /'
  refjob=$(echo "$msg" | grep -oE "^ref .*job [0-9A-Za-z_]+" | grep -oE "[0-9]+$" | head -1)
  for ((r = 1; r < REPLICAS; r++)); do
    msg=$(env REFDIR=$OUT/ref_$tag ${refjob:+DEPEND=$refjob} STATES=$P T=$T COUNTS="$COUNTS" MDPS=$MDPS EQPS=$EQPS HOURS=$HOURS SEED0=$((11 + 20 * r)) OUT=$OUT SUBMIT=$SUBMIT ${MODEL:+MODEL=$MODEL} bash mlp_pipeline/pwidom_campaign.sh 2>&1); echo "$msg" | sed 's/^/   /'
  done
done
[ "$SUBMIT" = 1 ] || echo "(dry run: add SUBMIT=1)"
