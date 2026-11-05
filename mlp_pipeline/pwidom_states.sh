#!/bin/bash
# PWidom at several (P, T) states as shared single-GPU sbatch jobs, via pwidom_campaign.sh, with the best model. Per state: make the DFT EOS json if it is missing (read-only inspect-eos on the DFT inventory; a state without
# enough DFT coverage is skipped with a message), then REPLICAS replicates (different seeds, one shared pure-H reference) of the hosts Hplus3 / Hplus9 / Hplus15 / Hplus27 (n = 0.028 / 0.083 / 0.139 / 0.25 extra H per O).
#   bash mlp_pipeline/pwidom_states.sh                       dry run (prints the sbatch lines)
#   SUBMIT=1 bash mlp_pipeline/pwidom_states.sh              submit
# MIX=1 (default): a state without DFT coverage gets a linear-in-T mix of the state-jsons at the same P (mix_eos.py): a starting point, PWidom corrects to the target pressure and reports P_host.
# env: STATES ("600:5000 600:7000 700:5000 700:7000" = P:T), COUNTS ("3 9 15 27"), REPLICAS (2), MDPS (2.0), EQPS (0.5), HOURS (6), MODEL (campaign default: ft_mn_f_lr1e-2_e20), OUTBASE (runs/pw_states), INV (runs/inventory_binned.csv)
# Results: python3 mlp_pipeline/pwidom_hull.py runs/pw_states/<P>GPa_<T>K/xh*_s* --pool   (per state; xh dirs of all replicates pool per host)
cd "$(dirname "$0")/.."; STATES=${STATES:-"600:5000 600:7000 700:5000 700:7000"}; COUNTS=${COUNTS:-"3 9 15 27"}; REPLICAS=${REPLICAS:-2}; MDPS=${MDPS:-2.0}; EQPS=${EQPS:-0.5}; HOURS=${HOURS:-6}
OUTBASE=${OUTBASE:-runs/pw_states}; INV=${INV:-runs/inventory_binned.csv}; SUBMIT=${SUBMIT:-0}; mkdir -p logs_pw "$OUTBASE"
PY=${PY:-/pscratch/sd/v/victornr/grace-env/bin/python}
have_eos() { [ -e "runs/dft_eos_${1}GPa_${2}K.json" ] || [ -e "runs/dft_eos_${1}GPa_${2}K_48h.json" ]; }
make_eos() { have_eos "$1" "$2" && return 0; bash mlp_pipeline/perlmutter_pwidom.sh --P "$1" --T "$2" --inspect-eos "$INV" --save-eos "runs/dft_eos_${1}GPa_${2}K.json" > "logs_pw/eos_${1}GPa_${2}K.log" 2>&1 || { rm -f "runs/dft_eos_${1}GPa_${2}K.json"; return 1; }; }
for st in $STATES; do
  P=${st%%:*}; T=${st##*:}; tag=${P}GPa_${T}K; EOS=runs/dft_eos_${tag}.json
  if ! have_eos "$P" "$T"; then
    if [ "$SUBMIT" = 1 ] || [ "${MAKE_EOS:-0}" = 1 ]; then
      if make_eos "$P" "$T"; then echo "== $tag: DFT EOS made from the inventory"
      elif [ "${MIX:-1}" = 1 ]; then      # not enough DFT coverage: linear-in-T mix of the state-jsons at this P (3000 K / 5000 K are tried first), see mix_eos.py
        for T0 in 3000 5000; do make_eos "$P" "$T0" > /dev/null 2>&1 || true; done
        if $PY mlp_pipeline/mix_eos.py "$P" "$T" > "logs_pw/mix_${tag}.log" 2>&1; then echo "== $tag: no DFT coverage: EOS = linear T mix (starting point, check P_host):"; sed 's/^/     /' "logs_pw/mix_${tag}.log" | cut -c1-200
        else echo "== $tag: SKIPPED, no DFT EOS and no mix possible; see logs_pw/mix_${tag}.log"; continue; fi
      else echo "== $tag: SKIPPED, no DFT EOS (not enough DFT coverage); see logs_pw/eos_${tag}.log"; continue; fi
    else echo "== $tag: no EOS json yet (dry run: it would be made with --inspect-eos from $INV, else mixed in T)"; fi
  fi
  OUT=$OUTBASE/$tag; echo "== $tag: $REPLICAS replicates, hosts Hplus [$COUNTS], model ${MODEL:-default}"
  msg=$(env STATES=$P T=$T COUNTS="$COUNTS" MDPS=$MDPS EQPS=$EQPS HOURS=$HOURS SEED0=11 OUT=$OUT SUBMIT=$SUBMIT ${MODEL:+MODEL=$MODEL} bash mlp_pipeline/pwidom_campaign.sh 2>&1); echo "$msg" | sed 's/^/   /'
  refjob=$(echo "$msg" | grep -oE "^ref .*job [0-9A-Za-z_]+" | grep -oE "[0-9]+$" | head -1)
  for ((r = 1; r < REPLICAS; r++)); do
    msg=$(env REFDIR=$OUT/ref_$tag ${refjob:+DEPEND=$refjob} STATES=$P T=$T COUNTS="$COUNTS" MDPS=$MDPS EQPS=$EQPS HOURS=$HOURS SEED0=$((11 + 20 * r)) OUT=$OUT SUBMIT=$SUBMIT ${MODEL:+MODEL=$MODEL} bash mlp_pipeline/pwidom_campaign.sh 2>&1); echo "$msg" | sed 's/^/   /'
  done
done
[ "$SUBMIT" = 1 ] || echo "(dry run: add SUBMIT=1)"
