#!/bin/bash
# PWidom campaign as SINGLE-GPU shared sbatch jobs (one host per job; shared jobs get one GPU, the launcher normally needs four). Per (P, T) state:
#   1. a pure-H reference job  (perlmutter_pwidom.sh --hosts pureH)
#   2. one job per extra-H composition (--extra-h <four counts> --hosts Hplus<k> --reference-run <ref>), each depending on the reference (--dependency=afterok).
# The launcher wants exactly four extra-H counts; each job passes all four but runs only its own host. 108 O per cell: count k -> n = k/108 extra H per O (3: 0.028, 9: 0.083 = H2.083O, 15: 0.139, 21: 0.194, 27 = H2.25O).
#   STATES="400 500 600" T=5000 bash mlp_pipeline/pwidom_campaign.sh              prints what it would do (DRY=1 is the default)
#   SUBMIT=1 STATES="500" COUNTS="3 9 15 21" bash mlp_pipeline/pwidom_campaign.sh  submits
# env: MODEL (default the best fine-tune), T, STATES, COUNTS, MDPS (production ps, 1.0), EQPS (0.4), SEL (insertions per frame, 24), KK (proxy candidates, 3000), SEED0 (11; one seed per job), HOURS (wall limit per job, 6), OUT (runs/pw_prod),
#      ICEPROXY / HPROXY (default: the launcher's). Needs the DFT EOS json per state: runs/dft_eos_<P>GPa_<T>K.json or ..._48h.json (make with --inspect-eos ... --save-eos).
# Afterwards:  $PY mlp_pipeline/pwidom_hull.py $OUT/xh*_500GPa_5000K_*   (pooled per host; one column per run in plot_pwidom_hull.py)
set -e; cd "$(dirname "$0")/.."
MODEL=${MODEL:-grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model}; T=${T:-5000}; STATES=${STATES:-500}; COUNTS=${COUNTS:-"3 9 15 21"}; MDPS=${MDPS:-1.0}; EQPS=${EQPS:-0.4}
SEL=${SEL:-24}; KK=${KK:-3000}; SEED0=${SEED0:-11}; HOURS=${HOURS:-6}; OUT=${OUT:-runs/pw_prod}; SUBMIT=${SUBMIT:-0}; MIN=$((HOURS * 60 - 15)); mkdir -p "$OUT" logs_pw
read -r -a CNT <<< "$COUNTS"; [ "${#CNT[@]}" = 4 ] || { echo "COUNTS must be exactly four distinct extra-H counts (launcher rule), e.g. \"3 9 15 21\""; exit 1; }
PROXY=""; [ -n "${ICEPROXY:-}" ] && PROXY="$PROXY --ice-proxy $ICEPROXY"; [ -n "${HPROXY:-}" ] && PROXY="$PROXY --h-proxy $HPROXY"
n=0
submit() {   # name  launcher-args  dependency-job-id-or-empty  -> echoes the job id (or DRY)
  local name=$1 cmd=$2 dep=$3 script="$OUT/job_$1.sh"
  printf '#!/bin/bash\ncd %s\nexport PYTHONPATH=${PYTHONPATH:-}:%s/mlp_pipeline\nbash mlp_pipeline/perlmutter_pwidom.sh %s\n' "$PWD" "$PWD" "$cmd" > "$script"
  local sb="sbatch --parsable -A m3349_g -C gpu -q shared -G 1 -c 32 -N 1 -t ${HOURS}:00:00 -J pw_$name -o logs_pw/$name.%j.out ${dep:+--dependency=afterok:$dep} $script"
  if [ "$SUBMIT" = 1 ]; then $sb; else echo "DRY: $sb" >&2; echo "DRY_$name"; fi
}
for P in $STATES; do
  EOS=runs/dft_eos_${P}GPa_${T}K.json; [ -e "$EOS" ] || EOS=runs/dft_eos_${P}GPa_${T}K_48h.json
  [ -e "$EOS" ] || { echo "skip $P GPa / $T K: no DFT EOS json (runs/dft_eos_${P}GPa_${T}K*.json)" >&2; continue; }
  tag=${P}GPa_${T}K; ref="$OUT/ref_$tag"
  common="--P $P --T $T --state-json $EOS --model $MODEL --ps $MDPS --equil-ps $EQPS --sel $SEL --K $KK --minutes $MIN --plain --interval 300 $PROXY"
  jid=$(submit "ref_$tag" "$common --hosts pureH --seed $((SEED0 + n)) --out $ref" ""); n=$((n + 1)); echo "ref  $tag  job $jid" >&2
  for k in "${CNT[@]}"; do
    j=$(submit "xh${k}_${tag}_s$((SEED0 + n))" "$common --reference-run $ref --extra-h ${CNT[*]} --hosts Hplus$k --seed $((SEED0 + n)) --out $OUT/xh${k}_${tag}_s$((SEED0 + n))" "$jid"); echo "  Hplus$k ($(python3 -c "print(round($k/108,3))") H/O)  job $j  (after $jid)" >&2; n=$((n + 1))
  done
done
[ "$SUBMIT" = 1 ] || echo "(dry run: nothing submitted; add SUBMIT=1)" >&2
