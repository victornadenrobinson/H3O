#!/bin/bash
# Prints the command lines (one fine-tune each) for bash mlp_pipeline/run_lines_gpus.sh <file> 1 4  -> 4 trainings in parallel, one per GPU of a node.
#   bash mlp_pipeline/ft_series_cmds.sh r1 > ft_r1.sh          round 1: 4 recipes on the full new tier (weights, replay, frozen/unfrozen)
#   bash mlp_pipeline/ft_series_cmds.sh r2 "10 10 1" frozen mn  > ft_r2.sh      round 2: learning curve 25/50/75 for the winning weights / mode / base ('n' new only, 'mn' replay)
#   bash mlp_pipeline/ft_series_cmds.sh len "10 10 1" frozen n  > ft_len.sh     training-length ladder 1000 / 3000 / 10000 updates
# EPOCHS (default 10) sets the length for r1/r2/len; lrA/lrB use fixed epoch counts (5 and 20). Speed measured on Young: ~2.3 updates/s on an A100 for the frozen fine-tune; the unfrozen one (all weights) crashed on Young (exit 1 after ~9 min, cause not yet seen), so run lrB on 80 GB GPUs: sbatch -C 'gpu&hbm80g'.
round=${1:-r1}; PY=${PY:-${SCRATCH:-$HOME}/grace-env/bin/python}
# training length in EPOCHS (as in the Young sweep: 21 epochs ~ 7,800 updates for 747 frames at batch 2); updates = epochs * frames / 2. Measured on a Young A100 (frozen, batch 2): ~2.3 updates/s.
nfr() { $PY -c "import pandas as pd; print(len(pd.read_pickle('runs/grace/train_$1.pkl.gz')))" 2>/dev/null || echo 3000; }
upd() { echo $(( $1 * $(nfr $2) / 2 )); }       # upd EPOCHS TAG
UPD=${UPDATES:-$(upd ${EPOCHS:-10} mn)}
case $round in
  r1)  echo "bash mlp_pipeline/ft_one.sh nBase   n  n  $UPD 1 5 10 frozen"
       echo "bash mlp_pipeline/ft_one.sh nEF     n  n  $UPD 10 10 1 frozen"
       echo "bash mlp_pipeline/ft_one.sh mnEF    mn mn $UPD 10 10 1 frozen"
       echo "bash mlp_pipeline/ft_one.sh mnEFnaive mn mn $UPD 10 10 1 naive 0.0003";;
  r2)  W=${2:-10 10 1}; M=${3:-frozen}; B=${4:-n}; V=$B; [ "$B" = mn ] && V=mn || V=n
       for p in 25 50 75; do echo "bash mlp_pipeline/ft_one.sh ${B}EF_f$p ${B}$p $V $UPD $W $M"; done
       echo "bash mlp_pipeline/ft_one.sh ${B}EF_f100 ${B} $V $UPD $W $M";;
  len) W=${2:-10 10 1}; M=${3:-frozen}; B=${4:-n}; V=n; [ "$B" = mn ] && V=mn
       for u in 1000 3000 10000; do echo "bash mlp_pipeline/ft_one.sh ${B}_u$u ${B} $V $u $W $M"; done;;
  lrA) W=${2:-10 10 1}   # overnight grid, node A: learning rate and length with the frozen (light) fine-tune, replay data
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-4_e5   mn mn $(upd 5 mn) $W frozen 0.0003"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-3_e5   mn mn $(upd 5 mn) $W frozen 0.001"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e5   mn mn $(upd 5 mn) $W frozen 0.003"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-3_e20  mn mn $(upd 20 mn) $W frozen 0.001";;
  lrB) W=${2:-10 10 1}   # node B: everything trainable (more capacity, forgetting risk), plus one no-replay control
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr1e-4_e5    mn mn $(upd 5 mn) $W naive 0.0001"
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr3e-4_e5    mn mn $(upd 5 mn) $W naive 0.0003"
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr1e-4_e20   mn mn $(upd 20 mn) $W naive 0.0001"
       echo "bash mlp_pipeline/ft_one.sh n_n_lr1e-4_e5     n  n  $(upd 5 n) $W naive 0.0001";;
  *) echo "usage: $0 r1 | r2 \"wE wF wS\" frozen|naive n|mn | len | lrA | lrB ..." >&2; exit 1;;
esac
