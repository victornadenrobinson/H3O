#!/bin/bash
# Prints the command lines (one fine-tune each) for bash mlp_pipeline/run_lines_gpus.sh <file> 1 4  -> 4 trainings in parallel, one per GPU of a node.
#   bash mlp_pipeline/ft_series_cmds.sh r1 > ft_r1.sh          round 1: 4 recipes on the full new tier (weights, replay, frozen/unfrozen)
#   bash mlp_pipeline/ft_series_cmds.sh r2 "10 10 1" frozen mn  > ft_r2.sh      round 2: learning curve 25/50/75 for the winning weights / mode / base ('n' new only, 'mn' replay)
#   bash mlp_pipeline/ft_series_cmds.sh len "10 10 1" frozen n  > ft_len.sh     training-length ladder 1000 / 3000 / 10000 updates
# UPDATES (default 3000) is a guess: look at the log of the first run (updates per minute) and the validation curve before the long ladder.
UPD=${UPDATES:-3000}; round=${1:-r1}
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
  *) echo "usage: $0 r1 | r2 \"wE wF wS\" frozen|naive n|mn | len ..." >&2; exit 1;;
esac
