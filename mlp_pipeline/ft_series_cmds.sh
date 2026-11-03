#!/bin/bash
# Prints the command lines (one fine-tune each) for bash mlp_pipeline/run_lines_gpus.sh <file> 1 4  -> 4 trainings in parallel, one per GPU of a node.
#   bash mlp_pipeline/ft_series_cmds.sh r1 > ft_r1.sh          round 1: 4 recipes on the full new tier (weights, replay, frozen/unfrozen)
#   bash mlp_pipeline/ft_series_cmds.sh r2 "10 10 1" frozen mn  > ft_r2.sh      round 2: learning curve 25/50/75 for the winning weights / mode / base ('n' new only, 'mn' replay)
#   bash mlp_pipeline/ft_series_cmds.sh len "10 10 1" frozen n  > ft_len.sh     training-length ladder 1000 / 3000 / 10000 updates
# EPOCHS (default 10) sets the length for r1/r2/len; lrA/lrB use fixed epoch counts (5 and 20). Speed measured on Young: ~2.3 updates/s on an A100 for the frozen fine-tune; the unfrozen one (all weights) ran out of memory on Young (a single 47 GB allocation on a 40 GB A100). So lrB uses batch 1, 80 GB GPUs (sbatch -C 'gpu&hbm80g') and the tier variants mnl / nl without cells above 500 atoms (build_tier.py --tag nl --max-atoms 500). The frozen grid (lrA) also uses batch 1 for safety with the new, denser cells.
round=${1:-r1}; PY=${PY:-${SCRATCH:-$HOME}/grace-env/bin/python}
# training length in EPOCHS (as in the Young sweep: 21 epochs ~ 7,800 updates for 747 frames at batch 2); updates = epochs * frames / 2. Measured on a Young A100 (frozen, batch 2): ~2.3 updates/s.
nfr() { $PY -c "import pandas as pd; print(len(pd.read_pickle('runs/grace/train_$1.pkl.gz')))" 2>/dev/null || echo 3000; }
upd() { echo $(( $1 * $(nfr $2) / ${3:-2} )); }       # upd EPOCHS TAG [BATCH]
UPD=${UPDATES:-$(upd ${EPOCHS:-10} mn)}
case $round in
  r1)  E=${EPOCHS:-5}   # round 1: weights (old 1/5/10 vs E/F-heavy 10/10/1), replay or not, frozen; plus one unfrozen run on the tier without the dense big cells. batch 1 everywhere (memory).
       echo "bash mlp_pipeline/ft_one.sh nBase   n  n  $(upd $E n 1) 1 5 10 frozen 0.001 1"
       echo "bash mlp_pipeline/ft_one.sh nEF     n  n  $(upd $E n 1) 10 10 1 frozen 0.001 1"
       echo "bash mlp_pipeline/ft_one.sh mnEF    mn mn $(upd $E mn 1) 10 10 1 frozen 0.001 1"
       echo "bash mlp_pipeline/ft_one.sh mnlEFnaive mnl mnl $(upd $E mnl 1) 10 10 1 naive 0.0003 1";;
  r2)  W=${2:-10 10 1}; M=${3:-frozen}; B=${4:-n}; V=$B; [ "$B" = mn ] && V=mn || V=n
       for p in 25 50 75; do echo "bash mlp_pipeline/ft_one.sh ${B}EF_f$p ${B}$p $V $UPD $W $M"; done
       echo "bash mlp_pipeline/ft_one.sh ${B}EF_f100 ${B} $V $UPD $W $M";;
  len) W=${2:-10 10 1}; M=${3:-frozen}; B=${4:-n}; V=n; [ "$B" = mn ] && V=mn
       for u in 1000 3000 10000; do echo "bash mlp_pipeline/ft_one.sh ${B}_u$u ${B} $V $u $W $M"; done;;
  lrA) W=${2:-10 10 1}   # overnight grid, node A: learning rate and length with the frozen (light) fine-tune, replay data
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-4_e5   mn mn $(upd 5 mn 1) $W frozen 0.0003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-3_e5   mn mn $(upd 5 mn 1) $W frozen 0.001 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e5   mn mn $(upd 5 mn 1) $W frozen 0.003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-3_e20  mn mn $(upd 20 mn 1) $W frozen 0.001 1";;
  lrB) W=${2:-10 10 1}   # node B: everything trainable (more capacity, forgetting risk), plus one no-replay control
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr1e-4_e5    mnl mnl $(upd 5 mnl 1) $W naive 0.0001 1"
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr3e-4_e5    mnl mnl $(upd 5 mnl 1) $W naive 0.0003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_n_lr1e-4_e20   mnl mnl $(upd 20 mnl 1) $W naive 0.0001 1"
       echo "bash mlp_pipeline/ft_one.sh n_n_lr1e-4_e5     nl nl $(upd 5 nl 1) $W naive 0.0001 1";;
  lrC) W=${2:-10 10 1}   # after the lrA/lrB comparison on valid_n: best frozen run was lr 3e-3 (replay, 5 epochs); lr still rising -> 1e-2; longer; and the NO-replay control (n tier) at the same lr
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-2_e5   mn mn $(upd 5 mn 1) $W frozen 0.01 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e20  mn mn $(upd 20 mn 1) $W frozen 0.003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-2_e20  mn mn $(upd 20 mn 1) $W frozen 0.01 1"
       echo "bash mlp_pipeline/ft_one.sh n_f_lr3e-3_e5    n n $(upd 5 n 1) $W frozen 0.003 1";;
  lc)  W=${2:-10 10 1}   # LEARNING CURVE (test A5): best frozen recipe (lr 3e-3, replay tier), the SAME number of updates (5 epochs of the full tier) for every fraction; 100 % = the existing mn_f_lr3e-3_e5 run. Score all on runs/valid_n.extxyz.
       for p in 25 50 75; do echo "bash mlp_pipeline/ft_one.sh lc_mn$p mn$p mn $(upd 5 mn 1) $W frozen 0.003 1"; done;;
  long) W=${2:-10 10 1}   # OVERNIGHT. Finished 20-epoch runs: lr 1e-3 16.5, 3e-3 14.5, 1e-2 13.9 meV/atom (5 epochs: 19.9, 16.2, 15.5): longer helps ~10-17 %, higher lr is better, no overfitting. So: 50 epochs at 1e-3 (the low-lr question), 3e-3 and 1e-2; plus a long UNFROZEN run with large cells (80 GB, low lr).
       # ~14 h each at 2.3 updates/s (about 6 h if the node is as fast as the e20 runs: 4.7-5.5 updates/s).  submit with TIME=24:00:00 bash mlp_pipeline/ft_submit.sh ft_long.sh
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-3_e50      mn mn $(upd 50 mn 1) $W frozen 0.001 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e50      mn mn $(upd 50 mn 1) $W frozen 0.003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr1e-2_e50      mn mn $(upd 50 mn 1) $W frozen 0.01 1"
       echo "bash mlp_pipeline/ft_one.sh mn_n_big_lr1e-4_e20  mn mn $(upd 20 mn 1) $W naive 0.0001 1";;
  hi)  W=${2:-10 10 1}   # is lr 1e-2 the top? quick 5- and 20-epoch runs at 3e-2
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-2_e5   mn mn $(upd 5 mn 1) $W frozen 0.03 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-2_e20  mn mn $(upd 20 mn 1) $W frozen 0.03 1";;
  cap) W=${2:-10 10 1}   # CAPACITY round (after ft_curves showed frozen train error = test error: capacity-limited, not data-limited). 5 epochs of the replay tier each.
       # (a,b) force-heavier loss, frozen; (c,d) UNFROZEN on the replay tier INCLUDING the 700+ atom cells (80 GB GPU, batch 1; the earlier unfrozen runs excluded them, which made their test error meaningless on pure H)
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e5_F50   mn mn $(upd 5 mn 1) 10 50 1 frozen 0.003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_f_lr3e-3_e5_F200  mn mn $(upd 5 mn 1) 10 200 1 frozen 0.003 1"
       echo "bash mlp_pipeline/ft_one.sh mn_n_big_lr1e-4_e5   mn mn $(upd 5 mn 1) $W naive 0.0001 1"
       echo "bash mlp_pipeline/ft_one.sh mn_n_big_lr3e-4_e5   mn mn $(upd 5 mn 1) $W naive 0.0003 1";;
  rep) W=${2:-10 10 1}   # MORE REPLAY: the old tier l (1027 frames, adds 2000 K and 108-O hosts at 7000 K) instead of m (747). Needs: build_tier.py ... --tag nB --fmax 60 --valid-from --old-train runs/train_l.extxyz --old-valid runs/valid_l.extxyz ; grace_prep_pkl.sh mnB
       echo "bash mlp_pipeline/ft_one.sh mnB_f_lr1e-2_e20  mnB mnB $(upd 20 mnB 1) $W frozen 0.01 1"
       echo "bash mlp_pipeline/ft_one.sh mnB_f_lr1e-2_e5   mnB mnB $(upd 5 mnB 1) $W frozen 0.01 1";;
  *) echo "usage: $0 r1 | r2 \"wE wF wS\" frozen|naive n|mn | len | lrA | lrB | lrC | lc | long | cap | hi | rep ..." >&2; exit 1;;
esac
