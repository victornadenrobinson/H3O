#!/bin/bash
# Convert tier files runs/train_<tag>.extxyz (and the matching valid file) to GRACE pickles runs/grace/<name>_<tag>.pkl.gz with extxyz2df. Perlmutter; plain bash, run from the repo root.
#   bash mlp_pipeline/grace_prep_pkl.sh n n25 n50 n75 mn mn25 mn50 mn75          (GBIN=$SCRATCH/grace-env/bin by default)
# valid file used for a tag: valid_n for n, n25, ...; valid_mn for mn, mn25, ...; otherwise valid_<tag>.
cd "$(dirname "$0")/.."
GBIN=${GBIN:-$SCRATCH/grace-env/bin}; mkdir -p runs/grace
[ -x "$GBIN/extxyz2df" ] || { echo "no $GBIN/extxyz2df: set GBIN to the bin dir of the environment that has gracemaker (ls \$SCRATCH/grace-env/bin | grep -i grace)"; exit 1; }
conv() { # name
  [ -e runs/grace/$1.pkl.gz ] && [ runs/grace/$1.pkl.gz -nt runs/$1.extxyz ] && { echo "have $1"; return; }
  echo "converting $1"; "$GBIN/extxyz2df" runs/$1.extxyz --output-dataset-filename runs/grace/$1.pkl.gz 2>&1 | grep -v -E "^(I0000|W0000|E0000|WARNING|20[0-9][0-9]-|\[tensorpotential\]|To enable)" | tail -2
}
for t in "$@"; do
  case $t in mn*) v=mn;; n*) v=n;; *) v=$t;; esac
  conv train_$t; conv valid_$v
done
ls -la runs/grace/*.pkl.gz | awk '{print $5, $9}'
