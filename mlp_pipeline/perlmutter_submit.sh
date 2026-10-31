#!/bin/bash
# Submit one of the Young .sbatch files on NERSC Perlmutter: the Young-only #SBATCH lines (partition, account, qos, gres, mem, constraint, output) are dropped from a temporary copy
# and replaced by Perlmutter's (-C gpu, your _g account, a qos, N GPUs). The script body is untouched, so its knobs are plain environment variables, exactly as on Young.
# usage (repo root on Perlmutter; runs/ may be a symlink to perlmutter-runs/):
#   PY=$SCRATCH/h3o/grace-venv/bin/python PS=4 bash mlp_pipeline/perlmutter_submit.sh mlp_pipeline/young_pwidom.sbatch
#   ACCT=m3349_g QOS=regular GPUS=1 TIME=04:00:00 DRYRUN=1 bash mlp_pipeline/perlmutter_submit.sh <file.sbatch> [extra sbatch args]     (DRYRUN=1 only prints)
# knobs: ACCT (m3349_g) QOS (regular; interactive/debug are for salloc / short tests) GPUS (1) TIME (default: the file's own --time). Needs on Perlmutter: the repo, a python env with tensorpotential
# (PY=...), and whatever inputs the job reads (stage0.json, frames, model checkpoint, inventory) under runs/ or the paths the script uses; outputs go to runs/ (= perlmutter-runs/).
f=${1:?"usage: bash mlp_pipeline/perlmutter_submit.sh <young .sbatch> [extra sbatch args]"}; shift
ACCT=${ACCT:-m3349_g}; QOS=${QOS:-regular}; GPUS=${GPUS:-1}; mkdir -p runs
tmp=$(mktemp ${SCRATCH:-/tmp}/pm_$(basename $f .sbatch)_XXXX.sbatch) || exit 1
grep -v -E '^#SBATCH --(partition|account|qos|gres|mem|constraint|output|gpus)' $f > $tmp
cmd=(sbatch -A $ACCT -C gpu -q $QOS --gpus $GPUS ${TIME:+-t $TIME} --output=runs/slurm_%x_%j.out "$@" $tmp)
echo "${cmd[@]}"; echo "(from $f; dropped: $(grep -c -E '^#SBATCH --(partition|account|qos|gres|mem|constraint|output|gpus)' $f) Young #SBATCH lines; script copy $tmp)"
[ -n "$DRYRUN" ] && { grep '^#SBATCH' $tmp; exit 0; }
"${cmd[@]}"
