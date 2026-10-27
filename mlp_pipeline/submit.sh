#!/bin/bash
# Submit a Slurm script AND record it:  bash mlp_pipeline/submit.sh "note" [sbatch options] mlp_pipeline/<script>.sbatch
# e.g.  JOBS="500 6000 4;500 6000 11" PS=10 NODES=12 bash mlp_pipeline/submit.sh "insertion n curve 500/6000" --array=0-1 --time=14:00:00 mlp_pipeline/young_tdi_insert.sbatch
# Appends one tab-separated line to runs/JOBS_LOG.tsv (on Lustre, not in git): date, job id, script, sbatch options, the run knobs that were set, git commit, note. Read it back with  bash mlp_pipeline/jobs_status.sh
note=${1:?usage: submit.sh "note" [sbatch options] script.sbatch}; shift
script=${@: -1}; [ -f "$script" ] || { echo "last argument must be the .sbatch file (got: $script)"; exit 1; }
mkdir -p runs; id=$(sbatch --parsable "$@") || { echo "sbatch failed"; exit 1; }
knobs=""; for v in JOBS PS NODES NPAR DT EQUIL HOT HOTT MODEL PLIST TLIST HLIST CLIST SEED HOLD ONLYTAG RULE NGAUSS ONLYNODES PREPHOLD PREPFREE FRESH; do [ -n "${!v}" ] && knobs="$knobs $v=${!v}"; done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "${id%%;*}" "$(basename $script)" "${*:1:$#-1}" "${knobs# }" "$(git rev-parse --short HEAD 2>/dev/null)" "$note" >> runs/JOBS_LOG.tsv
echo "submitted job ${id%%;*} ($note); logged in runs/JOBS_LOG.tsv"
