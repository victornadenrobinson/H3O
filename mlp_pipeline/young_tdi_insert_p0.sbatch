#!/bin/bash -l
# Add the pressure correction (stage 0 of the insertion code, commit 29ccd28) to EXISTING insertion runs WITHOUT redoing any lambda sampling: for every runs/tdi*/insert_*_stage0.json that has no P_host_GPa yet,
# the original command is rebuilt from its json (tdi/extend.py cmd) and run with ONLY0=1: stage 0 does the two short free full-MLP NVT runs (host alone, host + n H) and rewrites the json with the correction; node files are untouched.
# Tasks split the tags: tag i goes to task i mod 4. Only runs whose json lacks the correction are done (rerun-safe).   ONLYTAG=insert_400GPa_5000K_n2 sbatch --array=0-0 ...   tests one tag first.
# NOT yet run on real data (sandbox: bash -n only). Cost: two free NVT runs of EQUIL + PS1 ps per tag, about 5-15 min each (estimate).
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_insert_p0.sbatch
#SBATCH --job-name=insp0
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
. mlp_pipeline/array_env.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true
if [ -n "$ONLYTAG" ]; then TAGS=(runs/tdi/${ONLYTAG}_stage0.json); else mapfile -t TAGS < <(ls runs/tdi/insert_*_stage0.json runs/tdi_long/insert_*_stage0.json 2>/dev/null | sort); fi
echo "== ${#TAGS[@]} insertion runs, task ${SLURM_ARRAY_TASK_ID:-0} of 4 $(date) code $(git rev-parse --short HEAD)"
for ((i = ${SLURM_ARRAY_TASK_ID:-0}; i < ${#TAGS[@]}; i += 4)); do
  js=${TAGS[$i]}; base=${js%_stage0.json}
  grep -q '"P_host_GPa"' $js && { echo "-- $base: correction already present"; continue; }
  ps=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['args']['ps'])" $js) || { echo "-- $base: cannot read ps"; continue; }
  line=$(python3 mlp_pipeline/tdi/extend.py cmd $base $ps) || { echo "-- $base: extend.py cmd failed"; continue; }
  echo "-- $base: $line"; ONLY0=1 bash -c "$line" 2>&1 | grep -E "pressure correction|MLP pressures|refus|Error|Traceback|stage 0" ; echo "   exit ${PIPESTATUS[0]}"
done
