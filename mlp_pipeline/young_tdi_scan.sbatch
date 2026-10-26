#!/bin/bash -l
# Pressure scan of G_fcc - G_bcc at 3000 K: array task i -> phase (i even: bcc, odd: fcc), pressure PLIST[i/2]. Default PLIST="300 500" -> tasks 0-3. Each task = one GPU, about 40 min
# (6 min lambda=1 stage that measures k, then 10 lambda nodes x (0.4 + 3 ps) in parallel on the GPU). Restartable: resubmit the same command to continue unfinished nodes.
# usage on Young:   cd /lustre/scratch/mmm0037/H3O && git pull && sbatch mlp_pipeline/young_tdi_scan.sbatch
#   other pressures: PLIST="200 600 800" sbatch --array=0-5 mlp_pipeline/young_tdi_scan.sbatch       longer: PS=6 sbatch --time=02:00:00 mlp_pipeline/young_tdi_scan.sbatch
# results: runs/tdi/<phase>_<P>GPa_3000K_* ; compare with  python3 mlp_pipeline/tdi/compare.py runs/tdi/tdi_fcc_300GPa_3000K.json runs/tdi/tdi_bcc_300GPa_3000K.json
#   or the dashboard: ~/grace-venv/bin/python mlp_pipeline/tdi/live.py runs/tdi fcc_300GPa_3000K bcc_300GPa_3000K
# literature G_fcc - G_bcc, meV/atom at 3000 K (Cheng 2021 / Wilson 2013): 200 GPa -10.5 / -9.0, 300 GPa -17.6, 400 GPa -22.1, 500 GPa -26.5 / -24.7
#SBATCH --job-name=tdiscan
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PLIST=(${PLIST:-300 500}); I=${SLURM_ARRAY_TASK_ID:-0}
PHASES=(bcc fcc); PH=${PHASES[$((I % 2))]}; P=${PLIST[$((I / 2))]}
[ -n "$P" ] || { echo "no pressure for array task $I"; exit 2; }
echo "== task $I: $PH at $P GPa, 3000 K on $(hostname) $(date)"
P=$P T=3000 NODES=10 NPAR=10 DT=1.0 EQUIL=0.4 PS=${PS:-3} CHUNK=0.25 PS1=1.5 bash mlp_pipeline/young_tdi_par.sh $PH
