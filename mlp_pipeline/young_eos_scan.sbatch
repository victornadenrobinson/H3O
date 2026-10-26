#!/bin/bash -l
# MLP equation-of-state scan with the fine-tuned GRACE model: NVT MD at volumes taken from the DFT EOS for the target pressures 100, 200, ..., 1200 GPa (12 volumes per task, 6 processes at a time on one GPU),
# for several compositions and separately for SOLID and LIQUID branches. Result per run: the MLP pressure at that volume, structure label, Lindemann ratio, D_O and D_H. Summary: python3 mlp_pipeline/22_eos_summary.py
# Task table (33 tasks, default): fcc crystal H(2+n)O with H per O = 2, 2.25, 2.5, 2.75, 3 -> solid branch at 3000, 5000, 7000 K; liquid branch (the O lattice is melted at 12000 K for 1 ps first, then cooled) at 5000, 7000, 9000 K;
# pure H (250 atoms, fluid) at 5000, 7000, 9000 K. A solid run that melts, or a liquid run that did not melt, is flagged in the summary and left out of the fits.
# Volumes: DFT EOS of the composition (fcc path words) if the inventory has it, else ideal mixing of the H2O crystal and pure H (the scan measures the MLP pressure at these volumes anyway); pressures beyond the DFT data
# range reuse the edge volume (warned in the log), so the highest targets may be missing for some groups: check the 'in_range' columns of the summary.
# Cost per task is an ESTIMATE, not measured: 12 volumes x 3.5 ps (0.5 equilibration + 3 production; the liquid branch adds 1 ps of melting) = 7000 steps at about 0.25 s/step with 6 processes sharing the GPU, two waves: about 1 h.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_eos_scan.sbatch
#   subset:   sbatch --array=0-5 ...      own table:  JOBS="fcc solid 5000 2;fcc liquid 7000 3" sbatch --array=0-1 ...      longer/shorter:  PS=5 EQUIL=1 sbatch ...
#   restartable by resubmitting only the missing tasks (a task rewrites its own runs).
# results: runs/eos_scan/eos_<branch>_h<hper>_<phase>_<T>K_i<k>.csv (+ one folder per volume with thermo.csv, traj.extxyz, final.extxyz) and logs in runs/eos_scan/logs
#SBATCH --job-name=eosscan
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH --array=0-32
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
table() { for h in 2 2.25 2.5 2.75 3; do for T in 3000 5000 7000; do echo "fcc solid $T $h"; done; for T in 5000 7000 9000; do echo "fcc liquid $T $h"; done; done; for T in 5000 7000 9000; do echo "H fluid $T 2"; done; }
if [ -n "$JOBS" ]; then IFS=';' read -ra TABLE <<< "$JOBS"; else mapfile -t TABLE < <(table); fi
I=${SLURM_ARRAY_TASK_ID:-0}; read -r PH BR T H <<< "${TABLE[$I]}"
[ -n "$T" ] || { echo "no table entry for array task $I"; exit 2; }
export TF_FORCE_GPU_ALLOW_GROWTH=true H3O_RUNS=$PWD/runs
SNAP=runs/code/eos_$(date +%Y%m%d_%H%M%S)_$$; mkdir -p $SNAP runs/eos_scan/logs && cp -r mlp_pipeline $SNAP/ && CODE=$SNAP/mlp_pipeline   # run from a snapshot: a git pull cannot change a running task
PY=${PY:-$HOME/grace-venv/bin/python}; MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; PS=${PS:-3}; EQUIL=${EQUIL:-0.5}; BASE=eos_${BR}_h${H}_${PH}_${T}K
echo "== task $I: $PH $BR T=$T K H per O $H on $(hostname) $(date) code $(git rev-parse --short HEAD)"
seq 0 11 | xargs -P ${NPAR:-6} -I{} bash -c "$PY -u $CODE/17_eos_md.py $MODEL --phase $PH --branch $BR --T $T --hper $H --from-dft inventory_binned.csv --P 100 200 300 400 500 600 700 800 900 1000 1100 1200 --ps $PS --equil-ps $EQUIL --dt 0.5 --only {} --tag ${BASE}_i{} --outdir runs/eos_scan > runs/eos_scan/logs/${BASE}_i{}.log 2>&1"
echo "== done $(date)"; grep -h "^RESULT" runs/eos_scan/logs/${BASE}_i*.log | cut -c1-300
