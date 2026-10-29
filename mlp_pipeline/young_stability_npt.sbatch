#!/bin/bash -l
# Long NPT (fully flexible cell) stability runs of superionic fcc H(2+n)O with GRACE (mlp_pipeline/tdi/stability_npt.py): does H2.1O / H2.2O stay a superionic fcc solid over tens of ps,
# or does it deform, change O sublattice, melt or phase-separate? H2O is the control. Per-ps lines: cell, shear strain, O sublattice type, Lindemann, O MSD, D_H; VERDICT line at the end.
# Table (6 tasks): H per O 2, 2.1, 2.2 at 400 GPa 4500 K and 500 GPa 5000 K (inside the solid + liquid field of the T-x diagram, near the warm Uranus / Neptune profiles).
# Own table: JOBS="2.1 400 4000;2.2 500 4500" sbatch --array=0-1 ...    env: PS (ps, default 30), NREP (default 4 = 256 O), MODEL
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && sbatch mlp_pipeline/young_stability_npt.sbatch
# then:  grep VERDICT runs/slurm_stabnpt_*.out      (per-ps table: less runs/slurm_stabnpt_<job>_<task>.out)
#SBATCH --job-name=stabnpt
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
table() { for PT in "400 4500" "500 5000"; do for h in 2 2.1 2.2; do echo "$h $PT"; done; done; }
if [ -n "$JOBS" ]; then IFS=';' read -ra TABLE <<< "$JOBS"; else mapfile -t TABLE < <(table); fi
I=${SLURM_ARRAY_TASK_ID:-0}; read -r H P T <<< "${TABLE[$I]}"
[ -n "$T" ] || { echo "no table entry for array task $I"; exit 2; }
export TF_FORCE_GPU_ALLOW_GROWTH=true
SNAP=runs/code/stabnpt_$(date +%Y%m%d_%H%M%S)_$$; mkdir -p $SNAP && cp -r mlp_pipeline $SNAP/   # code snapshot: a git pull during the run cannot change it
echo "== task $I: H${H}O at $P GPa, $T K, ${PS:=30} ps, nrep ${NREP:=4} on $(hostname) $(date) code $(git rev-parse --short HEAD)"
${PY:-$HOME/grace-venv/bin/python} -u $SNAP/mlp_pipeline/tdi/stability_npt.py --hper $H --P $P --T $T --ps $PS --nrep $NREP --model ${MODEL:-grace-ft:runs/grace/ft_m2} -o runs/stability_npt; exit $?
