#!/bin/bash -l
# Finite-size test of G at fixed state points, 8 ps per lambda: pure H with N = 250, 500, 750 atoms (500 GPa / 6000 K and 400 GPa / 5000 K), fcc H2O with 32, 108, 256 O (NREP 2, 3, 4) at 500 GPa / 6000 K,
# liquid H2O with 32, 108 O (NREP 2, 3) at 500 GPa / 6000 K. Array task i = line i of the table. The report (tdi/report.py) draws G(N) - G(largest N) and the error vs N from every pair of sizes with equal dt it finds.
# Restartable; an existing run of the same tag (e.g. fcc_500GPa_6000K_rep3, H_500GPa_6000K_N250) is CONTINUED to 8 ps with its own settings only if you launch it with extend.py; this table starts new tags in runs/tdi_fs and never touches them.
# Cost per task is an ESTIMATE, not measured: scales roughly with N (750 atoms about 3x a 250-atom run), several hours; --time=14:00:00 (the 750-atom H and 256-O fcc tasks may need more: resubmit, it restarts).
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_finite_size.sbatch
#SBATCH --job-name=tdifs
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --array=0-10%6
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
ARRAY_KEEP="PS"; . mlp_pipeline/array_env.sh
TABLE=("H 500 6000 250" "H 500 6000 500" "H 500 6000 750" "H 400 5000 250" "H 400 5000 500" "H 400 5000 750" "fcc 500 6000 2" "fcc 500 6000 3" "fcc 500 6000 4" "liquid 500 6000 2" "liquid 500 6000 3")
read -r PH P T SZ <<< "${TABLE[${SLURM_ARRAY_TASK_ID:-0}]}"
[ -n "$PH" ] || { echo "no table entry for array task $SLURM_ARRAY_TASK_ID"; exit 2; }
echo "== task $SLURM_ARRAY_TASK_ID: $PH $P GPa $T K size parameter $SZ on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
case $PH in
  H) SIZE="NATOMS=$SZ"; NP=8; HP=2; HOTV="";;
  fcc) SIZE="NREP=$SZ"; NP=9; HP=2; HOTV="";;
  liquid) SIZE="NREP=$SZ"; NP=10; HP=2; HOTV=${HOT-0.5};;
esac
env $SIZE P=$P T=$T HPER=$HP HOT=$HOTV NODES=10 NPAR=$NP DT=0.5 EQUIL=1 PS=${PS:-8} PS1=1.5 CHUNK=0.5 MINRUNS=8 OUTDIR=runs/tdi_fs bash mlp_pipeline/young_tdi_par.sh $PH
