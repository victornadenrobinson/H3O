#!/bin/bash -l
# Extensive fcc vs bcc H2O scan: P = 100-600 GPa (steps of 50) x T = 2000 K upwards (steps of 500) up to a guessed melting limit TMAX(P) (tdi/fcc_bcc_grid.py: 95 (P, T) points), TDI of the fcc (108 O) and then the bcc (128 O) crystal at each point,
# 10 lambda values (Gauss-Legendre, power 4), 8 ps per lambda after 1 ps equilibration (default PS=8; PS=3 for a quick pass), dt 0.5 fs. Results in runs/tdi_fb (tags carry _rep3 / _rep4 because the volume is passed explicitly); summarize.py runs/tdi runs/tdi_fb pairs fcc and bcc at equal P, T.
# Volume: DFT EOS of the phase, else of the other phase (tdi/fcc_bcc_grid.py vpa; points with no volume at all are skipped and logged). Look at 'fcc_bcc_grid.py coverage' first.
# TMAX is a guess: points near it may be molten (a crystal held by springs gives a wrong G: check free NVT with young_melt_scan.sbatch, the quadrature estimate and the MLP pressure in summarize.py).
# Cost per task is an ESTIMATE from the 3 ps bcc 384-atom run (about 50 min with 10 processes): fcc + bcc about 1.5-2 h at 3 ps, so about 3.5-4.5 h at 8 ps = roughly 330-430 GPU-h for the whole grid. One task = one (P, T) point.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_fccbcc.sbatch          subset: sbatch --array=0-9 ...          more time per lambda: PS=6 sbatch --time=08:00:00 ...
#SBATCH --job-name=tdifb
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-94%8
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
ARRAY_KEEP="PS"; . mlp_pipeline/array_env.sh
export TF_FORCE_GPU_ALLOW_GROWTH=true; PY=$HOME/grace-venv/bin/python
mapfile -t GRID < <(python3 mlp_pipeline/tdi/fcc_bcc_grid.py grid)
I=${SLURM_ARRAY_TASK_ID:-0}; [ "$I" -lt "${#GRID[@]}" ] || { echo "task $I: grid has ${#GRID[@]} points"; exit 0; }
read -r P T <<< "${GRID[$I]}"
echo "== task $I: fcc and bcc H2O at $P GPa $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
for PH in fcc bcc; do
  r=$($PY mlp_pipeline/tdi/fcc_bcc_grid.py vpa $PH $T $P 2>&1 | tail -n1)
  case "$r" in NA|""|*Error*|*Traceback*) echo "-- $PH $P GPa $T K: no volume ($r): skipped"; continue;; esac
  V=${r%% *}; echo "-- $PH: V/atom $V A^3 (DFT EOS: ${r##* })"
  NREP=$([ $PH = fcc ] && echo 3 || echo 4) VPA=$V P=$P T=$T HPER=2 NODES=10 NPAR=10 DT=0.5 EQUIL=1 PS=${PS:-8} PS1=1.5 CHUNK=0.5 OUTDIR=runs/tdi_fb bash mlp_pipeline/young_tdi_par.sh $PH
  echo "   exit $?"
done
