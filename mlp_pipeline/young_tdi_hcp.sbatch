#!/bin/bash -l
# fcc (ABCABC, 'fccl') vs hcp (ABABAB) H2O TDI along the 3000 K isotherm, both in the SAME orthohexagonal 96-O cell (a x sqrt3 a x 2 sqrt6 a, 6 layers), so atoms, cell and finite-size error are identical and only the stacking differs. 10 lambda values, PS ps per lambda (default 10), dt 0.5.
# One task per pressure (P = 100 ... 600 GPa in steps of 50: 11 tasks), fccl first then hcp. Results in runs/tdi_hcp; summarize.py prints 'G_fcc(ABC, layered cell) - G_hcp'; report.py draws it.
# Volume: the DFT fcc EOS (same for both stackings: the volume difference is tiny); where the DFT fcc EOS has no data (fcc exists only at about 330-540 GPa at 3000 K) the volume comes from tdi/fcc_bcc_grid.py (bcc EOS or scaling); both volumes are passed with VPA.
# EXPECT A SMALL, HARD NUMBER: stacking-fault free energies are a few meV/atom at most (Andriambariarijaona et al. 2025 report mixed fcc/hcp close-packed stacking at 0.4-0.6 TPa), and each phase has a statistical error of about 1.3 meV/atom at 8-10 ps: the difference is +-2 meV/atom. A control: fccl vs the cubic fcc 108-O result of the same state point should agree within the finite-size difference.
# NOT TESTED on Young: the layered builder was only checked geometrically in the sandbox (12 neighbours, one nearest-neighbour distance, volume per atom) and by py_compile. Cost per task is an ESTIMATE: 2 x (96 O + 192 H = 288 atoms, 10 lambda, 11 ps) = 2-3 h. --time=08:00:00.
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && bash mlp_pipeline/submit.sh "fcc vs hcp 3000 K" mlp_pipeline/young_tdi_hcp.sbatch       own pressures: PLIST="300 400 500" bash ... --array=0-2 mlp_pipeline/young_tdi_hcp.sbatch
#SBATCH --job-name=tdihcp
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-10%4
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
ARRAY_KEEP="PS"; . mlp_pipeline/array_env.sh
PLIST=(${PLIST:-100 150 200 250 300 350 400 450 500 550 600}); T=${T:-3000}
P=${PLIST[${SLURM_ARRAY_TASK_ID:-0}]}; [ -n "$P" ] || { echo "no pressure for array task $SLURM_ARRAY_TASK_ID"; exit 2; }
echo "== task $SLURM_ARRAY_TASK_ID: fccl and hcp H2O at $P GPa $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true; PY=$HOME/grace-venv/bin/python
r=$($PY mlp_pipeline/tdi/fcc_bcc_grid.py vpa fcc $T $P 2>&1 | tail -n1)
case "$r" in NA|""|*Error*|*Traceback*) echo "no volume ($r): skipped"; exit 0;; esac
V=${r%% *}; echo "V/atom $V A^3 (source ${r##* })"
for PH in fccl hcp; do
  NREP=4 VPA=$V P=$P T=$T HPER=2 NODES=10 NPAR=10 DT=0.5 EQUIL=1.0 PS=${PS:-10} PS1=1.5 CHUNK=0.5 OUTDIR=runs/tdi_hcp bash mlp_pipeline/young_tdi_par.sh $PH
  echo "   $PH exit $?"
done
