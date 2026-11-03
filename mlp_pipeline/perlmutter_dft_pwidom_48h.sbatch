#!/bin/bash -l
#SBATCH --job-name=dft-pwidom
#SBATCH --nodes=2
#SBATCH --constraint=cpu
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=dft-pwidom-%j.out
# sbatch --account=m3349 --qos=regular ... PREPARED_PLAN
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export GRACE_ENV="${GRACE_ENV:-$SCRATCH/grace-env}"
export VASP_MODULE="${VASP_MODULE:-vasp/5.4.4-cpu}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load "$VASP_MODULE"
command -v vasp_gam >/dev/null || { echo "vasp_gam unavailable"; exit 2; }
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot --dry-run
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/dft_pwidom_runner.py "${1:?Provide prepared plan directory}" --minutes "${DFT_MINUTES:-2820}"
