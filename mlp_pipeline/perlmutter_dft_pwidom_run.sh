#!/usr/bin/env bash
# Bounded runner: PLAN --minutes 20 --smoke, or PLAN --minutes 2820.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VASP_MODULE="${VASP_MODULE:-vasp/5.4.4-cpu}"
module load "$VASP_MODULE"
command -v vasp_gam >/dev/null || { echo 'vasp_gam unavailable'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot --dry-run
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/dft_pwidom_runner.py "$@"
