#!/bin/bash
# Submit the 4 one-snapshot smoke tests on Young (each uses its own env's python). Run from the repo root.
cd "$(dirname "$0")/.."
mkdir -p runs
sub() { sbatch --parsable --job-name="smoke_$1" "${@:2}" mlp_pipeline/smoke_test.sbatch; }
MODEL=mace      PY=$HOME/work/envs/mace-ng/bin/python       NAME=medium-omat-0 sub mace_omat
MODEL=mace      PY=$HOME/work/envs/mace-ng/bin/python       NAME=medium        sub mace_mp0
MODEL=mattersim PY=$HOME/work/envs/mattersim-env/bin/python                    sub mattersim
MODEL=grace     PY=$HOME/grace-venv/bin/python              NAME=GRACE-2L-OMAT sub grace
echo "submitted; watch: squeue -u $USER ; outputs: runs/smoke_*.out"
