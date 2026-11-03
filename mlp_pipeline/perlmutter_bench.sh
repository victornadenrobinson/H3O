#!/usr/bin/env bash
# Quick throughput screening + whole-node validation inside an existing allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python run "$@"
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/perlmutter_bench.py "$@"
