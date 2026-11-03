#!/usr/bin/env bash
# CPU-only pair curves and an optional, unexecuted same-frame baseline replay plan.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot "$@"
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/proxy_compare.py "$@"
