#!/usr/bin/env bash
# Read-only diagnostics in the configured GRACE environment. No GPU/model calls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot "$@"
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/pwidom_analysis.py "$@"
