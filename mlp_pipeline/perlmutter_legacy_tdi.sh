#!/usr/bin/env bash
# Read-only original DFT/pair/CMC discovery; no VASP module or calculation needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot --dry-run
exec "${H3O_PYTHON[@]}" mlp_pipeline/discover_legacy_tdi.py "$@"
