#!/usr/bin/env bash
# Read-only status of a particular scan (pass its output directory).
# On the allocated node, nvidia-smi also reports the four GPU loads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python status "$@"
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/perlmutter_pwidom.py status "$@"
