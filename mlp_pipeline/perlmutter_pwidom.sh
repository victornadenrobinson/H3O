#!/usr/bin/env bash
# Inside an existing Perlmutter GPU allocation; discover/check the environment.
# Preview: bash mlp_pipeline/perlmutter_pwidom.sh --dry-run
# Run:     bash mlp_pipeline/perlmutter_pwidom.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source mlp_pipeline/perlmutter_python.sh
h3o_select_python run "$@"
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/perlmutter_pwidom.py run "$@"
