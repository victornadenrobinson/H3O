#!/usr/bin/env bash
# Inside an existing Perlmutter GPU allocation, with the GRACE environment active.
# Preview: bash mlp_pipeline/perlmutter_pwidom.sh --dry-run
# Run:     bash mlp_pipeline/perlmutter_pwidom.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PY:-python}" -u mlp_pipeline/tdi/perlmutter_pwidom.py run "$@"
