#!/usr/bin/env bash
# Prepare/check paired VASP energies with the existing GRACE Python environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action=${1:---help}
# Only run/preflight need the licensed VASP executable and site modules.
if [[ "$action" == run || "$action" == preflight || "$action" == vasp-config || "$action" == dft-plan || "$action" == dft-worker ]]; then
    if [[ -n "${VASP_MODULE:-}" ]]; then
        if ! type module >/dev/null 2>&1; then
            echo 'module command unavailable; use a Perlmutter login shell or load your site setup.' >&2
            exit 2
        fi
        module load "$VASP_MODULE"
    fi
    type module >/dev/null 2>&1 && module list >&2 || true
fi
source mlp_pipeline/perlmutter_python.sh
h3o_select_python plot --dry-run
exec "${H3O_PYTHON[@]}" -u mlp_pipeline/tdi/perlmutter_vasp_check.py "$@"
