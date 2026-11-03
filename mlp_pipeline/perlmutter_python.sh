#!/usr/bin/env bash
# Shared interpreter discovery for the Perlmutter shell entry points. Source only.
# PY is authoritative. H3O_ENV_INIT can name a local module/conda setup script;
# GRACE_ENV can name an environment directory. No installation or login changes.

h3o_python_ready() {
    "$@" -c 'import importlib.util, sys
missing = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]
sys.exit(0 if sys.version_info >= (3, 11) and not missing else 1)' \
        "${h3o_required[@]}" >/dev/null 2>&1
}

h3o_select_python() {
    local task=$1; shift
    local arg previous="" model="" cheap=0 cpu=0 candidate base
    local -a h3o_required=(numpy pandas ase scipy) candidates=()
    if [[ "$task" == plot ]]; then h3o_required+=(matplotlib); fi
    for arg in "$@"; do
        case "$arg" in
            --help|-h|--dry-run|--inspect-eos|--inspect-eos=*) cheap=1 ;;
            --cpu) cpu=1 ;;
            --model=*) model=${arg#--model=} ;;
        esac
        if [[ "$previous" == --model ]]; then model=$arg; fi
        previous=$arg
    done
    if [[ "$task" == run && "$cheap" == 0 && "$cpu" == 0 && "$model" != toyice ]]; then
        h3o_required+=(tensorflow tensorpotential matscipy)
    fi
    if [[ -n "${H3O_ENV_INIT:-}" ]]; then
        if [[ ! -r "$H3O_ENV_INIT" ]]; then
            echo "Cannot read H3O_ENV_INIT=$H3O_ENV_INIT" >&2; return 1
        fi
        # An explicit user-provided recipe can load site modules/activate conda.
        source "$H3O_ENV_INIT" || return 1
    fi
    if [[ -n "${PY:-}" ]]; then
        if ! h3o_python_ready "$PY"; then
            echo "PY=$PY is unavailable or lacks Python >=3.11 / ${h3o_required[*]}." >&2
            echo "PY is explicit: fix it or unset PY to enable environment discovery." >&2
            return 1
        fi
        H3O_PYTHON=("$PY")
    elif [[ -n "${GRACE_ENV:-}" ]]; then
        candidate=$GRACE_ENV/bin/python
        if ! h3o_python_ready "$candidate"; then
            echo "GRACE_ENV=$GRACE_ENV has no usable Python >=3.11 / ${h3o_required[*]}." >&2
            echo "If site modules are needed, set H3O_ENV_INIT to your activation recipe." >&2
            return 1
        fi
        H3O_PYTHON=("$candidate")
    else
        # Active environments take precedence over documented/discoverable paths.
        for base in "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}"; do
            [[ -z "$base" ]] || candidates+=("$base/bin/python")
        done
        candidates+=(python python3)
        if [[ -n "${SCRATCH:-}" ]]; then
            candidates+=("$SCRATCH/h3o/grace-venv/bin/python"
                         "$SCRATCH/h3o/grace-env/bin/python"
                         "$SCRATCH/grace-env/bin/python"
                         "$SCRATCH/grace-venv/bin/python")
        fi
        candidates+=("$PWD/grace-env/bin/python" "$PWD/.venv/bin/python"
                     "$HOME/.conda/envs/grace-env/bin/python"
                     "$HOME/miniconda3/envs/grace-env/bin/python"
                     "$HOME/anaconda3/envs/grace-env/bin/python")
        H3O_PYTHON=()
        for candidate in "${candidates[@]}"; do
            if h3o_python_ready "$candidate"; then
                H3O_PYTHON=("$candidate"); break
            fi
        done
        # conda run also applies activation hooks; no shell initialization needed.
        if [[ ${#H3O_PYTHON[@]} == 0 ]] && command -v conda >/dev/null 2>&1; then
            if h3o_python_ready conda run --no-capture-output -n grace-env python; then
                H3O_PYTHON=(conda run --no-capture-output -n grace-env python)
            fi
        fi
        if [[ ${#H3O_PYTHON[@]} == 0 ]]; then
            echo "No usable Python >=3.11 with ${h3o_required[*]} found." >&2
            echo "Set GRACE_ENV=/path/to/grace-env or PY=/path/to/grace-env/bin/python." >&2
            echo "For module/conda activation, set H3O_ENV_INIT=/path/to/your/setup.sh." >&2
            echo "No environment was installed and no GPU calculation started." >&2
            return 1
        fi
    fi
    # Child scripts use this interpreter's directory through PATH as well.
    local executable
    executable=$("${H3O_PYTHON[@]}" -c 'import sys; print(sys.executable)')
    export PATH="${executable%/*}:$PATH"
    echo "Perlmutter Python: $executable (checked: ${h3o_required[*]})" >&2
}
