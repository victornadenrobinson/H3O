#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PS=(500)
TS=(2000)
NS=(5 11 16 22)

SAMPLE_PS=1.0
EQUIL_PS=0.4
MINUTES=30
CANDIDATES="runs/dft_candidates"
DRY_RUN=0

usage() {
cat <<'EOF'
Usage:
  bash mlp_pipeline/pwidom_auto.sh [options]

Options:
  --P P [P ...]          Pressures in GPa
  --T T [T ...]          Temperatures in K
  --n N [N ...]          Extra H atoms per 108-O host
  --ps VALUE             Production sampling, ps (default 1.0)
  --equil-ps VALUE       Equilibration, ps (default 0.4)
  --minutes VALUE        Existing launcher limit (default 30)
  --candidate-dir PATH   Candidate output directory
  --dry-run              Show commands without running MD
  --help

Missing EOS JSONs are generated automatically.

Requires four GPUs in the current Slurm allocation
for real PWidom runs.
EOF
}

# Parse one or more values following --P, --T or --n.
read_values() {
    local kind="$1"
    shift

    VALUES=()

    while (($#)); do
        if [[ "$1" == --* ]]; then
            break
        fi
        VALUES+=("$1")
        shift
    done

    if ((${#VALUES[@]} == 0)); then
        echo "ERROR: $kind requires values" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --P)
            shift
            read_values "--P" "$@"
            PS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --T)
            shift
            read_values "--T" "$@"
            TS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --n)
            shift
            read_values "--n" "$@"
            NS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --ps)
            SAMPLE_PS="${2:?Missing --ps value}"
            shift 2
            ;;
        --equil-ps)
            EQUIL_PS="${2:?Missing --equil-ps value}"
            shift 2
            ;;
        --minutes)
            MINUTES="${2:?Missing --minutes value}"
            shift 2
            ;;
        --candidate-dir)
            CANDIDATES="${2:?Missing candidate directory}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ((${#NS[@]} != 4)); then
    echo "ERROR: --n requires exactly four counts" >&2
    exit 2
fi

# Validate integer counts, range and uniqueness.
declare -A SEEN_N=()

for N in "${NS[@]}"; do
    if [[ ! "$N" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid extra-H count: $N" >&2
        exit 2
    fi

    if ((10#$N > 27)); then
        echo "ERROR: n=$N exceeds 0.25 H/O for 108 O" >&2
        exit 2
    fi

    if [[ -n "${SEEN_N[$N]:-}" ]]; then
        echo "ERROR: repeated extra-H count: $N" >&2
        exit 2
    fi

    SEEN_N[$N]=1
done

# Use the same Python environment as the working
# Perlmutter installation.
PYTHON="${PWIDOM_PYTHON:-/pscratch/sd/v/victornr/grace-env/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter unavailable" >&2
    exit 1
fi

if [[ ! -f "runs/inventory_binned.csv" ]]; then
    echo "ERROR: runs/inventory_binned.csv missing" >&2
    exit 1
fi

if [[ ! -f "build_eos_2000.py" ]]; then
    echo "ERROR: build_eos_2000.py missing" >&2
    exit 1
fi

if [[ ! -f "mlp_pipeline/perlmutter_pwidom.sh" ]]; then
    echo "ERROR: Perlmutter launcher missing" >&2
    exit 1
fi

# Return success only for an explicitly completed manifest.
completed() {
    local directory="$1"
    local manifest="$directory/manifest.json"

    [[ -f "$manifest" ]] || return 1

    "$PYTHON" - "$manifest" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    assert data.get("status") == "complete"
except (OSError, ValueError, AssertionError):
    sys.exit(1)
PY
}

# A directory with previous outputs must never be silently
# overwritten if its manifest is incomplete.
check_existing() {
    local directory="$1"

    if [[ -e "$directory" ]] && ! completed "$directory"; then
        echo "ERROR: existing incomplete run: $directory" >&2
        echo "Inspect its manifest and logs before retrying." >&2
        echo "Refusing to overwrite or restart it automatically." >&2
        exit 1
    fi
}

run_command() {
    printf '\nCOMMAND:'
    printf ' %q' "$@"
    printf '\n\n'

    if ((DRY_RUN)); then
        return 0
    fi

    "$@"
}

printf '\nPWidom automatic scan\n'
printf 'Pressures: %s\n' "${PS[*]}"
printf 'Temperatures: %s\n' "${TS[*]}"
printf 'Extra H: %s\n' "${NS[*]}"
printf 'Python: %s\n' "$PYTHON"

if ((DRY_RUN)); then
    echo "DRY RUN: no EOS writes or MD launches"
fi

for P in "${PS[@]}"; do
    for T in "${TS[@]}"; do

        echo
        echo "=================================================="
        echo "PWIDOM: $P GPa / $T K"
        echo "=================================================="

        # The EOS generator names files using %g formatting.
        P_TAG="$("$PYTHON" -c \
            'import sys; print(f"{float(sys.argv[1]):g}")' "$P")"
        T_TAG="$("$PYTHON" -c \
            'import sys; print(f"{float(sys.argv[1]):g}")' "$T")"

        STATE_JSON="envs/pwidom_${P_TAG}GPa_${T_TAG}K.json"

        REFERENCE_OUT="runs/pwidom_${P_TAG}GPa_${T_TAG}K_reference"
        EXTRA_OUT="runs/pwidom_${P_TAG}GPa_${T_TAG}K_extra"

        if [[ ! -f "$STATE_JSON" ]]; then
            echo "Missing EOS: $STATE_JSON"
            echo "Generating from DFT inventory..."

            run_command \
                "$PYTHON" build_eos_2000.py \
                --P "$P" --T "$T"

            if ((DRY_RUN)); then
                echo "EOS generation planned."
            elif [[ ! -f "$STATE_JSON" ]]; then
                echo "ERROR: EOS generation failed: $STATE_JSON" >&2
                exit 1
            fi
        else
            echo "EOS exists: $STATE_JSON"
        fi

        # Check the state matches this requested condition.
        if ((!DRY_RUN)); then
            "$PYTHON" - "$STATE_JSON" "$P" "$T" <<'PY'
import json
import sys

filename, pressure, temperature = sys.argv[1:]
pressure = float(pressure)
temperature = float(temperature)

with open(filename) as f:
    state = json.load(f)

assert abs(float(state["P"]) - pressure) < 1e-6
assert abs(float(state["T"]) - temperature) < 1e-6

for key in ("ice", "h225", "H"):
    assert float(state[key]["vpa"]) > 0
    assert float(state[key]["B_GPa"]) > 0

assert int(state["ice"]["nrep"]) == 3
assert int(state["h225"]["nrep"]) == 3
assert int(state["H"]["natoms"]) == 250

print(f"EOS validated: {filename}")
PY
        fi

        # --------------------------------------------------
        # Stage 1: four reference hosts
        # --------------------------------------------------

        if completed "$REFERENCE_OUT"; then
            echo
            echo "SKIP completed reference: $REFERENCE_OUT"
        else
            if ((!DRY_RUN)); then
                check_existing "$REFERENCE_OUT"
            fi

            echo
            echo "STAGE 1: Four reference hosts"

            run_command \
                bash mlp_pipeline/perlmutter_pwidom.sh \
                --P "$P" \
                --T "$T" \
                --state-json "$STATE_JSON" \
                --ps "$SAMPLE_PS" \
                --equil-ps "$EQUIL_PS" \
                --minutes "$MINUTES" \
                --out "$REFERENCE_OUT" \
                --candidate-dir "$CANDIDATES"
        fi

        if ((!DRY_RUN)) && ! completed "$REFERENCE_OUT"; then
            echo "ERROR: reference did not complete successfully" >&2
            echo "See: $REFERENCE_OUT/manifest.json" >&2
            exit 1
        fi

        # --------------------------------------------------
        # Stage 2: four extra-H compositions
        # --------------------------------------------------

        if completed "$EXTRA_OUT"; then
            echo
            echo "SKIP completed extra-H scan: $EXTRA_OUT"
        else
            if ((!DRY_RUN)); then
                check_existing "$EXTRA_OUT"
            fi

            echo
            echo "STAGE 2: Extra-H scan: ${NS[*]}"

            run_command \
                bash mlp_pipeline/perlmutter_pwidom.sh \
                --P "$P" \
                --T "$T" \
                --state-json "$STATE_JSON" \
                --extra-h "${NS[@]}" \
                --reference-run "$REFERENCE_OUT" \
                --ps "$SAMPLE_PS" \
                --equil-ps "$EQUIL_PS" \
                --minutes "$MINUTES" \
                --out "$EXTRA_OUT" \
                --candidate-dir "$CANDIDATES"
        fi

        if ((!DRY_RUN)) && ! completed "$EXTRA_OUT"; then
            echo "ERROR: extra-H scan did not complete" >&2
            echo "See: $EXTRA_OUT/manifest.json" >&2
            exit 1
        fi

        echo
        echo "FINISHED: $P GPa / $T K"
    done
done

echo
echo "All requested P-T conditions processed."
