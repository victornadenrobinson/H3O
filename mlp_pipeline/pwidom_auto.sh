#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PS=(500)
TS=(2000)
NS=(5 11 16 22)

SAMPLE_PS=1.0
EQUIL_PS=0.4
MINUTES=30
CHECKPOINT_SECONDS=300
SCAN_OPTIONS=0
RESUME_RUN=""
RESUME_OUT=""
CANDIDATES="runs/dft_candidates"
DRY_RUN=0
INVENTORY="${PWIDOM_INVENTORY:-runs/inventory_binned.csv}"

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
  --resume-run DIR       Resume one prior stage (inherits its inputs)
  --out NEW_DIR          Required fresh output with --resume-run
  --checkpoint-seconds N  MD checkpoint wall interval (default 300)
  --minutes VALUE        Existing launcher limit (default 30)
  --candidate-dir PATH   Candidate output directory
  --inventory PATH       Binned H/O DFT inventory (needed only for missing EOS)
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
            SCAN_OPTIONS=1
            shift
            read_values "--P" "$@"
            PS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --T)
            SCAN_OPTIONS=1
            shift
            read_values "--T" "$@"
            TS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --n)
            SCAN_OPTIONS=1
            shift
            read_values "--n" "$@"
            NS=("${VALUES[@]}")
            shift "${#VALUES[@]}"
            ;;
        --ps)
            SCAN_OPTIONS=1
            SAMPLE_PS="${2:?Missing --ps value}"
            shift 2
            ;;
        --equil-ps)
            SCAN_OPTIONS=1
            EQUIL_PS="${2:?Missing --equil-ps value}"
            shift 2
            ;;
        --resume-run)
            RESUME_RUN="${2:?Missing previous run}"
            shift 2
            ;;
        --out)
            RESUME_OUT="${2:?Missing fresh output}"
            shift 2
            ;;
        --checkpoint-seconds)
            CHECKPOINT_SECONDS="${2:?Missing checkpoint interval}"
            shift 2
            ;;
        --minutes)
            MINUTES="${2:?Missing --minutes value}"
            shift 2
            ;;
        --candidate-dir)
            SCAN_OPTIONS=1
            CANDIDATES="${2:?Missing candidate directory}"
            shift 2
            ;;
        --inventory)
            SCAN_OPTIONS=1
            INVENTORY="${2:?Missing inventory path}"
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

# Share interpreter discovery and explicit overrides with the underlying launcher.
source mlp_pipeline/perlmutter_python.sh
if [[ -n "${PWIDOM_PYTHON:-}" ]]; then
    export PY="$PWIDOM_PYTHON"
fi
h3o_select_python run --dry-run
PYTHON="$("${H3O_PYTHON[@]}" -c 'import sys; print(sys.executable)')"
export PY="$PYTHON"

# Resume is an explicit single-stage operation; source outputs stay untouched.
if [[ -n "$RESUME_RUN" ]]; then
    if ((SCAN_OPTIONS)); then
        echo "ERROR: --resume-run inherits scan inputs; omit P/T/n/ps/equil/inventory/candidate options" >&2
        exit 2
    fi
    if [[ -z "$RESUME_OUT" ]]; then
        echo "ERROR: --resume-run requires --out NEW_DIR" >&2
        exit 2
    fi
    RESUME_COMMAND=(bash mlp_pipeline/perlmutter_pwidom.sh
        --resume-run "$RESUME_RUN" --out "$RESUME_OUT"
        --minutes "$MINUTES" --checkpoint-seconds "$CHECKPOINT_SECONDS")
    if ((DRY_RUN)); then RESUME_COMMAND+=(--dry-run); fi
    exec "${RESUME_COMMAND[@]}"
elif [[ -n "$RESUME_OUT" ]]; then
    echo "ERROR: --out is only supported with --resume-run" >&2
    exit 2
fi

"$PYTHON" - "${PS[*]}" "${TS[*]}" "$SAMPLE_PS" "$EQUIL_PS" "$MINUTES" "$CHECKPOINT_SECONDS" <<'PYVALID'
import math, sys
p, t, ps, eq, minutes, checkpoint = sys.argv[1:]
values = [float(x) for x in (p + " " + t).split()]
if not all(math.isfinite(x) and x > 0 for x in values):
    raise SystemExit("P/T must be finite and positive")
ps, eq, minutes, checkpoint = map(float, (ps, eq, minutes, checkpoint))
if not all(map(math.isfinite, (ps, eq, minutes, checkpoint))) or ps < .1 or eq < 0 or minutes <= 0 or checkpoint <= 0:
    raise SystemExit("Require ps >= .1, equil-ps >= 0, minutes > 0 (finite)")
PYVALID

if ((${#NS[@]} != 4)); then
    echo "ERROR: --n requires exactly four counts" >&2
    exit 2
fi

# Validate integer counts, range and uniqueness.
declare -A SEEN_N=()

for N in "${NS[@]}"; do
    if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid extra-H count: $N" >&2
        exit 2
    fi

    N=$((10#$N))
    if ((N < 1 || N > 27)); then
        echo "ERROR: n=$N exceeds 0.25 H/O for 108 O" >&2
        exit 2
    fi

    if [[ -n "${SEEN_N[$N]:-}" ]]; then
        echo "ERROR: repeated extra-H count: $N" >&2
        exit 2
    fi

    SEEN_N[$N]=1
done

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

    "$PYTHON" - "$manifest" "$P" "$T" "${NS[@]}" <<'PY'
import json
import sys
from pathlib import Path

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    assert data.get("status") == "complete"
    assert float(data["P"]) == float(sys.argv[2])
    assert float(data["T"]) == float(sys.argv[3])
    if sys.argv[1].endswith("_extra/manifest.json"):
        assert data.get("extra_h") == [int(x) for x in sys.argv[4:]]
    else:
        assert data.get("extra_h") is None
    assert len(data["jobs"]) == 4
    for job in data["jobs"]:
        assert job.get("exit_code") == 0
        result = json.loads(Path(job["result"]).read_text())
        assert "mu_ex_bar_meV" in result
except (OSError, ValueError, KeyError, AssertionError):
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
                --P "$P" --T "$T" --inventory "$INVENTORY"

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
        if [[ -f "$STATE_JSON" ]]; then
            "$PYTHON" - "$STATE_JSON" "$P" "$T" <<'PY'
import json
import math
import sys

filename, pressure, temperature = sys.argv[1:]
pressure = float(pressure)
temperature = float(temperature)

with open(filename) as f:
    state = json.load(f)

assert abs(float(state["P"]) - pressure) < 1e-6
assert abs(float(state["T"]) - temperature) < 1e-6

for key in ("ice", "h225", "H"):
    for field in ("vpa", "B_GPa"):
        value = float(state[key][field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}.{field}: must be finite and positive")
    if not state[key].get("source"):
        raise ValueError(f"Missing {key}.source")

for key, field, expected in (("ice", "nrep", 3), ("h225", "nrep", 3), ("H", "natoms", 250)):
    if type(state[key][field]) is not int or state[key][field] != expected:
        raise ValueError(f"Invalid {key}.{field}: expected integer {expected}")

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
                --checkpoint-seconds "$CHECKPOINT_SECONDS" \
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
                --checkpoint-seconds "$CHECKPOINT_SECONDS" \
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
