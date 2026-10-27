#!/bin/bash
# Short diagnostics of the insertion / mixing free energies (docs/CURRENT_ISSUES.md P1-P3), all in one go, for an interactive GPU job on Young (repo root):
#    bash mlp_pipeline/young_insert_checks.sh                 # default tags below
#    TAGS="insert_500GPa_6000K_n1 fcc_500GPa_6000K_h2.25o" bash mlp_pipeline/young_insert_checks.sh
# Output: runs/checks/insert_checks_<date>.txt (send that file back). Cost: steps 1-2 seconds on CPU; step 3 a few minutes per tag on the GPU (FRAMES x (2 x DIRS x 11 + UNIFORM) MLP calls).
#  1. node_stats.py: per lambda value, mean / error / median / max, share of the sum carried by the top 1% of samples (heavy tail), drift between halves, contribution
#     to G, closest H-H / H-O contacts: is the small-lambda end converged, and where does the integral come from?
#  1b. mu_breakdown.py: mu_H - G_H split into reference pieces, the TDI integral by lambda range (with its drift) and the pressure correction; figure runs/checks/..._mu.pdf.
#  2. collect.py: G versus simulation time of the insertion runs.
#  3. short_range.py: energy cost of an extra proton at 0.1-1.2 A from host O and H atoms and at uniform random points (the lambda -> 0 integrand), on frames of
#     an insertion run and of the H2.25O crystal: is the MLP repulsive at short range? ('NOT REPULSIVE AT SHORT RANGE' = the free energies are not trustworthy)
# Every tag is looked up in runs/tdi_long and runs/tdi (both are reported when both exist).
main() {
PY=${PY:-$HOME/grace-venv/bin/python}; FRAMES=${FRAMES:-5}; DIRS=${DIRS:-20}; UNIFORM=${UNIFORM:-200}
TAGS=${TAGS:-"insert_500GPa_6000K_n1 insert_500GPa_5500K_n1 insert_500GPa_5500K_n4 insert_400GPa_5000K_n2 fcc_500GPa_6000K_h2.25o fcc_500GPa_6000K H_500GPa_5500K_N250"}
PROBE=${PROBE:-"insert_500GPa_6000K_n1 fcc_500GPa_6000K_h2.25o"}   # step 3 (GPU) only for these
mkdir -p runs/checks; LOG=runs/checks/insert_checks_$(date +%Y%m%d_%H%M).txt
export TF_FORCE_GPU_ALLOW_GROWTH=true
{
echo "== insertion checks $(date) on $(hostname), code $(git rev-parse --short HEAD)$(git diff --quiet || echo +local)"
echo "== 1. per-lambda statistics"
for t in $TAGS; do for d in runs/tdi_long runs/tdi; do [ -f $d/${t}_stage0.json ] && python3 mlp_pipeline/tdi/node_stats.py $d $t; done; done
echo; echo "== 1b. breakdown of mu_H - G_H (reference pieces, TDI integral by lambda range, drift by lambda range, second quadrature) -> ${LOG%.txt}_mu.pdf/.png"
RUNS=$(for t in $TAGS; do case $t in insert_*) for d in runs/tdi_long runs/tdi; do [ -f $d/${t}_stage0.json ] && echo $d/$t; done;; esac; done)
[ -n "$RUNS" ] && $PY mlp_pipeline/tdi/mu_breakdown.py $RUNS -o ${LOG%.txt}_mu
echo; echo "== 2. G versus time (insertion runs)"
for t in $TAGS; do case $t in insert_*) for d in runs/tdi_long runs/tdi; do [ -f $d/${t}_stage0.json ] && { echo "-- $d/$t"; python3 mlp_pipeline/tdi/collect.py $d $t --every-ps 0.5 2>&1 | grep -v "^ *lambda"; }; done;; esac; done
echo; echo "== 3. short-range probe of the MLP (GPU)"
for t in $PROBE; do
  d=$(for x in runs/tdi_long runs/tdi; do [ -f $x/${t}_stage0.json ] && echo $x && break; done)
  [ -n "$d" ] || { echo "-- $t: not found"; continue; }
  echo "-- $d/$t"; $PY -u mlp_pipeline/tdi/short_range.py $d $t --frames $FRAMES --dirs $DIRS --uniform $UNIFORM 2>&1 | grep -vE "tensorflow|XLA|cuda|absl|E0000|W0000|I0000|ptxas|^$"
done
echo; echo "== done $(date)"
} 2>&1 | tee $LOG
echo "wrote $LOG"
}
main "$@"
