#!/bin/bash
# End-to-end test of the SOLUBILITY chain on the toy pair potential (CPU, no GPU): the production launcher young_tdi_par.sh -> summarize.py -> report.py -> solubility_check.py --solid.
# One (P, T) with a solid ice (default 450 GPa 5000 K). Two independent routes to the same free energy of H2O + n H(fluid) -> H(2+n)O:
#   insertion of n_add = 2, 4, 8, 16 H into the 32-O H2O host (a = n_add / 32 = 0.0625 ... 0.5 added H per O),
#   direct crystals H(2+a)O with 32 O and n_add = 4, 8, 16 (HPER = 2 + a; volume by ideal mixing of the toy H2O and pure-H volumes),
# plus the end members H2O and pure H (128 atoms). solubility_check.py then fits insertion and crystals together and per route: the n* of both routes must agree within the errors.
# The toy has no ZPE/NQE/DFT error, so this tests the bookkeeping and the fit, not the physics. ~1 h on 4 cores (PS=2, 10 lambda values); PS=1 NODES=8 for a 25-min look.
# usage (repo root, python with ASE + pandas + matplotlib):  bash mlp_pipeline/tdi/toy_solubility.sh [P] [T] [outdir]       defaults 450 5000 runs/toysol; rerunning continues unfinished runs
P=${1:-450}; T=${2:-5000}; R=${3:-runs/toysol}/P${P}_T${T}; mkdir -p $R; PY=${PY:-python3}
export MODEL=toyice ALLOW_NOGPU=1 PY P T NODES=${NODES:-10} PS=${PS:-2} EQUIL=${EQUIL:-0.5} CHUNK=0.5 DT=0.5 PS1=${PS1:-1.0} NPAR=${NPAR:-1} NREP=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
echo "== toy solubility test at $P GPa, $T K: eos $(date +%H:%M)"
[ -f $R/eos_${T}K.json ] || $PY mlp_pipeline/tdi/toy_validate.py eos --T $T --P $P --systems h2o,H --out $R 2>&1 | tee $R/eos.log || { echo "eos FAILED"; exit 1; }
read VW BW VH BH SOLID < <($PY -c "import json; d=json.load(open('$R/eos_${T}K.json')); w, h = d['h2o'], d['H']; print(round(w['vpa'], 5), round(w['B_GPa'], 1), round(h['vpa'], 5), round(h['B_GPa'], 1), int(w.get('solid', 1)))")
[ "$SOLID" = 1 ] || { echo "ice MELTED at $P GPa $T K (see $R/eos.log): pick a lower T"; exit 1; }
vmix() { $PY -c "print(round((3 * $VW + $1 * $VH) / (3 + $1), 5))"; }   # ideal mixing, $1 = added H per O
echo "   H2O vpa $VW (B $BW GPa), pure H vpa $VH (B $BH GPa); launching 9 runs in parallel $(date +%H:%M)"
go() { n=$1; ph=$2; shift 2; env OUTDIR=$R/tdi "$@" bash mlp_pipeline/young_tdi_par.sh $ph > $R/launch_$n.log 2>&1 || echo "   $n FAILED (see $R/launch_$n.log)"; echo "   $n finished $(date +%H:%M)"; }
go h2o   fcc    VPA=$VW BULK=$BW HPER=2 &
go H     H      VPA=$VH BULK=$BH NATOMS=128 &
for n in 2 4 8 16; do go ins$n insert VPA=$VW BULK=$BW NGHOST=$n & done
for n in 4 8 16; do a=$($PY -c "print($n / 32)"); go cry$n fcc VPA=$(vmix $a) BULK=$BW HPER=$($PY -c "print(2 + $n / 32)") & done
wait
echo "== summarize / report / solubility $(date +%H:%M)"
$PY mlp_pipeline/tdi/summarize.py $R/tdi -o $R/tdi_results.csv > $R/summary.txt 2>&1 || { echo "summarize FAILED: tail $R/summary.txt"; exit 1; }
$PY mlp_pipeline/tdi/report.py --base $R/tdi_results.csv --out $R/report --fmt png --no-gt --state-check none > $R/report.log 2>&1 || { echo "report FAILED: tail $R/report.log"; exit 1; }
$PY mlp_pipeline/tdi/solubility_check.py --solid $R/report_table_solid.csv -o $R/solubility.csv | tee $R/solubility.txt
