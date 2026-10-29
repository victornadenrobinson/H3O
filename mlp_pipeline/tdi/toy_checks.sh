#!/bin/bash
# Quick checks of the insertion method with the toy pair potential, all at one (P, T), all runs in parallel (CPU; ~6 x NPAR cores):
#   ref   insertion of 1 H into H2O (32 O), lambda = x^4                          the reference
#   path  the same with lambda = x^2                                             path independence: must equal ref
#   size  the same in a 108-O host (NREP=3)                                       finite size: must equal ref within a few meV
#   rev   insertion of 1 H into H2.25O (at its own volume at P)                  mu_H at x = 2.25
#   ins8  insertion of 8 H into H2O (average per added H, H2O -> H2.25O)          must equal (ref + rev) / 2 if mu_H is linear in x (regular solution)
#   eos first: volumes of H2O, H2.25O at (P, T) (toy_validate.py eos).
# usage (repo root, python with ASE): TOYICE_FIT=docs/data/toyfit_t4.json bash mlp_pipeline/tdi/toy_checks.sh [P] [T] [outdir]    (defaults 450 5000 runs/toychecks)
# Young: PY=~/grace-venv/bin/python on a node with >= 24 free cores; rerunning continues unfinished runs.
P=${1:-450}; T=${2:-5000}; R=${3:-runs/toychecks}/P${P}_T${T}; mkdir -p $R
export MODEL=toyice ALLOW_NOGPU=1 PY=${PY:-python3} P T NODES=${NODES:-10} PS=${PS:-2} EQUIL=${EQUIL:-0.5} CHUNK=${CHUNK:-0.5} DT=0.5 PS1=${PS1:-1.0} NPAR=${NPAR:-4} NGHOST=1 NREP=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
echo "== $P GPa $T K: eos $(date +%H:%M)"
[ -f $R/eos_${T}K.json ] || $PY mlp_pipeline/tdi/toy_validate.py eos --T $T --P $P --systems h2o,h2.25o --out $R > $R/eos.log 2>&1 || { echo "eos FAILED (see $R/eos.log)"; exit 1; }
read VW BW VM BM SOLID < <(python3 -c "import json; d=json.load(open('$R/eos_${T}K.json')); w, m = d['h2o'], d['h2.25o']; print(round(w['vpa'], 5), round(w['B_GPa'], 2), round(m['vpa'], 5), round(m['B_GPa'], 2), int(w.get('solid', 1) and m.get('solid', 1)))")
[ "$SOLID" = 1 ] || { echo "ice MELTED at $P GPa $T K (see $R/eos.log): pick a lower T"; exit 1; }
echo "   H2O vpa $VW, H2.25O vpa $VM (both solid); 5 runs in parallel $(date +%H:%M)"
go() { n=$1; shift; env OUTDIR=$R/$n "$@" bash mlp_pipeline/young_tdi_par.sh insert > $R/launch_$n.log 2>&1 || echo "   $n FAILED (see $R/launch_$n.log)"; echo "   $n finished $(date +%H:%M)"; }
go ref  VPA=$VW BULK=$BW &
go path VPA=$VW BULK=$BW POWER=2 &
go size VPA=$VW BULK=$BW NREP=3 &
go rev  VPA=$VM BULK=$BM HPER=2.25 &
go ins8 VPA=$VW BULK=$BW NGHOST=8 &
wait
python3 - $R <<'EOF'
import glob, json, sys
from math import hypot
R = sys.argv[1]; m = {}
for n in ("ref", "path", "size", "rev", "ins8"):
    f = glob.glob(f"{R}/{n}/tdi_*.json")
    if f: d = json.load(open(f[0])); m[n] = (d["G_meV_atom"], d["G_err_meV_atom"]); print(f"   {n:5s} mu_H = {m[n][0]:9.1f} +- {m[n][1]:5.1f} meV per added H   ({d['tag']})")
def chk(name, a, b, note):
    if a and b: d, e = a[0] - b[0], hypot(a[1], b[1]); print(f"CHECK {name:42s} {d:7.1f} +- {e:5.1f} meV  {'OK' if abs(d) < 2 * e else 'DIFFERS (> 2 sigma)'}   {note}")
chk("path: lambda=x^2 minus x^4", m.get("path"), m.get("ref"), "expect 0")
chk("size: 108 O minus 32 O", m.get("size"), m.get("ref"), "expect ~0 (a few meV)")
if all(k in m for k in ("ref", "rev", "ins8")):
    chk("linearity: n=8 average minus (ref+rev)/2", m["ins8"], ((m["ref"][0] + m["rev"][0]) / 2, hypot(m["ref"][1], m["rev"][1]) / 2), "expect ~0 if mu_H is linear in x")
    print(f"   slope d mu_H / d(H per O) = {(m['rev'][0] - m['ref'][0]) / 0.25:.0f} +- {hypot(m['rev'][1], m['ref'][1]) / 0.25:.0f} meV  (positive: mu_H rises as the ice fills up)")
EOF
echo "== checks done $(date +%H:%M)"
