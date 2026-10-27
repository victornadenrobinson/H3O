#!/bin/bash
# Pipeline runs for the toy validation (tdi/toy_validate.py): the production launcher young_tdi_par.sh, unchanged, with the 'toyice' pair potential on CPU.
# usage (repo root, after `toy_validate.py eos --T <T>`):  bash mlp_pipeline/tdi/toy_validate.sh 5000 [n8]     (n8: also insertion of 8 H = H2.25O by insertion)
# Volumes and bulk moduli from runs/toy/eos_<T>K.json (all systems at the same P). Output in runs/toy/tdi. PY: a python with ASE (default python3).
set -e
T=${1:?temperature}; EXTRA=$2; P=${P:-450}
J=runs/toy/eos_${T}K.json; [ -f $J ] || { echo "no $J: run toy_validate.py eos --T $T first"; exit 1; }
v() { python3 -c "import json; d=json.load(open('$J')); print(round(d['$1']['$2'], 5))"; }
VW=$(v h2o vpa); BW=$(v h2o B_GPa); VM=$(v h2.25o vpa); BM=$(v h2.25o B_GPa); VH=$(v H vpa); BH=$(v H B_GPa)
export MODEL=toyice ALLOW_NOGPU=1 PY=${PY:-python3} OUTDIR=runs/toy/tdi NODES=${NODES:-10} PS=${PS:-2} EQUIL=${EQUIL:-0.5} CHUNK=0.5 DT=0.5 PS1=${PS1:-1.0} NPAR=${NPAR:-4} P T
run() { echo "######## $* $(date +%H:%M:%S)"; env "$@" bash mlp_pipeline/young_tdi_par.sh "${PH}" > runs/toy/launch_${T}K_${NAME}.log 2>&1 || { echo "FAILED: see runs/toy/launch_${T}K_${NAME}.log"; return 1; }; grep -E "FINAL|G =|done" runs/toy/launch_${T}K_${NAME}.log | tail -n 3; }
mkdir -p runs/toy
PH=fcc NAME=h2o run VPA=$VW NREP=2 BULK=$BW HPER=2
PH=fcc NAME=h2.25o run VPA=$VM NREP=2 BULK=$BM HPER=2.25
PH=H NAME=H run VPA=$VH NATOMS=128 BULK=$BH
PH=insert NAME=ins1 run VPA=$VW NREP=2 BULK=$BW NGHOST=1
PH=insert NAME=insH run VPA=$VH NATOMS=128 BULK=$BH NGHOST=1 HOSTPHASE=H
if [ "$EXTRA" = n8 ]; then
  PH=insert NAME=ins8 run VPA=$VW NREP=2 BULK=$BW NGHOST=8
fi
echo "######## all done $(date +%H:%M:%S)"
