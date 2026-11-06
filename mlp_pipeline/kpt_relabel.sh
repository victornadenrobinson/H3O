#!/bin/bash
# 2x2x2 relabelling of ~300 stratified training/validation frames (for the k-point correction of the Gamma labels). Login node, repo root.
#   bash mlp_pipeline/kpt_relabel.sh                 template + selection + VASP inputs + job plan (nothing submitted)
#   TEST=1 bash mlp_pipeline/kpt_relabel.sh          + sbatch --test-only on each job;   SUBMIT=1 ... submits
# Template = runs/kpt/template_k222 (your template with the Gamma-centred 2x2x2 KPOINTS of the k-point test, validated against 3x3x3) plus KPAR = 8 (4 ranks per k-point
# on each 32-rank lane; changes only the parallel layout, not the result), NPAR removed, NCORE lowered to divide 4. SIGMA per frame = the electronic temperature of its Gamma label
# (kpt_select.py). Runs with vasp_std through dgcheck_jobs.sh (4 lanes x 32 ranks per node; a resubmission resumes unfinished points).
# env: N (300), NODES (8), MINUTES (600 per job), FILES (runs/train_mfin.extxyz runs/valid_mfin.extxyz), OUTD (runs/kpt222), KPAR (8), PY
set -e; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline
OUTD=${OUTD:-runs/kpt222}; T=$OUTD/template; KP=${KPAR:-8}; FILES=${FILES:-"runs/train_mfin.extxyz runs/valid_mfin.extxyz"}
[ -d runs/kpt/template_k222 ] || { echo "need runs/kpt/template_k222 (made by kpoint_jobs.sh)"; exit 1; }
mkdir -p $OUTD
if [ ! -d $T ]; then
  cp -rL runs/kpt/template_k222 $T
  grep -viE "^\s*(KPAR|NPAR)\s*=" $T/INCAR > $T/INCAR.new || true
  per=$((32 / KP)); nc=$(grep -iE "^\s*NCORE\s*=" $T/INCAR.new | head -1 | sed 's/.*=\s*//; s/[^0-9].*//')
  if [ -n "$nc" ] && [ $((per % nc)) -ne 0 ]; then sed -i -E "s/^\s*NCORE\s*=.*/NCORE = $per/I" $T/INCAR.new; echo "NCORE $nc -> $per (must divide 32/KPAR)"; fi
  echo "KPAR = $KP" >> $T/INCAR.new; mv $T/INCAR.new $T/INCAR
fi
echo "template $T: $(grep -iE '^\s*(KPAR|NCORE|ISMEAR|ENCUT|PREC)\s*=' $T/INCAR | tr '\n' ' ')| KPOINTS: $(sed -n 3,4p $T/KPOINTS | tr '\n' ' ')"
[ -e $OUTD/frames.extxyz ] || $PY mlp_pipeline/kpt_select.py $FILES --n ${N:-300} --out $OUTD/frames.extxyz
FRAMES_EXTXYZ=$OUTD/frames.extxyz OUTD=$OUTD TEMPLATE=$T EXE=vasp_std NODES=${NODES:-8} MINUTES=${MINUTES:-600} bash mlp_pipeline/dgcheck_jobs.sh
echo "after the jobs: \$PY mlp_pipeline/points_collect.py $OUTD/points_* --out $OUTD/labelled_k222.extxyz --db-set K222   (labels keep the source frame info: kpt_src, E_gamma_eV)"
