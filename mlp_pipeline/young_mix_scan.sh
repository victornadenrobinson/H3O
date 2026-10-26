#!/bin/bash
# Quick mixing scan at one (P, T): fcc H2O (108 O), fcc H(2+NH/108)O (NH added H) and pure H (250 atoms), one after the other with the same settings, then the summary table.
# usage on Young, inside a GPU job, repo root:   P=500 T=5000 bash mlp_pipeline/young_mix_scan.sh        NH=27 by default (= H2.25O)
# env: P (400) T (5000) NH (27) NODES (8) NPAR (8) PS (0.5) EQUIL (0.15) CHUNK (0.25) MINRUNS (8) DT (0.5)
# logs: runs/mix/<name>.out (everything the launcher printed); a failed run does not stop the next one; the status lines at the end say which runs produced a result.
P=${P:-400}; T=${T:-5000}; NH=${NH:-27}; NODES=${NODES:-8}; NPAR=${NPAR:-8}; PS=${PS:-0.5}; EQUIL=${EQUIL:-0.15}; CHUNK=${CHUNK:-0.25}; MINRUNS=${MINRUNS:-8}; DT=${DT:-0.5}
PYSUM=${PYSUM:-$HOME/grace-venv/bin/python}
HP=$(python3 -c "print(round(2 + $NH / 108, 6))"); mkdir -p runs/mix
echo "== mixing scan P=$P GPa T=$T K: fcc H2O, fcc H$HP O ($NH added H), pure H; $NODES lambda values, $PS ps each; $(git rev-parse --short HEAD) $(date)"
for spec in "fcc 2" "fcc $HP" "H 2"; do
  set -- $spec; name=${1}_${2}_${P}GPa_${T}K
  echo "-- $name $(date +%H:%M)"
  NATOMS=$([ $1 = H ] && echo 250) HPER=$2 P=$P T=$T NODES=$NODES NPAR=$NPAR DT=$DT EQUIL=$EQUIL PS=$PS CHUNK=$CHUNK MINRUNS=$MINRUNS \
    bash mlp_pipeline/young_tdi_par.sh $1 2>&1 | tee runs/mix/$name.out | grep --line-buffered -E "NOTE|WARNING|G\(t\)|failed|Error|Traceback|FINAL"
done
echo "== status"
for f in runs/mix/*_${P}GPa_${T}K.out; do
  printf "%-45s %s\n" "$(basename $f .out)" "$(grep -qE 'stage 0 failed|Traceback' $f && echo FAILED || (grep -q '== done' $f && echo finished || echo incomplete))"
done
echo "== MLP pressure at the highest lambda (target $P GPa)"; grep -h "MLP pressure at the highest" runs/mix/*_${P}GPa_${T}K.out | sort -u | cut -c1-110
$PYSUM mlp_pipeline/tdi/summarize.py runs/tdi
