#!/bin/bash
# Quick low-P / cold / hydrate snapshot campaign for DFT labelling, in two steps (run from the repo root in the 4-GPU job):
#   bash mlp_pipeline/lowp_quick.sh check     files, python, model, GPUs, a REAL 1-minute smoke test (model relax of C3 + a short MD), then the plan with counts and time
#   bash mlp_pipeline/lowp_quick.sh run       the whole campaign (liquids, cold, C2, C3) on all GPUs, with a summary at the end
# Coverage (override any variable on the command line): liquids PS="10 30 60" GPa x TS="3000 5000" K x HPERS "2 3 4 6 H" (H = pure H, 686 atoms);
# cold 1000/2000 K at 10/30/50 GPa for "2 4 H"; hydrates C2 (360 atoms) and C3 (224 atoms), 4 scales x 3 rattles. MDPS 0.4 ps keeps each MD short.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; MODEL=${MODEL:-grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model}
export PS=${PS:-"10 30 60"} TS=${TS:-"3000 5000"} HPERS=${HPERS:-"2 3 4 6 H"} COLDP=${COLDP:-"10 30 50"} COLDT=${COLDT:-"1000 2000"} COLDHPERS=${COLDHPERS:-"2 4 H"} MDPS=${MDPS:-0.4} COLDPS=${COLDPS:-0.3} CMDMIN=${CMDMIN:-10} PERGPU=${PERGPU:-3} MODEL PY
MODE=${1:-check}; ok=1; bad() { echo "  FAIL: $*"; ok=0; }
if [ "$MODE" = check ]; then
  echo "== files"; for f in mlp_pipeline/lowp_snapshots.sh mlp_pipeline/supercell_snapshots.py mlp_pipeline/17_eos_md.py mlp_pipeline/eos_low_p.py docs/data/hydrates/c2_pnas_10GPa.vasp docs/data/hydrates/c3_pnas.vasp; do [ -e "$f" ] || bad "missing $f"; done
  [ -d "${MODEL#grace-ft:}" ] || bad "model dir ${MODEL#grace-ft:} not found"; ls runs/eos_scan >/dev/null 2>&1 || bad "runs/eos_scan missing (EOS table for volumes)"
  echo "== python / GPUs"; $PY -c "import ase, numpy, pandas, tensorpotential; print('  python ok')" 2>&1 | tail -1 || bad "python imports"
  nvidia-smi -L 2>/dev/null | sed 's/^/  /' | head -4; [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 1 ] || bad "no GPU visible"
  if [ $ok = 1 ]; then
    echo "== smoke test: C3 relax with the real model (5 steps, 1 scale)"; mkdir -p runs/lowp_smoke
    $PY mlp_pipeline/supercell_snapshots.py docs/data/hydrates/c3_pnas.vasp --name C3 --out runs/lowp_smoke/c3.extxyz --min-atoms 150 --max-atoms 250 --min-edge 8.0 --scales 1.0 --rattle 0.03 --seeds 1 --T 1000 --relax "$MODEL" --relax-steps 5 2>&1 | tail -4 | cut -c1-200
    [ -s runs/lowp_smoke/c3.extxyz ] || bad "hydrate smoke test wrote nothing"
  fi
  echo "== plan"; OUT=runs/lowp3_check bash mlp_pipeline/lowp_snapshots.sh 2>&1 | grep -E "^compositions|^cold|^hydrates|^plan|missing|no line|failed" | cut -c1-220
  echo; [ $ok = 1 ] && echo "CHECK OK: now run   bash mlp_pipeline/lowp_quick.sh run" || echo "CHECK FAILED: fix the lines above first"
  echo "(the plan's time uses ~4 min per MD; with MDPS=$MDPS it is usually shorter. Anything over ~100 min: reduce TS / PS / HPERS)"
elif [ "$MODE" = run ]; then
  RUN=1 bash mlp_pipeline/lowp_snapshots.sh 2>&1 | tee logs_pw/lowp3.log | grep -E "^compositions|^cold|^hydrates|^plan|FAILED|finished|relaxed|snapshot sources|^  runs/hydrates|^ +[0-9.]+ +lowP|^ +-?[0-9.]+ +"
else echo "usage: $0 check|run"; fi
