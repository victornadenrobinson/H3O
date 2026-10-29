#!/bin/bash
# Rebuild everything downstream of runs/mined_h3o.csv: inventory -> snapshots (s/m/l) -> checks -> splits -> plots -> bundle.
# usage (from the repo root): [BY_PHASE=1] [MIN_BOX=9.0] bash mlp_pipeline/build_all.sh [workers]   (MIN_BOX = drop runs with cell edge V^(1/3) below this many A)
set -euo pipefail
W=${1:-64}
cd "$(dirname "$0")/.."
M=mlp_pipeline
mkdir -p runs; exec > >(tee runs/build.log) 2>&1   # send_bundle.sh checks this log for FAIL/PASS lines
python $M/01_bin_inventory.py mined_h3o.csv data-h3o-convex-hull.csv | tee runs/inventory_summary.txt | tail -45
python $M/02_extract_snapshots.py inventory_binned.csv snapshots --workers "$W" ${MIN_BOX:+--min-box $MIN_BOX} ${BY_PHASE:+--by-phase}
python $M/10_check_snapshots.py snapshots_s.extxyz snapshots_m.extxyz snapshots_l.extxyz --workers "$W"
for t in s m l; do python $M/03_split.py snapshots_$t.extxyz; done
python $M/10_check_snapshots.py train_m.extxyz valid_m.extxyz --workers "$W"
python $M/06_plot_coverage.py mined_h3o.csv data-h3o-convex-hull.csv
python $M/11_bundle.py --include-plots
echo "done: plots in runs/plots, bundle in runs/"
