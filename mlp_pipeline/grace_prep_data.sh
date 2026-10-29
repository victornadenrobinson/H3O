#!/bin/bash
# Convert a dataset tier (default s) to GRACE's dataframe pickles and show what is inside, so the stress/energy columns can be checked.
# usage (Young, repo root):  bash mlp_pipeline/grace_prep_data.sh [tier]
cd "$(dirname "$0")/.."
T=${1:-s}; G=$HOME/grace-venv/bin
mkdir -p runs/grace
for k in train valid; do
  $G/extxyz2df runs/${k}_${T}.extxyz --output-dataset-filename runs/grace/${k}_${T}.pkl.gz 2>&1 | grep -v -E "^(I0000|W0000|E0000|WARNING|20[0-9][0-9]-|\[tensorpotential\]|To enable)" | tail -4
done
$G/python - <<PY 2>/dev/null
import pandas as pd, numpy as np
for k in ("train", "valid"):
    df = pd.read_pickle(f"runs/grace/{k}_$T.pkl.gz")
    print(f"\n{k}: {len(df)} rows; columns: {list(df.columns)}")
    r = df.iloc[0]
    for c in df.columns:
        v = r[c]
        print(f"  {c:18s} {type(v).__name__:10s} {getattr(v, 'shape', '')}", (str(v)[:70] if np.ndim(v) == 0 else ""))
    if "energy" in df: print("  energy/atom range:", float((df.energy / df.ase_atoms.map(len)).min()), float((df.energy / df.ase_atoms.map(len)).max()))
PY
ls -la runs/grace
echo "next: gracemaker -t   (dialog: fine-tune foundation model GRACE-2L-SMAX-OMAT-medium, train=runs/grace/train_${T}.pkl.gz, test=runs/grace/valid_${T}.pkl.gz, fit stress, stress_units eV/A3)"
