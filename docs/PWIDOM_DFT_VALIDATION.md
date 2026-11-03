# Four-part PWidom model validation and retraining workflow

Run from the Perlmutter repository root. Keep the current model and production
results. These commands create separate diagnostics and a challenger dataset.
No job submission, new MD, or fitting happens during preparation.

```bash
git pull --ff-only origin feature/pwidom-checkpoint-resume
export GRACE_ENV="$SCRATCH/grace-env"
export PY="$GRACE_ENV/bin/python"
export VASP_MODULE=vasp/5.4.4-cpu
```

## 1. Original DFT labels and current GRACE baseline

Let `runs/vasp_training_40_1` finish. Its live LABEL lines check VASP reproducibility.
The following evaluates GRACE directly against the original saved labels; it can
run before the VASP reruns finish and requires no new VASP calculations:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh baseline runs/vasp_training_40_1 \
  --model grace-ft:runs/grace/ft_m2/seed/1/final_model --device cpu \
  --out runs/plots/grace_training_baseline_1
```

Outputs baseline.csv and summary.json with per-structure energy/force/stress
errors. The selected sources mix training and validation; this diagnostic is
not a new held-out accuracy estimate. Use the exact model from the PWidom
manifest if its path differs.

## 2. Ordinary MD configurations, with live VASP–GRACE comparisons

First use the same state as the previously labelled edge cases for a controlled
comparison. The known run is 300 GPa / 3000 K. A later 500 or 600 GPa / 5000 K
comparison can substitute an existing completed run at that state.

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh normal \
  --run runs/pwidom_300GPa_3000K_reference --template vasp_inputs \
  --hosts H2O H2p083O H2p25O pureH --per-host 10 \
  --discard-fraction 0.5 --device cpu --out runs/vasp_normal_40_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_normal_40_1 \
  --exe vasp_gam --ranks 32 --parallel 8 --minutes 50
```

These are forty evenly spaced host configurations, with no inserted/deleted
atoms. Model fingerprints, true cell and source temperature are checked. GRACE
energy, forces and stress are evaluated once per prepared geometry, individually
rather than through batched predictions, and saved as prediction references.
They are explicitly labelled GRACE, not DFT training labels. Completion prints
VASP−GRACE energy (whole-cell and per atom), force and stress RMSE. Each point has
label_comparison.json. Eight 32-rank points require 256 tasks total over two CPU
nodes. A current allocation with 128 tasks total supports `--parallel 4`.

Discarding half the cache does not establish equilibration or independence. Check
MD pressure, energy and structural diagnostics. Template settings must match the
training DFT. Pure H Gamma-point finite-size/k-mesh effects remain a separate
question; retain the original paired 2×2×2 comparison.

## 3. Insertion/deletion edge cases and independent replay

Use the existing forty VASP structures; no repeat VASP run is needed:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_saved_40_1 \
  --compare-mlp --batch 1 --forces-stress --device cpu \
  --out runs/plots/vasp_edge_replay_1
```

Prints all four absolute paired energies, saved versus independently recomputed
GRACE addition energy, and replay mismatches. CSV/plots include paired energy
errors and matched force/stress errors. Both insertion and deletion use
E(with H)−E(without H). These importance-weight/proxy-selected configurations
are diagnostic edge cases, not an unbiased free-energy correction sample.

## 4. Dataset, separate test set, and fitting preparation

Wait until the ordinary and edge VASP queues are complete:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh dataset \
  runs/vasp_normal_40_1 runs/vasp_saved_40_1 \
  --out runs/pwidom_dft_dataset_1
bash mlp_pipeline/perlmutter_vasp_check.sh fitprep runs/pwidom_dft_dataset_1 \
  --base-train runs/train_m.extxyz --base-valid runs/valid_m.extxyz \
  --out runs/grace/pwidom_challenger_1
```

Only converged OUTCAR labels are exported: energy=TOTEN, forces, ASE tensile stress
in eV/Å³. Exact geometry/cell/order are checked. Both members of each pair and
matching source/cache frame stay together. Duplicate geometries connect their
groups; conflicting duplicate energies stop export. An approximately 60/20/20
split by source-frame group creates train/valid/test files and a split manifest.
This controls exact-frame leakage, not time correlations between adjacent frames.
For stronger validation reserve whole independent MD runs when more are available.
Do not mix 2×2×2 recalculations into the Gamma-label training set: those are a
separate convergence experiment, not duplicate labels for one DFT target.

fitprep combines the new train/valid portions with the corresponding original
sets, deduplicates geometries, and removes held-out geometries from training.
It leaves new test untouched and stops if original validation overlaps test.
Original held-out test quality must still be monitored separately.

On a GPU fitting environment with the existing GRACE tools (do not fit on the
current VASP CPU allocation), prepare a challenger using the existing foundation
fine-tune recipe:

```bash
export FIT=runs/grace/pwidom_challenger_1
"$GRACE_ENV/bin/extxyz2df" "$FIT/train.extxyz" --output-dataset-filename "$FIT/train.pkl.gz"
"$GRACE_ENV/bin/extxyz2df" "$FIT/valid.extxyz" --output-dataset-filename "$FIT/valid.pkl.gz"
"$PY" mlp_pipeline/grace_make_input.py "$FIT/input.yaml" \
  --train train.pkl.gz --test valid.pkl.gz --updates 8000 --batch 2 --naive
```

This prepares a new foundation-model fine-tune, not a continuation of ft_m2.
Review installed trainer compatibility and compute resources before launching;
keep the test file out of trainer input and hyperparameter choices. The existing
`grace_ft_run.sh` documents fitting/export mechanics and its environment.
Acceptance should compare old/new models on unchanged original validation data,
new held-out normal snapshots, and held-out paired ΔU errors. Do not judge only
per-atom energy RMSE. Refresh ordinary MD and PWidom sampling with the challenger
before claiming corrected chemical potentials; single-point improvement alone
is not a validated free-energy result.
