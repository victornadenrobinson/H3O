# PWidom audit and safe validation

Base: `gpt/pwidom-diagnostics`, `eef2f6f`. No production MD or data changes
were performed. Tests use synthetic inputs, dry runs and existing CPU diagnostics.

## Ranked findings

1. **Scientific qualification: finite sampling is not unbiased free energy.**
   The Rosenbluth *exponential averages* are unbiased under equilibrated NVT
   host/full ensembles, uniform candidate positions, Boltzmann selection with
   replacement, identical H labels, and a fixed Hamiltonian and cell. Their
   logarithms and finite-sample BAR are generally biased; consistency requires
   ensemble sampling and overlap. Eight insertion trials share both a frame and
   candidate pool; four deletions share a full frame. Kish fractions count weight
   concentration, not independent observations. Five contiguous blocks in
   production do not establish decorrelation. No estimator rewrite was warranted.
2. **Foreign-material inventories:** EOS previously trusted `comp_snap` alone.
   Now requires H/O counts and empty `Other`, checks total counts when available,
   and checks labels against counts. The intended default remains
   `runs/inventory_binned.csv` (the user's approximately 10,025-row inventory),
   never the unrelated untracked `mlp_pipeline/inventory_binned.csv`.
   Neither inventory is present/tracked in this checkout: row counts, real
   coverage and exact temperature support could not be independently verified.
   `--inventory` / `PWIDOM_INVENTORY` selects a different binned inventory.
   Raw inventories must first pass the existing binning pipeline, which adds
   kinetic pressure once and removes non-H/O materials. No DFT data were created.
3. **Skew-cell tagged-H distances:** fractional rounding is not general minimum
   image geometry. Replaced by ASE `find_mic`. The earlier independent-trajectory
   mixing fix was already present (`full[:-1] - full[-1]`). Host pair counts use
   directed neighbors, normalized by `N_a*(N_b-delta_ab)*shell/V`; tagged-H
   normalization uses the N remaining neighbors. Both use the finite-size
   ideal-gas reference g=1. Cell heights restrict cutoff to avoid multiple images.
   Histogram edges now end exactly at the neighbor cutoff. Missing cell metadata
   still requires explicit `--assume-cubic`; it is never silently inferred.
4. **EOS extrapolation:** the prior grid search stopped at a finite log-volume
   expansion. Direct polynomial roots now remove that artificial bound, keeping
   only negative-slope roots. `B=-P*d(log P)/d(log V)` is correct. Quadratics
   must decrease throughout the fitted interval, otherwise a linear fit is used.
   There can be at most one stable quadratic root. Temperature interpolation
   linearly combines V(P,T) and dV/dP, so `B=-V/(dV/dP)` stays consistent.
   Temperature extrapolation uses the nearest two bins. Extrapolation has no
   arbitrary P/T distance limit but can still fail for no stable root, overflow,
   negative volume or modulus. Such failures are deliberate: arbitrary state
   requests do not guarantee a physically meaningful extrapolated EOS.
   Pressure and temperature extrapolation are marked and printed; interpolated
   results retain method metadata. Fits are empirical and do not certify phases.
5. **Automatic orchestration:** removed unconditional inventory requirement when
   EOS already exists; added configurable inventory, finite P/T/time preflight,
   positive canonical H counts, and checks for completed result files and matching
   state/counts before skipping. Existing incomplete or running directories are
   refused. This is safe rerunning of completed stages, **not automatic recovery
   of partial MD**. A stale running manifest is not proof a process is alive.
   Verify allocation/process status before choosing a fresh output directory.
6. **Convergence diagnostics were missing:** added production BAR reconstruction,
   forward/reverse exponentials, work histograms, Fermi acceptance overlap, Kish
   fraction, contiguous frame blocks, and CSV/PDF outputs. Corresponding JSON
   results are checked automatically. No independent-trial uncertainty is reported.

## Mathematical checks

With `W=mean(exp(-phi/kT))`, insertion work is
`w=dU-phi-kT*log(W)`. Reverse pools contain the actual deleted H plus K-1
uniform candidates. The same expression is evaluated as deletion log-work.
Production saves its **negative** as `reverse_work_eV`; pass it directly to BAR.
BAR solves
`sum sigmoid((mu-M-wf)/kT) = sum sigmoid((M-mu-wr)/kT)`,
where `M=kT*log(nf/nr)` and sums are over trials. Unequal counts are correctly
normalized. Tests enumerate an extended-space insertion/deletion example and
check unequal-count constant works and Gaussian Crooks distributions.

Ideal contribution `kT*log((NH+1)*Lambda_H**3/V)` has the correct indistinguishable
particle factor for insertion at fixed V. Pressure correction
`-V*((Pfull-Ptarget)**2-(Phost-Ptarget)**2)/(2*B)` (converted GPa A^3 to eV)
is a local elastic approximation using common V and B, not an exact NPT
estimator. Large pressure mismatch, composition-dependent compressibility,
phase change, and pressure noise can invalidate it. Production's measured-pressure
correction and `pwidom_scan.dg` robust correction are different reporting choices;
the new convergence tool reconstructs **excess fixed-V BAR only**.

## Launcher behavior verified from source

Both `--extra-h` and `--reference-run` are parsed and validated. Four GPU IDs
must be distinct; one process is assigned to each. Reference jobs are H2O,
H2.0833O, H2.25O and pure H. Extra hosts use per-O volume interpolation between
H2O/H2.25O and reuse the completed pure-H reference with matching P/T/model
fingerprint. Interpolated hosts use the ice bulk modulus, an existing approximation.
P/T loops run sequentially, references before extra hosts. Each system performs
host equilibration/sampling then independent N+1 equilibration/sampling. Dashboard
interleaving across four systems therefore alternates MD and equilibration normally.
Extensions intentionally start fresh velocities and re-equilibrate both ensembles;
they retain old frames but do not continue integrator state exactly.

The launcher handles TERM/INT/HUP and wall caps by terminating child process groups,
waiting, then killing survivors and updating its manifest. A hard allocation kill
can prevent finalization. Unfinished MD is not checkpointed. One failed child does
not immediately stop other children; they finish or reach the cap. The cap is per
stage, not a guarantee the entire P/T sweep fits the allocation. Automatic wrapper
signal propagation and Slurm GPU execution were not exercised here. Use the status
viewer and scheduler state; do not delete outputs or manually mark them complete.

## Commands and required inputs

CPU dependencies: Python, NumPy, SciPy, pandas, matplotlib, ASE and pytest.
GPU launches additionally require the existing GRACE environment, exported model,
ice/H proxies and a four-GPU allocation. EOS JSON requires P, T, and `ice`,
`h225`, `H` objects containing `vpa`, `B_GPa`, `source`, plus crystal `nrep=3`
and H `natoms=250`. The builder writes these sizes and all three compositions.
Existing JSONs are skipped; only explicit `--force` permits replacement.

```bash
python build_eos_2000.py --inventory /path/to/inventory_binned.csv \
  --P 200 400 600 --T 2000 4000 6000 --outdir /tmp/pwidom-eos-review
bash mlp_pipeline/pwidom_auto.sh --dry-run --P 500 --T 2000 4000 \
  --n 5 11 16 22 --inventory /path/to/inventory_binned.csv
python mlp_pipeline/plot_pwidom_diagnostics.py --frames /path/to/cache_frames.npz \
  --rmax 3 --out /tmp/pwidom-rdf-review
python mlp_pipeline/plot_bar_convergence.py /path/to/*_work.npz \
  --out /tmp/pwidom-bar-review --blocks 5
```

Choose rmax below half the smallest cell height. The RDF shows equilibrium host
pairs and an equilibrium N+1 tagged hydrogen environment, **not** a true weighted
insertion-conditioned RDF. That needs saved candidate positions and weights with
explicit conditioning. Fractions in convergence figures are sampling fractions,
not physical time. Parent/extension arrays overlap and must not be pooled as
independent samples. New convergence output directories must be fresh.

## Validation and remaining Perlmutter work

Regression commands (no MD):

```bash
python -m pytest -q tests/test_pwidom_audit.py tests/test_pwidom_diagnostics.py
python -m pytest -q tests/test_perlmutter_pwidom.py -k \
 'preflight or inspect_eos or gpu_metrics or progress_eta or old_run_log or terminal_dashboard or discovers or bad_python or initialization_recipe or conda_named or missing_environment'
bash -n mlp_pipeline/pwidom_auto.sh
git diff --check
```

Results: 13 tests passed in the first group, 11 passed/6 deselected in the
second; shell syntax and whitespace checks passed. Only an external pytest
plugin deprecation warning. MD tests intentionally excluded. Real inventory
EOS residuals/phase coverage, legacy cell provenance, actual result reconstruction,
correlation times, model batching and Slurm/interruption behavior remain unverified.

## Safe Perlmutter deployment

After transferring/pushing the review branch, use a fresh worktree. These commands
leave both existing production and diagnostics worktrees and all runs untouched:

```bash
cd /pscratch/sd/v/victornr/H3O
git fetch origin audit/pwidom-validation
git worktree add --detach /pscratch/sd/v/victornr/H3O-pwidom-audit \
  origin/audit/pwidom-validation
cd /pscratch/sd/v/victornr/H3O-pwidom-audit
python -m pytest -q tests/test_pwidom_audit.py
bash mlp_pipeline/pwidom_auto.sh --dry-run --P 500 --T 2000 4000 \
  --n 5 11 16 22 \
  --inventory /pscratch/sd/v/victornr/H3O/runs/inventory_binned.csv
```

Use the configured Python with dependencies. To inspect production caches, pass
absolute input paths to the diagnostics commands above and fresh `/tmp` output
paths. No symlinking, copying, overwriting or regeneration of production runs is
needed. Real MD requires a separate explicit allocation/runtime validation.
