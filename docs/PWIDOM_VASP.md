# Perlmutter VASP check: one snapshot first

Use the existing GRACE Python for preparation/analysis. The matching training
inputs are in local `vasp_inputs/` (ignored by Git because it contains POTCAR).
The user reported CPU modules including `vasp/5.4.4-cpu`; start with that module,
not the `vasp-tpc` build, and inspect the training OUTCAR version/settings.
No real VASP/Perlmutter run has been executed from cloud.

```bash
# On Perlmutter, request CPU resources. Use the correct CPU project account.
salloc -N 1 -C cpu -q interactive -t 00:30:00 -n 32 -A m3349
# Once allocated, from the H3O repository:
git pull --ff-only origin feature/pwidom-checkpoint-resume
export GRACE_ENV="$SCRATCH/grace-env"
export PY="$GRACE_ENV/bin/python"
export VASP_MODULE=vasp/5.4.4-cpu
"$PY" -c 'import sys, numpy, ase; print(sys.executable); print("Python/NumPy/ASE OK")'
bash mlp_pipeline/perlmutter_vasp_check.sh preflight --template vasp_inputs
bash mlp_pipeline/perlmutter_vasp_check.sh snapshot \
  --frames runs/pwidom_300GPa_3000K_reference/pw_H2O_300GPa_3000K_frames.npz \
  --template vasp_inputs --out runs/vasp_one_snapshot_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_one_snapshot_1 \
  --ranks 32 --max-points 1 --minutes 20
bash mlp_pipeline/perlmutter_vasp_check.sh status runs/vasp_one_snapshot_1
```

The CPU account `m3349` is the expected counterpart of the reported GPU account
`m3349_g`; verify your allocation/account if it differs. No allocation is requested
or calculation submitted by these source scripts. Preflight does not execute
VASP. It reports the module/executable/settings, without printing POTCAR contents.
Use `--train-outcar ../tdi/H3O/7000K/bcc-128mol-7000K/9.4/OUTCAR.gz` with preflight
if that relative training path is available. `VASP_EXE` can select a different
executable name supplied by the module.

Exactly one existing host configuration is selected (last saved host frame by
default), with its actual cell and temperature. No new MD, proxy sampling or model
calls are made by `snapshot`. This full-size point is a more relevant cost test
than a tiny cell. Optional `smoke --template vasp_inputs --out runs/vasp_small_1`
prepares a separate 12-atom check if the full-size SCF is too slow to diagnose.

VASP preparation reuses the existing `dft_check.py` template writer: original
PAW blocks/species order, cutoff, XC and k-points are retained; MD tags are removed
and NSW=0/IBRION=-1 is used. For Fermi smearing, SIGMA=kB*T of the cached frame.
NBANDS follows the template's bands/electron ratio. Charge/coupling/atom-specific
magnetic templates require explicit review rather than silently using wrong
settings. NCORE/KPAR must divide the selected ranks; parallel settings are not
silently rewritten.

`run --parallel 4 --ranks 32` packs four independent MPI points into a 128-rank
allocation. Each srun step uses `--exclusive --exact` CPU resources; allocation
capacity is validated. Outputs, locks and convergence records are independent.
No new GRACE calls are made when analyzing exact saved queue pairs. Stop launching
new work after a failure; already-running points can finish within the wall cap.
TERM/INT/HUP or the cap stop every active process group and preserve its attempt.

Results go under each point's `attempts/<timestamp>/`. Existing attempts are
never overwritten. Only an exit-zero OUTCAR with finite free-energy TOTEN,
EDIFF convergence and the timing footer gets a `DONE.json`; completed points are
skipped after input-hash validation. Interrupted/unconverged points can be retried
with the same command, retaining all earlier outputs. A per-point lock prevents
concurrent execution. The wall cap stops active child process groups and leaves
unfinished attempts unlabelled. Individual SCF progress is in `vasp.out`/OUTCAR.
These are single-point restarts, not exact electronic-WAVECAR continuation.

## Prefer the exact saved production pairs when available

`runs/dft_candidates/request_*.json` already links each without-H/with-H geometry
and its production GRACE **energy difference** `dU_mlp_eV`. The extxyz files do
not need absolute GRACE-energy labels for this paired comparison. Filter to one
source run rather than mixing P/T/compositions across the whole queue:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh queue \
  --run runs/pwidom_300GPa_3000K_reference
bash mlp_pipeline/perlmutter_vasp_check.sh prep-queue \
  --run runs/pwidom_300GPa_3000K_reference --hosts H2O --pairs 1 \
  --template vasp_inputs --out runs/vasp_saved_pair_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_saved_pair_1 \
  --ranks 32 --max-points 2 --minutes 20
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_saved_pair_1 \
  --out runs/plots/vasp_saved_pair_1
```

This performs two VASP single points for one ΔU comparison. `--max-points 1` runs
one geometry first; repeating it skips that completed geometry and runs its
partner. Later request more pairs, or `--hosts H2O pureH`, subject to the available
counts printed by `queue`. No samples are manufactured if fewer pairs exist.
Each pair requires both VASP energies even though its GRACE difference is saved.
The runner reports ΔU_VASP, ΔU_GRACE and their difference immediately after both
members complete. `watch DIRECTORY` provides the same live read-only report for
a job already running with an older runner; Ctrl-C stops the viewer only.
No GRACE/model calls are made by queue preparation or the default saved-pair
analysis. The source model identity and state must match the selected run.

The importer verifies geometry hashes, the one-H difference and common cell,
preserves direction/frame/tag/model/reasons/log-weight provenance, and deduplicates
identical geometry pairs. Multiple eligible groups are selected round-robin.
Reasons such as `dominant_estimator_weight` and `large_centered_proxy_residual`
make these targeted diagnostics, not an unbiased equilibrium sample. They must
not be passed off as a new pure-DFT BAR/free-energy estimate. Source queue and
run files remain untouched.

## Next: paired insertion/deletion energies

If exact saved requests are unavailable, prepare 40 NEW pairs (80 single-point structures),
split equally between H2O/pure-H and insertion/deletion, from a completed run:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh prep \
  --run runs/pwidom_300GPa_3000K_reference --template vasp_inputs \
  --hosts H2O pureH --pairs 40 --out runs/vasp_pwidom_pairs_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_pwidom_pairs_1 \
  --ranks 32 --max-points 2 --minutes 20
bash mlp_pipeline/perlmutter_vasp_check.sh status runs/vasp_pwidom_pairs_1
```

These are NEW reproducible proxy trials drawn on cached GRACE frames, not a replay
of original saved work sites. Default selection excludes the first half of each
saved ensemble; that is a diagnostic selection, not proof of equilibration.
Forward insertion uses Boltzmann proxy candidates; reverse deletion chooses a
real H from its own N+1 configuration. Both without-H/with-H geometries are kept.
No cubic cell is assumed. Metadata retain seed, frame, proxy, correction and model
fingerprint. Existing source outputs are never altered.

After timings are known, a separate 24-hour CPU submission can use:

```bash
export DFT_MINUTES=1380
sbatch -A m3349 -q regular mlp_pipeline/perlmutter_vasp_check.sbatch runs/vasp_pwidom_pairs_1
```

Verify the account/queue and resource choice first. Override `--ntasks` if a
scaling test justifies it. A 48-hour job requires supported queue limits plus
`--time=48:00:00` and `DFT_MINUTES=2820`. The script is a resumable point queue,
not pure-VASP PWidom MD. No long submission was made during development.

Analyze paired differences after completion, with optional GRACE comparisons:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_pwidom_pairs_1 \
  --compare-mlp --device cpu --out runs/plots/vasp_pwidom_pairs_1
```

Use a fresh analysis directory. The CSV/scatter compare VASP and GRACE ΔU on the
same geometries. Per-element offsets can only cancel against a matched pure-H
reference. GRACE trajectories plus VASP probe energies are NOT automatically
unbiased pure-DFT free energies; genuine DFT ensembles or validated ensemble
reweighting/overlap are needed. PWidom avoids lambda windows, but the fastest
converged VASP route is not established from the GRACE benchmarks.


## Gamma executable versus k-point convergence

Preflight prints KPOINTS and `gamma_only`. `VASP_EXE=vasp_gam` is supported for
future launches only when the prepared mesh is Γ-only (zero shift, 1×1×1 automatic
mesh or one explicit origin point) without SOC/noncollinear features. Incompatible
inputs are rejected before launch. An already-running vasp_std Γ calculation is
valid and need not be restarted. Template KPOINTS are copied to all points,
including the reservoir. Whether that mesh is converged for metallic hydrogen is
a separate calculation: compare paired energy differences against a denser mesh,
not just cell size or the availability of vasp_gam.


## Forces, stress and k-mesh controls

`analyze DIRECTORY --forces-stress --device cpu --out FRESH` reads completed
OUTCAR forces/stresses and reevaluates the saved model on `structure.extxyz`.
It checks species/order, periodic geometry and cell; OUTCAR coordinate rounding
is tolerated up to 1e-4 A, while GRACE sees the original prepared geometry.
It produces force RMSEs (component/vector and H/O species), full matched arrays,
stress/pressure differences and scatter plots. Unavailable OUTCAR quantities
are recorded rather than replaced by zeros. No new VASP/MD is run. Do this after
the packed VASP jobs finish, so the model evaluation does not compete for CPUs.
The queue did not save GRACE forces or full stress tensors; those require this
reevaluation. Comparisons use ASE tensile stress and Voigt xx yy zz yz xz xy,
with eV/A^3 converted to GPa. Both exclude the ions' ideal-gas kinetic pressure.

`mesh-check DIRECTORY --hosts H2O pureH --pairs-per-host 1 --mesh 2 2 2 --out FRESH`
selects one already-complete pair for each host and copies its exact structures,
POTCAR, INCAR and POSCAR into a new prepared set. It writes explicit KPOINTS and
removes unused KSPACING/KGAMMA tags. No original file is changed. Baseline ΔU
is recorded in the new manifest; `analyze` reports its difference as
`kmesh_shift_eV` in the CSV. Force/stress comparison is also available there.
Use vasp_std, not vasp_gam, for the denser mesh. Comparing Γ and 2×2×2 is the first
check, not proof that 2×2×2 is converged; metallic H may need further tests.

All requests in the example queue selection come from the 300-GPa/3000-K
reference run: different host compositions and independent N/N+1 trajectories,
not mixed state points. The pressure and temperature labels are target values;
the instantaneous ionic state is not determined by those labels or a static SCF.
