# mlp_pipeline

Run order is encoded in `build_all.sh`. Typical flows:

```bash
# Perlmutter (data lives there): after mining runs/mined_h3o.csv
bash mlp_pipeline/build_all.sh 64 2>&1 | tee runs/build.log    # inventory -> s/m/l datasets -> checks -> splits -> plots -> bundle
bash mlp_pipeline/send_bundle.sh                               # upload to Young (needs ssh key id_young), checksum-verified

# Young: models (each in its own env; use that env's python)
<env>/bin/python mlp_pipeline/smoke_test.py --model mace --name medium-omat-0
<env>/bin/python mlp_pipeline/04_benchmark.py mace:medium-omat-0 valid_m.extxyz
sbatch mlp_pipeline/05_finetune_young.sbatch                   # TAG=m selects train_m/valid_m (placeholders to fill)
```
Conventions and data manifest: see ../PROJECT.md. Inputs given as bare filenames resolve to `runs/<name>` first.

## Perlmutter: four-GPU short Widom scan

Inside an **existing** four-GPU interactive allocation, activate the GRACE Python
environment, then run from the repository (or invoke the script by absolute path):

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/<output-directory> --gpu
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/<output-directory> --follow
```

Both shell entry points check the environment before starting. `PY=/path/to/python`
is authoritative; otherwise `GRACE_ENV=/path/to/grace-env` selects its interpreter.
Without either, they try active venv/conda interpreters, python/python3, known
Perlmutter scratch paths (including `$SCRATCH/h3o/grace-venv`), local `grace-env`/
`.venv` and common conda paths, then `conda run -n grace-env` when available.
They print the selected interpreter, check Python >=3.11 and analysis packages,
and check TensorFlow/tensorpotential/matscipy for actual GRACE launches. Inspection,
status, help and dry-run do not require the GPU packages. This checks package
availability; real GRACE/model/GPU checks still happen in the existing evaluator.
There is no automatic package installation or guessed module load.

For site-specific module/conda activation, save your **existing working recipe**
outside Git and set `H3O_ENV_INIT=/path/to/setup.sh`. The scripts source it before
discovery; it can load modules and activate the environment. Set `GRACE_ENV` or
`H3O_ENV_INIT` once in your shell setup to reuse it on future allocations.
An explicit broken PY/GRACE_ENV produces an error rather than silently selecting
another environment. Standard venvs can be used directly without activation.
The launcher submits no job. Defaults: 500 GPa, 3000 K, H2O/H2.083O/H2.25O/pure-H,
one process per GPU, 0.4 ps sampling + 0.2 ps equilibration per MD, K=2000,
8 insertions and 4 deletions per frame, batch 8, a 10-minute wall cap.
The intermediate composition uses the existing per-O volume interpolation.
State/volume/bulk-modulus metadata come from `runs/tdi*/*_stage0.json`;
`--roots DIR ...` limits discovery. The exact choices are recorded in the manifest.
Model: `grace-ft:runs/grace/ft_m2` (one unambiguous SavedModel export required).
Proxies: `runs/pwidom_fit/proxy_ice.json` and `proxy_H.json`.
Override with `--model`, `--ice-proxy`, `--h-proxy`; inspect `--help` for run knobs.

Run `--dry-run` first: it checks inputs and constructs hosts without GPU inference
or output creation. A real run imports the required model packages and invokes
the existing pwidom evaluator (including its batched/single self-check).
Every run requires a fresh `--out` or creates a timestamped directory: existing
results are never overwritten or silently reused. `manifest.json` records the
commands, source paths, resolved model, Git revision, allocation, PIDs and exits;
each host has a log and the usual pwidom frames/result files. Status is read-only.
The launcher defaults to a compact bordered table with per-host progress bars,
stage elapsed/remaining ETA, GPU utilization and used/total VRAM. It uses colour
and refreshes in place on interactive terminals; `--plain` selects ASCII output
without colour/screen clearing. Redirected output has no ANSI formatting.
`--verbose` additionally streams full emitted host output with prefixes, while
the complete files are always preserved. `--quiet` suppresses raw streaming. The
status script's `--follow` streams logs from an already running scan and prints
the result summary when the launcher finishes. Ctrl-C stops that viewer only.
The 30-second heartbeat also reports utilization and used/total memory per GPU
from nvidia-smi. Optional progress sidecars record equilibration/sampling steps,
proxy preparation and batched single-point counts, percentages and measured
**stage** ETA. The first positive timing sample can include compilation and is
excluded from ETA prediction; two positive samples are required before estimating
from recent increments. Initial model loading/first-call compilation has unknown duration;
ETA says estimating until timed work is available. ETA is not a prediction for
all remaining stages. Progress is emitted at roughly 10-second intervals and
stage boundaries, with no extra force evaluations or changes to the sampler.
Time fields explicitly show `elapsed HH:MM:SS / remaining HH:MM:SS` for the
current stage; the launcher separately counts down time to the wall cap.
For older runs without sidecars, status estimates current-MD time remaining
from successive frame/time log lines; unmeasured later stages remain unknown.
Missing telemetry is reported without failing the calculation. Already running
processes keep their original output behavior until the next run.
For sampling extensions, the table can show the previous completed run's transfer
free energy, explicitly labeled `prior`. A `new` value is shown only after the
host and pure-H insertion/deletion estimates are complete; MD alone cannot give
a new Widom free energy. This display adds no model calls or estimator changes.
`--gpu` runs nvidia-smi on the machine where status is invoked. On the login node,
use an overlapping allocation step for GPU monitoring rather than querying login GPUs.

The deadline, Ctrl-C, or SIGTERM stops and reaps child process groups. Completed
results remain; interrupted MD is not checkpointed. Full result completion is
reported separately from scientific convergence. The status summary uses the
existing scan's robust pressure correction and also reports measured-P dg,
Widom/deletion/BAR, Kish, melting and sampled pure-H close contacts.
3000 K is outside the supplied proxy-fit label temperatures: treat this as an
extrapolation/sampling test. Missing physical inputs cause an error; no EOS,
bulk modulus, model or proxy is invented. No GRACE result is verified by local
toy checks. For CPU workflow tests, `--cpu --model toyice` is supported.

No previous TDI at the requested state? Supply `--state-json PATH` containing
`P`, `T` and three objects: `ice`, `h225`, `H`. Each object needs measured/fitted
`vpa` (A^3/atom), `B_GPa` (isothermal bulk modulus) and a `source` description.
The two crystal objects need `nrep` (normally 3: 108 O); H needs `natoms`
(normally 250). Both crystals must use the same nrep. This avoids creating fake
stage0 files. P/T must match the command. The intermediate host retains the
existing scan convention: per-O volume interpolation and the H2O bulk modulus.
The full supplied EOS metadata is copied into the run manifest. Do not reuse an
EOS from another temperature or invent bulk moduli. Existing DFT lookup functions
are `17_eos_md.dft_vpa` and `h_vpa`; `tdi/run_real.bulk_modulus` derives B from
the same EOS slope. Their data coverage/phase filters must be reviewed first.
To inspect the inventory without launching MD, using those existing fits and
requiring exact-temperature-bin coverage (no temperature extrapolation), run:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --inspect-eos runs/inventory_binned.csv
```

This prints volume/B estimates at P and P +/- 3%, or reports insufficient
coverage. It writes no files and needs no GPU. Check that the inventory path
exists on Perlmutter; the cloud checkout does not include it.

The user's Perlmutter report on 2026-10-31 confirmed the interpreter at
`$SCRATCH/grace-env/bin/python` and sufficient exact-3000K EOS coverage for all
three hosts. Those printed values are recorded with provenance in
`envs/pwidom_500GPa_3000K.json` (108-O crystals, 250-atom pure-H reference):

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --state-json envs/pwidom_500GPa_3000K.json --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh --state-json envs/pwidom_500GPa_3000K.json
```

These use the existing environment discovery; manual activation is unnecessary
for this standard environment. `export GRACE_ENV="$SCRATCH/grace-env"` pins it
explicitly for both wrappers if desired. The saved EOS values are rounded to the
precision of the supplied terminal report. This is a DFT-volume starting point:
check actual GRACE pressures and phase stability in the run results.

### New state points, additional sampling, and diagnostics

Prepare a new state from the inventory without copying EOS numbers by hand:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --P 400 --T 3000 \
  --inspect-eos runs/inventory_binned.csv --save-eos runs/eos_400GPa_3000K.json
bash mlp_pipeline/perlmutter_pwidom.sh --P 400 --T 3000 \
  --state-json runs/eos_400GPa_3000K.json --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh --P 400 --T 3000 \
  --state-json runs/eos_400GPa_3000K.json --candidate-dir runs/dft_candidates
```

Saving requires complete exact-bin coverage and a fresh filename. No GPU work
occurs until the last command. The default run cap remains ten minutes.

Repeating a launch starts fresh; it does not extend an earlier scan. For a
**completed** run, append more sampling in a fresh output directory:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --extend-run runs/<completed-run> \
  --ps 0.4 --equil-ps 0.2 --minutes 10 --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh --extend-run runs/<completed-run> \
  --ps 0.4 --equil-ps 0.2 --minutes 10 --candidate-dir runs/dft_candidates
```

This inherits the parent state, model, proxies and insertion settings; `--ps`
is **additional** production time per ensemble. Model fingerprint, T, species
and volume must match before any child starts. Existing caches contain no
velocities/integrator state, so the added segment starts at the last saved
coordinates, draws fresh seeded thermal velocities, re-equilibrates, then
combines old and new production frames. It is not exact trajectory continuation.
The original stays byte-for-byte unchanged. Insertion single points are
recomputed over the combined sample, so an extension may need more walltime than
the initial scan. Do not launch overlapping extensions of one parent. Errors
remain short-run statistical estimates; extension does not guarantee convergence.

Compare the pair curves (CPU only) and prepare an **unexecuted** baseline replay:

```bash
bash mlp_pipeline/perlmutter_proxy_compare.sh \
  --run-dir runs/<completed-run> --out runs/<fresh-comparison-directory>
```

Outputs: `proxy_pairs.png` (energy and force, including the short-range core),
`pair_curves.csv`, input hashes and `replay_old.sh`. The old proxy is
`docs/data/toyfit_t4.json`; fitted ones are `runs/pwidom_fit/proxy_{ice,H}.json`.
No model work is launched. Review the replay script/cost before executing it:
it reuses cached, fingerprinted ensembles and changes only proxy sampling, not MD.
The existing probe-fit Kish diagnostics describe starting-proxy-selected sites;
they are not the production Kish of new-proxy draws.

With `--candidate-dir`, future scans can save a bounded unlabeled extxyz queue
(default ten trial pairs per host; `--candidate-limit` changes this). Selection
includes dominant estimator weights, large centered proxy residuals, and
representative trials. Each request links host-without-H and host-with-H
geometries plus model/fingerprint, state, source frame, predicted insertion
energy, proxy value, log weight and reason. A lock serializes writers and exact
geometries are deduplicated. These are **sampling/proxy diagnostics**, not an
uncertainty estimate or proof of GRACE error. No files are labeled as DFT and no
VASP job/training is started. Include representative equilibrium configurations
alongside unusual sites; do not automatically train on every close-contact probe.

Older archives lack exact trial positions/weights. Their representative MD
snapshots can still be queued without model calls:

```bash
"$SCRATCH/grace-env/bin/python" mlp_pipeline/tdi/pwidom_candidates.py \
  runs/<completed-run>/*_frames.npz --out runs/dft_candidates --per-source 3
```

Existing `tdi/dft_check.py prep-widom` prepares host/probe VASP checks from cached
frames and a training DFT template. It samples new sites rather than recovering
the old estimator's exact influential sites. Before labeling queue geometries,
audit VASP version, PAW datasets, XC, cutoff, k-mesh, electronic smearing and
convergence against the actual training OUTCAR/template. The user's module list
includes VASP 5.4.4 CPU and VASP 6 GPU builds; no Perlmutter VASP launcher is
configured by this change. POTCAR and raw training frames stay outside Git.

The small reported 500-GPa/3000-K table is committed separately from raw data.
Reproduce its provisional formation plot with:

```bash
"$SCRATCH/grace-env/bin/python" mlp_pipeline/tdi/pwidom_formation.py \
  --out runs/<fresh-figure-prefix>
```

Lower error bars use the maximal-positive-correlation bound on propagated
statistical sigma because shared-reference covariance is unavailable. They
exclude sampling and model bias. This figure is not a validated equilibrium
convex hull or a fitted saturation composition.

### Quick four-GPU throughput benchmark

Finish other GPU calculations first, then run inside the allocation:

```bash
bash mlp_pipeline/perlmutter_bench.sh --dry-run
bash mlp_pipeline/perlmutter_bench.sh --minutes 10
```

The default `--workflow pwidom` measures direct NVT MD processes/GPU and the
actual pwidom energy evaluator's structures/batch separately. It validates a
best observed MD layout and a best observed energy-batch layout across four GPUs;
MD steps/s and structures/s are never ranked against each other. This is the
benchmark to use for composition scans or independent replicas of one composition.

`--workflow tdi` reuses `tdi/bench.py` and the real lambda engine on the 325-atom H2O+H
insertion system (host and full energies per lambda step). It screens 1/2/4/8
independent processes per GPU on four GPUs concurrently, then 2/4/8/16 batched
lambda replicas in one process/GPU. Finally it repeats the best observed layout
across **all four GPUs** to measure aggregate throughput under node contention.
Screening layouts use different GPUs and share CPU resources, so ranking is
preliminary until the whole-node validation finishes. `--system H` or `insH`
checks the hydrogen reference separately; `--system h2o` checks one-call MD.
Volumes and P/T come from `--state-json` (default the saved 500-GPa/3000-K file).

Default: five warm-up and 30 timed steps/calls, two TensorFlow CPU threads
per process, neighbor subprocess pool disabled, 180-second limit per phase and
ten-minute global cap (including loading/compilation). All processes warm up
before synchronized timing. Aggregate rates use total completed lambda steps
divided by the shared timing interval, not a sum of incomparable individual
rates. Batched GRACE verifies energies/forces/stress against the single calculator
before timing. Failed/OOM/self-check/timeout cases are excluded from ranking;
no whole-node result is claimed unless all four validation cases finish.

GPU memory/utilization is sampled during the run; sampled peak is not guaranteed
to capture the instantaneous allocation peak. GraceBatch may split OOM batches;
the raw result records its effective max-atoms setting. Larger batches are not
automatically better. The benchmark changes no MPS settings and refuses an active
MPS environment or GPUs with more than 512 MiB already allocated. It reports CPU
affinity/thread settings; CPU preparation can limit throughput despite free VRAM.
Before packed configurations, a one-worker/GPU calibration records sampled
device memory and TensorFlow allocator peak. Admission projects independent
context or batched disconnected-graph memory with a default 1.5 safety factor,
and reserves at least 20% of VRAM (or --reserve-gib, whichever is larger).
Layouts that do not fit, or lack calibration/telemetry, are skipped before
workers start. This reduces OOM risk; it cannot guarantee an unobserved peak
never exceeds a projection. No intentional run-to-OOM sweep is performed.

Outputs are timestamped `runs/bench_*/` logs, `manifest.json`, `results.jsonl` and
`summary.json`. `RUN_ORIGIN.txt` identifies Perlmutter, host and Slurm allocation
in new benchmark and scan directories. The scientific inputs/outputs stay in
`runs/` for existing analysis; nothing is moved into `perlmutter-runs/`.
The benchmark writes no production trajectories. CPU toy mode exists only for
testing the launcher and is never a GPU-performance measurement.

For an eventual 1–50 extra-H scan in a 108-O cell, check H2.5O EOS coverage as
well: 50 H means H2.463O, beyond the current H2.25O anchor at 27 H. Use
`--inspect-eos runs/inventory_binned.csv --include-h25` before planning volumes.
No volume extrapolation or 50-composition production job is started by this
benchmark. One shared pure-H reference can serve multiple compositions at the
same P/T/model, with shared-reference covariance retained in later analysis.
Independent replicas at one composition are another way to fill the node once
safe processes/GPU are measured. Use distinct seeds and equilibrated starts.
Future pwidom results include a small `_work.npz` containing frame-grouped
forward/reverse work arrays and provenance for later pooling. No replica-pooling
estimator is introduced here: do not average free energies/Kish fractions or
double-count frames present in both a parent and its extension.
