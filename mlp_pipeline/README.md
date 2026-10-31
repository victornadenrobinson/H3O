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
The launcher streams full emitted host output with `[H2O]` / `[pureH]` prefixes,
while preserving the full files. `--quiet` restores summary-only output. The
status script's `--follow` streams logs from an already running scan and prints
the result summary when the launcher finishes. Ctrl-C stops that viewer only.
The 30-second heartbeat also reports utilization and used/total memory per GPU
from nvidia-smi. Optional progress sidecars record equilibration/sampling steps,
proxy preparation and batched single-point counts, percentages and measured
**stage** ETA. Initial model loading/first-call compilation has unknown duration;
ETA says estimating until timed work is available. ETA is not a prediction for
all remaining stages. Progress is emitted at roughly 10-second intervals and
stage boundaries, with no extra force evaluations or changes to the sampler.
Time fields explicitly show `elapsed HH:MM:SS / remaining HH:MM:SS` for the
current stage; the launcher separately counts down time to the wall cap.
For older runs without sidecars, status estimates current-MD time remaining
from successive frame/time log lines; unmeasured later stages remain unknown.
Missing telemetry is reported without failing the calculation. Already running
processes keep their original output behavior until the next run.
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
