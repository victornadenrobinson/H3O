# PWidom checkpoint and resume

PWidom now checkpoints MD by default every **300 wall-clock seconds**. Change
this with `--checkpoint-seconds 60` on `pwidom_auto.sh`,
`perlmutter_pwidom.sh`, or `tdi/pwidom.py run`. The published CPU environment
already contains ASE and the analysis dependencies; new shells need activation,
not reinstallation. These new source changes require a separate update from the
previously published version. This change does not publish an environment.

## What is saved

For each host/tag:

- `<tag>_host_md.npz` and `<tag>_full_md.npz`: current positions, momenta, cell,
  species/masses, MD step, PCG64 random state, sampled positions and pressures,
  initial positions, model fingerprint and integrator/software settings.
- `<tag>_frames.npz`: the existing combined host/N+1 cache, after both ensembles
  complete. Its writes are now atomic too.
- `<tag>_<label>_K<K>_energies.npz`: completed single-point energies, saved after
  each batch, with a hash of the model and ordered structures.

Writes use temporary files in the destination directory, flush/fsync, and atomic
replacement. An interrupted write leaves the previous complete checkpoint.
MD checkpoints are also written at startup, equilibration completion, ensemble
completion, and on TERM/INT/HUP once the current integration step finishes.
Partial sampled trajectories are in the checkpoint `frames` arrays: this is not
an ASE `.traj` file. The existing RDF tool expects the final combined frame
cache; `--inspect` can inspect checkpoint arrays.

MD continues from saved momenta/random state without rerunning completed
integration steps or equilibration. Tested CPU deterministic trajectories match
uninterrupted runs exactly. This is supported for unconstrained ASE Langevin,
with the same ASE/NumPy versions, settings, species/cell/initial configuration
and model fingerprint. Existing Langevin friction and center-of-mass behavior
are preserved. GPU floating-point nondeterminism means bitwise identity is not
promised for GRACE; GPU continuation still requires Perlmutter validation.

## Resume or restart in the same directory

After verifying the previous processes have stopped, this explicit mode keeps
all files under the original run directory:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --resume-run runs/OLD_RUN --in-place \
  --minutes 50 --checkpoint-seconds 300 --dry-run
# Then repeat without --dry-run.
```

`pwidom_auto.sh` accepts the same `--resume-run ... --in-place` form. Logs append
with a timestamped boundary. Previous manifests, result JSONs and work arrays
are copied to `_resume_history/<timestamp>/` before restarting. Those copies are
historical records whose paths may still refer to the live directory, not
independent relocated runs. Checkpoints update by atomic replacement; sampled
frames resume from the saved step without duplicates. Completed combined frame
caches are reused unchanged. Result/work files are recomputed after their prior
versions have been preserved. Unknown files are left untouched. This is not
blindly concatenating coordinate or NPZ files, which would corrupt them.

A directory lock protects new launcher runs and in-place operations from each
other. Older running launchers did not hold that lock: verify their processes
have stopped before using this mode. As with fresh-output resume, hosts without
any saved state restart from their original initial conditions, with an explicit
message. The former logs remain. No production directory is modified by these
development tests.

## Resume safely

First ensure the previous allocation/processes have actually stopped (`squeue`
and the status viewer). A manifest can still say `running` after SIGKILL.
The default resume writes **into a fresh directory**; the explicit in-place mode above is also supported:

```bash
# In a NEW four-GPU Perlmutter allocation, with the existing GRACE environment:
cd /pscratch/sd/v/victornr/H3O
bash mlp_pipeline/perlmutter_pwidom.sh \
  --resume-run runs/OLD_RUN \
  --out runs/OLD_RUN_resumed_1 \
  --minutes 50 --checkpoint-seconds 300 --dry-run

# After inspecting that plan, same command without --dry-run:
bash mlp_pipeline/perlmutter_pwidom.sh \
  --resume-run runs/OLD_RUN \
  --out runs/OLD_RUN_resumed_1 \
  --minutes 50 --checkpoint-seconds 300
```

`OLD_RUN` is the directory holding that stage's `manifest.json`, not a frame
file. The automatic entry point also accepts the same explicit single-stage
operation:

```bash
bash mlp_pipeline/pwidom_auto.sh --resume-run runs/OLD_RUN \
  --out runs/OLD_RUN_resumed_1 --minutes 50 --checkpoint-seconds 300
```

Resume inherits P/T, model, proxy, sampling duration, equilibration and candidate
counts from the saved run. Omit `--state-json`, `--roots`, `--extra-h`,
`--reference-run` and `--extend-run`. It keeps the four-GPU requirement; it does
not serialize four jobs onto one GPU. `--minutes` is a NEW launcher cap, not the
Slurm allocation's remaining time. Request enough allocation time and leave a
margin; a nominal 30-minute launcher cap cannot extend an expiring allocation.
For example, an allocation request for this four-host launcher is:

```bash
salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus 4 -A m3349_g
```

Follow site queue limits. `--dry-run` still validates model/proxy inputs and the
four-GPU allocation for real GRACE plans.

When an ensemble has not started, it starts from its original initial state.
Completed MD caches are reused and the proxy proposals are deterministically
rebuilt; completed single-point batches are reused after checking their hash.
Proxy preparation itself is not checkpointed. Results are recomputed rather
than blindly trusting potentially incomplete result JSONs. To resume again,
use the most recent resumed directory as `--resume-run`.

For an interrupted extra-H stage, resume that stage's manifest; its saved pure-H
reference is inherited. If only the reference stage was resumed, start the extra-H
stage explicitly with `--reference-run` pointing to the newly completed resumed
reference. The automatic P/T sweep does not discover resumed directory names or
rewrite its canonical paths; it still refuses an incomplete canonical run.

Direct single-host operation uses a prefix instead of a manifest:

```bash
python mlp_pipeline/tdi/pwidom.py run [same original physics/settings] \
  --resume-from runs/OLD_RUN/pw_TAG --out runs/FRESH_RUN --tag pw_TAG
```

Use the exact original `--extend-from` too when directly resuming an extension.
The manifest-based launcher inherits it for you. `--extend-run` remains a
different operation: it adds sampling to a completed run, resets velocities and
re-equilibrates the new segment; `--resume-run` finishes the original target.

## Limits and legacy runs

The example allocation was revoked while host MD was in progress. Older PWidom
versions saved the combined trajectory only after both MD ensembles completed;
the logs/progress JSON contain no recoverable positions or velocities. If no
checkpoint or final frame cache exists, that host must start again. The resume
launcher prints this explicitly and starts it in the fresh destination; it never
claims those lost steps were recovered.

A hard kill can lose up to the checkpoint interval plus the time needed to finish
one integration step/write. Signals cannot interrupt an in-flight GPU calculation
safely. The parent launcher allows five seconds for child shutdown before killing
survivors, so a long GPU step may leave only the previous periodic checkpoint.
Storage must support atomic same-directory rename and durable writes; fsync does
not guarantee survival of every distributed-filesystem failure. Filesystem costs
scale with sampled trajectory size; checkpoint time is wall time, not MD time.

Fresh-output resume preserves the source; in-place resume preserves historical results and updates live checkpoints. Do not treat overlapping source/resume frames
as independent datasets in convergence analysis. No checkpoints or production
outputs were created on Perlmutter during development. This implementation covers
the PWidom workflow, not other TDI/EOS MD scripts in the repository.

## Tests

`tests/test_pwidom_checkpoint.py` checks interruption during equilibration,
sampling and final step, real TERM handling, exact deterministic continuation,
metadata mismatch refusal, failed-write preservation and batch-energy reuse.
`test_resume_run_from_md_checkpoints_preserves_parent` checks the four-host shell
launcher, checkpoint reuse, final frames/BAR and preservation of source bytes.
All MD in these tests is tiny synthetic CPU MD, not production or GRACE MD.

Validation command:

```bash
python -m pytest -q tests/test_pwidom_checkpoint.py tests/test_pwidom_audit.py \
  tests/test_pwidom_diagnostics.py tests/test_perlmutter_pwidom.py
```

Result: **46 passed** on the restored CPU environment. ASE emitted deprecation
warnings for the pre-existing Maxwell/Langevin center-of-mass API; the integrator
convention was intentionally kept unchanged. Shell syntax, Python compilation
and whitespace checks also passed. No GRACE/GPU or Slurm runtime tests ran.
