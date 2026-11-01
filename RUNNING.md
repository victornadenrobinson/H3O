# Terminal quick reference

Run from the repository root. `runs/` holds inputs and outputs; each new
Perlmutter run gets `RUN_ORIGIN.txt`. Use fresh output names unless explicitly
resuming. Check the source computer: Young paths and Perlmutter paths differ.
Long calculations belong inside an allocation, not on login nodes. Examples
below are commands to choose/run deliberately, not a sequence to execute all at once.

## Environment

**Perlmutter** (`victornr`, GPU node):

```bash
cd /pscratch/sd/v/victornr/H3O
export GRACE_ENV="$SCRATCH/grace-env"
export PY="$GRACE_ENV/bin/python"
export MODEL=grace-ft:runs/grace/ft_m2
# Optional interactive activation, if this is a standard venv:
source "$GRACE_ENV/bin/activate"
```

The `perlmutter_*.sh` entry points discover/check the environment automatically;
manual activation is unnecessary for them. `PY` overrides discovery. An existing
site module/conda recipe can be supplied with `H3O_ENV_INIT=/path/to/setup.sh`.
`nvidia-smi` and `squeue -j "$SLURM_JOB_ID" -h -o '%i %L'` check GPUs/time left.

**Young GPU** (`mmm0037`, GPU allocation):

```bash
cd /lustre/scratch/mmm0037/H3O
source "$HOME/grace-venv/bin/activate"
export PY="$HOME/grace-venv/bin/python"
export MODEL=grace-ft:runs/grace/ft_m2
```

**Young CPU** (SGE/RHEL7): `source mlp_pipeline/young_cpu_env.sh`, then
`export PY="$(command -v python)"`. The GPU-side venv is not compatible there.
See [environment details](envs/README.md). MACE/MatterSim need separate environments.

## Widom/BAR scans — no lambda path

One host and one host+H MD trajectory per composition, then proxy-selected
insertion/deletion energies. `--minutes` limits real time; `--ps` is MD sampling.
`--ice-proxy PATH` chooses the proposal pair potential for H/O hosts;
`--h-proxy PATH` chooses it for a newly sampled pure-H host. The default fitted
JSONs are under `runs/pwidom_fit/`; the old DFT fit is `docs/data/toyfit_t4.json`.
These guide probe selection and importance weights, not GRACE MD. Correctly
reweighted, converged estimates target the same GRACE free energy; finite
sampling/overlap can cause substantial differences. The direct `pwidom.py run`
command calls this option `--proxy`. This is distinct from the pure-H
thermodynamic reference selected by `--reference-run`.

**Fresh 500 GPa / 3000 K test**, four hosts on four GPUs:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --state-json envs/pwidom_500GPa_3000K.json --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh \
  --state-json envs/pwidom_500GPa_3000K.json --minutes 10 \
  --out runs/pwidom_500GPa_3000K_test2 --candidate-dir runs/dft_candidates
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/pwidom_500GPa_3000K_test2
```

**New P/T**, e.g. 400 GPa / 3000 K: derive valid EOS inputs first, then use
`--P 400 --T 3000 --state-json runs/eos_400GPa_3000K.json` in the launcher:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --P 400 --T 3000 \
  --inspect-eos runs/inventory_binned.csv --save-eos runs/eos_400GPa_3000K.json
```

No fit coverage means no configuration saved. Pure-H coverage at 3000 K starts
near 380 GPa, so this lookup cannot currently supply a 300-GPa reference.
Add `--include-h25` to inspect the H2.5O anchor before proposing 1–50 extra-H scans.

**Four intermediate compositions**, near n = 0.05, 0.10, 0.15, 0.20 at 500 GPa /
3000 K: 5, 11, 16, 22 extra H in the 108-O cell. One worker per GPU, reusing
extension1's completed pure-H result (shared reference uncertainty):

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --state-json envs/pwidom_500GPa_3000K.json \
  --extra-h 5 11 16 22 --reference-run runs/pwidom_500GPa_3000K_extension1 \
  --ps 1.0 --equil-ps 0.2 --minutes 30 \
  --out runs/pwidom_500GPa_3000K_nscan1 --candidate-dir runs/dft_candidates --dry-run
```

Review, then remove `--dry-run` to execute inside an allocation with sufficient
time. This uses the existing per-O volume interpolation and ice bulk modulus,
and refuses counts outside the H2.25O anchor. It changes no estimator settings.
After completion, compare old/new proxies on identical cached ensembles:

```bash
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/pwidom_500GPa_3000K_nscan1
bash mlp_pipeline/perlmutter_proxy_compare.sh \
  --run-dir runs/pwidom_500GPa_3000K_nscan1 --out runs/proxy_compare_nscan1
bash runs/proxy_compare_nscan1/replay_old.sh
rg 'proxy quality:' runs/pwidom_500GPa_3000K_nscan1/*.log
```

The replay uses the original DFT-fit pair proxy `docs/data/toyfit_t4.json` and the
same K/selection/deletion counts and seed. It performs GPU single points only;
allow up to ten extra minutes. MD remains GRACE in both cases. Compare each
composition with itself; the four jobs share the existing H reference.

**Additional sampling** from the known completed 500-GPa run:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --extend-run runs/pwidom_pm_20261031_105811 \
  --ps 0.4 --equil-ps 0.2 --minutes 10 \
  --out runs/pwidom_500GPa_3000K_extension_quick2 --candidate-dir runs/dft_candidates
```

`--ps` is additional production time. This retains old samples, starts from
their endpoint positions, assigns fresh thermal velocities, re-equilibrates and
recomputes estimates on the combined frames. It is not exact trajectory
continuation. A 3-ps + 0.5-ps extension needs roughly an hour allocation/cap,
not ten minutes; inspect `--dry-run` before choosing its runtime.

To extend **in the same directory**, add `--in-place` and omit `--out`:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --extend-run runs/pwidom_500GPa_3000K_nscan1 --in-place \
  --ps 1.0 --equil-ps 0.2 --minutes 30
```

The directory must contain a completed run. The launcher holds an exclusive lock,
copies the completed outputs into `_history/<timestamp>_<id>/` before replacing
any managed file, then writes the combined old+new results at the usual paths.
Historical manifest/JSON paths are relocated to their archived copies. Previous
old-proxy replays are archived too: replay them again on the new combined frames.
This consumes backup disk space. A failed extension remains marked incomplete;
the last completed parent can be recovered with `--extend-run <parent_archive>`
and a fresh output directory. Do not delete `_history` while its provenance/caches
are referenced. `--dry-run` makes no archive or changes.

Without `--in-place`, extensions retain the existing fresh-directory behavior.
Do not combine parent and extension as independent data: the extension already
includes the parent's frames. New PWidom runs save atomic MD checkpoints every
300 wall seconds by default (`--checkpoint-seconds`), plus completed single-point
energy batches. To finish an interrupted run in the same directory:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh \
  --resume-run runs/YOUR_INTERRUPTED_RUN --in-place \
  --minutes 50 --checkpoint-seconds 300
```

Verify the old processes have stopped first. Logs append; previous manifests,
results and work arrays are preserved under `_resume_history/`. Saved MD resumes
without repeated equilibration or duplicate frames. A legacy host with no saved
checkpoint/cache starts again from its initial state, retaining its old logs.
Omit `--in-place` and give a fresh `--out` to preserve the whole source directory.
`--resume-run` finishes the original target; `--extend-run` adds new sampling and
re-equilibrates. See [checkpoint/resume details](docs/PWIDOM_RESUME.md).

Compact progress/GPU tables are default; `--verbose` streams full logs and
`--plain` uses plain ASCII. Full logs are always in the run directory. For a log
stream: `tail -n 20 -F runs/<scan>/*.log`. Prior free energies are labeled prior;
new free energies require completed insertion/deletion estimates.

New launcher runs also save `telemetry.jsonl` every five seconds: GPU memory,
capacity, utilization, stage/progress and main-process CPU RSS. Plot a completed
or running trace (use a fresh output prefix):

```bash
"$PY" mlp_pipeline/tdi/pwidom_telemetry.py runs/pwidom_500GPa_3000K_nscan1 \
  --out runs/nscan1_resources
```

Older runs need a saved compact terminal transcript instead of the run path.
Only recorded intervals can be plotted. Stage shading has sampling uncertainty;
instantaneous utilization can miss short GPU bursts. In-place extensions archive
the old telemetry and start a new trace. Capacity comes from the device: one
reported scan had 40-GiB GPUs, while the earlier benchmarks had 80 GiB.

Retrospective running chemical potentials, transfer costs and insertion counts
use the completed run's `_work.npz` and frame archives, with no GPU calls:

```bash
"$PY" mlp_pipeline/tdi/pwidom_convergence.py runs/pwidom_500GPa_3000K_nscan1 \
  --out runs/nscan1_convergence
```

These plot cumulative **production MD time**, not wall time; both host and host+H
prefixes grow while the completed pure-H reference is held fixed. Curves are
correlated diagnostics, not independent estimates or confidence bands. Widom
does not accept/reject MD insertions: counts show evaluated insertion trials and
Kish effective weight counts, with candidate-site totals in the CSV. The latter
are not independent ESS. Older results without `_work.npz` cannot reconstruct
running chemical potentials from final summaries alone.

### μ/dG summaries and time-coloured hulls

Use the same GRACE environment (`$PY`) selected in **Environment** above; do not
use an unrelated system `python3`. The new analysis entry point reads the saved
JSON/work/pressure files directly, with no inline Python or intermediate export.
Collect all finished results, errors and exploratory formation hulls:

```bash
"$PY" mlp_pipeline/pwidom_analysis.py collect \
  --root runs --out runs/plots/pwidom_summary_1 \
  --pressure-mode robust --include-flagged
```

At one P/T, plot μH(t), the hydrogen-reservoir G_H(t), transfer dG(t), and hulls
every **0.1 ps coloured by time**:

```bash
"$PY" mlp_pipeline/pwidom_analysis.py time \
  --runs runs/pwidom_300GPa_3000K_reference runs/pwidom_300GPa_3000K_extra \
  --step-ps 0.1 --reference-mode cumulative --block-ps 0.1 \
  --out runs/plots/pwidom_300GPa_3000K_time_1
```

Alternatively, `bash mlp_pipeline/perlmutter_pwidom_analysis.sh ...` accepts the
same `collect`/`time` arguments and uses the existing `PY`/`GRACE_ENV` interpreter
discovery with dependency checks. Both forms are CPU-only; no GPU allocation,
model calls or MD is required. Use a fresh output directory for every report.

The time command writes `mu_G_dg_vs_time.png/.pdf`,
`drift_components_vs_time.png/.pdf` (changes in excess BAR and pressure terms),
`hulls_coloured_by_time.png/.pdf`, `time_series.csv`, `time_hulls.csv` and
`prefix_input.json`. The 300-GPa/3000-K results record work-file paths and 50
selected frames per ensemble: the 20-fs sampling cadence supports ten snapshots
from 0.1 to 1.0 ps if the saved files pass the reconstruction checks.

To redraw an existing time report with the current plot layout, reuse its
`prefix_input.json` instead of reconstructing BAR from work arrays again:

```bash
"$PY" mlp_pipeline/pwidom_analysis.py replot-time \
  --input runs/plots/pwidom_300GPa_3000K_time_1/prefix_input.json \
  --out runs/plots/pwidom_300GPa_3000K_time_2
```

The main hull figure separates formation nodes and convex envelopes on the same
zoomed composition axis. `hull_full_composition.png/.pdf` shows the pure-H
endpoint separately. Chemical-potential and transfer panels share composition
colours and an outside legend. Replotting leaves the numerical estimates intact.

Light shading on μH, G_H and transfer dG is **±1 contiguous-block jackknife
sampling standard error**, computed from the saved works and pressure cache.
`--block-ps 0.1` deletes all trials in each contiguous 0.1-ps MD block; host
pressure is recomputed with the same block omitted. Host/N+1 ensembles are
handled independently. At least three complete blocks are required (change with
`--min-blocks`, minimum 3), so early prefixes have no band. Band boundaries are
connected between complete-block sampling cutoffs. No trial-level independence
or stationarity of these short runs is established.

For transfer dG, the host and reservoir sampling variances are added because the
MD trajectories are independent. The same reservoir remains shared across host
compositions: their dG curves are correlated. The bands exclude unvisited work
tails, model bias, nonstationarity, and partial-volume-parameter uncertainty.
Vary `--block-ps` (for example 0.05, 0.1, 0.2) to assess block-size sensitivity;
short runs may not have enough blocks for the larger choice.

Old `prefix_input.json` files contain no sampling errors. Rerun the `time`
command once to compute them; `replot-time` can restyle old curves but cannot
recover uncertainty from their final μ values alone. Missing errors stay blank,
not zero. Estimator disagreements and low Kish remain reasons for caution even
when a shaded band is narrow.

`cumulative` grows the reservoir prefix at the same sampling cutoff as every
host; `fixed` uses its final value, matching the older convergence plots above.
Times count production sampling per independent ensemble, excluding equilibration
and extension re-equilibration gaps. These are correlated cumulative prefixes,
not independent 0.1-ps blocks or wall time. Missing exact times are skipped as
whole snapshots, and final JSON values cannot replace missing work arrays.
Choose one baseline or extension, not overlapping copies of both.

Formation ΔG is reconstructed by composition integration of transfer dG. The
coloured hull is provisional, not an independent absolute crystal G calculation
or equilibrium solubility measurement. Static `--include-flagged` allows visibly
flagged sampling issues; melted/unverified inputs stay excluded. Time hulls also
refuse melted inputs. See [results conventions](docs/PWIDOM_RESULTS.md) and
[time-plot details](docs/PWIDOM_TIME_HULLS.md).

## Perlmutter VASP: first single-point test

Use the existing GRACE `PY`/`GRACE_ENV` setup above. The local template supplied
for these tests is `vasp_inputs/` (INCAR/KPOINTS/POSCAR/POTCAR); it is ignored by
Git. In a CPU allocation, start with the reported `vasp/5.4.4-cpu` module:

```bash
export VASP_MODULE=vasp/5.4.4-cpu
bash mlp_pipeline/perlmutter_vasp_check.sh preflight --template vasp_inputs
bash mlp_pipeline/perlmutter_vasp_check.sh snapshot \
  --frames runs/pwidom_300GPa_3000K_reference/pw_H2O_300GPa_3000K_frames.npz \
  --template vasp_inputs --out runs/vasp_one_snapshot_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_one_snapshot_1 \
  --ranks 32 --max-points 1 --minutes 20
bash mlp_pipeline/perlmutter_vasp_check.sh status runs/vasp_one_snapshot_1
```

For an actual saved production insertion/deletion pair, use the existing
candidate queue and its recorded GRACE ΔU instead of regenerating a trial:

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

To use a 128-rank interactive allocation for **four independent 32-rank VASP
points**, prepare 20 saved pairs (40 structures) and pack the runner:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh prep-queue \
  --run runs/pwidom_300GPa_3000K_reference \
  --hosts H2O H2p083O H2p25O pureH --pairs 20 \
  --template vasp_inputs --out runs/vasp_saved_40_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_saved_40_1 \
  --ranks 32 --parallel 4 --minutes 25
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_saved_40_1 \
  --out runs/plots/vasp_saved_40_1
```

Each concurrent `srun` uses `--exclusive --exact`, one OpenMP thread per rank,
and its own directory/log/lock. Allocation rank count is checked before launch;
completed points are skipped and interrupted attempts remain available. The
analysis uses the queue's saved GRACE ΔU without model calls. At the measured
87.1 s per point, 40 structures would take roughly 15 minutes in four lanes,
not 10; shared-node throughput and SCF costs must be measured. Forty pairs means
80 structures and roughly twice that time. Check allocation time remaining.

Completed OUTCARs also supply forces and static stress without new VASP runs.
To compare against GRACE on the same prepared geometries (a cheap model
reevaluation, after CPU resources are available):

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_saved_40_1 \
  --forces-stress --device cpu --out runs/plots/vasp_saved_40_forces_1
```

This writes per-structure/species force errors, matched force/stress arrays,
scatter plots and unavailable-data reasons. ASE sign/order/units are used for
both models; stresses are configurational, with no ionic NkBT/V contribution.
The default energy comparison still uses the stored GRACE ΔU without model calls.

To isolate k-point error, copy one completed ice and one completed reservoir
pair into a separate explicit 2×2×2 mesh check:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh mesh-check runs/vasp_saved_40_1 \
  --hosts H2O pureH --pairs-per-host 1 --mesh 2 2 2 \
  --out runs/vasp_kmesh_2_1
bash mlp_pipeline/perlmutter_vasp_check.sh run runs/vasp_kmesh_2_1 \
  --exe vasp_std --ranks 32 --parallel 4 --minutes 20
bash mlp_pipeline/perlmutter_vasp_check.sh analyze runs/vasp_kmesh_2_1 \
  --out runs/plots/vasp_kmesh_2_1
```

There are four VASP structures (two pairs). Source inputs/results stay untouched;
the same geometry, PAW, cutoff, smearing, and parallel settings are retained.
The CSV `kmesh_shift_eV` reports the change in ΔU relative to the completed source
mesh, not a total energy per-atom comparison. `vasp_std` is required for 2×2×2.
Costs can be substantially higher than Γ; check allocation time remaining before
starting. This is a convergence control, not new MD or a DFT free-energy estimate.

The runner prints `PAIR ... ΔU_VASP ... ΔU_GRACE ... VASP−GRACE ...` as soon as
both geometries finish. A single total energy cannot yet supply that difference.
For a job started with an older runner, a separate terminal can watch the same
files without restarting the calculations:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh watch runs/vasp_saved_40_1
```

Ctrl-C stops only the viewer. Preflight prints the actual KPOINTS and a
conservative `gamma_only` check. Future runs may use `export VASP_EXE=vasp_gam`
when the input is strictly Γ-only with no SOC/noncollinear flags. The runner
rejects `vasp_gam` on incompatible inputs. `vasp_std` is also valid at Γ; let
active calculations finish. Every prepared point currently copies the template
KPOINTS, including pure H: convergence of Γ-only energies/paired ΔU with k-mesh
must be tested for each cell/material, especially metallic hydrogen.

A pair is two single-point geometries. Queue import filters to the selected
run/host tags and checks the saved geometry hashes. Analysis uses `dU_mlp_eV`
directly; no new GRACE evaluation is required. Weight-selected/extreme requests
remain labelled diagnostics, not representative DFT free-energy samples.

This prepares exactly one saved host configuration and runs one VASP SCF, with
no MD/model calls. Outputs and retries are retained in timestamped attempt dirs;
SCF convergence is required before accepting TOTEN. Run VASP inside an allocation,
never on a login node. See [VASP test instructions](docs/PWIDOM_VASP.md) for the
allocation example, settings audit, paired 40–100-structure check and later batch
submission. First establish a working executable and actual single-point timing.

## Other calculations

Use `$PY`/`$MODEL` from the source computer's setup above. Inspect `--help` before
choosing scientific parameters. Commands containing `<...>` need real paths/tags.

| Task | Terminal entry point |
| --- | --- |
| One GRACE energy/force check | `"$PY" mlp_pipeline/04_benchmark.py "$MODEL" valid_m.extxyz --n 1 --device cuda` |
| CPU toy calculation | `"$PY" mlp_pipeline/04_benchmark.py toyice valid_m.extxyz --n 1 --device cpu` (checks software only, not GRACE) |
| GPU packing benchmark | `bash mlp_pipeline/perlmutter_bench.sh --workflow pwidom --minutes 10` (after other GPU work finishes) |
| Plot reported benchmark runs | `"$PY" mlp_pipeline/tdi/plot_pwidom_benchmark.py docs/data/perlmutter_pwidom_benchmarks_20261031.csv --out runs/benchmark_comparison` (also accepts actual benchmark directories) |
| TDI lambda throughput | `bash mlp_pipeline/perlmutter_bench.sh --workflow tdi --minutes 10` |
| Benchmark against DFT frames | `"$PY" mlp_pipeline/04_benchmark.py "$MODEL" valid_m.extxyz --n 30 --device cuda` |
| Free NVT EOS/phase MD | `"$PY" mlp_pipeline/17_eos_md.py "$MODEL" --phase fcc --T 3000 --P 500 --from-dft inventory_binned.csv --ps 0.4 --equil-ps 0.2 --outdir runs/eos_test_new --tag fcc_test` |
| Flexible-cell NPT stability | `"$PY" mlp_pipeline/tdi/stability_npt.py --help` (choose P/T, size, length and fresh `-o`; `--report DIR` reads a completed run) |
| TDI/ghost-H insertion | `PY="$PY" MODEL="$MODEL" bash mlp_pipeline/young_tdi_par.sh insert` (review header defaults first; `fcc`, `bcc`, `H`, `liquid` are other modes) |
| TDI collection | `"$PY" mlp_pipeline/tdi/collect.py runs/<tdi-directory> <tag>` |
| TDI summary / mixing | `"$PY" mlp_pipeline/tdi/summarize.py` (explicit run roots and `-o fresh.csv` export tables) |
| Proxy fitting: inspect options | `"$PY" mlp_pipeline/tdi/proxy_fit.py label --help` and `fit --help` (cached frames → GRACE labels → CPU spline fit) |
| Old/new pair curves | `bash mlp_pipeline/perlmutter_proxy_compare.sh --run-dir runs/<completed-scan> --out runs/<fresh-comparison>` (CPU; replay script is not executed) |
| Reported 500-GPa formation plot | `"$PY" mlp_pipeline/tdi/pwidom_formation.py --out runs/<fresh-figure-prefix>` |
| Combined method/hull plot | `"$PY" mlp_pipeline/tdi/fig_methods.py -o runs/<fresh-figure>.png` (local scan indices must first be rescored with `pwidom_scan.py --rescore DIR`) |
| Queue representative old snapshots | `"$PY" mlp_pipeline/tdi/pwidom_candidates.py runs/<scan>/*_frames.npz --out runs/dft_candidates --per-source 3` |
| Prepare paired VASP probes | `"$PY" mlp_pipeline/tdi/dft_check.py prep-widom runs/<scan>/<tag>_frames.npz --template <training-DFT-directory> --proxy docs/data/toyfit_t4.json -o runs/<fresh-DFT-check>` |
| Audit DFT settings | `"$PY" mlp_pipeline/tdi/dft_check.py audit runs/<DFT-check> --template <training-DFT-directory>` |
| DFT job status / analysis | `"$PY" mlp_pipeline/tdi/dft_check.py status runs/<DFT-check>` / `analyze runs/<DFT-check>` |
| Data extraction/bundling | `bash mlp_pipeline/build_all.sh 4` (needs raw OUTCAR paths and mined inventory; updates downstream data) |
| CPU software checks | `"$PY" -m pytest tests -q` (pytest required); `"$PY" mlp_pipeline/tdi/solubility_check.py --selftest` |

The legacy TDI launcher defaults to one GPU; independent/batched lambdas are
different from pwidom's host jobs. Its identical command resumes checkpointed
lambda samples. Use a fresh `OUTDIR` for a new campaign and never set `FRESH`
unless deliberately discarding that tag's data. See [pipeline recipes](mlp_pipeline/README.md).

For GRACE training, Young's `grace_prep_data.sh`, `grace_make_input.py`, and
`grace_ft_run.sh` are the existing workflow; the training shell wrapper hardcodes
`~/grace-venv`, so do not blindly use it on Perlmutter. Review the prepared YAML
and run the active environment's `gracemaker input.yaml` in a fresh training
directory. See [project guidance](HANDOFF.md) before changing loss/training settings.

The available VASP version/PAW/XC/cutoff/k-mesh/smearing must match the labeling
plan. VASP 5.4.4 CPU and VASP 6 GPU modules are distinct choices; inspect a training
OUTCAR and template before preparing a submission. No automatic VASP submission
or training is performed by the candidate queue. Raw structures/POTCARs stay out
of Git. Source-specific summary snapshots: [docs/status](docs/status/README.md).

For 40 original DFT-labelled snapshots with live energy/force/stress comparisons,
see [training-label reproduction](docs/PWIDOM_VASP.md#reproduce-original-training-labels-two-cpu-nodes).
Use `training --count 40 --source-root .. --out runs/vasp_training_40_1`, then
`run runs/vasp_training_40_1 --exe vasp_gam --ranks 32 --parallel 8 --minutes 50`
through `bash mlp_pipeline/perlmutter_vasp_check.sh`. Two nodes need
`--ntasks-per-node=128` (256 tasks total); `-n 128` only supports four such points.

The complete four-part sequence (original-label baseline, ordinary MD, paired
edge-case replay, and a split retraining dataset) is in
[PWidom DFT validation](docs/PWIDOM_DFT_VALIDATION.md).

For the **same PWidom MD + insertion/deletion/BAR calculation with VASP as its
ASE force engine**, including an interactive smoke test, isolated restart caches,
CPU timing, and two-node 24-hour job recipes, see
[Pure-VASP PWidom](docs/PWIDOM_PURE_VASP.md). Smoke-test before submitting pilots;
this backend has local mock-executable tests, not a completed Perlmutter run.

For four DFT PWidom systems × 64 ranks, use `dft-layout` followed by
`bash mlp_pipeline/perlmutter_dft_pwidom_run.sh PLAN --smoke --minutes 20`.
The 48-hour batch script is `mlp_pipeline/perlmutter_dft_pwidom_48h.sbatch`;
see the final section of [Pure-VASP PWidom](docs/PWIDOM_PURE_VASP.md).

Numerical evidence and current concerns are recorded in
[GRACE/DFT audit, 2026-11-01](docs/status/GRACE_DFT_AUDIT_2026-11-01.md),
including all supplied validation rows and the later three-job production tails.
