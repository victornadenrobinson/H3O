# Project memory

Tool-agnostic orientation, 2026-10-31, checkout 7a8de56. Supplements rather than
replaces HANDOFF.md, PROJECT.md, or any Claude instructions. No AGENTS.md or
CLAUDE.md was found in this workspace. Read HANDOFF.md, CURRENT_ISSUES.md section
0, SOLUBILITY_VALIDATION.md, CODEBASE.md and MU_METHODS.md before scientific work.
Historical queue states and branch recipes in those documents are not live state;
inspect the current checkout and jobs before pulling, restarting, or submitting.

Current campaign correction: the user is running proxy fitting on PERLMUTTER,
in `/pscratch/sd/v/victornr/H3O`, interactive Slurm allocation 60617748 on
nid200360 (user-reported). Young recipes below are historical/other workflows,
not this campaign. User has GPU allocation hours and reports a free GPU overrun
queue; do not infer its submission flags without checking Perlmutter's setup.
Uploaded terminal output confirms label 109/109, 13,625 single points in 29.2 min,
then both fits saved to runs/pwidom_fit/proxy_{ice,H}.json and .png; PROD=0
stopped after fitting, with no production comparison. Ice: 90 sources, 13 wholly
held out; selected held-out gap 1.94 -> 1.57 kT, pooled Kish .084 -> .191.
Pure H: 19 sources, 3 wholly held out; .50 -> .47 kT, Kish .848 -> .861.
Overlap diagnostics worsen (ice held-out .711 -> .983, H .346 -> .692), and
some rich-H ice states regress. Treat this as a candidate proxy, not production
validation: probes were selected by the starting proxy. First inspect fitted
short-range curves and then test new-proxy sampling on independent held-out
frames, only after user approval. Do not relabel completed sources by default.
This is remote terminal evidence; run artifacts remain absent from this cloud
checkout. Live allocation state after the reported completion is unknown.
The subsequent attempted 500 GPa / 3000 K launch stopped at missing ice stage0;
no MD started. The state may be new. Use the new perlmutter_pwidom.sh
`--inspect-eos <inventory.csv>` report to check exact-bin DFT coverage, then
provide reviewed volume/B inputs with `--state-json` rather than assuming TDI
metadata exists. Four-GPU launcher/status commands are in mlp_pipeline/README.md.

## Layout and workflows

- `h3o_mlp/`, `scripts/`: earlier inventory/extraction/benchmark package.
- `mlp_pipeline/00-03*`, `10_check_snapshots.py`, `11_bundle.py`, `build_all.sh`:
  DFT inventory -> snapshots -> grouped train/validation tiers s/m/l -> bundle.
- `mlp_pipeline/04_benchmark.py:get_calc_spec`: shared calculator factory;
  `grace-ft:runs/grace/ft_m2` resolves a SavedModel below that run directory.
  Multiple SavedModels are resolved by sorted-first match: inspect the actual export.
  `grace_make_input.py`, `grace_ft_run.sh`: GRACE training, distinct from proxy fitting.
- `mlp_pipeline/tdi/toy_fit.py`: DFT force/energy/stress least squares for OO/OH/HH
  spline pairs; tracked baseline `docs/data/toyfit_t4.json`.
- `mlp_pipeline/tdi/proxy_fit.py label`: existing host frames -> model single-H
  insertion differences `U(host+H)-U(host)` at selected/uniform probe sites.
  `fit`: weighted spline least squares for OH/HH only, with a nuisance intercept;
  no MD or neural-model training. Default base core/cutoff and OO remain fixed.
  Separate ice/pure-H fits, state holdouts and frame holdouts; inspect held-out
  gap std in kT, Kish fraction and overlap diagnostics before production.
- `mlp_pipeline/tdi/pwidom.py`: proxy-guided Rosenbluth insertion/deletion + BAR;
  `Model` uses `04_benchmark` and `batched.GraceBatch` for TPCalculator batches.
  Its first batch self-check compares against individual calculator evaluations.
  `run --like <stage0.json>` inherits state/model; missing frame cache starts MD.
  `dg <ice.json> <pure-H.json>` compares pressure-corrected chemical potentials.
- `mlp_pipeline/tdi/run_real.py`, `young_tdi_par.sh`: restartable ghost-H TDI,
  stage0 pressures/reference, lambda sampling, collect/summarize/report downstream.
- `mlp_pipeline/tdi/dft_check.py`, `young_dft_check.sge`: independent VASP checks,
  including prep-widom. GRACE probe labels are model predictions, not DFT labels.
- `docs/data/`: small scientific tables; `docs/figures/`: plots; `tests/`: CPU tests.

## Data, logs and fit state

Large files belong in ignored `runs/` or external storage, not public Git history.
DFT snapshots `runs/train_m.extxyz`, `valid_m.extxyz`; model `runs/grace/ft_m2`;
cached Widom hosts `runs/pwidom*/pw_*_frames.npz`; TDI `runs/tdi*/`.
`H3O_RUNS` affects common pipeline helpers; proxy_fit accepts literal paths.
`young_proxy_fit.sbatch` defaults FIT=runs/pwidom_fit, BASE=docs/data/toyfit_t4.json:
`*_probes.npz`, `label.log`, `fit_ice.log`, `fit_H.log`, `proxy_ice.json/.png`,
`proxy_H.json/.png`; outer log `runs/slurm_proxyfit_<jobid>.out` or `runs/proxyfit.log`.
PROD=1 also evaluates old/new proxies; PROD=0 stops after both fits.

The supplied [32/109], [33/109] lines exactly match proxy_fit.py label output:
five frames, 120 probes, 125 single points/source, cumulative 4000/4125 calls.
The 2.21/2.18 kT numbers are STARTING-proxy residual standard deviations, not
fitted errors or convergence. Filename identifies parent source directory
`pwidom_k1d` and its frame-file stem; it does not identify the model or command.
Five frames is consistent with PER_SOURCE=5, or a source with only five frames.
There are no optimizer checkpoints: each completed source is saved as a probes
file; unfinished sources rerun. Existing files are skipped by existence alone.
Labels record source/base-proxy path/temperature/temper, but not complete model,
fingerprint, seed or invocation provenance. Retain original manifest/order/logs;
the label RNG seed depends on source index. Never resume blindly or use --force.
Label loading reconstructs cubic cells from V and does not check saved model
fingerprints: audit geometry and Hamiltonian before trusting cached sources.
The shell job lacks pipefail/fail-fast checks, so a failed label stage can be
followed by a fit on partial files; final job exit alone does not prove completeness.
Default holdouts match one directory prefix only: check equivalent state copies
in other source directories for leakage before approving holdout configuration.

Observed here: no runs directory, model, probe archives or fit logs; no local
Python fitting process or Slurm/SGE clients. Remote job activity/completion and
actual FIT/MODEL/holdouts cannot be determined from this checkout. Do not submit
a duplicate. On Young login, inspect `bash mlp_pipeline/jobs_status.sh`,
`squeue -u "$USER"`, `sacct`, `runs/JOBS_LOG.tsv`, original log and batch script.

## Environments and constraints

Cloud CPU: `cd /workspace/H3O; source /workspace/.venvs/h3o/bin/activate`;
set MPLCONFIGDIR=/workspace/.cache/matplotlib, XDG_CACHE_HOME=/workspace/.cache,
OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Python 3.11.17; baseline 14 tests
passed during onboarding. Commands: `python -m pytest tests -q` and
`python mlp_pipeline/tdi/solubility_check.py --selftest`.
Four single-point toy energies through pwidom.Model were checked this session;
three insertion differences match proxy_energy with the same toyfit_t4 potential
to 1e-10. This tests evaluator plumbing only, not physical accuracy or a GRACE fit.
TensorFlow, tensorpotential, matscipy, nvidia-smi and model exports are absent here.
User reports prior TF 2.20/TPCalculator four-structure GPU inference succeeded;
that result is not verification of this instance or the fitting configuration.

Young GPU: `/lustre/scratch/mmm0037/H3O`, Slurm, `~/grace-venv/bin/python`;
separate model environments. Young CPU: SGE/RHEL7, source young_cpu_env.sh;
GPU RHEL9 venv is incompatible; request all 40 cores for exclusive CPU nodes.
Perlmutter data: `/pscratch/sd/v/victornr/H3O`, nersc-python; raw data under
`~/work/tdi/H3O` and `~/work/quartz-work/tdi/h3o`. Scratch is purgeable.
Use existing isolated checkout; no automatic worktree, PR, push or job submission.
Preserve Claude/project guidance; do not overwrite results or scientific settings.
Obtain user go-ahead for expensive runs. Validate paths/model/source provenance first.
Report meV and kT, use 'lambda values', dt=0.5 fs; count insertion per added H.
DFT energy is TOTEN; virial excludes ionic kinetic pressure; avoid double-counting
N kT/V. Keep model error, finite sampling and statistical errors distinct.
