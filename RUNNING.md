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

Extensions use a new directory to preserve the completed parent and prevent
two writers from racing over caches/results. Do not combine parent and extension
as independent data: the extension already includes the parent's frames. An
in-place mode would need locking, atomic replacement and a preserved prior version;
it is not implemented. By contrast, `run_real.py` TDI supports checkpoint restarts.

Compact progress/GPU tables are default; `--verbose` streams full logs and
`--plain` uses plain ASCII. Full logs are always in the run directory. For a log
stream: `tail -n 20 -F runs/<scan>/*.log`. Prior free energies are labeled prior;
new free energies require completed insertion/deletion estimates.

## Other calculations

Use `$PY`/`$MODEL` from the source computer's setup above. Inspect `--help` before
choosing scientific parameters. Commands containing `<...>` need real paths/tags.

| Task | Terminal entry point |
| --- | --- |
| One GRACE energy/force check | `"$PY" mlp_pipeline/04_benchmark.py "$MODEL" valid_m.extxyz --n 1 --device cuda` |
| CPU toy calculation | `"$PY" mlp_pipeline/04_benchmark.py toyice valid_m.extxyz --n 1 --device cpu` (checks software only, not GRACE) |
| GPU packing benchmark | `bash mlp_pipeline/perlmutter_bench.sh --workflow pwidom --minutes 10` (after other GPU work finishes) |
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
