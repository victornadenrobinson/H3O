# widom-calculator

Free energies of hydrogen in water at planetary-interior conditions (about 10-800 GPa, 1000-9000 K), computed with machine-learning interatomic potentials and checked against DFT.

The main question is whether hydrogen dissolves in, or mixes with, superionic and fluid H2O (the H2O-H3O-H system relevant to ice-giant interiors), and where the mixed phases melt. The code:

- **Computes chemical potentials of H** in a host (`mu_H(host) - mu_H(pure H)`) by proxy-guided Widom insertion and deletion combined with BAR (`mlp_pipeline/tdi/pwidom*.py`), with ideal-dilution, pressure and finite-size corrections, checkpoint/resume, and per-state uncertainty.
- **Cross-checks** those numbers with thermodynamic integration and Frenkel-Ladd free energies (TDI), static formation enthalpies, and DFT (VASP) single points on the exact configurations that carry the Widom weight, including a DFT-over-model reweighting correction.
- **Builds and fine-tunes the potentials**: mining VASP MD into snapshot databases, DFT labelling jobs, training tiers with held-out state groups, and fine-tuning GRACE (also MACE and MatterSim benchmarks) on Perlmutter A100s.
- **Analyses the results**: mu/dG tables, convex hulls versus time, melting and liquid hulls.

The project was first called H3O, and some file names and older docs still use that name.

## Where to start

- **Current state, open items and next steps: [docs/HANDOFF.md](docs/HANDOFF.md).** (The root [HANDOFF.md](HANDOFF.md) is the older Young-cluster handoff.)
- Commands and cluster setup (Perlmutter, Young): [RUNNING.md](RUNNING.md), [envs/README.md](envs/README.md).
- Plan, conventions and data manifest: [PROJECT.md](PROJECT.md). Code map: [docs/CODEBASE.md](docs/CODEBASE.md).
- Method notes: [MU_METHODS](docs/MU_METHODS.md), [TDI_PLAN](docs/TDI_PLAN.md), [DFT_PLAN](docs/DFT_PLAN.md), [MODEL_TEST_SUITE](docs/MODEL_TEST_SUITE.md); code review: [PWIDOM_BUG_REVIEW](docs/PWIDOM_BUG_REVIEW.md).

## Layout

| path | contents |
|---|---|
| `mlp_pipeline/` | numbered data pipeline (mining, snapshots, benchmarks, fine-tuning), DFT job builders (`dgcheck_jobs.sh`, `points_collect.py`, `build_tier.py`), low-P / hydrate campaign (`lowp_*.py`) |
| `mlp_pipeline/tdi/` | PWidom, TDI / Frenkel-Ladd, BAR/MBAR, proxy fitting, VASP check runners |
| `h3o_mlp/`, `scripts/` | earlier data-collation package (inventory, extraction, zero-shot benchmark) |
| `docs/` | plans, results, figures and the numbers behind them |
| `tests/` | `pytest tests` (no GPU needed) |

Reference EOS / hull data: `data-h3o-convex-hull.csv`. Large data products live in `runs/` (gitignored).

## PWidom

Use the existing GRACE setup and commands in **[RUNNING.md](RUNNING.md)** for
simulations, checkpoint/resume, μ/dG tables and time-coloured formation hulls.
The analysis commands run directly on saved results; they require no GPU or MD.
Detailed conventions: [results](docs/PWIDOM_RESULTS.md),
[time plots](docs/PWIDOM_TIME_HULLS.md), [resume](docs/PWIDOM_RESUME.md).

## Earlier data-collation package (`h3o_mlp/`, `scripts/`)

Benchmark and fine-tune MLPs (MACE, GRACE, MatterSim) against H2O-H DFT data.

Workflow (`h3o_mlp/`, `scripts/`):
1. `scripts/fast_inventory.sh ROOT` → `mined_inventory_ground_truth.csv` (Path, True_T_K, True_P_GPa, True_V_A3).
2. `inventory.load_inventory` / `coverage` – bin by x, P (±25 GPa), T (±250 K) and check coverage of `data-h3o-convex-hull.csv`.
3. `extract.build_dataset` – write 5–15 decorrelated snapshots per state to `.extxyz`.
4. `benchmark.run` – zero-shot energy MAE, force RMSE, stress MAE.
5. `finetune.split` + `finetune.mace_command` – 80/20 split and force/stress-weighted fine-tuning on Perlmutter A100s.

Tests: `pytest tests`
