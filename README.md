# H3O

Benchmarking and fine-tuning machine-learning interatomic potentials on DFT data for H2O-H mixtures at extreme pressure and temperature.

- Plan, conventions, data manifest and status: [PROJECT.md](PROJECT.md)
- Pipeline scripts: [mlp_pipeline/](mlp_pipeline/) (run order in `build_all.sh`)
- Environments (Perlmutter, Young): [envs/README.md](envs/README.md)
- Figures and their source numbers: [docs/](docs/)
- Reference EOS/hull data: `data-h3o-convex-hull.csv`. Large data products live in `runs/` (gitignored) and are shipped as bundles.

## Earlier data-collation package (`h3o_mlp/`, `scripts/`)

Benchmark and fine-tune MLPs (MACE, GRACE, MatterSim) against H2O–H3O DFT data.

Workflow (`h3o_mlp/`, `scripts/`):
1. `scripts/fast_inventory.sh ROOT` → `mined_inventory_ground_truth.csv` (Path, True_T_K, True_P_GPa, True_V_A3).
2. `inventory.load_inventory` / `coverage` – bin by x, P (±25 GPa), T (±250 K) and check coverage of `data-h3o-convex-hull.csv`.
3. `extract.build_dataset` – write 5–15 decorrelated snapshots per state to `.extxyz`.
4. `benchmark.run` – zero-shot energy MAE, force RMSE, stress MAE.
5. `finetune.split` + `finetune.mace_command` – 80/20 split and force/stress-weighted fine-tuning on Perlmutter A100s.

Tests: `pytest tests`
