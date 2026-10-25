# Figures (docs/figures)

Drawn by `docs/make_figures.py` from the tables in `docs/data/*.csv`. Those numbers were **transcribed from the Perlmutter terminal output**
of `01_bin_inventory.py` and `07_crosscheck_eos.py` run on `runs/mined_h3o.csv` (2026-10-25): lambda_* TDI runs excluded, runs with <100 MD steps
excluded from the T x P and composition x T tables, pressure = OUTCAR external pressure + N kB T / V, temperature from TEBEG / folder name.
They are not recomputed from the raw CSV here; to regenerate from the data use `mlp_pipeline/06_plot_coverage.py` and `07_crosscheck_eos.py`.

| file | shows |
|---|---|
| inventory_T_P.png | runs per T bin x 100 GPa pressure bin |
| inventory_composition_T.png | runs per composition x T |
| hull_target_coverage.png | runs near each hull EOS isotherm target (201/201 covered) |
| eos_crosscheck.png | mined P minus hull P at equal density, by composition and T |

Note: `snapshots_*.extxyz` built before the `inp()` fix did not contain the 11000/13000 K runs; rebuild with `build_all.sh`.
