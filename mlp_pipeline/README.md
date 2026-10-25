# MLP benchmark / fine-tune pipeline (H2O-H mixtures, extreme P-T)
Run order: 01 -> 02 -> 03 -> 04 (zero-shot) -> 05 -> 04 (mace-ft:model).
Not yet run on real data. Verify: pressure units (kB vs GPa), GRACE/MatterSim APIs, mace_run_train flags, sbatch account.
Hull EOS targets span ~264-1400 GPa (not 200-500), so extraction uses 250-1425 GPa. No H2.5O/H2.75O EOS in hull CSV.
Unknown-composition runs (e.g. `10.20/`) need composition from OUTCAR ion counts; skipped for now.
