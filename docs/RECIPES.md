# Recipes: how to run each calculation

One entry point: **`bash run.sh <task> key=value ...`** from the repo root on Perlmutter. It translates the keys to the underlying script and prints the exact command (`+ ...`) before running it; **`dry=1`** only prints. `bash run.sh help` lists the tasks. Defaults below are what you get when a key is left out.

Where to run: tasks marked **GPU** need an interactive GPU node (`salloc -A m3349_g -C gpu -q interactive -N 1 -G 4 -c 128 -t 02:00:00`); tasks marked **login** submit batch jobs and must run on a login node (submitting from inside a GPU allocation gives "Job request does not match any supported policy").

Common keys: `model=` (a run name like `mfin_f_lr1e-2_e50` or a full `grace-ft:...` spec; default the current best model), `dry=1`.

## Tasks

### eos — liquid / fluid EOS scan with live dashboard (GPU, ~15-25 min per composition at 4 GPUs)
`bash run.sh eos comp="h2o h3o h" T=7000`
- keys: `comp` (h2o h2.25 h3o h5o h7o h; several = one job queue), `T` (7000), `nv` (16 volumes), `vmax` / `vmin` (A^3/atom; default per composition), `ps` (0.75; pure H 0.5), `eq` (0.25; pure H 0.2), `per_gpu` (4).
- cells ~384 atoms (pure H 729); mixtures premelted at 9000 K for 0.25 ps; output `runs/eos_live/<model>_<comp>_<T>K_<N>at/`; finished volumes are skipped on rerun; flags FROZE / COLLAPSE / CLOSE-HH / T-off / P-drift; per volume `rdf.csv`, coordination, H2 fraction.

### eos_plot — all scans vs DFT (and French 2009 for H2O), P-T maps (login or GPU node, CPU only)
`bash run.sh eos_plot T=7000`
- keys: `T`, `pmin` / `pmax`, `frames` (DFT-MD frames for the energy comparison; default train_mfin + valid_mfin).
- writes `runs/eos_live/eos_all_<T>K_<model>.png` and `runs/eos_live/eos_maps_<model>.png`.

### tdi — full TDI free energy of one phase (GPU here, or one shared-GPU batch job)
`bash run.sh tdi phase=fcc P=500 T=5000 hper=2 quad=gk17 npar=6 ps=2`
`bash run.sh tdi phase=liquid P=500 T=5000 hper=2.25 quad=gl10 where=batch hours=8`
`bash run.sh tdi phase=H P=500 T=5000 quad=gl10 npar=4`
- `phase`: fcc | bcc | liquid | H (pure-H fluid, 729 atoms) | insert (`nghost=` protons into fcc H2O).
- `quad`: `gl<N>` = Gauss-Legendre in lambda = x^4 with N values (gl10, gl12); `gk<2n+1>` = Gauss-Kronrod: the n Gauss values plus n+1 added (gk17 = 8 + 9, gk21 = 10 + 11). A gk run gives G_Kronrod and G_Gauss(n) from the same data: their difference is the quadrature error. An existing gl run can be extended to gk later without redoing it (see `young_tdi_par.sh` header, ONLYNODES).
- keys: `P` (500), `T` (5000), `hper` (2; 2.25 = H2.25O), `npar` (4: lambda processes at a time on the GPU), `ps` (4 ps per lambda), `ps1` (1.5), `equil` (1), `power` (4), `gpu` (0), `where` (here | batch), `hours` (8, batch), `natoms`, `nghost`, `hot` / `hott` (liquid melt; default 0.5 ps at 12000 K), `k` (Einstein constant), `nrep`, `prephold`, `tag`, `outdir`.
- volumes from the DFT EOS (`inventory_binned.csv`, linked from runs/ if needed); restartable (rerun the same command); output `runs/tdi/`; results `python3 mlp_pipeline/tdi/summarize.py runs/tdi`.
- cost (fcc 324 atoms, measured 2026-11-06 with the F10 model): **npar 4 is the best** (2/4/6/8 processes per GPU = 12.3/23.8/22.2/22.4 steps/s per GPU); one full TDI on one GPU at npar 4: 10 lambda x 2 ps 0.8 h, 10 x 4 ps 1.4 h, 17 x 2 ps 1.4 h, 17 x 4 ps 2.3 h, 21 x 2 ps 1.7 h. Four TDIs side by side: gpu=0..3. Large cells (> 400 atoms) need npar <= 8 on a 40 GB card, 729-atom H npar 4.

### tdi_bench — TDI throughput, different numbers of lambda processes per GPU at once (GPU, ~10 min)
`bash run.sh tdi_bench system=h2o layout="2 4 6 8"`
- `system`: h2o (fcc crystal, 324 atoms) | H (pure H 250) | ins | insH; `layout` (processes on GPU0..3), `steps` (200), `vpa`.
- prints steps/s per process and per GPU and the full-TDI wall time for 10 / 17 / 21 lambda values.

### finetune — one GRACE fine-tune as a batch job (login)
`bash run.sh finetune name=mfin_f_lr1e-2_e50_S10 train=mfin epochs=50 wE=10 wF=10 wS=10 lr=0.01`
- keys: `name` (required), `train` (mfin), `valid` (= train), `epochs` (50) or `updates`, `wE` / `wF` / `wS` (10 / 10 / 1), `mode` (frozen | naive), `lr` (0.01), `batch` (1), `gpus` (2: the 4.3k-frame tier needs the host RAM of 2 shared GPUs), `time` (24:00:00).
- always starts from the foundation model (ft_one.sh); output `runs/grace/ft_<name>/`; status `python3 mlp_pipeline/ft_status.py`.

### score — compare fine-tunes (GPU, minutes per model)
`bash run.sh score only="mfin_f_lr1e-2_e50"`
- held-out frames (`compare_D` table), DFT-MD composition test (`composition_check`), sealed pair test (pair c, pass |c| <= 50 meV/H; only reference models that did not train on the pairs are a real test).

### dft — VASP single points for any set of geometries (login)
`bash run.sh dft frames=runs/x/frames.extxyz outd=runs/x_dft nodes=8 submit=1`
- keys: `frames`, `outd` (required), `template` (vasp_inputs; a k-mesh template for k-point labels), `exe` (vasp_gam; vasp_std for k-meshes), `nodes` (8 jobs), `minutes` (170), `submit` (0 = prepare only).
- 4 lanes x 32 ranks per node; a resubmission resumes unfinished points; collect with `points_collect.py` (skips unconverged SCF, prints a health table).

### kpt — k-point relabelling of a stratified frame selection (login)
`bash run.sh kpt n=300 outd=runs/kpt222mp submit=1`
- MP 2x2x2 (4 k-points, 1.8x faster than Gamma-centred, as accurate), KPAR 1; keys `mesh` (mp222 | g222), `kpar`, `n`, `outd`, `nodes` (20), `minutes` (120), `submit`.
- check before submitting: `python3 mlp_pipeline/kpt_sanity.py runs/kpt222mp runs/kpt222mp2 --ref runs/kpt222/frames.extxyz`.

## Not (yet) behind run.sh
- PWidom chemical potentials: `mlp_pipeline/pwidom_states.sh` / `perlmutter_pwidom.sh` (headers and RUNNING.md).
- DFT labelling campaigns of snapshot databases: `snapdb.py`, `lowp_quick.sh`.
- Training tiers: `build_tier.py` (exact commands in docs/HANDOFF.md section 2.1), `grace_prep_pkl.sh`.
- T-x phase diagram from free energies: `tdi/tx_diagram.py` (header).

## Earlier quadrature findings (old ft_m2 model; redo with the final model)
From docs/CURRENT_ISSUES.md (2026-10-27/28): Gauss-Kronrod 10 -> 21 lambda moved the H2.25O crystal G by -14.1 meV/atom but H2O by only -1.4 (x13 amplification per added H: ~85 meV/H quadrature uncertainty of the crystal mixing value at 500 GPa / 6000 K); crystals that melt in free MD but are held by the springs show huge Kronrod - Gauss (-327 / -451 meV/atom: not valid solids); MBAR over the saved lambda series (`tdi/bar_lambda.py`) is a quadrature-free cross-check; the insertion runs' quadrature error (P3) was never measured. Suggested test with the final model: gk17 for fcc H2O, fcc H2.25O, liquid H2O and pure H at 500-600 GPa / 5000 K, one GPU each.
