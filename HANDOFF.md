# HANDOFF: read this first in a new session (written 2026-10-26)

Branch `claude/eloquent-ramanujan-2tjx2p`, open PR victornadenrobinson/H3O#2 (do not open another; push to the branch). Read this file, then `PROJECT.md` (plan, conventions, zero-shot table), `docs/TDI_PLAN.md` (free-energy plan), `docs/LITERATURE_NOTES.md` (paper facts), `mlp_pipeline/README.md`, `envs/README.md`.

## Goal
Benchmark foundation MLIPs (MACE, MatterSim, GRACE) against VASP AIMD data of H2O-H mixtures at about 200-1400 GPa and 3000-13000 K; build clean production-only fine-tuning sets; fine-tune; then use the fine-tuned potential for thermodynamic integration (TDI) to get G_fcc - G_bcc of superionic H2O versus pressure and compare with the literature (Wilson et al. 2013 DFT numbers, Cheng et al. 2021 MLP numbers). The user ultimately wants free-energy-grade accuracy, and the hull of H2O-H mixtures.

## Where things live (not in git: `runs/`, the data, the papers)
- **Perlmutter** (data processing, OUTCARs under `~/work`): repo `/pscratch/sd/v/victornr/H3O`, python = `nersc-python` (pandas/ase). `runs/` is gitignored; bundle with `bash mlp_pipeline/build_all.sh 64` then `send_bundle.sh --force`.
- **UCL Young** (GPU, Slurm): repo `/lustre/scratch/mmm0037/H3O`; envs `~/grace-venv` (py3.11, tensorpotential/gracemaker 0.6.1), `~/work/envs/mace-ng`, `~/work/envs/mattersim-env`. At most 2 interactive GPU jobs: `srun -p intergpu --qos=intergpu -A allusers --gres=gpu:1 --cpus-per-task=8 --mem=120G --time=04:00:00 --pty bash`. **`--mem=120G` is required**: gracemaker's data conversion needs more than the 8 GB default (exit 137 otherwise). Slurm prints a harmless `_handle_getgr` error every minute; `git config --global core.pager cat` avoids the `less` trap.
- The sandbox/cloud session cannot reach arxiv/nature/osti hosts (network policy) and cannot see Young/Perlmutter: the user pastes outputs. Papers are in `docs/papers/` locally (not committed; may be gone in a new container: ask the user to re-upload).
- Python versions: Young venv is 3.11 (no same-quote nesting inside f-strings), Perlmutter default 3.9; all scripts are checked with `python3.11 -m py_compile`.

## Conventions (verified)
Pressure = OUTCAR external pressure + N kB T / V (ideal-gas term added ONCE, in `01_bin_inventory.load_inventory`; `inventory_binned.csv` already contains it and the loader now detects that). Energy label = VASP free energy (TOTEN), sigma->0 kept as `energy_sigma0`. Stress = ASE 3x3 eV/A^3. Frames come from one OUTCAR per run chain (default: segment at 0.75 of the chain, `02 --late-segment`). lambda_* VASP-TDI runs, test/broken/small-box runs are excluded (`exclude_paths.txt`, `--min-box 8.5`). VASP: PBE, ENCUT 1100, gamma-mostly, SIGMA = kB T.

## State of the data (rebuilt 2026-10-26)
tiers s/m/l = 212 / 932 / 1281 frames (train/valid grouped by run directory: train_m 747, valid_m 185). `16_phase_census.py` counts runs by path words; `09_label_phase.py` labels dynamics (joins restart segments, removes COM drift, O Lindemann ratio, fcc/bcc from time-averaged O sites). Its thresholds are NOT settled: use `--lindemann 0.25`, `--h-super 1.0` as a first guess, and rerun with `--comp h2o --sample 300` then `--reuse` after the COM-drift fix; paste the by-T-bin tables. `02 --phase-csv` then balances fcc/bcc/liquid/other buckets. fcc DFT data exist only for about 330-540 GPa at 3000 K (after cleaning); bcc down to about 100 GPa.

## Results so far
- Zero-shot (see PROJECT.md): GRACE-2L-SMAX-OMAT-medium and MatterSim 5M are usable, MACE variants and OMat24-only models are not (mh-1 heads all fail).
- Fine-tune GRACE-2L-SMAX-OMAT-medium (`grace_make_input.py`, `grace_ft_run.sh`, only `I1/I2/rho reducing_` layers trainable, Huber delta 0.01, weights E1/F5/S10, 2000 updates, batch 2, about 33 min, GPU peak 38.9 GB of 40): `ft_m2` on the rebuilt data, validation: energy MAE 18.4 meV/atom, force MAE 511 meV/A (RMSE 730), stress MAE 2.05 GPa; zero-shot start on the same frames 96 meV/atom, 1017 meV/A, 20.7 GPa. Train about equal to validation: **underfitting, not data-limited**. Loss composition (ft_m2, reconstructed from the log, Huber delta 0.01, weights E1/F5/S10): forces 96.7%, stress 2.7%, energy 0.6% of the training loss; all three are in the linear (L1) regime (errors 49x, 1.9x, 1.2x delta). So stress is NOT over-weighted; energy is nearly absent from the loss (matters for free energies; `grace_make_input.py --wE`). Not yet tried: unfreezing all weights (`--naive`), many more updates (only about 5 epochs so far), larger model.
- EOS by MD (`17_eos_md.py`, `18_plot_eos_md.py`): bcc 3000 K, medium model within 3 GPa / 0.35% in volume of DFT (small model 7 GPa / 0.5%). An apparent -15 to -22 GPa "bias" was an analysis bug (double-counted kinetic term); all conclusions drawn from it were withdrawn (PROJECT.md). H diffusion 6-8e-4 cm2/s, O frozen. fcc EOS-MD and fcc frame check: frame errors <= 1.5 GPa; fcc EOS-MD not yet run (use P = 350-500 GPa).
- Literature (docs/LITERATURE_NOTES.md, docs/data/dg_fcc_bcc.csv, `docs/plot_dg_vs_P.py`): Wilson et al. Table I, G_fcc - G_bcc at 3000 K per atom: -0.7(3.7) meV at 100 GPa, -9.0(2.7) at 200, **-24.7(2.3) at 500**, -71(2) at 4000 GPa. Cheng et al. (public data tables, github.com/BingqingCheng/superionic-water, read-only clone not in git; derived table docs/data/cheng2021_dmu_bcc_fcc.csv, script docs/extract_cheng_mu.py) agree at 3000 K: **targets for our TDI: G_fcc - G_bcc = -26.5 meV/atom at 500 GPa, -22.1 at 400 GPa**. Their NNP was trained from scratch on about 17.5k structures; the 2022 Nat. Commun. paper (Reinhardt et al.) confirms the FEP correction to PBE and gives their TI consistency checks (see docs/LITERATURE_NOTES.md).

## TDI engine (`mlp_pipeline/tdi/`)
`refs.py` analytic references (Einstein crystal with fixed O-COM, ideal gas, harmonic network, kinetic terms), `engine.py` (stages direct/stage1/stage2, `FixGroupCom`, Langevin run per lambda), `analyze.py` (Gauss-Legendre in lambda^p, block errors). `python mlp_pipeline/tdi/test_harmonic.py` reproduces an analytic harmonic-crystal free energy to 0.5 meV total (PASS, 8 lambda nodes). Use Langevin: Bussi does not thermalise the modes of a harmonic crystal. Not yet written: structure builder + runner for the real system (builder exists in `17_eos_md.py::build`), GRACE components (full and O-only), proton reference (ideal gas; Uhlenbeck-Ford fluid is the alternative used by Cheng et al.), the DFT-level correction, Slurm array scripts. Plan: first state point 500 GPa 3000 K (Wilson's 5 Mbar row), then 400 GPa; DFT-sized cells (128 O bcc / 108 O fcc); check bcc stays bcc in 20 ps NVT first.

`mlp_pipeline/young_stability.sbatch`: 20 ps NVT check (bcc/fcc at 400 and 500 GPa, 3000 K) that prints O_struct / delta_L; not yet run.

`mlp_pipeline/tdi/run_real.py` (+ `young_tdi.sbatch`, `tdi/compare.py`): real-system TDI, direct path (Einstein O + ideal-gas H -> GRACE), all lambda nodes in chunks, prints G(t); smoke-tested only with the LJ stand-in. Known weak point: near lambda=0 the integrand is huge (H overlaps), Wilson used 48 nodes there; G uses P_target*V.

TDI parallel mode: `mlp_pipeline/young_tdi_par.sh <bcc|fcc>` = stage 0 (lambda=1) then lambda nodes as separate processes sharing one GPU (A100 is only ~21-30% busy per process; needs TF_FORCE_GPU_ALLOW_GROWTH), `tdi/collect.py runs/tdi <tag>` integrates the node files (works on partial runs). Verbose per-node health lines (T, U, spring energy, min H-H/H-O distance, O displacement). Untested with GRACE.

TDI monitoring: `tdi/live.py runs/tdi <tagA> <tagB> [--loop 60 --png]` (text dashboard: per-node integrand, drift in sigma, preview integral, G(t), dG(t)), `tdi/plot_tdi.py` (PNG), nodes are restartable. First real fcc 400 GPa result (10 GL nodes, power 4, 2 ps/node, dt 1 fs, k 22.8): G = 1978.5 +- 2.2 meV/atom (incl. P V); integrand plateau +1217 eV at lambda<1e-5, steep between lambda 0.003-0.3, -881 eV at 0.95.

**TDI result, 400 GPa 3000 K, ft_m2, 10 GL nodes power 4, dt 1 fs, 6 ps/node:** G_fcc = 1979.50 +- 0.98, G_bcc = 2004.19 +- 1.08 meV/atom (incl. P_target V), **G_fcc - G_bcc = -24.7 +- 1.5 meV/atom** (Cheng 2021: -22.1, Wilson interpolated about -19). Between 2 and 6 ps bcc G fell by 3 meV/atom (the flagged node-7 drift was real), dG moved from -28.5 to -24.7. Not yet checked: dt 0.5 fs, a second quadrature, finite size, MLP-vs-DFT (FEP). Plans: 5000 K (check DFT volumes first), pure-H G (ideal-gas reference, U = lam U_MLP), ghost-H insertion into superionic H2O for the chemical potential.

## Open items, in order
1. `ft_s2` (small model on the new data, about 10 min), then `bash mlp_pipeline/bench_grace_compare.sh valid_m.extxyz 0 runs/grace/ft_m2 runs/grace/ft_s2` (figure `runs/bench_compare_valid_m.png`; points coloured by composition).
2. Reduce the force/energy error: unfrozen weights and 8000+ updates (compare train vs validation), then decide whether more data is needed.
3. fcc EOS-MD, bcc/fcc at 400-500 GPa with the new models.
4. Finish phase labelling thresholds, rebuild with `--phase-csv`.
5. Write the TDI runner for the real system; test the runtime of GRACE at 400-750 atoms (measured 172 ms/step for 384 atoms on an A100).
6. Cheng et al. numbers; optional DFT-level correction by perturbation using existing DFT frames.

User preference: say "lambda" / "lambda values", never "nodes" (nodes = compute nodes). In code the NODES env var, --node option and *_node<k>.* file names are kept for compatibility; printed text says lambda.

## Working agreements with the user
Report energies in meV (also relative to kB T), pressure on the x axis for EOS plots, say plainly when something is unverified, do not mock results, keep outputs short, never open a new PR, commit with the Co-Authored-By line, test with Python 3.11. The user runs commands on Perlmutter/Young and pastes output; give them copy-paste blocks that say where to run.

## Mistakes already made (do not repeat)
Double-counted kinetic pressure term (above); tier selection claimed "out1 unequilibrated" without evidence (withdrawn); `r.T` pandas transpose in a progress line; f-string quote nesting breaking py3.11; global thermostat on a harmonic test; `send_bundle.sh` needing a build log that was never written.
