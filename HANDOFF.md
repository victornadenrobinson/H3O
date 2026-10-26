# HANDOFF: read this first in a new session (written 2026-10-26)

Branch `claude/eloquent-ramanujan-2tjx2p`, open PR victornadenrobinson/H3O#2 (do not open another; push to the branch). Read this file, then `PROJECT.md` (plan, conventions, zero-shot table), `docs/TDI_PLAN.md` (free-energy plan), `docs/LITERATURE_NOTES.md` (paper facts), `mlp_pipeline/README.md`, `envs/README.md`.

Output formats, file list, log-line meanings, run commands and coding rules for new scripts: `docs/TDI_OUTPUT.md`.

## LATEST STATE (end of session 1, 2026-10-26 afternoon)  -- read this block first
All TDI results below: fine-tuned GRACE `ft_m2` (runs/grace/ft_m2), classical nuclei, G includes P_target V, errors are statistical (block) only. dt was 1.0 fs unless noted (new default 0.5 fs, user wants 0.5 for everything). Say "lambda" / "lambda values", never "nodes" (user preference).

| system (400 GPa) | T (K) | result (meV/atom) | notes |
|---|---|---|---|
| G_fcc - G_bcc | 3000 | -24.7 +- 1.5 (6 ps/lambda) | Cheng 2021 -22.1, Wilson interp. ~ -19 |
| G_fcc - G_bcc | 5000 | -10.4 +- 3.5 (1.5 ps) | Cheng PBE-corrected table ~ -10.0 |
| G_fcc - G_bcc | 3000, 200 GPa | -8.4 +- 2.4 (2 ps) | Cheng -10.5, Wilson -9.0 +- 2.7; bcc G still drifting up |
| G_bcc / G_fcc | 5000 | 908.9 +- 2.7 / 898.5 +- 2.3 | |
| pure H, N=500, dt 1 fs | 5000 | 243.9 +- 2.0 per atom | 250-atom run blew up in one lambda at dt 1 (T_kin millions K); needs dt 0.5 rerun (tag H_400GPa_5000K) |
| H2.25O fcc | 5000 | ~859 per atom after 0.25 ps (run in progress) | 27 extra H (108 O + 243 H) |
| H2.25O bcc | 5000 | preview 882.9 -> ~872 after the -11 meV preview offset | lambda 0 failed (GPU OOM); use `collect.py ... --copy 0:1` (weight 3e-7) or `ONLYNODES=0` rerun |
Preliminary mixing free energy per added H (H2O crystal + H fluid -> H2.25O crystal): fcc ~ +144 +- 47 meV, bcc ~ +185 +- 55 meV (weakly unfavourable, ~0.3-0.4 kT); P*dV_excess from the hull EOS is +340..810 meV/H at 5000 K, ~0 at 7000 K, negative at 9000 K (docs/mixing_pv.py, docs/data/mixing_pv.csv).
Open/unsure: dt 0.5 fs vs 1.0 fs comparison, second quadrature (power 3 or 6), finite size, MLP-vs-DFT (FEP) correction (Cheng's own MLP->PBE shift of dG_fcc-bcc is only 0.8-1.4 meV/atom at 200-400 GPa), nuclear quantum effects ignored.

Tools written this session (all in mlp_pipeline/, see their docstrings): `tdi/run_real.py` (phases bcc, fcc, H, liquid, insert; modes --stage0, --node K; restartable; stability guard; thermo CSV, snapshots), `young_tdi_par.sh` (launcher: lambda values as processes on one GPU; env T P HPER NGHOST NATOMS DT PS EQUIL NPAR K ONLYNODES FRESH TAG; default k scaled, no lambda=1 stage), `young_tdi_scan.sbatch` (pressure/T scan array), `young_tdi_hull.sbatch` (liquid convex hull at 400 GPa, 7000/9000 K), `young_melt_scan.sbatch` + `21_melt_summary.py` (NVT stability of bcc/fcc vs T; superheating caveat), `tdi/live.py` (dashboard), `tdi/plot_tdi.py`, `tdi/collect.py` (--truncate, --copy), `tdi/summarize.py` (table + fcc-bcc pairs + liquid hull mixing dG), `tdi/uq_check.py` (GRACE extrapolation grade gamma on saved snapshots; GRACE_UQ is OFF in production), `18b_inventory_summary.py` (run on Young/Perlmutter and commit docs/data/inventory_*.csv: not done yet).
New code paths NOT yet run with GRACE: `--phase insert` (ghost-proton chemical potential: U=(1-lam) U_host + lam U_full, mu_H = TDI integral + kT[ln((N+n)!/N!) - n ln V] + 3nkT ln Lambda, per added H, no PV; compare with G per atom of pure H), `--phase liquid`, melt scan, hull sbatch, stability guard.

Immediate to-do for the next session (the user runs commands on Young interactive jobs / sbatch and pastes output):
1. 5-min insertion test: `TAG=test NGHOST=1 T=5000 P=400 NODES=4 NPAR=4 DT=0.5 EQUIL=0.1 PS=0.25 CHUNK=0.25 bash mlp_pipeline/young_tdi_par.sh insert`; then n=1,2,3 production (est. ~30 min each).
2. Finish fcc/bcc H2.25O at 5000 K (dt 0.5), rerun pure H (250 atoms) at dt 0.5, compute dG_mix per added H with tdi/summarize.py; compare with the insertion mu_H.
3. sbatch the scans: `young_tdi_scan.sbatch` (3000 K 300/500 GPa, 5000 K 400 GPa), `young_tdi_hull.sbatch`, `young_melt_scan.sbatch`; run `git pull --no-rebase --no-edit` first on Young (merge commits happen there; plain `git pull` complains about divergent branches).
4. Liquid H2O G at 5000/6000 K vs solid G as a melting criterion; check whether 5000 K at 400 GPa solid is stable (Cheng: Tm ~4.6-4.8 kK at 200 GPa, higher at 400).
5. Commit the inventory summaries and TDI results (`summarize.py -o docs/data/tdi_results.csv`) from Young to the repo; runs/ is on Lustre scratch (not in git, may be purged).
6. Still open from before: fine-tune sweep (young_ft_sweep.sbatch: ft_m4/ft_m3/ft_l3/ft_m5), phase-labelling thresholds on Perlmutter, DFT single points on MLP configurations for the FEP correction.

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

## Session 2 additions (2026-10-26 evening)
Results: fcc H2.25O 5000 K 400 GPa dt 0.5: G = 860.2 +- 2.5 meV/atom (1.5 ps); pure H N=250 dt 0.5: 235.8 +- 1.8 (GL value; the PREVIEW line is a different estimator); mixing per added H with fcc H2O (898.5, still dt 1 fs: rerun!): +165 +- 43 meV (0.38 kT). Insertion 4-lambda test (0.25 ps): mu_H = -111 +- 167 meV (harness test only). Insertion n=1 production (12 lambda values, PS=4) was running on Young.
New: young_tdi_insert.sbatch (array over n), NREP=2/3 for 32/54-O cells (tag _rep<n>), summarize.py prints crystal and insertion mixing per added H and pairs fcc/bcc only for equal cell size and prefers equal dt, restart guard (dt, lambda, T, natoms in the node state files), live.py shows the pressure of the highest lambda.
Known weak points (not fixed): block errors at short times (blocks of ~50 fs at 0.25 ps) are probably optimistic; with K given the MLP pressure at the TDI volume is only seen in the highest lambda value (G error ~ (dP)^2 V/2B, second order); insertion costs two MLP evaluations per step.

### Working rules learned this session (add to the agreements above)
- Command blocks for the user: NEVER pipe a long run through `| tail -N` (it prints nothing until the script ends, and the user thinks it hung). Use `... 2>&1 | tee runs/<name>.out | grep --line-buffered -v "_handle_getgr"`; the Slurm `_handle_getgr` lines are harmless noise every minute.
- Ctrl-C on a launcher does NOT stop the lambda processes (background xargs ignores SIGINT): check `ps -u $USER -o pid,etime,args | grep run_real.py` before relaunching (a second launch writes to the same state files). They keep running and writing logs; run `collect.py` / `summarize.py` by hand when they end.
- On Young the login python (3.9, user-site pandas on Lustre) is slow and fragile: use `~/grace-venv/bin/python` for `summarize.py` / `collect.py`.
- The launcher now runs from a code snapshot (runs/code/<time>_<pid>) as one bash function, so `git pull` during a run is safe. Runs started before commit efca126 use the live files (a pull only changes python for processes started later).
- Small cells: `NREP=2` (fcc, 32 O) / `NREP=3` (bcc, 54 O), tags `_rep2`/`_rep3`. Result so far: fcc 32 O, 400 GPa 3000 K, dt 0.5, 3 ps: G = 1954.8 +- 2.7 meV/atom, still drifting down ~4 meV over the last 1.25 ps; absolute G is 25 meV/atom below the 108 O cell (cell size, dt and run length not separated). bcc 54 O running (pid group started 16:23). Compare only fcc - bcc with -24.7 +- 1.5 (108/128 O).
- Compositions between H2O and H2.25O: HPER = (216+n)/108 (n=2: 2.018519, 5: 2.046296, 11: 2.101852); volumes by ideal mixing of the DFT H2O and pure-H EOS when the inventory has no DFT EOS (run_real.solid_vpa). Direct-G error per added H is about 1150 meV / n at 1.5 ps (n=27: 43; n=11: 105; n=5: 230; n=2: 575): small excess is not resolvable by the direct route, insertion is the efficient route (error does not scale with the cell).
- Next candidates: 400 GPa 6000/6500 K (liquid hull mixing becomes favourable between 5000 and 7000 K; P*Vexcess +809 meV/H at 5000 K, -76 at 7000 K) after `young_melt_scan.sbatch` (PLIST=400 TLIST="5500 6000 6500 7000", fcc task); alternative 500 GPa 5000 K (P*Vexcess 338 meV/H). Units of the hull dG columns (header says eV/atom, values look like meV/atom) are unverified.

### Quadrature: add lambda values instead of repeating (2026-10-26 evening)
`analyze.gauss_kronrod`, `analyze.rule_nodes(A)`; run_real `--rule gk --ngauss n --nodes 2n+1` (launcher env RULE, NGAUSS, NODES, ONLYNODES): an n-value Gauss-Legendre run keeps its n lambda values and gets n+1 new ones (indices n..2n); all 2n+1 are integrated and `quad_diff_meV` in summarize.py (Kronrod minus Gauss) estimates the quadrature error. Unit checks: exact to degree 3n+1 for n = 4, 8, 10; the first n lambda values equal the GL(n) ones; on a synthetic steep integrand GL8 was off by 0.43 eV per cell and K17 - G8 gave 0.43 eV. summarize.py shows a pending extension as "Gauss n only". Untested on Young. Not the cause of the 8 vs 12 lambda difference at 400 GPa 6500 K (about 8 meV/atom): that is larger than the synthetic quadrature error (about 1.3 meV/atom per 324 atoms), so sampling noise / drift is the likelier source.

### Free NVT (no springs) stability check at the TDI volumes, 500 GPa, 6 ps after 1 ps equilibration (2026-10-26 night; 17_eos_md.py --hper, runs/melt_tdi)
fcc H2O 6000 K: fcc, delta_L 0.131, D_O 0, P 506.5 GPa. fcc H2.25O 6000 K: fcc, delta_L 0.135, D_O 0, P 493.8. fcc H2O 7000 K: fcc, delta_L 0.156, D_O 0, P 512.2 (solid for 6 ps; superheating not excluded). **fcc H2.25O 7000 K: MELTED** (label 'other', delta_L 1.82, D_O 2e-4 cm2/s, O MSD 52 A^2). Consequences: the 7000 K H2.25O TDI (and its mixing value -108 +- 46 meV/H) describes a spring-stabilised metastable crystal: drop it; the 6000 K solid mixing numbers stand (+45.5 +- 40 all samples, about -22 second half only: not converged); melting temperature of H2.25O at 500 GPa is between 6000 and 7000 K (H lowers it), that of H2O above 7000 K or superheated. The tdi/health.py flag threshold (ratio 0.14) is too tight: solid runs sit at 0.13-0.16, the melted one at 1.8; use the free NVT run and the liquid G for melting. Pending: 6500 K bracket runs, liquid TDI at 500 GPa (young_tdi_hull.sbatch, queued) for G_liquid vs G_solid, convergence runs with EQUIL=1.0.
