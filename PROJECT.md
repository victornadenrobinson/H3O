# H3O: benchmarking and fine-tuning MLPs on extreme-condition H2O / H mixtures

Living project plan + data manifest. Last updated 2026-10-25. Branch: `claude/eloquent-ramanujan-2tjx2p`.

## Goal
Benchmark foundation MLPs (MACE, MatterSim, GRACE; optionally CHGNet/SevenNet) against VASP AIMD data for H2O-H mixtures
(H2O, H2.25O, H2.5O, H2.75O, H3O, H5O, H7O, H2) at ~200-1400 GPa and 3000-13000 K, then fine-tune the best candidates
(force/stress-weighted) and check EOS/stability. Compute: NERSC Perlmutter (data, extraction), UCL Young A100 (MLP runs).

## Decisions and conventions (all verified on the data unless marked)
| topic | decision | evidence |
|---|---|---|
| Which runs | only OUTCARs under `~/work/tdi/H3O` and `~/work/quartz-work/tdi/h3o`; all `lambda_*` TDI runs excluded (1.0 = DFT, 0.0 = pair potential, others = mixed) | user rule |
| Composition | from OUTCAR ion counts (POSCAR/CONTCAR fallback); snapped to nearest known H:O ratio within 4% | 18.7k runs parsed |
| Pressure | OUTCAR `external pressure` **+ N kB T / V** (ions' ideal-gas term) = eos.dat / hull CSV convention | fit vs 238 eos.dat lines: slope 1.001, intercept -0.1 GPa, resid std 0.5 GPa |
| NVT averaging | run pressure = mean over 2nd half of the run | all runs NVT |
| Temperature | `TEBEG`, else folder name, else MD mean. VASP prints `TEBEG=******` and `(temperature13378 K)` (no space) at >=10000 K | folder T vs measured T agree within ~1% (11000/13000 K) |
| Energy label | VASP **free energy (TOTEN)** as `energy`; E(sigma->0) kept as `energy_sigma0` | VASP forces are derivatives of F |
| Stress label | ASE 3x3 virial (eV/A^3), excludes ions' kinetic term (right for MLP training); add N kB T/V when comparing MLP-MD pressure with the EOS | stress/pressure check: median 0 GPa, 95% < 9 GPa |
| Production only | composition whitelist (no pure O, no H-poor), path exclusions in `mlp_pipeline/exclude_paths.txt` (kp* tests, pure-o2, too-small, broken, bad*), `--min-box` 9 A | user rule: no test/convergence systems in training |
| Dataset tiers | nested s (1 run x 1 frame / state) < m (3x3) < l (6x5); state = (composition, T bin, 100 GPa bin) | checker: s in m in l |
| Split | 80/20 grouped by run directory | no shared run dirs |
| GRACE model | NOT the small `GRACE-2L-OMAT` we first tried (deprecated, fp64, blew up); candidates `GRACE-2L-SMAX-OMAT-medium/large`, `GRACE-2L-OMAT-large-ft-E`, 1L/3L variants | gracemaker docs (user pasted) |
| Not verified | what energy convention foundation models' own training sets used; GRACE/MatterSim fine-tune APIs | see envs/README.md |

## Data status (2026-10-25)
Inventory `runs/mined_h3o.csv`: 17,851 OUTCARs scanned; 10,041 non-lambda; 10,025 H/O-only; T up to 13000 K.
Hull EOS targets (5000/7000/9000 K isotherms): **201 / 201** have mined data within +-25 GPa, +-250 K.
Cross-check at equal density vs hull EOS (5,447 runs): median |dP| 1.0 GPa; all compositions within ~0.5 GPa except
H2O at 5000/7000 K (-4.2 / -8.9 GPa; fcc-solid and small-cell runs sit on another branch than the bcc-128 hull curve).

Bundle `H3O_bundle_20261025.tar.gz` (73.4 MB, sha256 `de289910ae41af783b280d57bd3bd16d8052e31e2de3bb8c4a01487d46602b1c`), sent to Young `/lustre/scratch/mmm0037/H3O/runs/`:
| file | frames |
|---|---|
| snapshots_s / train_s / valid_s | 230 / 184 / 46 |
| snapshots_m / train_m / valid_m | 1091 / 874 / 217 |
| snapshots_l / train_l / valid_l | 1583 / 1266 / 317 |
Frames at 11000 K / 13000 K: s 11/12, m 37/39, l 38/41 (only H2O, H3O, H2 there). Large forces: 5 frames > 50 eV/A (max 62.6).
Authoritative per-file counts and checksums: `runs/MANIFEST.json` (written by `11_bundle.py`).

## Pipeline (mlp_pipeline/)
`build_all.sh` runs 01 -> 02 -> 10 -> 03 -> 10 -> 06 -> 11 (needs `runs/mined_h3o.csv`). Outputs go to `runs/` (gitignored, `H3O_RUNS` overrides).
| script | purpose |
|---|---|
| 00_mine_outcars.py | mine T/P/V/composition from OUTCARs; `--scan DIR... --known old.csv` mines only new files |
| 01_bin_inventory.py | bin by composition/P/T, hull-target coverage, kinetic pressure correction, lambda exclusion |
| 02_extract_snapshots.py | nested s/m/l extxyz (free-energy labels), per-run cache in `runs/cache` |
| 03_split.py | grouped 80/20 split |
| 04_benchmark.py | zero-shot / fine-tuned model vs DFT (`mace[:name]`, `mattersim`, `grace[:name]`, `mace-ft:path`) |
| 05_finetune*.sbatch | MACE fine-tune (Perlmutter / Young); placeholders for partition/account |
| 06_plot_coverage.py, docs/make_figures.py | coverage plots (real CSV / pasted tables) |
| 07/08_crosscheck_*.py | mined data vs hull EOS and vs eos.dat |
| 09_label_phase.py | MSD-based solid/superionic/liquid label (not yet run on real data) |
| 10_check_snapshots.py | integrity/format/consistency check (parallel) |
| 11_bundle.py, send_bundle.sh | bundle + manifest; upload to Young via ssh+cat with sha256 check |
| smoke_test.py/.sbatch | one snapshot through each foundation model |

## Environments
Perlmutter: `nersc-python` (pandas, ase). Young (Slurm): `~/work/envs/mace-ng` (mace-torch 0.3.16, torch 2.6.0+cu126),
`~/work/envs/mattersim-env` (mattersim 1.2.3), `~/grace-venv` (tensorpotential 0.6.1). Details in `envs/README.md`.

## Itinerary
- [x] Mine and validate inventory; EOS/eos.dat cross-checks; pressure/temperature/energy conventions fixed
- [x] Build and verify nested datasets s/m/l with 11000-13000 K data; bundle sent to Young
- [x] Smoke test: one snapshot through MACE, MatterSim, GRACE on a Young GPU (all load and run)
- [~] Zero-shot benchmark on valid_m: done for MatterSim, MACE-OMAT/MP-0, GRACE (6 variants); MACE mpa-0/mh-1 pending
- [ ] Choose models: criteria = pressure error at 400-1400 GPa, 11000/13000 K error, learning curve s/m/l, MD stability
- [ ] Fine-tune (MACE first, MatterSim second, GRACE third); E0 re-estimation; consider multihead replay
- [ ] Short MD with fine-tuned models; compare EOS (add N kB T/V) and RDFs against DFT
- [ ] Expand data: more 11000/13000 K compositions (H2.25O, H5O, H7O have none above 9000 K); run `09_label_phase.py` to tag solid/liquid

## Zero-shot results and model choice (2026-10-25, Young A100 40 GB; 30 frames/model, cells >= 8.5 A, 1-3 frames per state; indicative)
| model | force RMSE (eV/A) | pressure error (GPa) | memory (inference, 40 GB) |
|---|---|---|---|
| MatterSim 5M | 0.9-1.8 | -3..-5 (<=9000 K) | OOM at 432+ atoms |
| **GRACE-2L-SMAX-OMAT-medium** | 1.0-1.6 (H 0.55-0.70) | +49,+16,+20,+15,+5,-1.5 (3000..13000 K) | 729 atoms 10.1 GiB, 0.17 GiB/Mtriplet |
| GRACE-2L-SMAX-OMAT-large | 1.1-1.8 | +46,+11,+19,+6,-9,-5 | 729 atoms 10.4 GiB |
| MatterSim 1M | 1.1-2.1 | -34..+19 | OOM at 512+ |
| GRACE-1L-SMAX-OMAT-large | 1.7-2.5 | mixed | (pending) |
| GRACE-3L-OMAT-large | 1.9-3.5 | +120..+333 | - |
| MACE-OMAT-0 | 3.9-22 | hundreds | OOM at 512 (fp64); fp32 fits 512, not 729 |
| GRACE-2L-OMAT-large-ft-E | 6-13 | thousands | - |
| MACE MP-0, GRACE-2L-OMAT (small) | blow-up | 1e6+ | - |

Pattern: SMAX-trained models stay finite at 300-1400 GPa; OMat24-only models fail (MACE-OMAT-0, GRACE OMAT variants). GRACE needs 7-20x less memory per triplet than MACE/MatterSim.
Decision: first fine-tune = GRACE-2L-SMAX-OMAT-medium (large as second). MatterSim 5M is the best zero-shot but cannot hold production cells for training here.
Open: MACE candidates (medium-mpa-0, mh-1) and MatterSim fine-tune feasibility on an 80 GB GPU not yet benchmarked; real training-memory test for GRACE pending (inference x3 is only an estimate).

## Open questions
1. H2O offset at 5000/7000 K: tag branch (fcc solid, kp222 small cells) via `09_label_phase.py` and keep as separate config types?
2. Large-force frames (> 50 eV/A): clip or down-weight in the loss?
3. Include or exclude the pure-O2 state and the h6o/h4o/h5.5o minor states in training?
4. Young: partition/account/gres names for the sbatch templates; GRACE GPU stack.
5. Scratch purge: keep a permanent copy of `runs/` (CFS/HPSS on NERSC, or UCL storage).
