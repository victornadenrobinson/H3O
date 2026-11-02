# Model test suite: how a fine-tuned potential is judged before it is used

Written 2026-11-02, after the ft_m2 audit. The model looked good on one number (per-atom energy MAE after a per-element shift on ~30-185 frames) and was then used for free energies,
where the quantities that matter are per-cell energy differences, composition differences, and the thermostatted ensemble. This suite tests those things *before* any production run.
Every candidate model runs the whole suite; results are written as one JSON per test under `runs/suite/<model_tag>/` and aggregated into a scorecard that compares models side by side.
Thresholds below are provisional starting points (reasons given); change them in one place when the scorecard is built.

## Design rules
* **Independent tests, run in parallel.** One test = one command that writes `runs/suite/<model>/<test>.json` (`{name, model, metrics, thresholds, passed, notes}`). A Slurm array (GPU for MD tests, CPU nodes for the frame-scoring tests) or a local process pool runs them concurrently.
* **Held-out by construction.** Whole runs / states the model never saw; the sealed set (below) is never in any training tier.
* **Gauge-invariant and per-cell.** Composition differences and per-cell scatter, not shifted per-atom MAE.
* **Ensemble tests, not only frames.** Stability, drift, structure and dynamics under the thermostat used in production.
* **Report scatter, not only means.** sd of the DFT-model difference per cell decides whether free-energy perturbation is usable at all.

## Data the suite needs
| set | content | status |
|---|---|---|
| sealed frames | lambda = 1 DFT frames of the 500 GPa TDI campaigns held out of training: 5000 K `fcc-h2.25o/tdi-v1`, 5000 K ice host, 5000 K pure H, one 3000 K H2.25O campaign | exist (frame_batch scores them); must be excluded from training |
| held-out pool runs | whole runs per (comp, T, P) state, by-run split (valid tiers) | exist, small; to be enlarged |
| insertion / deletion sets | host + n H, n = -8 ... +54, labelled with VASP, hosts from valid tiers | being built (`build_insertion_set.py`, `prepare_points.py`) |
| paired chains | n -> n+1 on the same geometry, `--chain` | same |
| DFT-MD reference | structure/dynamics of DFT AIMD at the key states (RDFs, MSD, D_H) | from pool OUTCARs (to extract) |
| literature | Cheng 2021 / Wilson 2013 fcc-bcc and melting, Huang H3O static enthalpies | in docs/data |

## Tests
Columns: id, scenario, what it detects, tool (exists = already in the repo; write = to be written), provisional pass criterion, cost.

### A. Static accuracy on held-out DFT frames (CPU or GPU, minutes)
| id | test | tool | pass criterion (provisional) |
|---|---|---|---|
| A1 | state-resolved error map: per (comp, T, P) mean/sd of D = E_DFT - E_model per cell, force RMSE and relative error, pressure error | `tdi/frame_batch.py` + `04_benchmark.py` | per-cell sd(D) <= 1 eV at ~350 atoms in every H-rich state (the FEP-usable level); relative force error <= 8 %; |P error| <= 3 GPa mean |
| A2 | composition error c(n) per added H from ice / ice+H / pure-H triplets, mean-only and with var/2kT, over all (P, T, lattice) triplets | `tdi/frame_batch.py` (exists) | |c| <= 50 meV/H, both versions, at every triplet; spread over ice hosts <= 20 meV/H |
| A3 | tail frames: largest-D frames, what they have in common (min H-O, H-H distances, local coordination) | write | no cluster of frames with D > 5 sd |
| A4 | per-species energy offsets and their T dependence (gauge drift) | `tdi/frame_batch.py` (exists) | offset change with T <= 5 meV/atom across a 2000 K step for the same composition |
| A5 | learning curve: sd(D) and c against training fraction (25/50/100 %) | write (needs 3 trainings) | sd(D) still falling = data-limited; flat = settings-limited |

### B. Insertion / composition sensitivity (needs the labelled insertion set)
| id | test | tool | pass criterion |
|---|---|---|---|
| B1 | paired insertion energy DeltaE_ins = E(host+H) - E(host), DFT vs model, over n = 0 ... 54 | write | mean error <= 100 meV, sd <= 200 meV per H, no trend with n |
| B2 | n ladder: gauge-invariant c as a function of n (0 ... 54) at 3 (P, T) states | write | smooth, |c| <= 50 meV/H |
| B3 | deletion (n -> n-1) symmetric with insertion | write | |DeltaE_ins(n -> n+1) + DeltaE_del(n+1 -> n)| <= 20 meV in the model |
| B4 | reverse test: DFT on model-MD frames (the ensemble the model actually samples); mean offset minus var/2kT equals the forward estimate | write | forward and reverse c agree within 100 meV/H |
| B5 | H probe curves near host H/O (the existing DFT probe) | `tdi/dft_check.py prep-probe` (exists) | model dE(r) within 100 meV of DFT for r >= 0.7 A |

### C. Thermodynamic consistency (GPU, hours)
| id | test | tool | pass criterion |
|---|---|---|---|
| C1 | EOS by MD: P(V) for each composition, T = 3000/5000/7000/9000 K vs DFT hull EOS | `17_eos_md.py`, `18_plot_eos_md.py` (exist) | volume within 0.5 % / 3 GPa |
| C2 | fcc - bcc G difference at 400/500 GPa, 3000 K | `tdi/` TDI engine (exists) | within 3 meV/atom of Cheng/Wilson (-22 ... -26.5) |
| C3 | H2O melting line / liquid-solid G difference | `young_melt_*.sbatch` (exist) | within 300 K of Cheng 2021 |
| C4 | static formation enthalpies of H3O and partly filled variants vs DFT and Huang | `h3o_static.py` + `dft_check.py prep-static` (exist) | within 20 meV/atom of DFT (GRACE was 90-100 above) |
| C5 | n -> 0 chemical potential consistency across methods (insertion TDI, crystal TDI, pwidom) | existing engines | agree within 50 meV/H, independent seeds |

### D. Stability and dynamics (GPU, 20 ps runs)
| id | test | tool | pass criterion |
|---|---|---|---|
| D1 | long NVT stability at the corners of the domain (400/600 GPa x 3000/5000/9000 K x H2O, H2.25O, H3O): no blow-up, O lattice keeps its structure (Lindemann), no H2 clustering | `young_stability.sbatch` (exists, not yet run) + write summary | all runs finish; O Lindemann ratio within 20 % of DFT-MD |
| D2 | NVE energy drift, dt = 0.5 and 1 fs | write | drift <= 1 meV/atom/ps |
| D3 | time-step sensitivity of <P>, <E>, D_H (dt 0.25 / 0.5 / 1 / 2 fs) | write | <P> within 1 GPa between dt 0.5 and 1 fs |
| D4 | thermostat sensitivity: friction 1/(100 fs) against 1/(500 fs), instantaneous kinetic T | write | kinetic T within 2 % of target |
| D5 | structure: RDFs O-O, O-H, H-H and coordination vs DFT-MD | write | first-peak positions within 0.03 A, integrated difference <= 10 % |
| D6 | dynamics: H diffusion D_H and O MSD vs DFT-MD (DFT: 6-8e-4 cm^2/s) | write | within 30 % |
| D7 | minimum distances reached during MD (H-H, O-H, O-O) vs DFT-MD | write | no distance below the DFT-MD minimum by more than 0.1 A |

### E. Short range and extrapolation
| id | test | tool | pass criterion |
|---|---|---|---|
| E1 | isolated dimer curves H-H, O-H, O-O: finite, monotone core, no spurious wells at r < 0.6 A | `tdi/pair_scan.py` (exists) | no E(r) minimum below r = 0.6 A |
| E2 | out-of-distribution states (P < 200 or > 1450 GPa, T 13000 K, untrained compositions/T such as H2.5O above 3000 K) | write | D and forces degrade gracefully (<= 2x in-domain); no crash |
| E3 | committee disagreement of 3 seeds on the held-out frames (uncertainty proxy) vs actual error | write (needs 3 seeds) | rank correlation >= 0.5 so it can steer active learning |

### F. Numerical and practical
| id | test | pass criterion |
|---|---|---|
| F1 | repeatability: same frame twice, float32 vs float64 energies | difference <= 0.1 meV/atom |
| F2 | k-mesh sensitivity of the DFT labels themselves (Gamma vs 2x2x2 vs 3x3x3 on 20 H-rich frames) | composition-dependent difference <= 50 meV/H, else relabel |
| F3 | throughput and memory at 324 / 432 / 729 atoms | recorded, not pass/fail |

## Running it
`runs/suite/<model>/` collects one JSON per test; `suite_report.py` (to write) prints a scorecard with a column per model and a pass/fail line per test, plus a short list of the worst offenders (states and frames).
Parallel layout: A, B, F on CPU nodes (frame scoring is cheap there); C, D, E on GPU nodes as a Slurm array, one job per (test, state); the aggregator runs when the array finishes.
Policy: no model is used for TDI or Widom production until A1, A2, B1, D1 and E1 pass; the full suite runs for every candidate and for every later retrain.

## Order of implementation
1. A1/A2 (done as `frame_batch.py`) and the sealed-set definition.
2. B1/B2/B4 as soon as the first labelled insertion set exists.
3. D1/D2/D7 (stability, drift, minimum distances), which would also have caught problems with the old thermostat settings early.
4. The runner and `suite_report.py`, then the rest.
