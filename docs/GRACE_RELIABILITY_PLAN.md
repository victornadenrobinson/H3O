# Which GRACE results can be trusted? Review of the repo state and a plan (2026-11-01)

Written from a read of `main` plus the branches that `main` does not contain, and from arithmetic on the tables
they hold. Nothing was re-run: no `runs/`, no model and no VASP are available in the cloud checkout, so every
number below is a user-pasted transcription that lives in the repo. All energies in meV or eV as marked;
"DFT-GRACE" means E_DFT - E_GRACE.

## 1. What is new since `main` (branch map)

`main` (= `claude/eloquent-hopper-kfjdei`, 2026-10-31) is the Young/GRACE work. The Codex ("gpt") and Perlmutter
work is **not on main**; it is on these branches, which are nested, so `feature/pwidom-checkpoint-resume` is the
superset (41 commits ahead of main, 0 behind, 14k added lines, 15 new test files):

| branch | what it adds |
|---|---|
| `gpt/perlmutter-pwidom` | 4-GPU Perlmutter pwidom launcher/status/benchmarks, composition scans (`--extra-h`), in-place extensions, EOS inspection from the DFT inventory |
| `gpt/pwidom-diagnostics` | EOS automation, RDF diagnostics |
| `audit/pwidom-validation` | code audit of pwidom (`docs/PWIDOM_AUDIT.md`), BAR convergence diagnostics, EOS/inventory validation |
| `feature/pwidom-checkpoint-resume` | checkpoint/resume, time-resolved hulls, **VASP backends: ordinary-MD check, saved-pair check, VASP-driven PWidom MD (729-H reservoirs)**, `docs/status/GRACE_DFT_AUDIT_2026-11-01.md` with every pasted number |
| `claude/jolly-mccarthy-0zrnfe`, `victornadenrobinson-fictional-goggles` | small, also not on main by SHA: proxy refit `--select/--shard`, ARCHER2/Perlmutter label scripts, `proxy_smoke.py`, `docs/HANDOFF_PROXYFIT.md` (candidate proxy: held-out gap 1.94 -> 1.57 kT, Kish 0.084 -> 0.191, overlap diagnostics worse) |

Scientific content of the new work:

1. **Perlmutter GRACE pwidom scans** over composition at 300/400/500/600 GPa and 2000-5000 K (44 finished results).
2. **A GRACE-vs-VASP validation campaign at 300 GPa / 3000 K** (this is the "DFT and GRACE do not agree" result): 7 original-label
   reproductions, 40 ordinary GRACE-MD configurations, 20 saved insertion/deletion pairs (40 VASP points), a Gamma vs 2x2x2 test on pure H,
   and GRACE on 40 + 24 original labels.
3. **Three production DFT PWidom jobs** (60649379/80/81, 500/400/600 GPa, 5000 K, 729-H reservoirs). By the last paste they were still in
   host equilibration (checkpoint 242-528 of 3000 host steps, 0/25 frames).

## 2. What the numbers say

### 2.1 DFT vs GRACE on the same configurations (300 GPa, 3000 K, GRACE-MD frames)

VASP reproduces the original training labels to <1 meV per cell (7 rows), so the DFT setup is not the problem.
Ten frames per host, raw DFT-GRACE:

| host | mean (eV/cell) | sd (eV) | force RMSE (eV/A) | stress RMSE (GPa) |
|---|---|---|---|---|
| H2O (108 O, 216 H) | 7.63 | 0.73 | 0.41 | 0.85 |
| H2.083O (+9 H) | 26.30 | 3.60 | 0.65 | 3.6 |
| H2.25O (+27 H) | 48.88 | 7.86 | 1.16 | 8.3 |
| pure H (250) | 21.18 | 1.33 | 0.49 | 6.3 |

Elemental energy shifts are a gauge (constant per O and per H), so the meaningful quantity is the composition dependence that survives them:
c(n) = [mean(host) - mean(H2O) - n_add * mean(pure H)/250] / n_add.

* H2.083O: **c = 1.99 eV per added H** (bootstrap sd 0.12; 7.7 kT at 3000 K)
* H2.25O: **c = 1.44 eV per added H** (sd 0.09; 5.6 kT)

Independent confirmation from the 20 saved pairs, which share nothing with the ordinary frames except the state: DFT-GRACE of the
one-H addition energy is **+1.84 eV/H in water hosts (n=16, sd 0.68) and +0.08 eV/H in pure H (n=4, sd 0.26)**; difference 1.76 +- 0.21 eV.
All 20 GRACE dU values replay to <0.07 meV, so this is not a batching/caching bug. The error lives in the water host, not the H reservoir.

For scale: the GRACE dilute mu_H - G_H that the whole solubility story rests on is -0.3 ... -0.55 eV, and `SOLUBILITY_VALIDATION.md` found
the sign survives a shift of mu0 only up to +0.25 ... +0.49 eV. An error of 1.4-2 eV per H, if it applied at the TDI states, would reverse the sign.

Why this cannot yet be applied to the headline numbers:

* **State mismatch.** The test is 300 GPa / 3000 K. The TDI/insertion results are 400-500 GPa and 5000-7000 K. The
  existing held-out benchmark (shifted-MAE, `valid_m`) is clearly worse cold: 59.5 / 33.7 / 14.1 / 14.3 / 7.9 meV/atom at 2000 / 3000 / 5000 / 7000 / 9000 K.
* **Configuration source.** The 40 frames are GRACE-generated. Their H2.25O force RMSE (1.16 eV/A) is worse than anything in the
  DFT-generated benchmark (H-rich 3000 K: 0.63-0.74; H2.25O 5000-9000 K: 0.71-0.96). That points at GRACE-MD wandering into regions the
  fine-tune never saw, not only at a smooth energy error. Needs the diagnosis in plan step 2.
* **Selection.** Pair selection is deliberately the largest weights / residuals; use pairs as a detector, not a correction.
* **FEP is hopeless at this width.** The per-cell sd of DFT-GRACE is 0.7-7.9 eV against kT = 0.26-0.43 eV; reweighting GRACE ensembles to DFT
  is not possible. Only retraining or DFT sampling can fix it. The earlier bias-proxy estimate (+40 +- 15 meV/H, DFT-generated frames, >=4000 K)
  and this one (1.4-2 eV/H, GRACE-generated frames, 3000 K) differ by ~35x; the data cannot yet say whether that is temperature, pressure,
  or configuration source.
* **k-points.** Gamma-only labels in ~9-10 A cells are what GRACE was trained on. The pure-H Gamma vs 2x2x2 test shifts dU by -150 and -403 meV/H
  and (2 pairs only) brings VASP-GRACE from +204/+371 to +53/-32 meV: GRACE's pure-H reservoir looks closer to converged DFT than to
  Gamma DFT. What "DFT truth" means (Gamma 324-atom as in training, or k-converged) is a decision for you, and it moves c by an unknown ~0.3 eV.

### 2.2 The GRACE-only pwidom results are not converged, and their error bars are too small

* **Same state, more sampling, 500 GPa / 3000 K.** Original scan -> extension1 (which contains the original frames), n = 0 / 0.083 / 0.25:
  -1146.5 -> -759.3 (+387, 3.9 sigma), -738.5 -> -132.7 (+606, 4.8 sigma), +185.7 -> +854.4 (+669, 4.9 sigma). Estimators are unbiased for any
  proxy, and the proxy changed between the two runs, so this is sampling error 4-5x the quoted error. Kish fractions 0.004-0.1.
* **mu(n) must not decrease in a single stable phase.** In the selected 500 GPa / 3000 K data it drops by 223 meV between 9 and 11 extra H
  (2.2 sigma, and neighbouring slopes swing +13400 / -12000 / +10500 meV per H/O). Also 300/3000 and 300/4000 drop between 22 and 27 H (1.1 sigma).
  Flags: 30 of 37 host results carry low-Kish and/or forward-reverse-disagreement flags. The "hull minima" (-5 ... -33 meV/atom) are built from these.
* **Young light scans vs the insertion TDI at the same state** (`pwidom_scan_2026-10-29.csv` vs `insertion_regsol_fit.csv`): n = 0 values are
  200-370 meV more negative than the TDI mu0 in all four unconverged cases (400/5000: -805 vs -439; 500/5000: -875 vs -528; 400/6000: -546 vs -318),
  every one with Widom - deletion split of -300 ... -840 meV. The **single** run with converged diagnostics (500/6000, K 4000, Kish 0.17, W-D -66 meV)
  agrees: -549 +- 93 vs TDI -552 +- 35 (uncorrected basis). That agreement is the only direct cross-validation of the insertion TDI against another route.
* **Composition dependence disagrees between routes by ~0.45 eV/H.** At 400/5000 and 500/5000 the crystal TDI gives an average over 0 -> 0.25 added
  H/O of +192 and +92 meV/H. The pwidom differential values are -805 -> +276 and -875 -> +109; their mean over the range is -264 / -383 for a straight
  line, and no monotone curve through those end points can reach +92 unless mu jumps by ~1 eV at n -> 0+. So at least one route is off by ~0.4-0.5 eV/H
  at 400-500 GPa, 5000 K, **within GRACE alone**. (Caveats: the pwidom runs were light 0.4 ps settings; the 400/5000 crystal pairs a dt 0.5 crystal
  with a dt 1.0 H2O, CURRENT_ISSUES D5.)

### 2.3 What the earlier validated pieces still support

* Same-composition free-energy differences: G_fcc - G_bcc = -24.7 +- 1.5 (3000 K, 400 GPa; Cheng -22.1), -10.4 +- 3.5 (5000 K; Cheng ~-10.0), -8.4 +- 2.4 (200 GPa; Cheng -10.5).
* EOS/pressure from the fine-tune (within 3 GPa, 0.35 % in volume of DFT; held-out signed pressure error -0.7 GPa, N=40).
* Bookkeeping of the insertion/pwidom machinery (toy T0 exact; Widom/TDI agreement at 500/6000).
These are *internal-consistency or cancelling* quantities. None tests the energy of an extra H in ice.

## 3. Reliability tiers (current assessment)

| tier | results | why |
|---|---|---|
| **A: usable now** | EOS, pressures; fcc-bcc and other same-composition differences; the TDI/pwidom machinery itself | validated against DFT EOS / literature / exact toy checks; H-count errors cancel |
| **B: GRACE-self-consistent, DFT accuracy unknown** | dilute mu0 (-0.3 ... -0.55 eV) at 400-500 GPa, 5000-7000 K from insertion TDI; hosts that stay solid | pwidom agrees at 500/6000 only; benchmark shifted MAE ~14 meV/atom at these T; bias proxy +40 +- 15 meV/H; but no paired VASP at these states and a 1.4-2 eV/H signal at 300/3000 |
| **B-: shape of G(n), n*, T-x diagram, hull minima of the TDI route** | n* = 0.08-0.11 | s from crystal at n=0.25 and n=1,2,4 only; pwidom route disagrees on the composition dependence by ~0.45 eV/H |
| **C: do not use** | every H-rich result at <=3000 K (Perlmutter pwidom 2000-3000 K, all hull minima); light Young scans other than 500/6000; items D1, D2, D4, D7 in CURRENT_ISSUES (overwritten H references, uncorrected insertion rows, crystals that melt in free NVT, frozen 500 GPa liquids); pure-H runs with min d(H-H) < 0.25 A | model error >= 1.4 eV/H measured there, or sampling not converged by 4-5 sigma, or known-invalid inputs |

## 4. Plan, ordered by cost and by what each result decides

**Step 0 (free).** Merge `feature/pwidom-checkpoint-resume` into `main` (fast-forward, no conflicts: 0 behind) so Young, Perlmutter and cloud sessions see one history; I have not done it. Quarantine tier C in the summary CSVs (flag, do not delete).

**Step 1 (~1 h on the existing VASP chain): repeat the DFT check where the headline numbers live.** Same protocol as the 300/3000 test (`perlmutter_vasp_check.sh normal --run <pwidom run dir>` for ordinary frames, `queue`/`prep-queue` for pairs, then `run` and `analyze`), at
(a) 500 GPa / 5000 K and (b) 400 GPa / 5000 K or 6000 K, hosts H2O, H2.083O, H2.25O, pure H, using existing GRACE hosts (the `pwidom` frame caches or TDI trajectories), 10 frames/host + 5 pairs/host.
Report c(n) with a frame-block error. Decision rule (from the sensitivity in SOLUBILITY_VALIDATION: a 100 meV shift moves n* by >20 %, |c| > |mu0| flips the sign):
* |c| <= 50 meV/H: tier B stands; the model is fine at the TDI states and the 3000 K failure is a low-T problem.
* 50-150 meV/H: GRACE numbers need a DFT correction (retrain, step 3) before any quantitative claim.
* > 150 meV/H: tier B is void; skip to step 3.
Use n_add = 27 in the ordinary frames (it amplifies a per-H offset by averaging over 27 H); pairs are a detector only.

**Step 2 (~0 DFT): decide whether it is model error or configuration drift.** Because the failing frames are GRACE-generated:
(a) evaluate GRACE on *DFT-generated* H2.25O / H2.5O / H2.75O / H3O frames at 2000-3000 K and 300-500 GPa (the inventory has 368 / 178 / 158 / 252 runs at 3000 K) and compute the same c(n) (`perlmutter_vasp_check.sh training --T 3000 --P 300 --pressure-window 100` + `baseline` already select and score original labels; restrict to H-rich compositions and keep composition resolution). Small c there with large c on GRACE-MD frames means GRACE-MD leaves the training distribution (active learning cures it); large on both means the fine-tune is deficient there (train ~ validation = underfit, per HANDOFF).
(b) structural comparison of the 40 GRACE frames vs DFT frames at the same state: min and mean H-H, O-H coordination, OH-/H2-like species, O-O RDF; and VASP forces on the worst frames (the numerical `force_stress.csv` for the 40 pairs was not pasted; get it).
(c) the 5000 K / 600 +- 100 GPa baseline (24 original labels, MAE 9.3 meV/atom, bias -3.7) shows the model is fine on DFT-generated frames at 5000 K; extend it with per-composition resolution (H2.25O only).

**Step 3 (only if step 1 or 2 says so): retrain.** Challenger with labelled GRACE-MD frames (ordinary + paired insertion/deletion, both members in one split), unfrozen or longer, energy weight raised (energy is 0.6 % of the loss), whole-run hold-out, acceptance = paired H-addition bias <= 50 meV/H and scatter <= 100 meV/H at the production states, plus unchanged EOS and fcc-bcc. Codex's `prepare` datasets already support this. Then rerun tier B quantities with the challenger; do not patch old numbers with measured offsets (host and reservoir ensembles differ, a mean residual is not a chemical potential).

**Step 4 (GRACE-only, can run now in parallel, ~1-2 GPU-days): fix the internal inconsistency of section 2.2.** At 500/5000 and 400/5000 run pwidom for n = 0, 5, 11, 27 with settings like the converged 500/6000 run (PS 1, K 4000; >= 4 independent seeds, not extensions of one trajectory) and require Kish > 0.1 and Widom ~ deletion; compare (i) pwidom differential curve integrated 0 -> 0.25, (ii) crystal TDI at H2.25O (same dt for host and reference), (iii) insertion TDI n = 8, 16, 27 (already specified in SOLUBILITY_VALIDATION step 2). Three routes, one model: they must agree within ~50 meV/H before any hull is drawn. Errors: use independent-seed scatter, not the BAR block error (it was 4-5x too small).

**Step 5: the DFT PWidom jobs (60649379/80/81).** At ~60-70 s per 729-H step and 30-50 s per water step the projected remaining MD is 50-115 h *before* trials against a 48 h allowance, so they will not finish in one allocation even with checkpoint/resume (~288 node-h reserved). Decide now whether to (i) resubmit with 2-3x allowance for the 500 GPa state only and truncate to n = 0 and n ~ 0.08, or (ii) rely on steps 1-3. They remain the only physical DFT mu_H; but a result at 5000 K with a Gamma-only 729-H reservoir needs the k-point decision first.

**Step 6: k-point decision (cheap, do with step 1).** Matched Gamma vs 2x2x2 VASP on 2-3 water-host pairs (324 atoms x 8 k-points, ~15 min each): is the training label itself composition-biased by k-point error? If yes, "DFT truth" and "training label" differ and retraining on Gamma labels teaches GRACE the k-point error.

## 5. What I did not verify

* All numbers are transcriptions in the repo; I recomputed the derived quantities above (c(n), pair statistics, convexity checks, route comparison) from the CSV/markdown tables, but not from logs or run directories.
* The 300 GPa / 3000 K and 500 GPa / 3000 K failures may be specific to low T; step 1 exists to find out. Section 2.1's 35x gap to the earlier proxy is a measured disagreement between two methods, not an explained one.
* No code was run or reviewed beyond reading docs; the pwidom audit (`docs/PWIDOM_AUDIT.md`) found the estimator maths correct but states that Kish fractions and 5 contiguous blocks do not establish decorrelation, which matches the under-stated errors found in 2.2.

## 6. Code review of `feature/pwidom-checkpoint-resume` (partial, 2026-11-01)

Read in full: `tdi/pwidom.py`, parts of `pwidom_checkpoint.py`, `pwidom_uncertainty.py`, `vasp_normal_check.py`. Not yet reviewed: `collect_pwidom_results.py`, `pwidom_formation.py` (hull integral), `vasp_backend.py`, `dft_pwidom.py`, `perlmutter_pwidom.py`, `perlmutter_vasp_check.py` (most of it).

**Checked and correct.** Insertion Rosenbluth algebra (selection with probability proportional to exp(-phi/kT) times W gives the uniform average); deletion weight W_t with K-1 fresh candidates and the real H; BAR argument convention (`bar(wf, -wr)` matches `toy_validate.bar`); ideal term kT ln((N_H+1)/V) + 3 kT ln Lambda; the pressure-correction formula; ASE stress excludes the ideal-gas term, so adding N kT/V by hand is not a double count.

**Finding 1 (sampling, probably significant): the pwidom thermostat is ~10x weaker than intended.** `pwidom.py:168`, `pwidom_checkpoint.py:90` (production path), `bench.py:88`, `toy_validate.py:38` use `Langevin(..., friction=0.01)`. ASE takes friction in inverse ASE time units, so this is 0.01 per 10.18 fs, i.e. a damping time of ~1 ps. The TDI engine (`run_real.py`, `engine.py`) uses `1/(tau*units.fs)` with tau = 100 fs (damping 100 fs). The intended value was almost certainly 0.01/fs. Consequence: a run started from a perfect lattice (fresh scans, pure H bcc start, `add_ghosts` starts) loses ~half its kinetic energy to potential energy in the first ~50 fs and the thermostat takes ~1 ps to restore it. Toy check (Cu, EMT, 108 atoms, nominal 1000 K, lattice start, 3 seeds), mean T after 0.1 / 0.2 / 0.4 / 1.0 ps: 509 / 566 / 639 / 786 K with the pwidom friction, against 695 / 883 / 978 / 990 K with 0.01/fs. The pwidom equilibration windows are 0.2-0.5 ps (the audit notes 0.2/0.4 ps were already questioned). So fresh-start hosts are sampled well below the nominal temperature and still drifting, which fits the observed non-convergence (4-5 sigma shifts on extension, non-monotone mu(n)). It also means the cached "3000 K" frames used by `vasp_normal_check.py` (second half of a fresh run) may be colder than 3000 K, which could contribute to the DFT-GRACE composition offsets: the 300/3000 DFT test needs a temperature/pressure-drift check on those frames before it is read as pure model error. The cached `P_host` arrays are the quickest diagnostic (a rising pressure trend over the frames = still heating). Extensions restart from sampled positions, so they suffer less. The TDI/insertion results use the proper friction and are not affected.
**FIXED 2026-11-03 (merged into claude/youthful-brown-4qfaeg):** `pwidom.thermostat_friction` defaults to a 100 fs damping time (`--friction-fs`), also `bench.py` and `toy_validate.py`; a checkpoint written with the old 0.01 is detected on resume and keeps it (warning), an explicit contradicting `--friction-fs/--legacy-friction` is refused (tests in tests/test_pwidom_checkpoint.py); new `--thermostat bussi` (stochastic velocity rescaling, exact checkpoint continuation tested) and a launcher `--seed` (every host used seed 3 before, so identical settings repeated the same trajectory). Original text: Fix: `friction=1/(100*units.fs)` in both places (note that changing it changes the checkpoint metadata, so old checkpoints will refuse to resume, by design), longer equilibration, and report the instantaneous kinetic T per frame in the frame cache.

**Finding 2 (statistics): the new uncertainty code is a delete-one-block jackknife over 5-frame blocks.** Its own docstring says it is conditional on the observed trajectory and does not correct for non-stationarity, missing tails or proxy bias. Consistent with the 4-5 sigma shifts seen in section 2.2: use independent seeds, not these errors, for any go/no-go decision.

## 7. Addendum: direct crystal TDI rows (from `docs/status/young_tdi_summary_received_2026-10-31.txt` on the feature branch)

* There is **no H2.125O** crystal run (a 108-O cell cannot hold 13.5 extra H). Direct crystals in the table: H2.25O (n=27) at 300-600 GPa / 4000-7000 K, H2.5O (n=54), and one dilute point, **H2.1111O (n=12) at 400 GPa / 5000 K: -190.9 +- 46.2 meV per added H** (pressure term 6.4).
* That n=12 point is a useful, previously unlisted check on the shape of G(n): the regular-solution fit (mu0 = -439, s pinned by n=27) predicts an average of -139 at n=12/108 (quadratic), -306 (cubic), +11 (p=1.5). Measured -191 +- 46 favours the quadratic shape (single state point; section 2.1 of the plan overstated "nothing measured between the ends" for 400 GPa / 5000 K).
* Replicates of the same state agree within their errors where the run is sound: 500/6000 H2.25O +33 +- 31, +41 +- 22, +56 +- 27, +76 +- 20; 400/5000 +198 +- 22, +224 +- 26, +252 +- 38.
* **Rows that are artefacts and should be quarantined:** every H2.5O row with a value of -1600 to -3700 meV/H (400/5000, 400/6000, 500/6000, 600/6000), and the 300 GPa 5000/6000 K H2.25O/H2.5O rows (-3100 ... -6700): thousands of meV cannot be physical; they match the known failure modes (melted crystal held by springs, overwritten or collapsed pure-H reference). Also melted in free NVT: 400/6000 and 400/6500 H2.25O, 500/7000.
* Crystal TDI does **not** claim favourable mixing at n=0.25 in most states: +200 ... +400 meV/H at 300-400 GPa, +40 ... +110 at 500-600 GPa and 5000 K, ~0 only at 600 GPa/6000 K (-36). The solubility claim comes from the negative dilute end (insertion) plus the curvature. If the 300/3000 DFT-GRACE sign (+1.4 ... +2 eV/H, DFT less favourable) carried over, the n=0.25 values would become more positive, and the dilute end could turn positive.

## 8. Further analysis

Free-energy analysis that used non-public calculations is kept in a private location and is not part of this repository.
