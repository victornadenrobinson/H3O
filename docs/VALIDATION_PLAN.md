# Validation plan: does H mix into superionic (SI) H2O, or is GRACE telling us it does?

2026-11-01. Companion to `docs/GRACE_RELIABILITY_PLAN.md` (evidence and review). This file is the executable plan. It is built so that
**"no mixing in the SI region" is a possible, and cleanly reachable, outcome**, not a failure.

## What would make "no mixing" true, in numbers
GRACE: dilute mu_H(SI ice) - G_H(pure H) = mu0 = -0.3 ... -0.55 eV at 400-500 GPa, 5000-7000 K (insertion TDI), rising with H content (s ~ 2-6 eV per H/O, crystal average at 0.25 H/O: +0.04 ... +0.4 eV).
Let Delta = mu0(DFT) - mu0(GRACE). Dilute H dissolves only if Delta < |mu0|, i.e. below ~+0.3 ... +0.55 eV. The 300 GPa / 3000 K VASP test measured DFT-GRACE of +1.4 ... +2.0 eV per added H (ordinary frames, selected pairs: +1.8). If anything like that holds at the production states, Delta > |mu0| and there is no dilute solubility in SI ice.
What argues against reading that as the answer yet: one cold state; GRACE-generated frames that may have been under-thermalised (friction bug, `GRACE_RELIABILITY_PLAN.md` section 6); pairs chosen at the largest weights / residuals; Gamma-only labels; and two older DFT-based checks that point the other way or are far smaller (static H3O: GRACE dH_f 90-100 meV/atom *above* DFT; bias proxy at >=4000 K: +40 +- 15 meV/H).

## Stage 0: hygiene before any new compute (CPU, hours)
1. Merge `feature/pwidom-checkpoint-resume` into `main`; one history for Young, Perlmutter, cloud.
2. Fix `friction=0.01` -> `1/(100*units.fs)` in `pwidom.py`, `pwidom_checkpoint.py`, `bench.py`, `toy_validate.py`; store instantaneous kinetic T per frame in the frame cache. New runs only; old caches stay labelled "weak thermostat".
3. Quarantine flags in the summary CSVs: H2.5O rows < -1000 meV/H, 300 GPa 5000/6000 K crystal rows, rows melted in free NVT (400/6000, 400/6500, 500/7000), D1/D2/D4/D7.
4. Thermalisation check of what was already tested (login node, seconds): in the 300 GPa / 3000 K caches `pw_*_frames.npz`, plot `P_host` and `P_full` against frame index. A trend = the 40 VASP frames were not equilibrated. Report the per-host mean pressure of the 10 frames used against the 300 GPa target.
5. Round-trip test of the DFT writer on H-rich data: `perlmutter_vasp_check.sh training --T 3000 --P 300 --pressure-window 100`, restricted to H2.25O/H2.5O labels, through the same `write_dft` path, VASP on 5 of them, reproduce the stored TOTEN to < 1 meV/cell (the earlier 7 reproductions did not specifically cover H-rich cells).

Gate 0: if step 4 shows drift or a pressure far off target, the 300/3000 numbers are provisional and Stage 1 repeats them properly; if the writer fails the round trip, every DFT-GRACE number so far is void.

## Stage 1: unbiased repeat at the states where the headline numbers live (VASP ~ 1-2 h wall on 2 CPU nodes per state)
States: **(500 GPa, 5000 K)**, **(400 GPa, 5000 K)** (both have insertion TDI and crystal TDI), then (500 GPa, 6000 K) (the one converged pwidom agreement).
Hosts: H2O, H2.111O (n=12), H2.25O (n=27), pure H (729 atoms for production-like, 250 for comparison with Gamma training labels).
GRACE-MD frames from the fixed-friction thermostat, >= 1 ps equilibration, kinetic T and pressure logged, frames only after the pressure trend is flat.
For each host:
* 10 evenly spaced frames (as before): gives c(n) = [mean(DFT-GRACE)_host - mean(DFT-GRACE)_H2O - n_add*mean(DFT-GRACE)_pureH/N_H]/n_add. Use block errors; use n_add = 27 to average over many H.
* **30 pairs drawn at random from the GRACE insertion/deletion trials in proportion to their importance weights**, not top-weight/top-residual (needs a `random_weighted` mode in `pwidom_candidates.export_trials`; record the draw probabilities).
* Gamma vs 2x2x2 on 3 water-host pairs (mesh-check) to size the k-point question.
Cost: ~ (40 + 90 + 15) points x ~100 s / 8 parallel ~ 30-40 min per state.

Analysis (new, small): composition offset c(n) with errors; from the weighted random pairs the first-order and cumulant estimate Delta_ins = <delta>_w - var(delta)_w/(2kT) where delta = dU_DFT - dU_GRACE and w are the GRACE weights (this corrects the insertion step only: the host-ensemble term needs the c(n) frames); report the effective sample size of the weights.

Gate 1 (per state, using the larger of c(12), c(27) and Delta_ins):
* all of |c|, |Delta_ins| < 50 meV/H: GRACE is adequate there; go to Stage 3.
* 50-150: GRACE numbers need a correction; go to Stage 2 (retrain) before any quantitative claim.
* > 150, or Delta > |mu0|: GRACE cannot support SI mixing at that state. **Provisional conclusion: no dilute mixing in SI ice at that state** (strong evidence, still needs Stage 4 to be firm).
* at 300/3000 and the production states give different answers: it is a temperature/state effect, and the 3000 K and 2000 K Perlmutter scans stay quarantined.

## Stage 2: separate model error from configuration drift, then repair if it is real (GPU, VASP; days)
1. Score GRACE on DFT-generated H2.25O/H2.5O/H3O frames at 300-500 GPa and 2000-5000 K (`baseline`; no new VASP). Compare c(n) on DFT-generated against GRACE-generated frames: large on both = model deficiency; large only on GRACE-MD frames = GRACE leaves the training distribution.
2. Active-learning challenger: add the Stage 1 VASP frames (ordinary + pairs, both members in one split), higher energy weight, unfrozen weights, whole-run hold-out (`dataset` / `fitprep` already exist). Acceptance: paired H-addition bias <= 50 meV/H and scatter <= 100 meV/H on a hold-out; EOS, fcc-bcc (-24.7 +- 1.5 at 400 GPa 3000 K), Cheng comparison unchanged.
3. Stage 1 again with the challenger on fresh GRACE-MD frames from the challenger's own MD (active learning is not complete until the new ensemble is tested).
4. Decision on labels: Gamma-only (as trained) vs k-converged. Settle with the 2x2x2 test; if composition-dependent k error is > 50 meV/H, relabel a subset with the denser mesh and train on it.

## Stage 3: GRACE-internal consistency (after Stage 1/2 says the model is usable; GPU, ~1-2 GPU days per state)
At each production state, with the (possibly retrained) model, the same quantity by three routes, **independent seeds, not extensions**:
1. insertion TDI n = 1, 4, 8, 16, 27 with 21 lambda values (Kronrod);
2. crystal TDI at n = 12, 27 (same dt for host and references; reject melted crystals using a >= 6 ps free-NVT check);
3. pwidom with the fixed thermostat, >= 4 seeds, n = 0, 12, 27 and PS 1+, K 4000, requiring Kish > 0.1 and |Widom - deletion| < 100 meV.
Require agreement of mu(n) within ~50 meV/H between routes. Where they disagree, believe none until the pair is resolved. Errors from seed scatter, not from BAR block errors (these were 4-5x too small).

## Stage 4: independent confirmation by direct DFT (days to weeks; only if it matters)
* Cheap first: **DFT-sampled** dilute H on the DFT PWidom machinery but truncated: n = 0 only at (500 GPa, 5000 K), with the 729-H reservoir already checkpointed. This is the direct DFT mu0 - the only number that does not use GRACE. Jobs 60649379/80/81 project to 50-115 h of MD before trials against 48 h, so resubmit one state with 2-3x the allowance, or stop two of three.
* Compare mu0(DFT) with mu0(GRACE challenger) and with Stage 1 Delta. Agreement within ~100 meV closes the model question.
* Optional: ZPE/NQE size (Huang static vs ZPE: 10-20 meV/atom for H3O = 40-80 meV/H), He, other ices (CURRENT_ISSUES 0.4).

## Outcomes
| Stage 1 result | Next | What you can say |
|---|---|---|
| Delta, c small (< 50 meV/H) at production states | Stage 3, then hull refinement (n = 8-27 insertion, NPT stability) | solubility ~5-15 % extra H stands as a GRACE result with a DFT check |
| 50-150 | Stage 2 retrain, then Stage 3 | solubility is real in sign, size uncertain |
| Delta > |mu0| | Stage 2 to confirm it is not a model artefact, Stage 4 n = 0 DFT | **no dilute mixing in SI ice**; the interesting physics becomes the liquid mixing and the melting line; the SI phase stays H2O + separate H |
| Only cold states fail | quarantine <= 3000 K results, rerun hot only | model valid above ~5000 K; the Perlmutter 2000-3000 K hulls are void |

## Order and cost
Stage 0: hours, no cluster time except the cache check. Stage 1: ~2 h of two CPU nodes per state, three states. Stage 2.1: GRACE only. Stage 3: ~5-10 GPU days. Stage 4: ~300 node-hours, optional. The decision at Gate 1 needs only Stages 0-1: **about a day**.
Things to implement first (none exist yet): the friction fix + kinetic T logging; random-by-weight pair selection; the composition-offset / paired-FEP analysis script (my arithmetic in `GRACE_RELIABILITY_PLAN.md` section 2.1 is the prototype); the P-trend plot for cached frames.
