# PWidom code review: bugs and physics issues

Review date: 2026-11-04. This review covers `main` (`7a8de56`) and `claude/youthful-brown-4qfaeg` (`37dfd82`, the newest unmerged branch).
In scope: `tdi/pwidom.py`, `pwidom_checkpoint.py`, `pwidom_scan.py`, `proxy_fit.py`, `reweight_dft.py`, `pwidom_finite_size.py`, `collect_pwidom_results.py`,
`pwidom_convergence.py`, `pwidom_formation.py`, `pwidom_uncertainty.py`, `wk_correction.py`, `bar_lambda.py`, `toy_validate.py`, and the thermostats in `run_real.py`, `engine.py` and `17_eos_md.py`.
Not reviewed in depth: `fe_stages.py`, the Perlmutter / VASP tooling, the viewers.

## Summary

| # | Issue | Kind | Severity | `main` | `youthful-brown` |
|---|-------|------|----------|--------|------------------|
| 1 | pwidom host MD thermostat ~10x too weak, short equilibration | physics | **high** | present | **fixed** |
| 2 | Rerun into the same `--out` crashes (energy checkpoint) | code | **high** | — | **new** |
| 3 | Pre-fix `frames.npz` caches reused silently | code / physics | **high** | n/a | **present** |
| 4 | `proxy_fit` diagnostics omit the core term | code | medium | present | present |
| 5 | DFT-over-model reweighting ignores host-ensemble reweighting | physics | medium | — | **new** |
| 6 | Reporting tools use different pressure corrections | consistency | medium | — | **new** |
| 7 | Frame cache reused without checking run settings | code | medium | present | partly (resume paths check; plain cache does not) |
| 8 | `pwidom_scan.pw_args` pure-H size ignores `nrep` | code | low | present | present |
| 9 | `model_fingerprint` ignores `TOYICE` env overrides | code | low | present | present |
| 10 | Crash with fewer than 5 frames | code | low | present | present |
| 11 | Minor / cosmetic items | code | low | present | present |

## What was verified correct

- **Estimators.** The proxy-Rosenbluth insertion / deletion estimators are exactly unbiased for any proxy. The forward and reverse works satisfy Crooks in the extended space, so BAR is valid. This was checked numerically on a 1D soft-sphere fluid with a deliberately wrong proxy. The brute-force reference is 2.158; the estimators give Widom 2.168, deletion 2.153 and BAR 2.157.
- **BAR.** `toy_validate.bar` and the call `bar(wf, -wr)` are correct, including with very unequal sample counts (checked on Gaussian work distributions).
- **Proxy energy.** `pwidom.proxy_energy` (KD tree) equals `toy_ice.ghost_energy` to 1e-13 eV.
- **Formulas.** The ideal term `kT ln((N_H+1)/V) + 3kT ln Lambda`, the pressure correction `-V/(2B)[(P_full-P)^2-(P_host-P)^2]`, the "robust" variant in `pwidom_scan.mu_P`, and the finite-size term `fs` are all correct. So are the Wigner–Kirkwood formula, its units and self-test, and the MBAR in `bar_lambda.py`.
- **Analysis scripts on youthful-brown.** The sign conventions in `pwidom_convergence.py` are correct. In `pwidom_formation.py`, the composition integral and per-atom normalisation (`/(3+n)`) are correct, and so is the delete-one-block jackknife in `pwidom_uncertainty.py`.
- **TDI thermostats.** `run_real.py` / `engine.py` use `friction = 1/(tau fs)` with tau = 100 fs, `fixcm=False`, and `FixGroupCom` on O. All fine.

---

## 1. pwidom host MD thermostat ~10x too weak (high, physics) — fixed on youthful-brown

**Where:** `tdi/pwidom.py` `md_frames` on `main` (`friction=0.01`). The same setting was in `toy_validate.md`.

ASE reads `friction` in inverse ASE time units (1 unit = 10.18 fs), so `0.01` gives a damping time of about 1 ps. The rest of the pipeline uses 100 fs.
The default equilibration is only 0.5 ps, and `pwidom_scan` uses 0.2 ps. Production frames therefore start well away from the target T, and nothing records the kinetic temperature.
The kinetic pressure term is also built from the *target* T, while the virial part reflects the actual T.

The toy model (fcc H2O, 32 O, target 5000 K, lattice start, 3 seeds) gave these mean kinetic temperatures:

| Thermostat | Scan window (0.2–0.6 ps) | Default window (0.5–3 ps) |
|---|---|---|
| `friction=0.01` | **7380 K** | **5790 K** |
| `1/(100 fs)` | 5177 K | 5128 K |

**Effect:** `mu_ex` is sampled at the wrong T, and `P_host` / `P_full` (and so the pressure correction) are biased. The toy reference in `toy_validate` shared the bias.
**Status:** fixed on youthful-brown (`thermostat_friction`, `--friction-fs`, `--thermostat bussi`, kinetic-temperature trace). This becomes moot once that branch is merged, **but see #3.**

## 2. Rerun into the same output crashes (high, code) — new on youthful-brown

**Where:** `tdi/pwidom.py` `Model.energies`, lines 128–139, and `pwidom_checkpoint.md_frames`, lines 52–53.

The single-point checkpoint `<tag>_<label>_K<K>_energies.npz` is written *before any energies are computed*. Any later run that finds it raises
`ValueError: Energy checkpoint exists ... choose fresh output` unless `--resume-from <same prefix> --resume-in-place` is given. The MD checkpoints (`_host_md.npz`, `_full_md.npz`) behave the same way.

**Reproduced:** two identical toy `pwidom.py run` commands into one `--out`; the second one crashes.

**Effect:**
- **The `pwidom_scan.py` "retry alone" path cannot succeed.** That file is unchanged on this branch, and the path reruns the identical command without the resume flags. A host that ran out of GPU memory in single points (the case the retry exists for) fails again at this check. A host that died during MD fails at the MD checkpoint check.
- Rerunning a scan whose hosts partly failed ("missing ones reuse their MD frames") also fails.
- The new proxy-stage checkpoint ("a rerun with identical inputs loads it") is mostly unreachable, because the rerun dies at the energy check first.
- `pwidom_farm.py`, `perlmutter_pwidom.py` and `dft_pwidom.py` pass the resume flags and are not affected.

**Fix options:** resume automatically from a matching checkpoint in the same output (the fingerprint already guards against mismatches), or have `pwidom_scan` pass `--resume-from <out>/<tag> --resume-in-place` on retry.

## 3. Pre-fix frame caches reused silently (high) — youthful-brown

**Where:** `tdi/pwidom.py` `cmd_run`, lines 282–296 (cache load) and 333 (cache write).

`<tag>_frames.npz` records V, T, natoms, numbers, cell and the model fingerprint. It does **not** record friction, thermostat, `dt`, `ps`, `equil-ps` or `every-fs`.
Every existing `runs/pwidom*` cache made with the old `friction=0.01` (bug #1) is loaded by the fixed code with no warning. The result JSON looks like a post-fix run, but it is computed from frames that are not equilibrated at T.
`--extend-from` on such a cache also mixes old, cold-start frames with new ones.

**Fix:** write `friction`, `thermostat`, `dt`, `ps`, `equil_ps` and `every_fs` into `frames.npz`. On load, warn or refuse when they are missing (legacy) or differ from the request, and record them in the result JSON.

## 4. `proxy_fit` diagnostics omit the core term (medium, code) — main and youthful-brown

**Where:** `tdi/proxy_fit.py:136`, `phi_new = X @ c - c[-1]`.

The fit is to `y = dU - core`, so the proxy energy is `X @ c - c[-1] + core`. The saved JSON is correct, because `toy_ice._spline_pair` adds `K dl^2` itself. But the printed diagnostics for the **new** proxy are wrong for probes inside `r_lo`:
std(dU - phi), the Kish fraction, and "phi too low by > 5 kT". `phi_old` (from `proxy_energy`) does include the core, so the old-vs-new comparison is skewed against the new fit.
**Fix:** `phi_new = X @ c - c[-1] + core`.

## 5. DFT-over-model reweighting is an approximation (medium, physics) — youthful-brown

**Where:** `reweight_dft.py`, with weighted export in `pwidom_candidates.py`.

`d_mu = -kT ln <exp(-beta dD)>` over weight-resampled trials (dD = DFT−model difference of the insertion pair) corrects only the insertion energy. The exact identity is

    exp(-beta (mu_DFT - mu_model)) = < e^{-beta dU} e^{-beta D_N} e^{-beta dD} >_N / ( < e^{-beta dU} >_N < e^{-beta D_N} >_N )

where D_N is the whole-cell DFT−model energy of the host frame. The script drops the `e^{-beta D_N}` reweighting of the host ensemble. That is valid only if D_N barely varies between frames (much less than kT) or is unrelated to the insertion weights.

- **Check with existing data:** compute the std of `D(small)` across frames in units of kT.
- **Sample size:** the default `--candidate-limit 10` gives only 5 draws per direction, which is too few for a meaningful bootstrap error.
- **Docstring:** should state that this is a first-order correction.

## 6. Reporting tools use different pressure corrections (medium) — youthful-brown

| Tool | Correction |
|---|---|
| `pwidom_scan`, `perlmutter_pwidom` | `robust` |
| `collect_pwidom_results.py` | `measured` (default `--pressure-mode measured`) |
| `pwidom_finite_size.py` | `measured` (uses `mu_P_meV`) |

The scan's own notes say the measured correction is noisy (±50–150 meV in light runs). That noise would dominate a finite-size G_H(N) comparison and makes tables from different tools disagree.
**Fix:** use one default (robust) everywhere and label the mode in every output.

## 7. Frame cache reused without checking run settings (medium, code)

**Where:** `tdi/pwidom.py` cache load. The plain-cache check covers V, natoms, T and the model only.

A rerun with a different `--ps`, `--dt`, `--every-fs`, `--equil-ps`, `--hper` or `--phase` that keeps the same atom count reuses the old frames silently. On youthful-brown the `--resume-from` path also checks numbers and cell, but the plain path does not. This overlaps with #3 and has the same fix.

## 8. `pwidom_scan.pw_args` pure-H size ignores `nrep` (low, code)

**Where:** `tdi/pwidom_scan.py` `pw_args`, `nat = g.get("natoms") or 250`.

A pure-H stage0 run made with `nrep != 5` and no `--natoms` is rebuilt with 250 atoms, not the `2*nrep^3` atoms of the TDI run. The density is the same, so only finite-size effects differ.
**Fix:** `g.get("natoms") or 2 * (g.get("nrep") or 5) ** 3`.

## 9. `model_fingerprint` ignores `TOYICE` env overrides (low, code)

**Where:** `tdi/pwidom.py` `model_fingerprint`.

`toy_ice.params()` applies `TOYICE="k=v,..."` overrides, but only `TOYICE_FIT` is hashed. Toy Hamiltonians that differ only in those overrides get the same fingerprint, and their cached frames pass the check.

## 10. Crash with fewer than 5 frames (low, code)

**Where:** `tdi/pwidom.py` block errors, `blocks(..., nb=5)`.

A small `--ps` or a large `--stride` leaves empty blocks, and `lse` / `np.max` fails on an empty array. `perlmutter_pwidom` guards against this (`ps >= 0.1`); `pwidom.py` itself does not.

## 11. Minor / cosmetic

- `wk_correction.py` pairs hosts only with a reference tagged `pureH`, which only scan tags use. Direct `pwidom.py` pure-H runs (`pw_H_...`) never get a reference.
- `pwidom_scan --hpers 2.25` creates a second job named `H2.25O`, so two processes write the same tag and frames file.
- `pwidom.py` resets `t0` before the deletion stage, so the reported run times cover only part of the run.
- The `pwidom.bar` docstring says "wr = minus the reverse works", but the argument *is* the reverse works. The call itself is correct.
- On youthful-brown the kinetic-temperature trace file name contains spaces (`<tag>_host + 1 H_trace.npz`).
- ASE >= 3.28 warns that `Langevin(fixcm=True)`, the default used by pwidom, does not strictly sample NVT. The effect is O(1/N), but `fixcm=False` matches the rest of the pipeline.

## Physics caveats (not bugs)

- **Proxy fitted on the same frames.** This does not bias the result: the estimator is exact frame by frame for any proxy, even a frame-dependent one. Holding frames out only keeps the variance diagnostics honest.
- **Error bars in short scans.** In 0.4 ps scans, 5-block errors over ~20 frames come from correlated blocks, so the error bars are likely too small.
- **Deletion over all H.** Choosing any H to delete assumes protons exchange places during the MD. That holds when superionic, but not at low T.
- **Robust correction.** It uses one constant partial volume (v_H = 1.0 A^3) at every P and T.

## Suggested fix order

1. #2: auto-resume, or pass the resume flags in `pwidom_scan`.
2. #3 and #7: record MD settings in `frames.npz` and warn on legacy or mismatched caches.
3. #4: one-line fix.
4. #6: one pressure-correction default everywhere.
5. #5: document the approximation and add the D_N-spread check.
6. #8–#11.

---

## Status on `claude/youthful-brown-4qfaeg` after the review (2026-11-04)

Every claim was checked against the code before fixing. Tests: the full suite passes except the known environment test (`test_missing_environment_has_actionable_error`); `test_mismatch_and_existing_destination_refused` was updated because it asserted the old crash (see #2).

| # | Status | What changed |
|---|--------|--------------|
| 1 | fixed earlier | thermostat default 100 fs, `--friction-fs`, `--thermostat bussi`, kinetic-T trace |
| 2 | **fixed** | a plain rerun into the same `--out` resumes from a MATCHING checkpoint: `Model.energies` compares the stored fingerprint (structures + model) and continues, `pwidom_checkpoint.md_frames` resumes when the metadata matches (another seed / model / settings is still refused). `pwidom_farm sp` already did this; `pwidom_scan`'s retry path now works too. The proxy-stage checkpoint is reachable again |
| 3, 7 | **fixed** | `frames.npz` and the result JSON store `md_settings` (friction, thermostat, dt, every_fs, equil_ps, ps). On reuse: physics settings that differ are refused (`--allow-cache-mismatch` overrides), equil / length differences and caches with no recorded settings warn. `pwidom_farm pool` carries the settings over when all replicas agree. Caches made before this change (all existing `runs/pwidom*`, `farm_*`) warn "no MD settings recorded" |
| 4 | **fixed** | `phi_new = X @ c - c[-1] + core` in `proxy_fit.py` |
| 5 | **documented + diagnostic** | `reweight_dft.py` is a first-order correction (docstring). It now prints sd of the host error D(small) in kT and its correlation with dD, and says whether the host-ensemble factor cancels (uncorrelated) or not (sd > 1 kT AND |corr| >= 0.2). Not an exact estimator |
| 6 | **fixed** | `pwidom_finite_size.py` and `collect_pwidom_results.py` default to the robust correction (`--pressure-mode`), mode printed |
| 8 | **fixed** | `pwidom_scan.pw_args`: pure-H size `2 * nrep^3` |
| 9 | **fixed** | `model_fingerprint` hashes the `TOYICE` overrides |
| 10 | **fixed** | `blocks()` uses at most one block per frame |
| 11 | partly | trace files are now `<tag>_host_trace.npz` / `<tag>_full_trace.npz` (no spaces; `plot_pwidom_time.py` reads them). `fixcm=True` left as is on purpose: changing it would break resuming existing checkpoints (the metadata records it). Other cosmetic items untouched |
