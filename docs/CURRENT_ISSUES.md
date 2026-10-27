# Current issues (TDI free-energy pipeline)

Last updated 2026-10-27. Living list of what is wrong, unverified or open, in priority order. Fixed items move to HANDOFF.md (dated sections) with their commit.
Code on branch `claude/eloquent-hopper-kfjdei` (PR #2 branch + review fixes). Nothing below has been re-run on Young with the fixed code yet.

## 1. Results that are invalid or need action on Young

| # | What | Affected numbers | Action |
|---|------|------------------|--------|
| D1 | `H_500GPa_5000K_N250`, `H_500GPa_6000K_N250`: stage0.json overwritten by young_tdi_liquid (10-value grid) over 17-value Gauss-Kronrod node files | G_H 2212 / 1616 meV/atom (expected ~1065 / ~518; tdi_long gives 517.7 at 6000 K); every 500 GPa 5000/6000 K mixing (-1060, -1036, -1059 meV/H) and insertion (-1710, -1655, ...) row | summarize now lists them under NOT USED. Restore: HANDOFF session 4, "later, same day" |
| D2 | Insertion runs made before the pressure correction | all insertion rows (flagged "NO PRESSURE CORRECTION") | relaunch each with its original settings (stage 0 measures P_host, P_full; nodes restart). Expected shift -26 to -80 meV per added H |
| D3 | tdiext 115468 (if it ran with the old extend.py) started new `_rep2/_rep3/_rep4` runs instead of extending | stray runs in runs/tdi | `ls runs/tdi/*_rep[0-9]_stage0.json`, delete unintended ones, resubmit young_tdi_extend.sbatch |
| D4 | Crystals that melt in free NVT but are held by the springs in TDI | 500 GPa 6500/7000 K and 400 GPa 6500 K H2.25O mixing values; tdi_prep fcc 300 GPa 5000 K h2.25o/h2.5o (Kronrod - Gauss -327 / -451 meV/atom) | not valid solids: exclude (the report already marks them) |
| D5 | 400 GPa 5000 K mixing rows pair dt 0.5 crystals with dt 1.0 H2O | +188..+225 meV/H | dt 0.5 H2O references queued (extend_list NEW lines) |
| D6 | DFT hull sheet: header says eV/atom, values look like meV/atom | DFT squares in the liquid-hull plot | check the source of data-h3o-convex-hull.csv |

## 2. Physics: open questions

**P1. Insertion vs crystal mixing are consistent, and the dilute limit is strongly negative.** Insertion gives the dilute chemical potential (mu_H - G_H ~ -0.5 eV, about -1 kT, after the pressure correction). The crystal value is the average over 27 added H (x = 0.25 H/O, e.g. +76 meV at 500 GPa 6000 K, +141 at 5500 K). A linear (regular-solution) fit through the corrected n = 1, 2, 4 averages extrapolates to +55 (5500 K), -83 (6000 K) and +275 (400 GPa 5000 K) at n = 27, against +141, +76 and +192 direct: agreement within ~100-150 meV, which is the noise of a 7x extrapolation. So mu_H rises steeply with concentration (~20 meV per added H in 108 O at constant P). The fixed-V pressure rise explains only ~25% of the raw insertion slope.
- Consequence: H is soluble in superionic fcc H2O up to roughly **x ~ 0.07-0.15 H/O** (differential mu_H = G_H(pure)) at 400-500 GPa, 5000-6000 K. A positive per-added-H average at x = 0.25 does NOT mean "insoluble" (summarize label fixed; the report generator, not in this repo, still says "positive = H not soluble").
- Needed: points at x ~ 0.05-0.15 to locate the solubility limit (insertion n = 8, 16, or crystal runs `hper` 2.074 / 2.111 / 2.148 with 108 O; the empty `fcc_400GPa_6500K_h2.0xxxxo` dirs were meant for this), and long runs (insertion n=1 has +-30-45 meV and 30-85 meV drift at 2 ps).

**P2 results so far (500 GPa 6000 K, tdi_long insert n1):** TDI integral split: lambda < 1e-2 ~150 meV, 1e-2..0.1 ~320 meV, 0.1..1 ~2480 meV, converged (halves agree), Gauss-Legendre vs trapezoid 14 meV; integrand finite (~14 eV) as lambda -> 0 (soft core, harmless for a state function unless lambda = 1 visits it). Widom on the host: mu_ex = 2.90 eV (blocks 2.4-3.7, not converged) vs TDI 2.95 +- 0.02 eV: agree. In-host probe: O-H strongly repulsive (25 eV at 0.5 A); H-H soft (8.0 eV at 0.6 A vs 7.0 at 1.0 A median) but the deepest state within 0.6 A of an H (3.3 eV) is ~5 kT above the deepest physical site (0.6 eV): no hole visited at lambda = 1 (the old 'NOT REPULSIVE' flag was a bad criterion, fixed). Dimer scan (tdi/pair_scan.py, ft_m2): H-H minimum at 0.70 A, monotonic, 1.9 eV above the minimum at 0.5 A (a Morse H2 gives ~1.7 eV: H-H shape physical), core soft vs ZBL (67 vs 93 eV at 0.1 A); O-H minimum at 1.00 A, O-O at 1.25 A; O-H and O-O turn over below 0.15-0.2 A after barriers of ~170 / ~680 eV (no ZBL term): far beyond anything sampled (kT 0.5 eV), so irrelevant for lambda = 1 free energies; dimer well depths relative to 6 A (-8.2, -8.1, -6.6 eV) are not meaningful (isolated atoms are out of the training distribution). Still to check: H-H softness in the host against DFT; pure-H method check (P2/P1).

**P2. MLP behaviour at short range.** Every path starts H as an ideal gas, so the small-lambda integrand samples protons overlapping O/H. On the toy model the small-lambda nodes contribute +240 meV, cancelled by -264 meV at large lambda: the result is a small difference of large pieces. A GRACE model that is not repulsive below ~0.7 A would bias insertion and crystal routes differently. Test: `mlp_pipeline/tdi/short_range.py runs/tdi insert_500GPa_6000K_n1` (flags "NOT REPULSIVE AT SHORT RANGE").

**P3. Quadrature error of the insertion runs is unknown** (10-value Gauss-Legendre, no Kronrod). The crystal mixing value has ~85 meV/H quadrature uncertainty at 500/6000 (Kronrod - Gauss of H2.25O and H2O, x13 amplification). Extend one insertion run: `RULE=gk NGAUSS=10 NODES=21 ONLYNODES="10 ... 20"` with its original settings.

**P4. G = F + P_target V when the MLP pressure misses the target**: second order, reported (not applied) as summarize's dG_P2 column, x13 in the per-added-H mixing value. Only for stage0 jsons that store B_GPa (new runs).

**P5. Classical nuclei.** Every number is classical (ideal-gas H with the thermal wavelength, classical MD). Nuclear quantum effects of H at 5000-6000 K and 500 GPa are not small for mixing free energies; state this in any comparison with DFT-PIMD literature.

## 3. Verified correct (do not re-check without a reason)
Einstein crystal with fixed O centre of mass (N^{3/2}(2 pi kT/k)^{3(N-1)/2}, -kT ln V), ideal-gas H with ln N!, 3N kT ln Lambda, P V units, insertion reference kT ln((N_H+n)!/N_H!) - n kT ln V + 3 n kT ln Lambda (same counting as the crystal route), Gauss-Legendre in lambda = x^4 and the Gauss-Kronrod rule, insertion integral vs Widom on a toy LJ fluid (-9 +- 13 vs -1 +- 9 meV), pressure correction vs exact Legendre transform (0.2% at 20 GPa offsets), shell array decoding of every sbatch table.

## 4. Known limitations, not fixed (judgement calls)
- `17_eos_md.build()`: H on random bond midpoints: exact composition but locally 0-4 H per O, H-H down to d_OO/2 in the start frame.
- Restarted nodes reseed the RNG from the seed (not a bias, not a bit-exact continuation).
- 5-block errors: +-35% relative uncertainty; +-1 SEM covers ~63% (4 dof). Errors of short (<= 2 ps) runs are optimistic; use the drift column.
- Finite size: ln N! Stirling terms (~ -6 meV per added H at 6000 K for 108 O / 250 H); 96-atom liquid cells; only one H finite-size pair (400 GPa 6500 K, N 250 vs 500: equal within 5 meV/atom).
- Toy LJ blows up at the smallest lambda (bare r^-12): toy-only, but any potential with a stiff core needs dt checks at lambda < 1e-6.
- The report/figure generator (tdi_report.pdf) is not in the repo: commit it so its labels and fits can be reviewed.

## 5. Next tests, cheapest first
Tests 1-2 and the mu breakdown (tdi/mu_breakdown.py: reference pieces, TDI integral by lambda range with drift, second quadrature, figure) (plus per-lambda tail/drift statistics of the crystal and pure-H runs) in one go, interactive GPU job: `bash mlp_pipeline/young_insert_checks.sh` -> runs/checks/insert_checks_<date>.txt.

MD diagnostics (sbatch; young_tdi_insert.sbatch, 4 h limit; every line keeps the original insertion settings PS 2, EQUIL 0.5, DT 0.5, CHUNK 0.5):
- Quadrature of the insertion integral (P3): Gauss-Kronrod extension of one run, 11 new lambda values, the old 10 kept:
  `RULE=gk NGAUSS=10 NODES=21 NPAR=11 ONLYNODES="10 11 12 13 14 15 16 17 18 19 20" JOBS="500 5500 1" sbatch --array=0 mlp_pipeline/young_tdi_insert.sbatch`  -> summarize quad_diff of insert_500GPa_5500K_n1
- Heavy tail at small lambda (P2): the 4 smallest lambda values again, independent seed, 4x longer, separate directory; compare node_stats of the two:
  `SEED=11 OUTDIR=runs/tdi_seed ONLYNODES="0 1 2 3" NPAR=4 PS=8 JOBS="500 5500 1" sbatch --array=0 mlp_pipeline/young_tdi_insert.sbatch`
- Same quantity by both routes (P1): H2.25O by insertion vs +141 / +76 meV direct:  `PS=4 JOBS="500 5500 27;500 6000 27" sbatch --array=0-1 --time=08:00:00 mlp_pipeline/young_tdi_insert.sbatch`
- METHOD CHECK (do first): insertion into the pure-H fluid at the same (P, T) must give mu_H - G_H ~ 0 (summarize row "insert into pure H (METHOD CHECK)"):
  `PS=4 JOBS="500 6000 1 - H;500 5500 1 - H" sbatch --array=0-1 mlp_pipeline/young_tdi_insert.sbatch`
- Solubility limit (P1): `PS=4 JOBS="500 5500 8;500 5500 16;500 6000 8;500 6000 16" sbatch --array=0-3 --time=08:00:00 mlp_pipeline/young_tdi_insert.sbatch`
1. `short_range.py` on insert_500GPa_6000K_n1 (P2).
2. `collect.py runs/tdi insert_500GPa_6000K_n1`: per-lambda integrand; compare with the toy profile (P2).
3. Restore D1, relaunch insertions (D2), rerun summarize.
4. Intermediate concentrations x ~ 0.05-0.15 (P1): insertion n = 8, 16 at 500 GPa 5500/6000 K, 5 ps.
5. `JOBS="500 6000 27" young_tdi_insert.sbatch`: H2.25O by the insertion route vs +76 meV direct (P1/P2).
6. Gauss-Kronrod extension of one insertion run (P3).
