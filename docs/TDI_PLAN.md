# Plan: MLP thermodynamic integration, superionic H2O fcc vs bcc

Status: PLAN ONLY, no code run yet. Numbers marked (guess) are not measured.

## Goal
G_fcc - G_bcc of superionic H2O (O on fcc / bcc sites, mobile protons) at one (P, T) where the literature says the difference is resolvable,
using the fine-tuned GRACE model, then a correction to the DFT level from single-point VASP snapshots.

## Where to run it (literature, from a quick search; confirm in the papers before fixing P, T)
- Wilson, Wong, Militzer (2013), DFT free energies: bcc becomes unstable against denser fcc above about 1.0 +- 0.5 Mbar; fcc has lower proton mobility. https://arxiv.org/pdf/1211.6482
- Millot et al. (2019) report an fcc-oxygen superionic phase (ice XVIII) above 2000 K by x-ray diffraction (secondhand, via https://ar5iv.labs.arxiv.org/html/2103.09035).
- A 2025 paper reports BCC and FCC signatures together in one P-T range: https://arxiv.org/pdf/2510.24305v1
I did not read Sun-Clark-Torquato-Car (2015) or Millot 2019 themselves.
Revised by the census: our data is richest at 400-500 GPa, 3000-7000 K, so start at 500 GPa (the DFT TDI folders are named 500gpa) at 2000-3000 K and treat the 150/300 GPa idea as a later extension. Original proposal: pick 2 state points bracketing the bcc/fcc crossover near 100 GPa-class pressures (e.g. about 150 and 300 GPa, T = 3000-5000 K), because
our fine-tune data starts at 200 GPa; below that the model is extrapolating. Decision needed from you: which (P, T), and is going below 200 GPa wanted.

## Data census (16_phase_census.py, 2026-10-26; path words only, so tags are the STARTING structure, not a verified end state)
h2o runs: 3554. fcc 798 + fcc/solid 370 + fcc/superionic 15; bcc 1132 + bcc/liquid 542 + bcc/liquid/solid 163 + bcc/superionic 18; untagged 514.
Both phases overlap at 3000 K (200-1000 GPa), 5000 K (200-1000 GPa) and 7000 K (300-1000 GPa), best at 400-500 GPa 3000 K (fcc 123/111 runs, bcc 66/28 runs per 100 GPa bin).
2000 K superionic is thin: about 15-18 runs per phase (paths `superionic-500gpa-2000K/fcc-h2o-108-2000K`, `bcc-h2o-128-2000K`).
The DFT cells are fcc-h2o-108 (108 O, 324 atoms) and bcc-h2o-128 (128 O, 384 atoms). Existing VASP TDI runs (`tdi/h3o/500gpa-3000K/...`, `500gpa-7000K-solid`) mean DFT free energies already exist at 500 GPa: compare the MLP-TDI against them, but do NOT treat them as reliable reference values (user's call: they are a cross-check only).
Caveats: the dataset tiers pool fcc and bcc in the same (T, 100 GPa) bin, so each phase gets only part of the 1 (s) / 3 (m) runs per bin; whether a run stayed solid, became superionic or melted is not in its folder name (09_label_phase.py).

## Literature on the fcc-bcc free-energy difference (2026-10-26; from search excerpts only, the papers themselves could not be opened: this sandbox blocks arxiv/nature/osti hosts)
NUMBERS NOW AVAILABLE for Wilson 2013 (Table I of the paper, entered by the user; see docs/LITERATURE_NOTES.md and docs/data/dg_fcc_bcc.csv): G_fcc - G_bcc at 3000 K is -0.7(3.7) meV/atom at 100 GPa, -9.0(2.7) at 200 GPa, -24.7(2.3) at 500 GPa, -71(2) at 4000 GPa (per atom = eV/molecule / 3). Cheng et al. 2021 (NNP + FEP to PBE, from the authors' public data tables) agree with Wilson at 3000 K: mu_bcc - mu_fcc = 0.0038(18), 0.0314(20), 0.0794(16) eV/molecule at 100, 200, 500 GPa and 0.0662(17) eV/molecule (= 22.1 meV/atom) at 400 GPa. TARGETS for our MLP-TDI at 3000 K: 26.5 meV/atom at 500 GPa, 22.1 meV/atom at 400 GPa (G_fcc lower). Their TI method (Hessian-based harmonic reference crystal, TI along several routes as a consistency check, error bands from polynomial fits, free-energy perturbation to PBE for the MLP error) is described in docs/LITERATURE_NOTES.md.
- Wilson, Wong, Militzer, PRL 2013, https://militzer.berkeley.edu/papers/Wilson_Militzer_superionic_transition.pdf (arXiv:1211.6482): DFT molecular dynamics; two-step coupling-constant integration from a simpler analytic reference; Table I lists G_bcc - G_fcc split into P dV, dU and T dS; points from about 1 Mbar to 40 Mbar; bcc unstable against fcc above 1.0 +- 0.5 Mbar; fcc has lower H mobility.
- Cheng, Bethkenhagen, Pickard, Hamel, Nature Physics 2021, https://www.nature.com/articles/s41567-021-01334-9 (arXiv:2103.09035; data and potential at https://github.com/BingqingCheng/superionic-water): neural-network potential trained on PBE DFT, free energies by TI at the PBE level. Reference system: Einstein crystal for the O lattice + Uhlenbeck-Ford potential for the protonic fluid. Result: close-packed (fcc with mixed stacking in finite cells) is stable over a wide P-T range; bcc is thermodynamically stable only in a small window but kinetically favoured.
- Qualitative statements seen in excerpts (not numbers): Wilson 2013 find G_fcc < G_bcc at all conditions studied except 1 Mbar and 3000 K, where the difference is close to zero; an excerpt of arXiv:2103.09035 says the fcc-hcp difference is under 10 meV per H2O formula unit (about 3 meV/atom by my division by 3), which is a stacking, not a bcc-fcc, number; a 2026 preprint (arXiv:2601.03659) says the bcc superionic transition at 200 GPa is at 2250-2600 K, about 500 K above fcc; Nature Communications 2025, https://www.nature.com/articles/s41467-025-67063-2, 'Observation of a mixed close-packed structure in superionic water' (experiment; not read). An excerpt of arXiv:1810.05658 lists dU = U_fcc - U_bcc of 0.03-1.0 eV/atom (internal energy, not G, and not checked).
- Militzer and Zhang (cited in a search excerpt): P2_1/c beats fcc above 22.8 +- 0.5 Mbar at 4000 K, outside our range.
- Related MLP work seen in the search (not read): JCP 2025 Deep Potential on stress-strain curves (https://pubs.aip.org/aip/jcp/article-abstract/163/22/224507/3374708/Phase-behavior-of-a-machine-learning-potential?redirectedFrom=fulltext); JCP 2023 hybrid solid/liquid TI for the fcc melting at 340 GPa (https://pubs.aip.org/aip/jcp/article-abstract/158/6/064502/2875200/Melting-conditions-and-entropies-of-superionic?redirectedFrom=fulltext).
Consequences for this plan: (1) replace the ideal-gas proton reference by an Uhlenbeck-Ford fluid (as Cheng et al.), which is soft and avoids the lambda -> 0 overlap problem of Stage 1; (2) the fcc cell can contain hcp-like stacking faults, so check stacking in the fcc runs; (3) the data and potential from Cheng et al. could serve as an independent PBE cross-check of our MLP at the same state point.

## TDI approaches to compare for G_fcc - G_bcc (goal of the research: the published energy differences as a function of P)
Run the same fcc/bcc cells with each; the spread between approaches is itself the systematic-error estimate.
- A. Einstein crystal (O) + ideal gas (H), two stages as above. Simplest; risk at lambda -> 0 for the protons.
- B. Einstein crystal (O) + Uhlenbeck-Ford fluid (H) (Cheng et al. 2021). Soft proton reference, smoother Stage 1.
- C. Two-step coupling-constant integration from a pair-potential reference (Wilson et al. 2013; also what the earlier VASP lambda_* runs look like).
- Variants of the quadrature on any of them: Gauss-Legendre in lambda^(1/m) versus MBAR/BAR over all windows.
Each approach is run at several P along 2000 K, 3000 K and (if the phases survive) 5000 K, then everything goes into docs/data/dg_fcc_bcc.csv next to the literature values (docs/plot_dg_vs_P.py draws one panel per temperature; the CSV is empty until values are filled in from the papers).

## Cells
- First use the SAME cells as the DFT runs (fcc 108 O, bcc 128 O): the FEP correction needs DFT single points, and 324/384 atoms is what VASP already ran. Then repeat with larger cells (below) as a finite-size check.
- Larger check: one cubic supercell per phase, about 250 O each so ΔG per atom is comparable: fcc 4x4x4 conventional (256 O + 512 H = 768 atoms), bcc 6x6x6 (432 O) or 5x5x5 (250 O + 500 H = 750 atoms). Edge must stay above 8.5 A (our box rule) - at about 200-300 GPa these are about 12-14 A.
- H start: random on interstitial sites, then equilibrate (the 11000+ K trick is not needed; superionic H diffuses within ps).
- Check first: the MLP has not seen cells this large; memory at 768 atoms is fine for GRACE (729 atoms was 10 GiB inference).

## Volume
Our rule: NVT at V* from an EOS. Recommendation: use the DFT EOS only as the starting guess (the repo's DFT hull data has a bcc-128 curve and fcc-solid runs sitting on another branch,
PROJECT.md line 30), then refine with the MLP itself: 3-4 volumes x 5 ps NVT per phase, fit P(V), take V* where <P_MLP> = P_target.
Reason: G = F(V*) + P V*, and the pressure error of the MLP shifts V* by dV = dP/B; using MLP-consistent V* removes that, and the DFT-vs-MLP pressure gap (a few GPa after fine-tune)
is then handled in the DFT correction. Each phase gets its own V*; do not force the same volume.

## Step A: plain NVT at lambda = 1 (diagnostics, 10-20 ps per phase)
Bussi velocity-rescaling NVT (canonical, perturbs dynamics less than Langevin), dt 0.5 fs (check by a 2 ps NVE energy-drift test first), 5 ps equilibration + 15 ps production.
Files per run (all in one folder, names fixed so postprocessing is trivial):
- `thermo.csv` every step or every 2 steps: step, t_fs, T, Epot, Ekin, Etot, P_GPa, the six stress components (GPa; for Green-Kubo viscosity), E_O_only (stage 1 below), dEdlambda.
- `traj.extxyz` every 20 fs: positions UNWRAPPED (ASE does not wrap, so do not call wrap), velocities, forces, cell, step; enough for MSD/D for O and H (no rewrap needed).
- `vacf` data derivable from velocities every 5 fs for the first 2 ps of production (separate `vel.extxyz`).
- `log.txt`: ms/step, ms/atom/step, GPU memory, running averages (T, P, D_H from MSD over the last 2 ps) every 1 ps, plus the gamma (UQ extrapolation grade) max/mean per frame so we see when the MLP leaves its training domain.
Postprocess `tdi_analyze.py` (to write): D_H and D_O from MSD slope, RDFs O-O/O-H/H-H, O-site occupancy (is the lattice intact), viscosity by Green-Kubo (only meaningful if the O sublattice melts; in a solid it will not converge - expected), P and T averages with block errors.
Sanity: fcc and bcc runs must keep their O sublattice (check structure factor peaks) and D_H must be superionic-size (about 1e-4 to 1e-3 cm2/s is the usual range, verify in the DFT D values you already have).

## Step B: free energy, two stages as you proposed
Reference system: O harmonic Einstein crystal (spring k per O about its starting lattice site) + H ideal gas in the box.
Define U_all(r) = U_MLP of all atoms, U_O(r_O) = U_MLP of the O sublattice alone (H deleted; one extra, small evaluation), U_E = sum_i k/2 |r_i - R_i|^2.
- Stage 1, proton gas -> interacting protons, O pinned by U_E:
  U1(lambda) = U_E + lambda * (U_all - U_O)   (lambda 0 -> 1). dF1 = integral of <U_all - U_O>_lambda.
  Start: F_E + F_ideal gas(H); end: O Einstein + fully interacting H, no O-O interaction.
- Stage 2, Einstein O -> interacting O, H fully interacting:
  U2(lambda) = (U_all - U_O) + (1 - lambda) U_E + lambda U_O ... equivalently U_all at lambda = 1. dF2 = integral of <U_O - U_E>_lambda.
- Total: F_MLP = F_E + F_ig(H) + dF1 + dF2 (+ the Einstein centre-of-mass correction). Then G = F + P V*.
Known risk (Stage 1): at lambda -> 0 protons sit on top of O and on each other and the MLP is evaluated at tiny distances it was never trained on; the integrand is then large and noisy.
Mitigation, to test before committing: integrate in x = lambda^(1/m) with Gauss-Legendre points (m = 3-4), 12-16 points, and plot <dU/dlambda> against lambda for both stages.
Fallback if the lambda -> 0 end is unphysical: add a short-range repulsive wall (soft-core or ZBL) to U1 only at small lambda, or start Stage 1 from an H lattice-gas reference instead of the ideal gas.
Choose k in U_E from the real <u^2> of O at lambda = 1 (k = 3 kT / <u^2>) so Stage 2 is flat; Einstein k does not change the final F, only the noise.
Closed forms to implement (and unit-test against the harmonic limit): F_E, F_ig(H) with the exact ln N!, the fixed-centre-of-mass correction for Stage 1/2 (we run momentum-conserving MD),
and the classical nuclei assumption (nuclear quantum effects for H at these T, P are ignored in both phases - state this in any result).
Cost: MEASURED 172 ms/step for 384 atoms on an A100 (GRACE-2L medium fine-tune, 0.5 fs) = 10.5 ps/h = 250 ps/day. If cost scales about linearly with atoms (not measured), 750 atoms is about 0.34 s/step = 5 ps/h, so 5 ps per lambda point = 1 h; 2 stages x about 12 points x 2 phases = 48 points about 48 GPU-hours plus Step A. With the DFT-sized cells (324/384 atoms) it is half of that. Use a Slurm array, one GPU per lambda point.

## Error budget (all numbers are ESTIMATES, to be replaced by measured fluctuations from the first lambda runs)
Structure: DeltaG = G_fcc - G_bcc; each phase needs Stage 1 + Stage 2; each stage = n lambda points (assumed 12 Gauss-Legendre points in x = lambda^(1/m)); each point = one independent NVT run. 2 phases x 2 stages x 12 = 48 runs.
Statistical error of one point: s / (sqrt(N) sqrt(n_eff)) per atom, s = per-atom std of dU/dlambda (unknown; assumed 0.2 eV), N = atoms (384), n_eff = independent samples (5 ps / correlation time of about 100 fs = 50): about 1.4 meV/atom per point.
Quadrature combines points with weights w_i (sum 1): error = sqrt(sum w_i^2 sigma_i^2) about sigma/sqrt(12) = 0.4 meV/atom per stage; 2 stages, 2 phases: about 0.8 meV/atom on DeltaG. Error falls as 1/sqrt(simulation time): 4x longer for half the error.
The 'MLP-to-DFT correction' row below is the OPTIONAL free-energy-perturbation step F_DFT - F_MLP = -kT ln <exp(-(U_DFT - U_MLP)/kT)>_MLP (cumulant form: <dU> - Var(dU)/2kT). It needs DFT energies of configurations, but NOT new DFT runs if we reuse the existing AIMD frames (reverse direction: sample = DFT frames, evaluate the MLP on them); the 2.5 meV/atom is only the statistical error of the mean of dU from about 100 frames (assumed sigma 25 meV/atom), not the size of the MLP error itself. A no-new-DFT first estimate: mean SIGNED energy error of the MLP on held-out fcc and on held-out bcc frames (same per-element shift for both); the difference is the first-order bias on G_fcc - G_bcc.
Systematic terms that do NOT shrink with more MD: quadrature (lambda grid / x-mapping), Einstein centre-of-mass and ideal-gas formulas, finite size (compare 128 O vs about 250 O), classical nuclei, and above all the MLP-vs-DFT difference: FEP correction error about sigma_diff/sqrt(n_snapshots) = 25 meV / sqrt(100) = 2.5 meV/atom per phase (sigma_diff assumed), correlated between the phases only partly.
So the MD statistics can reach below 1 meV/atom, but the DFT-level DeltaG is limited to a few meV/atom by the correction step.

## Accuracy: what this can and cannot settle today
- The fcc-bcc free-energy difference in the literature is small (a few meV/atom class; unverified, check the papers). Our fine-tune had energy MAE 10-70 meV/atom on 30 frames. The h2o data from Perlmutter very probably already contains fcc and bcc superionic runs (e.g. 3000 K; at 5000 K a mix of solid and liquid branches depending on P) but they are not labelled yet: run `mlp_pipeline/16_phase_census.py` and `09_label_phase.py` to count them.
- So the MLP number is a method check first, not a result. Before it counts: (1) check how many fcc/bcc superionic frames the training tiers hold at the chosen state points and add DFT frames (20-40 per phase) only where they are thin; (2) correct with free-energy perturbation: dF_corr = -kT ln < exp(-(U_DFT - U_MLP)/kT) >_MLP over 50-100 MLP snapshots per phase, DFT as single points; if the exponential average is noisy, use the cumulant (mean + variance/2kT) form and report both.
- The label convention matters: DFT free energy (TOTEN with sigma = kT) is what we trained on, so the correction uses the same quantity; the electronic entropy is then included in both.

## Order of work (updated 2026-10-26)
Done: data rebuilt (late segments), pressure-reference bug found and fixed (EOS-MD of bcc 3000 K: medium model within 3 GPa / 0.35% in volume of DFT), fcc frame check (errors <= 1.5 GPa).
GPU queue (two interactive jobs):
1. Job 1: ft_m2 (new data, running) -> ft_s2 (about 10 min) -> bench_grace_compare.sh with both.
2. Job 2: fcc EOS-MD with ft_m (P = 350-500 GPa; about 40 min) -> bench of the old models (zero-shot, ft_s, ft_m) on the rebuilt valid_m.
3. Quick tests of ft_m2/ft_s2: bench figure, then bcc and fcc EOS-MD at 400 GPa and 3000 K only.
No GPU needed (can start now): write and test the TDI engine on a cheap potential (EMT/LJ) in the sandbox: mixed Hamiltonian, Einstein-crystal and proton reference free energies, quadrature, analysis; test against a harmonic crystal where the answer is analytic.
4. First TDI state point: 500 GPa, 3000 K (equals Wilson's 5 Mbar row: G_bcc - G_fcc = 0.074(7) eV/molecule = 24.7(2.3) meV/atom, so we get a direct literature check), then 400 GPa (interpolated about 19 meV/atom; our EOS is best validated there). Both with DFT-sized cells (128 O bcc / 108 O fcc). Why: richest DFT data (fcc exists only for about 330-540 GPa at 3000 K), EOS validated there, superionic (D_H about 7e-4 cm2/s, O frozen), well above the superionic transition and, as far as the data show, below melting. The difference at these pressures is about 20-25 meV/atom (0.08-0.1 kT at 3000 K), several times the roughly 1 meV/atom statistical error I estimated: resolvable. Caveat: Wilson used small cells (32 and 54 molecules) and PBE with 2x2x2 k-points; their error bars are 2-3 meV/atom. Before any TDI: run 20 ps NVT at lambda = 1 at the TDI volumes to check that bcc does not transform to fcc.
5. Only after the fluctuation sizes are known: decide the lambda grid, the DFT-level correction, and whether to go to 5000 K or higher pressures.

## Decisions I need from you
1. The (P, T) points (and whether below 200 GPa is wanted).
2. How many GPU-hours we may spend; 5 ps per lambda point is my default.
3. After the phase census: whether fcc/bcc superionic coverage at the chosen (P, T) is enough, or extra DFT frames are needed.

## Thermostat choice for the lambda runs (tested 2026-10-26 on the harmonic-crystal test, 32 atoms, 3 nodes x 8000 steps)
- Langevin (friction 1/100 fs): kinetic T = 1000 K, integrand within error of the exact one at every node; full 8-node test PASS (-0.46 meV of 8.005 eV).
- Bussi velocity rescaling: integrand 3-20% too high, FAIL (global rescaling does not mix the modes of a harmonic crystal).
- Nose-Hoover chain (ASE, 3 chain thermostats, tdamp 100 fs): integrand -7% to +7% off, kinetic T 1015-1022 K, -4.3 sigma, FAIL (same non-ergodicity of deterministic thermostats on harmonic systems; one parameter set and one short run only).
Free-energy averages need only the configurational distribution, so Langevin is the safe choice; its friction perturbs the dynamics, so use NVE or a chain only for the lambda = 1 diagnostic runs (D, viscosity).
