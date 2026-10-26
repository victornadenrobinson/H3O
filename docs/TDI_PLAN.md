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
NUMBERS NOT OBTAINED. No meV/atom value for G_bcc - G_fcc or its error bar was visible in any excerpt; they are in the papers' tables (see below). Do not quote a number from this document.
- Wilson, Wong, Militzer, PRL 2013, https://militzer.berkeley.edu/papers/Wilson_Militzer_superionic_transition.pdf (arXiv:1211.6482): DFT molecular dynamics; two-step coupling-constant integration from a simpler analytic reference; Table I lists G_bcc - G_fcc split into P dV, dU and T dS; points from about 1 Mbar to 40 Mbar; bcc unstable against fcc above 1.0 +- 0.5 Mbar; fcc has lower H mobility.
- Cheng, Bethkenhagen, Pickard, Hamel, Nature Physics 2021, https://www.nature.com/articles/s41567-021-01334-9 (arXiv:2103.09035; data and potential at https://github.com/BingqingCheng/superionic-water): neural-network potential trained on PBE DFT, free energies by TI at the PBE level. Reference system: Einstein crystal for the O lattice + Uhlenbeck-Ford potential for the protonic fluid. Result: close-packed (fcc with mixed stacking in finite cells) is stable over a wide P-T range; bcc is thermodynamically stable only in a small window but kinetically favoured.
- Militzer and Zhang (cited in a search excerpt): P2_1/c beats fcc above 22.8 +- 0.5 Mbar at 4000 K, outside our range.
- Related MLP work seen in the search (not read): JCP 2025 Deep Potential on stress-strain curves (https://pubs.aip.org/aip/jcp/article-abstract/163/22/224507/3374708/Phase-behavior-of-a-machine-learning-potential?redirectedFrom=fulltext); JCP 2023 hybrid solid/liquid TI for the fcc melting at 340 GPa (https://pubs.aip.org/aip/jcp/article-abstract/158/6/064502/2875200/Melting-conditions-and-entropies-of-superionic?redirectedFrom=fulltext).
Consequences for this plan: (1) replace the ideal-gas proton reference by an Uhlenbeck-Ford fluid (as Cheng et al.), which is soft and avoids the lambda -> 0 overlap problem of Stage 1; (2) the fcc cell can contain hcp-like stacking faults, so check stacking in the fcc runs; (3) the data and potential from Cheng et al. could serve as an independent PBE cross-check of our MLP at the same state point.

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
Systematic terms that do NOT shrink with more MD: quadrature (lambda grid / x-mapping), Einstein centre-of-mass and ideal-gas formulas, finite size (compare 128 O vs about 250 O), classical nuclei, and above all the MLP-vs-DFT difference: FEP correction error about sigma_diff/sqrt(n_snapshots) = 25 meV / sqrt(100) = 2.5 meV/atom per phase (sigma_diff assumed), correlated between the phases only partly.
So the MD statistics can reach below 1 meV/atom, but the DFT-level DeltaG is limited to a few meV/atom by the correction step.

## Accuracy: what this can and cannot settle today
- The fcc-bcc free-energy difference in the literature is small (a few meV/atom class; unverified, check the papers). Our fine-tune had energy MAE 10-70 meV/atom on 30 frames. The h2o data from Perlmutter very probably already contains fcc and bcc superionic runs (e.g. 3000 K; at 5000 K a mix of solid and liquid branches depending on P) but they are not labelled yet: run `mlp_pipeline/16_phase_census.py` and `09_label_phase.py` to count them.
- So the MLP number is a method check first, not a result. Before it counts: (1) check how many fcc/bcc superionic frames the training tiers hold at the chosen state points and add DFT frames (20-40 per phase) only where they are thin; (2) correct with free-energy perturbation: dF_corr = -kT ln < exp(-(U_DFT - U_MLP)/kT) >_MLP over 50-100 MLP snapshots per phase, DFT as single points; if the exponential average is noisy, use the cumulant (mean + variance/2kT) form and report both.
- The label convention matters: DFT free energy (TOTEN with sigma = kT) is what we trained on, so the correction uses the same quantity; the electronic entropy is then included in both.

## Order of work (proposal, nothing started)
1. Finish the m-tier fine-tune (running) and benchmark; decide model.
2. Pick (P, T) points; build fcc/bcc supercells (`tdi_build.py`); EOS refinement runs (Step A at 3-4 volumes).
3. Step A diagnostics on both phases; fix dt, k, supercell if the lattice is unstable.
4. Stage 1 / Stage 2 lambda ladders on a Slurm array; analyse the integrands before trusting any total.
5. DFT single points + FEP correction; compare ΔG with the literature crossover.

## Decisions I need from you
1. The (P, T) points (and whether below 200 GPa is wanted).
2. How many GPU-hours we may spend; 5 ps per lambda point is my default.
3. After the phase census: whether fcc/bcc superionic coverage at the chosen (P, T) is enough, or extra DFT frames are needed.
