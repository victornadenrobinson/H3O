# Literature notes (read from the PDFs, 2026-10-26)

## Wilson, Wong, Militzer, PRL 110, 151102 (2013): fcc vs bcc superionic water
**Result.** fcc oxygen sublattice has the lower Gibbs free energy than bcc under all conditions studied except 1 Mbar / 3000 K (difference about zero); transition predicted at 1.0 +- 0.5 Mbar. fcc has lower H mobility and perhaps a higher melting temperature. hcp O sublattice is not stable in MD (rapid distortion).

**Table I of the paper: G_bcc - G_fcc and its components, eV per H2O molecule (divide by 3 for per atom)**

| P (Mbar) | T (K) | dG | P dV | dU | -T dS |
|---|---|---|---|---|---|
| 1 | 3000 | 0.002(11) | 0.017(16) | -0.037(20) | 0.023(28) |
| 2 | 3000 | 0.027(8) | 0.064(20) | 0.019(16) | -0.055(26) |
| 5 | 3000 | 0.074(7) | 0.072(26) | 0.087(13) | -0.084(30) |
| 10 | 5000 | 0.065(8) | 0.054(24) | 0.102(18) | -0.095(31) |
| 40 | 2000 | 0.256(8) | 0.198(14) | 0.229(10) | -0.165(19) |
| 40 | 3000 | 0.214(7) | 0.145(12) | 0.177(20) | -0.108(24) |
| 40 | 5000 | 0.180(11) | 0.033(19) | 0.188(18) | -0.042(28) |

Per atom at 3000 K: 200 GPa 9.0(2.7) meV, **500 GPa 24.7(2.3) meV**, 4000 GPa 71(2) meV. Linear interpolation in P between the 2 and 5 Mbar rows gives about 19 meV/atom at 400 GPa (crude).
P dV and dU favour fcc increasingly with pressure, -T dS (bcc has more entropy: H can use both tetrahedral and octahedral sites) opposes it. Check: dG = P dV + dU - T dS holds in every row.

**Method.** VASP, PBE, PAW, 900 eV, 2x2x2 k-points, NVT, timestep 0.2 fs, 2000-5000 steps per run; cubic cells with 32 (fcc), 54 (bcc), 48 (hcp) molecules (96/162/144 atoms). Volumes from the variable-cell method of Hernandez et al. Two-step coupling-constant integration:
1. DFT -> classical: two-body spline pair potentials fitted to the DFT-MD forces (force matching, Izvekov et al.) plus a harmonic one-body term holding O on lattice sites (spring 5 eV/A^2); 5 evenly spaced lambda, trapezoid rule (integrand nearly a straight line).
2. Classical -> reference: harmonic oscillators for O on lattice sites + noninteracting ideal gas for H; 48 lambda concentrated at small lambda (integrand strongly curved there), averages by classical Monte Carlo; halving the lambda points changed G_fcc - G_bcc negligibly.
Convergence: doubling the fcc cell (96 -> 182 atoms) or k-grid 2x2x2 -> 4x4x4 changed U by < 0.01 eV/molecule. Supplement Table II: 40 Mbar 3000 K energies per molecule fcc 32 mol 24.811(11) vs 64 mol 24.819(9); bcc 54 mol 25.006(10) vs 108 mol 24.987(6).
**Path test (supplement Table III):** integral computed directly (all potentials on at once) vs through two sub-steps (H-H and O-O first, then O-H), eV/molecule: bcc 1 Mbar 3000 K: 81.42(2) vs 81.51(3); fcc 40 Mbar 3000 K: 989.22(3) vs 989.09(3). The text says the difference is "on the order of 0.001 eV per molecule", but the printed numbers differ by 0.09 and 0.13 eV/molecule. Unresolved; do not rely on either statement.
**Supplement Table I (MD-averaged total energy, E_fcc - E_bcc, eV/molecule):** 2 Mbar 3000 K -0.076(11) [2x2x2 k], -0.052(11) [3x3x3]; 10 Mbar 5000 K -0.073(14), -0.053(18); 40 Mbar 3000 K -0.216(15), -0.200(10). These are energies, not free energies, and they do not obviously match the dU column of the main Table I (e.g. 2 Mbar: dU = +0.019(16) there). Unexplained.
**Table II (paper): H diffusion constants** fcc smaller than bcc at all densities (e.g. 5000 K, 11 g/cc: 2.48 bcc vs 1.33 fcc), stated in "A^2 fs^-1"; the magnitudes look scaled (a true 2.48 A^2/fs would be 0.25 cm^2/s), so treat the unit as unreliable.

## Soubiran and Militzer, ApJ 806, 228 (2015): H2O-H2 miscibility
Same group, same TI machinery: DFT -> classical non-bonding pair potentials fitted to DFT-MD forces (5-7 lambda points suffice, Fig. 2), then classical -> ideal gas reference with many lambda (classical runs are about 1e5 times faster than DFT). G = F + P V. 0.2 fs, PBE, 1100 eV, 2-70 GPa, 1000-6000 K. No fcc/bcc.

## Gupta, Stixrude, Schlichting (arXiv:2407.04685): H2-H2O miscibility
Direct DFT two-phase coexistence (PBEsol, gamma point, 500 eV, 300-450 atoms, 150-300 ps), critical curve up to 2000 GPa, no H3O molecules found in the fluid. No MLP and no free-energy method for solids; not used for the fcc/bcc work. Possibly relevant for the H-water hull.

## Two Nature Communications papers the user named as having Delta G figures (not readable here; hosts blocked; titles/abstract facts from search excerpts only)
- https://www.nature.com/articles/s41467-022-32374-1 : "Thermodynamics of high-pressure ice phases explored with atomistic simulations", Nat. Commun. 13, 4707 (2022). PBE machine-learning potential; says the MLP reproduces the free energies of the PBE reference with very small residual error; bcc-oxygen ices VII, VII', X are one thermodynamic phase, superionic VII'' has a first-order boundary with VII'; X -> Pbcm near 300 GPa; liquid-solid coexistence by TI; phase diagram below 100 GPa. Whether it contains an fcc-bcc Delta G is unknown.
- https://www.nature.com/articles/s41467-025-67063-2 : Andriambariarijaona et al., "Observation of a mixed close-packed structure in superionic water", Nat. Commun. 17, 374 (2025; PMC12796318). Ultrafast X-ray diffraction on shock-compressed water: above about 150 GPa and 2500 K the patterns favour a mixed close-packed (fcc + hcp stacking) superionic phase over pure fcc; signatures of bcc and fcc together at lower pressure. Energetics not seen.
To use their Delta G numbers: upload the PDFs or screenshots of the figures (axis labels and units visible) and say whether the quantity is per atom or per molecule and which sign (G_bcc - G_fcc or the reverse).

## Not obtained
Cheng, Bethkenhagen, Pickard, Hamel, Nat. Phys. 17, 1228 (2021) (MLP free energies, fcc/bcc/Pbcm; UF-fluid reference): not available in this sandbox (hosts blocked); the user believes the paper is public and has ML dG numbers. If you can upload the PDF it can be read like these.
