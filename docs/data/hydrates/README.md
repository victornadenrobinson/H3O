# Hydrogen-hydrate (filled-ice / clathrate) structures: H2O:H2 compounds

| file | what | H/O | notes |
|---|---|---|---|
| `c2_pnas_10GPa.vesta` / `.vasp` | C2, space group Pna2_1, 4 O + 16 H per cell | 4 (H2O.H2 = H4O) | the .vasp is the .vesta expanded with `mlp_pipeline/vesta_to_poscar.py` (5 asymmetric atoms x 4 operations) |
| `c2_pnas_40GPa.vasp` | C2 at 40 GPa, 4 O + 16 H | 4 | **check before use**: O-H neighbours per O are 0.83 / 1.07 / 1.28 A (a 0.83 A O-H is too short for water), b grows slightly while a, c shrink 14 %; not used for the snapshots. The 10 GPa cell compressed by 0.90 (linear) has the same volume (~77 Å^3) |
| `c2_pnas_10GPa_s222_160atoms.vasp` | the 2x2x2 supercell of the 10 GPa C2 (32 O + 128 H, edges 8.33 x 12.0 x 8.39 A), supplied; verified equal to the 2x2x2 of `c2_pnas_10GPa.vasp` | 4 | base of the C2 snapshots (cheaper than 360 atoms) |
| `c3_pnas.vesta` / `.vasp` | C3, P4_1, 4 O + 24 H per cell | 6 (H2O.2H2 = H6O) | the .vasp equals the expansion of the .vesta (checked) |
| `C2_snapshots.extxyz` (**unrelaxed**: rigid molecules, 360 atoms), `C3_snapshots.extxyz` | Gamma-sized supercells (C2: 3x2x3 of the 20-atom 10 GPa cell = 360 atoms; C3: 2x2x2 = 224 atoms). **For the final C2 set use `--relax <model>` on a GPU node** (positions relaxed at each scaled volume so bonds adapt to the density, then rattled), linear volume scales 0.86-1.0 (C2) / 0.88-1.06 (C3), rigid-molecule volume scaling, Gaussian rattle 0 / 0.03 / 0.06 A (redrawn if H-H < 0.55 or O-H < 0.85 A), 12 frames each | | made by `mlp_pipeline/supercell_snapshots.py`; T_K = 1000 (DFT electronic temperature floor); geometries only, no labels |
Checks (O-H, H-H): C2 10 GPa and C3 are normal: two O-H at ~1.01 A and two hydrogen bonds at 1.5-1.56 A per O, H2 molecules (H-H 0.72-0.74 A).
