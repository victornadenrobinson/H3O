"""Thermodynamic integration (TDI) engine for the fcc/bcc superionic H2O free-energy difference.
refs.py     analytic reference free energies (Einstein crystal with fixed centre of mass, ideal gas, harmonic network, kinetic terms)
engine.py   mixed Hamiltonians for the lambda stages, centre-of-mass constraint, one NVT run per lambda point
analyze.py  Gauss-Legendre nodes (optionally in lambda = x^p), block averages, integral with error bars
test_harmonic.py  end-to-end check against an exactly solvable harmonic crystal (no GPU needed)
"""
