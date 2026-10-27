"""Grid of (P, T) points for the fcc/bcc H2O TDI scan, and the starting volume per atom for one point.
usage (repo root):  python3 mlp_pipeline/tdi/fcc_bcc_grid.py grid                  -> lines 'P T' (stable, sorted by P then T)
                    ~/grace-venv/bin/python mlp_pipeline/tdi/fcc_bcc_grid.py vpa PHASE T P     -> 'V/atom source' (source = fcc/bcc: the DFT EOS used) or 'NA'
                    ~/grace-venv/bin/python mlp_pipeline/tdi/fcc_bcc_grid.py coverage          -> table of the volume source for every grid point and phase (login node, no MD)
Grid: P = 100-600 GPa in steps of 50, T = 2000 K upwards in steps of 500 up to TMAX(P), the highest T at which the crystal is expected to still be solid. TMAX is a GUESS, not a measurement:
Cheng et al. give Tm about 4.6-4.8 kK at 200 GPa; our free NVT runs (6 ps, superheating not excluded) kept fcc H2O solid at 6000 K / 300 GPa (bcc melted at >= 6000 K), up to 6500 K at 400 GPa and 7000 K at 500 GPa.
Check afterwards with young_melt_scan.sbatch / young_melt_comp.sbatch (free NVT) on the points near TMAX, and read the quadrature estimate and MLP pressure in summarize.py.
Volume: the DFT EOS of the phase itself (17_eos_md.dft_vpa, 1000 K bins with >= 20 runs, interpolated in T); if that phase has no data there (fcc exists only at about 330-540 GPa), the DFT EOS of the OTHER phase (fcc and bcc volumes differ by a small
fraction of a percent at equal P, T in the DFT data we have: not checked everywhere; the MLP pressure at the highest lambda shows any mismatch)."""
import os, sys
TMAX = {100: 3500, 150: 4000, 200: 4500, 250: 5000, 300: 5500, 350: 6500, 400: 7000, 450: 7000, 500: 7000, 550: 7000, 600: 7000}

def grid(tmax=TMAX, t0=2000, dT=500):
    for P in sorted(tmax):
        for T in range(t0, tmax[P] + 1, dT): yield P, T

def vpa(phase, T, P, inv="inventory_binned.csv", min_runs=20):
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
    from importlib import import_module
    import contextlib, io, math
    import warnings; warnings.filterwarnings("ignore")   # pandas FutureWarning / numpy RankWarning from 17_eos_md, once per lookup otherwise
    inv_mod = import_module("01_bin_inventory")
    if not getattr(inv_mod, "_grid_cached", False):   # dft_vpa reloads the inventory csv on every call: load it once per process
        orig, cache = inv_mod.load_inventory, {}
        inv_mod.load_inventory = lambda f, *a, **k: cache.setdefault(f, orig(f, *a, **k)); inv_mod._grid_cached = True
    eos = import_module("17_eos_md")
    for ph in (phase, "bcc" if phase == "fcc" else "fcc"):
        try:
            with contextlib.redirect_stdout(io.StringIO()): v = float(eos.dft_vpa(inv, ph, float(T), [float(P)], 2.0, min_runs=min_runs)[0])
        except SystemExit: continue
        if math.isfinite(v): return v, ph
    return None

if __name__ == "__main__":
    m = sys.argv[1]
    if m == "grid":
        for P, T in grid(): print(P, T)
    elif m == "vpa":
        r = vpa(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])); print(f"{r[0]:.6f} {r[1]}" if r else "NA")
    elif m == "coverage":
        for P, T in grid():
            f, b = vpa("fcc", T, P), vpa("bcc", T, P)
            print(f"{P:4d} GPa {T:5d} K  fcc: {('%.4f (%s)' % (f[0], f[1])) if f else 'NA':>14}   bcc: {('%.4f (%s)' % (b[0], b[1])) if b else 'NA':>14}")
