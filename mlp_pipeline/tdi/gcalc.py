"""The free-energy arithmetic shared by run_real.py, collect.py, live.py and summarize.py (needs numpy and ase.units only)."""
import numpy as np
from tdi import analyze, refs


def stats(series, nb=5):
    s = np.asarray(series, float)
    if len(s) < 5 * nb:
        return float("nan"), float("nan")
    return analyze.block_stats(s, nblocks=nb)


def G_total_eV(st, dF):
    """whole-cell G (eV) from the TDI integral dF; the -kT ln V translation term applies only when the O centre of mass is held fixed (not for the pure-H fluid)"""
    kt = refs.kT(st["T"]); tr = 0.0 if st.get("no_com_term") else -kt * np.log(st["V"])
    return st["f_ref_cons_eV"] + dF + tr + st["f_kin_eV"] + st["PV_eV"]


def G_from_nodes(st, lam, w, series_list, N):
    ms = [stats(s) for s in series_list]; m = np.array([x[0] for x in ms]); s = np.array([x[1] for x in ms])
    dF, dFe = analyze.integrate(m, s, w)
    return G_total_eV(st, dF) / N * 1e3, dFe / N * 1e3, dF / N * 1e3, m, s
