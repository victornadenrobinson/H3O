"""Free-NVT stability verdict of one 17_eos_md.py result row: the single rule used by plot_pt.py, melt_table.py, summarize.py, 21_melt_summary.py and 22_eos_summary.py.
O lattice kept = O-sublattice label equal to the starting phase and D_O < D_O_MAX; then Lindemann ratio delta_L < DL_SOLID solid, < DL_MARGINAL marginal; anything else melted.
'unknown' when the row has no O_struct / D_O / delta_L (17_eos_md.py writes them only with more than 20 stored production frames): a short run is not evidence of melting."""
import numpy as np

D_O_MAX, DL_SOLID, DL_MARGINAL = 1e-5, 0.2, 0.5   # cm2/s, Lindemann ratio, Lindemann ratio


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def verdict(r, phase=None):
    """r: dict or pandas row. phase: the starting O lattice (default r['phase']).
    Solid branch: 'solid' | 'marginal' | 'melted' | 'unknown'; liquid branch: 'liquid' | 'not melted' | 'unknown'; pure H: 'fluid'."""
    ph = phase or r.get("phase", "fcc")
    if ph == "H":
        return "fluid"
    lab, do, dl = r.get("O_struct"), _num(r.get("D_O_cm2s")), _num(r.get("delta_L"))
    if not isinstance(lab, str) or not np.isfinite(do) or not np.isfinite(dl):
        return "unknown"
    keep = lab == ph and do < D_O_MAX
    v = "solid" if keep and dl < DL_SOLID else "marginal" if keep and dl < DL_MARGINAL else "melted"
    if r.get("branch", "solid") == "liquid":
        return "not melted" if v in ("solid", "marginal") else "liquid"
    return v
