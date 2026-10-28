"""'toyice': a cheap pair potential that mimics the GRACE situation for testing the TDI / insertion pipeline end to end (no GPU, no MLP).
  O-O  Born-Mayer repulsion  A exp(-r / rho)                      (O sublattice: fcc solid of soft spheres)
  O-H  Morse  D [(1 - exp(-a (r - r0)))^2 - 1], r0 = 1.0 A          (proton bound to O; finite core like GRACE's soft core)
  H-H  soft repulsion  B exp(-r / rho_H)                           (no H2 molecules; pure H = soft-sphere fluid)
All terms are finite at r = 0, so the ideal-gas end of every TDI path (all H, and O in the fluid path) is regular. Shifted-force cutoff rc (energy and force go
to zero at rc), so MD conserves energy. Units eV, A. Use as a model spec in the pipeline: `run_real.py toyice ...` (04_benchmark.get_calc_spec).
Parameters can be overridden with TOYICE="A_OO=..,rho_OO=..,D=..,a=..,r0=..,B=..,rho_H=..,rc=.." in the environment.
TOYICE_FIT=<toy_fit.py json>: force-matched spline pairs fitted to DFT snapshots instead (tdi/toy_fit.py; same cutoff handling, quadratic soft core).
"""
import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

DEFAULTS = dict(A_OO=1500.0, rho_OO=0.30, D=2.0, a=2.2, r0=1.0, B=60.0, rho_H=0.30, rc=3.0)


def params():
    p = dict(DEFAULTS)
    for kv in filter(None, os.environ.get("TOYICE", "").split(",")):
        k, v = kv.split("="); p[k.strip()] = float(v)
    if os.environ.get("TOYICE_FIT"):   # force-matched spline pairs from tdi/toy_fit.py instead of the analytic forms
        import json
        p["fit"] = json.load(open(os.environ["TOYICE_FIT"])); p["rc"] = p["fit"]["rc"]
    return p


def _spline_pair(fit, kind, r):
    """phi, phi' of a toy_fit.py spline pair (quadratic core below r_lo); r array"""
    from scipy.interpolate import BSpline
    q = fit["pairs"][("OO", "OH", "HH")[kind]]
    if "_bs" not in q:
        t = np.array(q["t"]); c = np.r_[q["c"], 0.0, 0.0]   # the two dropped end functions have zero weight
        q["_bs"] = BSpline(t, c, 3, extrapolate=False); q["_dbs"] = q["_bs"].derivative()
    r = np.asarray(r, float); rl, rc, K = q["r_lo"], fit["rc"], fit["K"]
    x = np.clip(r, rl, rc); v = np.nan_to_num(q["_bs"](x)); dv = np.nan_to_num(q["_dbs"](x)); dl = np.where(r < rl, rl - r, 0.0)
    e = np.where(r < rc, v - dl * dv + K * dl ** 2, 0.0); de = np.where(r < rc, dv - 2 * K * dl, 0.0)
    return e, de


def _pair(kind, r, p):
    """phi(r), dphi/dr for arrays r (unshifted)"""
    if kind == 0:   # O-O
        e = p["A_OO"] * np.exp(-r / p["rho_OO"]); return e, -e / p["rho_OO"]
    if kind == 1:   # O-H
        x = np.exp(-p["a"] * (r - p["r0"])); return p["D"] * ((1 - x) ** 2 - 1), 2 * p["D"] * (1 - x) * p["a"] * x
    e = p["B"] * np.exp(-r / p["rho_H"]); return e, -e / p["rho_H"]   # H-H


def pair_energy(kind, r, p=None, fit=None):
    """shifted-force pair energy and its derivative (zero beyond rc); spline pairs when a toy_fit.py fit is given (argument, p['fit'] or TOYICE_FIT)"""
    p = p or params(); fit = fit or p.get("fit")
    if fit is not None:
        return _spline_pair(fit, kind, r)
    rc = p["rc"]; r = np.asarray(r, float)
    e, de = _pair(kind, r, p); ec, dec = _pair(kind, np.array(rc), p)
    m = r < rc
    return np.where(m, e - ec - (r - rc) * dec, 0.0), np.where(m, de - dec, 0.0)


class ToyIce(Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(self, fit=None, **kw):
        super().__init__(**kw); self.p = params()
        if fit is not None:
            self.p = dict(self.p, fit=fit, rc=fit["rc"])

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms; n = len(a); p = self.p; rc = p["rc"]; cell = np.asarray(a.cell)
        isH = a.numbers == 1
        kind = np.where(~isH[:, None] & ~isH[None, :], 0, np.where(isH[:, None] & isH[None, :], 2, 1))   # n x n
        inv = np.linalg.inv(cell); s = a.positions @ inv
        ds = s[None, :, :] - s[:, None, :]; ds -= np.round(ds)   # minimum-image base, then the images within rc
        width = a.get_volume() / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)   # perpendicular widths
        nimg = [int(np.floor(rc / w + 0.5)) for w in width]   # base image has |ds| <= 1/2: images k with |k| <= rc/w + 1/2 can be within rc
        E = 0.0; F = np.zeros((n, 3)); vir = np.zeros((3, 3))
        for sx in range(-nimg[0], nimg[0] + 1):
            for sy in range(-nimg[1], nimg[1] + 1):
                for sz in range(-nimg[2], nimg[2] + 1):
                    D = (ds + np.array([sx, sy, sz])) @ cell   # D[i, j] = r_j - r_i (+ image)
                    d = np.linalg.norm(D, axis=-1)
                    m = d < rc
                    if sx == sy == sz == 0:
                        np.fill_diagonal(m, False)
                    if not m.any():
                        continue
                    ii, jj = np.nonzero(m); dd = d[m]; kk = kind[m]; e = np.zeros(len(dd)); de = np.zeros(len(dd))
                    for k in (0, 1, 2):
                        q = kk == k
                        if q.any():
                            e[q], de[q] = pair_energy(k, dd[q], p)
                    f = (de / dd)[:, None] * D[m]   # each ordered pair (i, j, image): half of the pair energy
                    E += 0.5 * e.sum(); np.add.at(F, ii, f); vir += 0.5 * (D[m][:, :, None] * f[:, None, :]).sum(0)
        stress = vir / a.get_volume()
        self.results = dict(energy=float(E), free_energy=float(E), forces=F, stress=stress.flat[[0, 4, 8, 5, 2, 1]])


def ghost_energy(pos_ghost, host_pos, host_is_H, cell, p=None):
    """interaction energy (eV) of each trial proton (m x 3) with a host frame, all periodic images within rc (same image rule as ToyIce); host_is_H: bool array"""
    p = p or params(); cell = np.asarray(cell); inv = np.linalg.inv(cell); rc = p["rc"]
    V = abs(np.linalg.det(cell)); width = V / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    nimg = [int(np.floor(rc / w + 0.5)) for w in width]
    ds = (host_pos @ inv)[None, :, :] - (pos_ghost @ inv)[:, None, :]; ds -= np.round(ds)
    out = np.zeros(len(pos_ghost))
    for sx in range(-nimg[0], nimg[0] + 1):
        for sy in range(-nimg[1], nimg[1] + 1):
            for sz in range(-nimg[2], nimg[2] + 1):
                r = np.linalg.norm((ds + np.array([sx, sy, sz])) @ cell, axis=-1)
                out += pair_energy(1, r[:, ~host_is_H], p)[0].sum(1) + pair_energy(2, r[:, host_is_H], p)[0].sum(1)
    return out
