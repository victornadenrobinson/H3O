#!/usr/bin/env python3
"""Two-stage thermodynamic-integration free energy, analysed from the raw files: tethered (Einstein) / ideal-gas reference -> pair-potential system (classical Monte Carlo,
implemented here) -> DFT (OUTCAR series of a mixed-Hamiltonian run, MIX_EFFPOT = lambda).

  F(V,T) = F0 + dF_mc + dF_dft
    F0      analytic: harmonic tethered atoms (U = k/2 u^2, KSPRING) + ideal gas for the untethered species, exact ln N!
    dF_mc   = int_0^1 <V_pair>_lambda dlambda in the system tethered atoms + free atoms, U_lambda = V_springs + lambda V_pair (Metropolis MC in numba, Gauss-Legendre in lambda = x^m)
    dF_dft  = int_0^1 <E_DFT - V_eff>_lambda dlambda from the OUTCAR 'totenminuseffpot' series (blocking errors, Boole / trapezoid / spline / Hermite and MBAR)
  G = F + P_target V  (first order; V = cell volume of the runs)

Atomic units inside (bohr, Ha). Case directory layout: <case>/basis/{INCAR,REFCAR,POSCAR}, <case>/cmc/{POSCAR,Poten_11.dat,Poten_12.dat,Poten_22.dat (AU), cmc_1.00},
<case>/lambda_0.000 ... lambda_1.000/{INCAR, out*/OUTCAR[.gz], OUTCAR[.gz]}.  Species order is the POSCAR/REFCAR order; Poten_ab = species a, b (1-based).

subcommands:  survey ROOT       read-only census of every lambda_* campaign under ROOT: KSPRING active / commented / absent, composition, T, tag type, OUTCAR counts
              vcheck CASE      do the dynamics code and the reference code evaluate the same V_eff?  (pair + spring) on the start configuration
              vscan CASE       same, regression over the first steps of lambda_0.000 (spring convention, interpolation offset)
              ref CASE         F0
              cmc CASE         dF_mc by MC           (needs numba)
              dft CASE         dF_dft from OUTCARs, pressure
              assemble CASE --cmc cmc.json --dft dft.json  -> F, G json
              diff A.json B.json                 G per atom difference of two assembled cases
              mix ICE ICEH PUREH [--n-add 27]            mixing per added H from three assembled jsons
              selftest
"""
import argparse, glob, gzip, json, os, re, sys, time
from math import lgamma, log, pi, sqrt
import numpy as np

HA_EV = 27.211386245988
BOHR = 0.529177210903
KB_HA = 3.166811563455608e-6          # Ha / K
AMU = 1822.888486209                  # electron masses per amu
MASS = {"O": 15.9994, "H": 1.00794}   # amu
GPA_PER_HA_BOHR3 = 29421.02648438959


# ----------------------------------------------------------------------------------------------------------------------------------------------- parsing
def read_poscar(path, default_names=("O", "H")):
    """VASP4 or VASP5 POSCAR/CONTCAR/REFCAR -> dict(lat (A), names, counts, pos (A, cartesian), frac)"""
    L = open(path).read().splitlines()
    scale = float(L[1].split()[0])
    lat = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * scale
    tok = L[5].split(); names = None
    if re.fullmatch(r"\d+", tok[0]):
        counts = [int(c) for c in tok]; i = 6
    else:
        names = tok; counts = [int(c) for c in L[6].split()]; i = 7
    if names is None:
        names = list(default_names) if len(counts) == len(default_names) else (["H"] if len(counts) == 1 else None)
        if names is None:
            raise ValueError(f"{path}: no species line and {len(counts)} species: give default_names")
    if L[i].strip().lower().startswith("s"): i += 1
    direct = L[i].strip().lower().startswith("d"); i += 1
    n = sum(counts)
    X = np.array([[float(v) for v in L[i + j].split()[:3]] for j in range(n)])
    pos = X @ lat if direct else X * scale
    return dict(lat=lat, names=names, counts=counts, pos=pos, frac=X if direct else pos @ np.linalg.inv(lat))


def read_incar(path):
    d = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); d[k.strip().upper()] = v.strip()
    return d


def read_table(path, units="au"):
    """two-column potential table (r, V); au: bohr, Ha; ev: A, eV (VASP files carry 'EFFPOT = \\' and trailing backslashes)"""
    rows = []
    for line in open(path):
        line = line.replace("\\", " ").replace("EFFPOT", " ").replace("=", " ")
        t = line.split()
        if len(t) == 2:
            try: rows.append((float(t[0]), float(t[1])))
            except ValueError: pass
    a = np.array(rows)
    if units == "ev": a = np.column_stack([a[:, 0] / BOHR, a[:, 1] / HA_EV])
    return a[:, 0], a[:, 1]


def dense_table(r, v, dr=5e-5):
    from scipy.interpolate import CubicSpline
    grid = np.arange(0.0, r[-1] + 2 * dr, dr)
    t = CubicSpline(r, v)(np.minimum(grid, r[-1])); t[grid > r[-1]] = 0.0
    return t


def find_table(case, a, b):
    """AU pair table for species a, b (0-based): --pots dir, cmc/, then the potentials of the lambda directories"""
    pots = os.environ.get("FE_POTS"); n = f"{a + 1}{b + 1}"
    c = []
    if pots: c += [os.path.join(pots, f"Poten_{n}.dat"), os.path.join(pots, f"Poten_1_{n}_AU.dat")]
    c += [os.path.join(case, "cmc", f"Poten_{n}.dat"), os.path.join(case, "lambda_0.000", "potentials", f"Poten_1_{n}_AU.dat"), os.path.join(case, "lambda_0.000", f"Poten_1_{n}_AU.dat"),
          os.path.join(case, "basis", "potentials", f"Poten_1_{n}_AU.dat"), os.path.join(case, "basis", f"Poten_{n}.dat")]
    for q in c:
        if os.path.exists(q): return q
    raise SystemExit(f"{case}: no AU pair table for species pair {n} (tried {c[:4]}...); give --pots DIR")


def load_case(case, poscar=None, refcar=None, incar=None, units="au", sites="refcar"):
    case = os.path.abspath(case)
    def first(*c):
        for q in c:
            if q and os.path.exists(q): return q
        return None
    refcar = first(refcar, *[os.path.join(case, d, "REFCAR") for d in ("basis", "lambda_0.000", "lambda_0.500", "lambda_1.000")], os.path.join(case, "basis", "POSCAR"),
                   os.path.join(case, "lambda_0.000", "POSCAR"))        # pure H: no tether sites needed
    if refcar is None: raise SystemExit(f"{case}: no REFCAR/POSCAR found (use --refcar)")
    ref = read_poscar(refcar)
    start = first(poscar, os.path.join(case, "cmc", "POSCAR"), os.path.join(case, "lambda_0.000", "out1", "POSCAR"), os.path.join(case, "lambda_0.000", "POSCAR"))
    pos = read_poscar(start, default_names=ref["names"])
    if pos["names"] != ref["names"] or pos["counts"] != ref["counts"]:
        raise ValueError(f"species mismatch POSCAR {pos['names']} {pos['counts']} vs REFCAR {ref['names']} {ref['counts']}")
    inc = read_incar(first(incar, os.path.join(case, "basis", "INCAR"), os.path.join(case, "lambda_0.000", "INCAR")))
    if "KSPRING" not in inc:                                     # some cases keep the tether constants only in the lambda directories
        for alt in ("lambda_0.000", "lambda_0.500", "lambda_1.000"):
            q = os.path.join(case, alt, "INCAR")
            if os.path.exists(q) and "KSPRING" in read_incar(q):
                inc = dict(inc, KSPRING=read_incar(q)["KSPRING"]); print(f"NOTE: KSPRING taken from {alt}/INCAR", file=sys.stderr); break
    if os.environ.get("FE_KSPRING"):
        inc = dict(inc, KSPRING=os.environ["FE_KSPRING"]); print(f"NOTE: KSPRING overridden: {inc['KSPRING']}", file=sys.stderr)
    names, counts = pos["names"], pos["counts"]
    if "KSPRING" not in inc and any(n != "H" for n in names):
        raise SystemExit(f"{case}: no KSPRING in basis/INCAR or the lambda INCARs, but species {names} include a non-H atom: refusing to assume untethered")
    typ = np.repeat(np.arange(len(names)), counts)
    ks = [float(x) for x in inc["KSPRING"].split()] if "KSPRING" in inc else [0.0] * len(names)
    if len(ks) != len(names):
        print(f"NOTE: KSPRING {ks} for species {names}: using the first {len(names)} values, padded with 0", file=sys.stderr)
        ks = (ks + [0.0] * len(names))[:len(names)]
    lat = pos["lat"]
    if not np.allclose(lat, np.diag(np.diag(lat)), atol=1e-8): raise ValueError("MC kernel needs an orthorhombic cell")
    tabs = {}
    ns = len(names); rc = 0.0
    tab = None
    for a in range(ns):
        for b in range(a, ns):
            r, v = read_table(find_table(case, a, b), units)
            t = dense_table(r, v); rc = max(rc, r[-1]); tabs[(a, b)] = (t, r[-1])
    raw = {}
    for a in range(ns):
        for b in range(a, ns):
            raw[(a, b)] = raw[(b, a)] = read_table(find_table(case, a, b), units)
    npt = max(len(t) for t, _ in tabs.values())
    tab = np.zeros((ns, ns, npt))
    rcs = np.zeros((ns, ns))
    for (a, b), (t, rmax) in tabs.items():
        tab[a, b, :len(t)] = t; tab[b, a, :len(t)] = t; rcs[a, b] = rcs[b, a] = rmax
    site_pos = ref["pos"].copy()
    if sites == "poscar":                                       # hypothesis test: tether the O to the positions of the (disordered) start snapshot instead of the ideal lattice
        site_pos[:counts[0]] = pos["pos"][:counts[0]] if names[0] == "O" else site_pos[:counts[0]]
        for s_, nm in enumerate(names):
            if nm == "O": idx = typ == s_; site_pos[idx] = pos["pos"][idx]
    return dict(case=case, names=names, counts=counts, typ=typ, box=np.diag(lat) / BOHR, pos=pos["pos"] / BOHR, sites=site_pos / BOHR,
                k=np.array([ks[t] for t in typ]) / HA_EV * BOHR ** 2, tab=tab, rcs=rcs, dr=5e-5, T=float(inc["TEBEG"]), incar=inc, raw=raw,
                masses=np.array([MASS[names[t]] for t in typ]) * AMU)


def mic(d, box):
    return d - box * np.floor(d / box + 0.5)


# ----------------------------------------------------------------------------------------------------------------------------------------------- energies
def pair_energy_total(S, pos=None, mode="spline"):
    """total pair energy, minimum image. mode 'spline': cubic spline of the table (CMC-like); 'linear': linear interpolation between the 62 table nodes"""
    pos = S["pos"] if pos is None else pos
    N = len(pos); e = 0.0
    for i in range(N - 1):
        d = mic(pos[i + 1:] - pos[i], S["box"]); r = np.sqrt((d ** 2).sum(1)); tj = S["typ"][i + 1:]
        for t in np.unique(tj):
            m = (tj == t) & (r < S["rcs"][S["typ"][i], t])
            if mode == "linear":
                rr, vv = S["raw"][(S["typ"][i], t)]; e += float(np.interp(r[m], rr, vv).sum()); continue
            u = r[m] / S["dr"]; k = u.astype(int); f = u - k; tb = S["tab"][S["typ"][i], t]
            e += float(((1 - f) * tb[k] + f * tb[k + 1]).sum())
    return e


def spring_energy(S, pos=None, sites=None, how="mic"):
    pos = S["pos"] if pos is None else pos
    sites = S["sites"] if sites is None else sites
    m = S["k"] > 0
    d = pos[m] - sites[m]
    if how == "mic": d = mic(d, S["box"])
    return 0.5 * float((S["k"][m, None] * d ** 2).sum())


def F0_reference(S, T=None, V=None):
    """Einstein O (classical harmonic, includes kinetic) + ideal gas H/any untethered species with exact ln N!, Ha per cell. T in K."""
    T = S["T"] if T is None else T
    V = float(np.prod(S["box"])) if V is None else V
    kT = KB_HA * T; F = 0.0; parts = {}
    for s, name in enumerate(S["names"]):
        n = S["counts"][s]; idx = np.where(S["typ"] == s)[0]; m = S["masses"][idx[0]]; k = S["k"][idx[0]]
        if k > 0:
            w = sqrt(k / m); f = n * 3 * kT * log(w / kT)
        else:
            lam3 = (2 * pi / (m * kT)) ** 1.5
            f = kT * (lgamma(n + 1) - n * log(V / lam3))
        parts[name] = f; F += f
    return F, parts


# ----------------------------------------------------------------------------------------------------------------------------------------------- MC (numba)
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                           # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _vi(x, i, pos, typ, tab, dr, rcs, box):
    """pair energy of an atom of species typ[i] at x with every other atom (j != i), minimum image"""
    e = 0.0; ti = typ[i]
    for j in range(pos.shape[0]):
        if j == i: continue
        dx = x[0] - pos[j, 0]; dy = x[1] - pos[j, 1]; dz = x[2] - pos[j, 2]
        dx -= box[0] * np.floor(dx / box[0] + 0.5); dy -= box[1] * np.floor(dy / box[1] + 0.5); dz -= box[2] * np.floor(dz / box[2] + 0.5)
        r2 = dx * dx + dy * dy + dz * dz; tj = typ[j]; rc = rcs[ti, tj]
        if r2 < rc * rc:
            u = np.sqrt(r2) / dr; k = int(u); f = u - k
            e += (1.0 - f) * tab[ti, tj, k] + f * tab[ti, tj, k + 1]
    return e


@njit(cache=True)
def _sweeps(nsw, pos, typ, sites, kspr, tab, dr, rcs, box, lam, beta, delta, ep, series, spring_series, acc, tries):
    N = pos.shape[0]; xn = np.empty(3)
    for s in range(nsw):
        for i in range(N):
            t = typ[i]; d = delta[t]
            for a in range(3): xn[a] = pos[i, a] + d * (2.0 * np.random.random() - 1.0)
            eo = _vi(pos[i], i, pos, typ, tab, dr, rcs, box); en = _vi(xn, i, pos, typ, tab, dr, rcs, box)
            dE = lam * (en - eo)
            if kspr[i] > 0.0:
                so = 0.0; sn = 0.0
                for a in range(3):
                    so += (pos[i, a] - sites[i, a]) ** 2; sn += (xn[a] - sites[i, a]) ** 2
                dE += 0.5 * kspr[i] * (sn - so)
            tries[t] += 1
            if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
                for a in range(3): pos[i, a] = xn[a]
                ep += en - eo; acc[t] += 1
        series[s] = ep
        sp = 0.0
        for i in range(N):
            if kspr[i] > 0.0:
                for a in range(3): sp += 0.5 * kspr[i] * (pos[i, a] - sites[i, a]) ** 2
        spring_series[s] = sp
    return ep


def block_err(x):
    """standard error of the mean of a correlated series: largest block-average error over block sizes with >= 16 blocks"""
    x = np.asarray(x, float); n = len(x); errs = []; b = 1
    while n // b >= 16:
        m = n // b; errs.append(x[:m * b].reshape(m, b).mean(1).std(ddof=1) / sqrt(m)); b *= 2
    return max(errs) if errs else float("nan")


def gl_nodes(n, m=4):
    xg, wg = np.polynomial.legendre.leggauss(n)
    x = (xg + 1) / 2; w = wg / 2
    return x ** m, w * m * x ** (m - 1)


def mc_node(args):
    S, lam, nsw, neq, seed, nramp = args
    _seed(int(seed)); rng = np.random.default_rng(int(seed))
    box = S["box"]; N = len(S["typ"]); typ = S["typ"].astype(np.int64)
    pos = S["pos"].copy()
    own = S["k"] > 0
    pos[own] = S["sites"][own] + mic(S["pos"][own] - S["sites"][own], box)         # unwrapped near the sites
    pos[~own] = rng.random(((~own).sum(), 3)) * box                                 # start untethered species uniformly (ideal gas)
    beta = 1.0 / (KB_HA * S["T"]); ns = len(S["names"])
    delta = np.array([0.5 * box.min() if not (S["k"][typ == s] > 0).any() else 0.05 for s in range(ns)])
    acc = np.zeros(ns); tries = np.zeros(ns)
    ep = pair_energy_total(S, pos)
    series = np.empty(max(nsw, neq // nramp)); sp = np.empty_like(series)
    tune = 50
    for stage in range(1, nramp + 1):                                              # lambda ramp + step-size tuning
        lam_s = lam * (stage / nramp) ** 2
        for rep in range(max(1, (neq // nramp) // tune)):
            acc[:] = 0; tries[:] = 0
            ep = _sweeps(tune, pos, typ, S["sites"], S["k"], S["tab"], S["dr"], S["rcs"], box, lam_s, beta, delta, ep, series, sp, acc, tries)
            rate = acc / np.maximum(tries, 1)
            delta = np.minimum(delta * np.where(rate > 0.4, 1.2, np.where(rate < 0.3, 0.8, 1.0)), 0.5 * box.min())
    acc[:] = 0; tries[:] = 0
    ep = _sweeps(nsw, pos, typ, S["sites"], S["k"], S["tab"], S["dr"], S["rcs"], box, lam, beta, delta, ep, series, sp, acc, tries)
    check = pair_energy_total(S, pos)
    s = series[:nsw]
    return dict(lam=float(lam), mean=float(s.mean()), err=float(block_err(s)), spring=float(sp[:nsw].mean()), acc=(acc / np.maximum(tries, 1)).tolist(),
                drift=float(ep - check), mean_first_half=float(s[:nsw // 2].mean()), mean_second_half=float(s[nsw // 2:].mean()))


def cmc_integral(S, nodes=20, m=4, sweeps=20000, eq=4000, procs=1, seed=1, nramp=8):
    lams, w = gl_nodes(nodes, m)
    jobs = [(S, l, sweeps, eq, seed + 7919 * i, nramp) for i, l in enumerate(lams)]
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool: res = pool.map(mc_node, jobs)
    else:
        res = [mc_node(j) for j in jobs]
    f = np.array([r["mean"] for r in res]); e = np.array([r["err"] for r in res])
    return dict(dF_cmc=float((w * f).sum()), dF_cmc_err=float(np.sqrt(((w * e) ** 2).sum())), nodes=nodes, m=m, sweeps=sweeps, results=res, weights=w.tolist())


# ----------------------------------------------------------------------------------------------------------------------------------------------- DFT stage
def parse_outcar(path, tag="totenminuseffpot", field=0):
    op = gzip.open if path.endswith(".gz") else open
    tm, te, pe = [], [], []
    with op(path, "rt", errors="ignore") as f:
        for line in f:
            if tag in line: tm.append(float(line.split()[field - 1]) if field else float(line.split("=")[1].split()[0]))
            elif "free  energy   TOTEN" in line: te.append(float(line.split("=")[1].split()[0]))
            elif "external pressure" in line: pe.append(float(line.split("=")[1].split()[0]))
    n = min(len(tm), len(te), len(pe)) if pe else min(len(tm), len(te))
    return np.array(tm[:n]), np.array(te[:n]), (np.array(pe[:n]) if pe else np.array([]))


def outcar_files(ldir):
    segs = glob.glob(os.path.join(ldir, "out*", "OUTCAR*"))
    def num(p): return int(re.search(r"out(\d+)", p).group(1))
    by = {}
    for p in segs: by.setdefault(num(p), []).append(p)
    files = [sorted(by[k], key=lambda p: p.endswith(".gz"))[0] for k in sorted(by)]
    top = sorted(glob.glob(os.path.join(ldir, "OUTCAR*")), key=lambda p: p.endswith(".gz"))
    if top: files.append(top[0])
    return files


def lambda_series(ldir, tag="totenminuseffpot", verbose=True, field=0):
    inc = read_incar(os.path.join(ldir, "INCAR")); lam = float(inc["MIX_EFFPOT"].split()[0])
    dV, TE, PE = [], [], []; first = True
    for p in outcar_files(ldir):
        a, b, c = parse_outcar(p, tag, field)
        if verbose: print(f"    {os.path.relpath(p, ldir):28s} {len(a):6d} steps", flush=True)
        if len(a) == 0: continue
        if not first: a, b, c = a[1:], b[1:], (c[1:] if len(c) else c)               # restart repeats the last configuration
        first = False; dV.append(a); TE.append(b); PE.append(c)
    if not dV: raise SystemExit(f"{ldir}: no '{tag}' lines found in {[os.path.basename(f) for f in outcar_files(ldir)][:3]}...: check the tag (liquids: --tag V_KS-V_EP --field 4)")
    return lam, np.concatenate(dV), np.concatenate(TE), (np.concatenate(PE) if PE and len(PE[0]) else np.array([])), float(inc.get("POTIM", "nan").split()[0])


TAG_FIELD = ["totenminuseffpot", 0]


def lambda_series_com(ldir, S, stride=5, verbose=True):
    """dV and the translation-invariant dV' = dV + (1/2) k N_O |m|^2 (m = O centre of mass relative to the tether sites), sampled every `stride` steps"""
    own = S["k"] > 0; kk = S["k"][own][0]; n_o = int(own.sum()); N = len(S["typ"])
    inc = read_incar(os.path.join(ldir, "INCAR")); lam = float(inc["MIX_EFFPOT"].split()[0])
    dV, dVp, sc = [], [], []; first = True
    for p in outcar_files(ldir):
        a = []; b = []; c = []
        for i, (pos, te, tm) in enumerate(outcar_positions(p, N, stride, TAG_FIELD[0], TAG_FIELD[1])):
            if i == 0 and not first: continue
            u = mic(pos[own] / BOHR - S["sites"][own], S["box"]); m2 = float((u.mean(0) ** 2).sum()); e = 0.5 * kk * n_o * m2
            a.append(tm / HA_EV); b.append(tm / HA_EV + e); c.append(e)
        first = False
        if verbose: print(f"    {os.path.relpath(p, ldir):28s} {len(a):5d} sampled steps", flush=True)
        dV += a; dVp += b; sc += c
    if not dV: raise SystemExit(f"{ldir}: no POSITION blocks / steps found (is the OUTCAR written with NWRITE >= 2?)")
    return lam, np.array(dV), np.array(dVp), np.array(sc), float(inc.get("POTIM", "nan").split()[0]) * stride


def mbar(du, counts, betas_lam):
    """MBAR for states k with reduced potential u_k(n) = betas_lam[k] * du(n) (relative); du concatenated over states in order, counts per state.
    Returns dimensionless free energies f_k - f_0."""
    from scipy.special import logsumexp
    K = len(counts); N = np.asarray(counts, float)
    U = np.outer(betas_lam, du)                           # K x n
    f = np.zeros(K)
    for _ in range(5000):
        lw = logsumexp(f[:, None] - U + np.log(N)[:, None], axis=0)           # log sum_k N_k exp(f_k - u_k(n)) for each n
        fn = -logsumexp(-U - lw[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < 1e-10: f = fn; break
        f = fn
    return f


def boole(f, h=0.25):
    return 2 * h / 45 * (7 * f[0] + 32 * f[1] + 12 * f[2] + 32 * f[3] + 7 * f[4])


def dft_stage(case, discard=0.2, tag="totenminuseffpot", verbose=True, com=False, stride=5, field=0, end=1.0):
    case = os.path.abspath(case)
    ldirs = sorted(d for d in glob.glob(os.path.join(case, "lambda_*")) if os.path.isdir(d))
    lam, mean, err, ser, info = [], [], [], [], []
    S = load_case(case) if com else None; unprimed = []; scom_mean = []
    if com and not (S["k"] > 0).any(): com = False; S = None; print("no tethered species: COM correction not needed, using plain dV", flush=True)
    T = None
    for d in ldirs:
        if verbose: print(f"  {os.path.basename(d)}", flush=True)
        if com:
            l, dV0, dVp, sc, potim = lambda_series_com(d, S, stride, verbose)
            n0 = int(len(dVp) * discard); n1 = int(len(dVp) * end); x = dVp[n0:n1]; unprimed.append(float(dV0[n0:n1].mean())); scom_mean.append(float(sc[n0:n1].mean())); te = np.zeros(len(dVp)); dV = dVp * HA_EV; pe = np.array([])
            if verbose: print(f"    <dV> = {dV0[n0:n1].mean():.4f}, <S_com> = {sc[n0:n1].mean():.4f} (quarters {['%.4f' % q.mean() for q in np.array_split(sc[n0:n1], 4)]}), <dV'> = {x.mean():.4f}", flush=True)
        else:
            l, dV, te, pe, potim = lambda_series(d, tag, verbose, field)
            n0 = int(len(dV) * discard); n1 = int(len(dV) * end); x = dV[n0:n1] / HA_EV
        blocks = [float(b.mean()) for b in np.array_split(x, 4)]
        veff = (te[n0:n1] - l * dV[n0:n1]) / HA_EV if not com else np.zeros(1)
        lam.append(l); mean.append(float(x.mean())); err.append(float(block_err(x))); ser.append(x)
        info.append(dict(lam=l, steps=int(len(dV)), ps=float(len(dV) * potim / 1000), kept=int(len(x)), mean_Ha=float(x.mean()), err_Ha=float(block_err(x)),
                         quarter_means_Ha=blocks, veff_mean_Ha=float(veff.mean()), ext_pressure_kB=(float(pe[n0:n1].mean()) if len(pe) else None)))
        if verbose:
            i = info[-1]
            print(f"    lambda {l:.3f}: {i['steps']} steps = {i['ps']:.2f} ps, kept {i['kept']}, <dV> = {i['mean_Ha']:.6f} +- {i['err_Ha']:.6f} Ha, <V_eff> = {i['veff_mean_Ha']:.4f} Ha, quarter means {['%.4f' % q for q in blocks]}", flush=True)
    order = np.argsort(lam); lam = np.array(lam)[order]; mean = np.array(mean)[order]; err = np.array(err)[order]; ser = [ser[i] for i in order]
    res = dict(lambdas=lam.tolist(), info=[info[i] for i in order], com=bool(com))
    if com: res["dV_unprimed_means"] = [unprimed[i] for i in order]; res["S_com_means"] = [scom_mean[i] for i in order]
    from scipy.interpolate import CubicSpline
    wt = np.zeros(len(lam)); wt[:-1] += np.diff(lam) / 2; wt[1:] += np.diff(lam) / 2          # trapezoid weights (any node set)
    res["trapezoid"] = float(np.trapezoid(mean, lam)); res["trap_err"] = float(np.sqrt(((wt * err) ** 2).sum()))
    res["spline"] = float(CubicSpline(lam, mean).integrate(0, 1))
    if len(lam) == 5 and np.allclose(lam, [0, .25, .5, .75, 1.0]):
        w = np.array([7, 32, 12, 32, 7]) * 2 * 0.25 / 45
        res["boole"] = float(boole(mean)); res["boole_err"] = float(np.sqrt(((w * err) ** 2).sum()))
        res["simpson3"] = float((mean[0] + 4 * mean[2] + mean[4]) / 6)
    T = float(read_incar(os.path.join(ldirs[0], "INCAR"))["TEBEG"].split()[0])
    beta = 1.0 / (KB_HA * T)
    # fluctuation-dissipation check: d<dV>/dlambda = -beta var(dV) must equal the finite difference of the means if every window is in equilibrium
    sd = np.array([float(x.std()) for x in ser]); slope = -beta * sd ** 2
    res["fdt"] = dict(std_Ha=sd.tolist(), slope=slope.tolist(), fd_slope=(np.diff(mean) / np.diff(lam)).tolist(),
                      interval_obs=np.diff(mean).tolist(), interval_pred=[float(np.diff(lam)[i] * (slope[i] + slope[i + 1]) / 2) for i in range(len(lam) - 1)])
    res["hermite"] = float(sum((lam[i + 1] - lam[i]) / 2 * (mean[i] + mean[i + 1]) + (lam[i + 1] - lam[i]) ** 2 / 12 * (slope[i] - slope[i + 1]) for i in range(len(lam) - 1)))
    du = np.concatenate(ser); cnt = [len(s) for s in ser]
    f = mbar(du, cnt, beta * lam)
    res["mbar"] = float(f[-1] / beta) if abs(lam[-1] - 1) < 1e-9 else None
    res["T"] = T
    if "boole" in res: res["dF_dft"] = res["boole"]; res["dF_dft_err"] = res["boole_err"]                  # five equal nodes (the original ladders)
    else: res["dF_dft"] = res["spline"]; res["dF_dft_err"] = res["trap_err"]; print("  non-standard lambda ladder: dF_dft = cubic spline over the nodes (statistical error from the trapezoid weights); compare the other estimators", flush=True)
    return res


def pressure_gpa(case, discard=0.2, tag="totenminuseffpot", field=0):
    d = os.path.join(os.path.abspath(case), "lambda_1.000")
    l, dV, te, pe, potim = lambda_series(d, verbose=False, tag=tag, field=field)
    inc = read_incar(os.path.join(d, "INCAR")); T = float(inc["TEBEG"].split()[0])
    pos = read_poscar(os.path.join(d, "POSCAR")); V = abs(np.linalg.det(pos["lat"])); N = sum(pos["counts"])
    x = pe[int(len(pe) * discard):] * 0.1
    kin = N * KB_HA * T / (V / BOHR ** 3) * GPA_PER_HA_BOHR3
    return float(x.mean() + kin), float(block_err(x)), V / BOHR ** 3


def outcar_steps(path, nmax, N, tag="totenminuseffpot", field=0):
    """first nmax ionic steps of an OUTCAR: (positions A (N,3), TOTEN eV, totenminuseffpot eV)"""
    op = gzip.open if path.endswith(".gz") else open
    out = []; pos = None; te = tm = None
    with op(path, "rt", errors="ignore") as f:
        lines = iter(f)
        for line in lines:
            if "POSITION" in line and "TOTAL-FORCE" in line:
                next(lines); pos = np.array([[float(v) for v in next(lines).split()[:3]] for _ in range(N)])
            elif "free  energy   TOTEN" in line: te = float(line.split("=")[1].split()[0])
            elif tag in line:
                tm = float(line.split()[field - 1]) if field else float(line.split("=")[1].split()[0])
                if pos is not None and te is not None: out.append((pos, te, tm))
                if len(out) >= nmax: break
    return out



@njit(cache=True)
def _pair_by_type(pos, typ, tab, dr, rcs, box, out):
    N = pos.shape[0]
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]; dy = pos[i, 1] - pos[j, 1]; dz = pos[i, 2] - pos[j, 2]
            dx -= box[0] * np.floor(dx / box[0] + 0.5); dy -= box[1] * np.floor(dy / box[1] + 0.5); dz -= box[2] * np.floor(dz / box[2] + 0.5)
            r2 = dx * dx + dy * dy + dz * dz; ti = typ[i]; tj = typ[j]; rc = rcs[ti, tj]
            if r2 < rc * rc:
                u = np.sqrt(r2) / dr; k = int(u); f = u - k
                a = min(ti, tj); b = max(ti, tj)
                out[a, b] += (1.0 - f) * tab[ti, tj, k] + f * tab[ti, tj, k + 1]


def pair_by_type(S, pos):
    ns = len(S["names"]); out = np.zeros((ns, ns))
    _pair_by_type(np.ascontiguousarray(pos), S["typ"].astype(np.int64), S["tab"], S["dr"], S["rcs"], S["box"], out)
    return out


def outcar_positions(path, N, stride=1, tag="totenminuseffpot", field=0):
    """generator over ionic steps: (positions A, TOTEN eV, totenminuseffpot eV); only every stride-th step is parsed and yielded (steps counted from the start of the file)"""
    op = gzip.open if path.endswith(".gz") else open
    pos = te = None; step = 0
    with op(path, "rt", errors="ignore") as f:
        lines = iter(f)
        for line in lines:
            if "POSITION" in line and "TOTAL-FORCE" in line:
                next(lines)
                if step % stride == 0: pos = np.array([[float(v) for v in next(lines).split()[:3]] for _ in range(N)])
                else:
                    for _ in range(N): next(lines)
                    pos = None
            elif "free  energy   TOTEN" in line: te = float(line.split("=")[1].split()[0])
            elif tag in line:
                if pos is not None and te is not None: yield pos, te, (float(line.split()[field - 1]) if field else float(line.split("=")[1].split()[0]))
                step += 1; pos = None


def cmd_lam0(a):
    """direct (non-MCMC) sampling of the lambda = 0 CMC ensemble: Gaussian O around the tether sites (variance kT/k), uniform ideal-gas H; pair energy by species pair"""
    S = load_case(a.case, a.poscar, a.refcar); rng = np.random.default_rng(a.seed); kT = KB_HA * S["T"]
    own = S["k"] > 0; box = S["box"]; N = len(S["typ"]); ns = len(S["names"])
    if not own.any(): raise SystemExit("no tethered species in this case: nothing to compare")
    variants = [("REFCAR sites", S["sites"][own]), ("POSCAR O positions as sites", S["pos"][own] + 0 * S["sites"][own])]
    print("direct sampling of the lambda = 0 reference ensemble (Gaussian tethered atoms, uniform ideal-gas atoms), pair energy by species pair")
    print(f"{'sites':30s} {'k scale':>7s} {'<V_pair>':>10s} {'+-':>7s}   " + "  ".join(f"{S['names'][i]}-{S['names'][j]:s}".rjust(9) for i in range(ns) for j in range(i, ns)))
    for name, sites in variants:
        for ks in (1.0, 0.5, 2.0, 0.25):
            sig = np.sqrt(kT / (S["k"][own][0] * ks)); acc = []
            for _ in range(a.samples):
                pos = np.empty((N, 3)); pos[own] = sites + rng.normal(0, sig, sites.shape); pos[~own] = rng.random(((~own).sum(), 3)) * box
                acc.append(pair_by_type(S, pos))
            A = np.array(acc); tot = A.sum((1, 2))
            cols = "  ".join(f"{A[:, i, j].mean():9.4f}" for i in range(ns) for j in range(i, ns))
            print(f"{name:30s} {ks:7.2f} {tot.mean():10.4f} {tot.std() / sqrt(len(tot)):7.4f}   {cols}")
    print("O-O depends on the O distribution only; O-H and H-H of an ideal gas do not depend on where the O sit.")


def cmd_vensemble(a):
    """classical pair+spring ensemble as sampled by VASP at lambda = 0 (same Hamiltonian as the CMC at lambda = 1): <V_pair>, <U_spring> and <V_eff> from the OUTCAR trajectory"""
    S = load_case(a.case, a.poscar, a.refcar); own = S["k"] > 0; N = len(S["typ"]); kk = S["k"][own][0]
    l0 = os.path.join(S["case"], a.lam_dir); lam = float(read_incar(os.path.join(l0, "INCAR"))["MIX_EFFPOT"].split()[0])
    rows = []; step = 0; first = True
    for f in outcar_files(l0):
        for i, (pos, te, tm) in enumerate(outcar_positions(f, N, 1, a.tag, a.field)):
            if i == 0 and not first: continue
            if step % a.every == 0:
                P = pos / BOHR; u = mic(P[own] - S["sites"][own], S["box"])
                rows.append(((te - lam * tm) / HA_EV, pair_by_type(S, P).sum(), 0.5 * kk * (u ** 2).sum(), float((u.mean(0) ** 2).sum()), float(np.sqrt((u ** 2).sum(1).mean()) * BOHR), tm / HA_EV))
            step += 1
        first = False
        print(f"  read {os.path.relpath(f, l0)}: {len(rows)} samples so far", flush=True)
    R = np.array(rows); n0 = int(len(R) * a.discard); R = R[n0:]
    print(f"{len(R)} samples (every {a.every} steps, first {a.discard:.0%} dropped) of the VASP lambda = 0 run")
    for name, c in (("<V_eff> (TOTEN)", 0), ("<V_pair> own tables", 1), ("<U_spring>", 2)):
        q = [float(b.mean()) for b in np.array_split(R[:, c], 4)]
        print(f"  {name:20s} {R[:, c].mean():10.4f} +- {block_err(R[:, c]):.4f} Ha   quarters {['%.3f' % v for v in q]}")
    print(f"  V_eff - V_pair - U_spring = {np.mean(R[:, 0] - R[:, 1] - R[:, 2]):+.5f} +- {block_err(R[:, 0] - R[:, 1] - R[:, 2]):.5f} Ha (0 if VASP and own energies agree)")
    print(f"  O rms displacement from the ideal-lattice sites (A): quarters {['%.4f' % q.mean() for q in np.array_split(R[:, 4], 4)]}   <E_DFT - V_eff> quarters (Ha): {['%.3f' % q.mean() for q in np.array_split(R[:, 5], 4)]}")
    print("     (at lambda = 1 a falling dV together with a growing O displacement = the O sublattice is rearranging; a flat displacement = the drift is in the H subsystem)")
    n_o = int(own.sum()); kT = KB_HA * S["T"]; m2 = R[:, 3]
    print(f"  O centre of mass relative to the tether sites: <|m|^2> = {m2.mean():.5f} bohr^2 (rms {np.sqrt(m2.mean()) * BOHR:.4f} A), quarters {['%.5f' % q.mean() for q in np.array_split(m2, 4)]};")
    print(f"     free tethered O: 3kT/(N_O k) = {3 * kT / (n_o * kk):.5f} bohr^2 (rms {np.sqrt(3 * kT / (n_o * kk)) * BOHR:.4f} A); the coupled system gives <= this.  Spring energy of that mode: <(1/2) k N |m|^2> = {0.5 * kk * n_o * m2.mean():.4f} Ha (equipartition 3/2 kT = {1.5 * kT:.4f} Ha)")
    print("     a much larger <|m|^2> or a monotone trend over the quarters means the centre of mass is NOT held by the springs (drift removal / momentum handling in the edited VASP): the dV averages then carry a spring-energy bias of up to that size")



def _incar_flags(path):
    """(active KSPRING string or None, commented-out KSPRING string or None, MIX_EFFPOT float or None, TEBEG, POTIM) read from the raw lines"""
    act = com = mix = T = dt = None
    for line in open(path, errors="ignore"):
        t = line.strip()
        if t.startswith("#"):
            if "KSPRING" in t and com is None: com = t.lstrip("# ").split("=", 1)[-1].split("#")[0].strip()
            continue
        key = t.split("=", 1)[0].strip().upper() if "=" in t else ""
        val = t.split("=", 1)[1].split("#")[0].strip() if "=" in t else ""
        if key == "KSPRING": act = val
        elif key == "MIX_EFFPOT": mix = float(val.split()[0])
        elif key == "TEBEG": T = float(val.split()[0])
        elif key == "POTIM": dt = float(val.split()[0])
    return act, com, mix, T, dt


def _first_tag(ldir, maxlines=400000):
    for f in outcar_files(ldir)[:1]:
        op = gzip.open if f.endswith(".gz") else open
        try:
            with op(f, "rt", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if "totenminuseffpot" in line: return "totenminuseffpot"
                    if "V_KS-V_EP" in line: return "V_KS-V_EP"
                    if i > maxlines: break
        except OSError: return "unreadable"
    return "none"


def cmd_survey(a):
    root = os.path.abspath(os.path.expanduser(a.root)); camps = {}
    for d, dirs, files in os.walk(root):
        if os.path.basename(d).startswith("lambda") and "INCAR" in files:
            camps.setdefault(os.path.dirname(d), []).append(d)
        dirs[:] = [x for x in dirs if not (x.startswith("out") and x[3:].isdigit())]          # do not descend into OUTCAR segment directories
    rows = []
    for c, ls in sorted(camps.items()):
        ls = sorted(ls); fl = [_incar_flags(os.path.join(l, "INCAR")) for l in ls]
        act = sorted({f[0] for f in fl if f[0]}); com = sorted({f[1] for f in fl if f[1]})
        mix = sorted({f[2] for f in fl if f[2] is not None}); T = sorted({f[3] for f in fl if f[3]}); dt = sorted({f[4] for f in fl if f[4]})
        comp = "?"
        for l in ls:
            for nm in ("POSCAR", "CONTCAR"):
                q = os.path.join(l, nm)
                if os.path.exists(q):
                    try: ps = read_poscar(q); comp = "".join(f"{n}{k}" for n, k in zip(ps["names"], ps["counts"])); break
                    except Exception: pass
            if comp != "?": break
        nout = [len(outcar_files(l)) for l in ls]
        tag = _first_tag(ls[0]) if not a.fast else "-"
        ks = ("ACTIVE " + "|".join(act)) if act else (("commented " + "|".join(com)) if com else "absent")
        res = "".join(k for k, nm in (("F", "F_DFT.dat"), ("G", "G_DFT.dat"), ("c", "cmc-integrated-F.txt"), ("R", "basis/REFCAR")) if os.path.exists(os.path.join(c, nm)))
        rows.append(dict(path=os.path.relpath(c, root), comp=comp, T=",".join(f"{x:g}" for x in T), KSPRING=ks, nlam=len(ls), mix=",".join(f"{x:g}" for x in mix), dt=",".join(f"{x:g}" for x in dt),
                         outcars=f"{min(nout)}-{max(nout)}", tag=tag, results=res))
    rows.sort(key=lambda r: (not r["KSPRING"].startswith("ACTIVE"), r["path"]))
    w = max(len(r["path"]) for r in rows) if rows else 10
    print(f"{'campaign':{w}s} {'composition':16s} {'T':>6s} {'KSPRING':34s} {'nlam':>4s} {'dt':>4s} {'OUTCAR files':>12s} {'tag':>16s} res  (F=F_DFT G=G_DFT c=cmc-integrated R=REFCAR)")
    for r in rows:
        print(f"{r['path']:{w}s} {r['comp']:16s} {r['T']:>6s} {r['KSPRING'][:34]:34s} {r['nlam']:>4d} {r['dt']:>4s} {r['outcars']:>12s} {r['tag']:>16s} {r['results']}")
    print(f"{len(rows)} campaigns; {sum(r['KSPRING'].startswith('ACTIVE') for r in rows)} with an active KSPRING")
    if a.csv:
        import csv
        with open(os.path.expanduser(a.csv), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


def cmd_vscan(a):
    S = load_case(a.case, a.poscar, a.refcar)
    l0 = os.path.join(S["case"], "lambda_0.000"); lam = float(read_incar(os.path.join(l0, "INCAR"))["MIX_EFFPOT"].split()[0])
    st = outcar_steps(outcar_files(l0)[0], a.steps, len(S["typ"]), a.tag, a.field)
    if not st: raise SystemExit(f"no '{a.tag}' / POSITION steps in {outcar_files(l0)[0]}: check --tag/--field (liquid style: --tag V_KS-V_EP --field 4)")
    own = S["k"] > 0; kk = S["k"][own][0]; n_o = own.sum()
    rows = []
    for pos, te, tm in st:
        P = pos / BOHR; u = mic(P[own] - S["sites"][own], S["box"]); veff = (te - lam * tm) / HA_EV
        rows.append((veff, pair_energy_total(S, P), pair_energy_total(S, P, "linear"), 0.5 * kk * (u ** 2).sum(), 0.5 * kk * n_o * (u.mean(0) ** 2).sum()))
    R = np.array(rows); print(f"{len(R)} configurations from the start of lambda_0.000")
    for name, col in (("cubic spline", 1), ("linear nodes", 2)):
        d = R[:, 0] - R[:, col] - R[:, 3]                               # unaligned springs
        da = R[:, 0] - R[:, col] - (R[:, 3] - R[:, 4])                  # COM-aligned springs
        A = np.column_stack([-R[:, 4], np.ones(len(R))]); coef, *_ = np.linalg.lstsq(A, R[:, 0] - R[:, col] - R[:, 3], rcond=None)
        print(f"{name:13s}: V_VASP - pair - spring(REFCAR sites)  mean {d.mean():+.5f} sd {d.std():.5f} Ha | with COM-aligned spring mean {da.mean():+.5f} sd {da.std():.5f} | fit V = pair + spring - alpha*(1/2)kN|m|^2 + c: alpha = {coef[0]:.2f}, c = {coef[1]:+.5f}")
    print(f"no springs at all : V_VASP - pair(spline)  mean {np.mean(R[:, 0] - R[:, 1]):+.5f} sd {np.std(R[:, 0] - R[:, 1]):.5f} Ha   (if this sd is the smallest the run was NOT tethered)")
    print("a convention is right if its sd is far below the others and the mean is ~0; a constant mean with small sd is an offset of the potential (interpolation / cutoff shift)")


# ----------------------------------------------------------------------------------------------------------------------------------------------- commands
def cmd_vcheck(a):
    S = load_case(a.case, a.poscar, a.refcar)
    ep = pair_energy_total(S)
    own = S["k"] > 0
    com_o = S["pos"][own].mean(0); com_s = S["sites"][own].mean(0)
    disp = mic(S["pos"][own] - S["sites"][own], S["box"])
    sites_shift = S["sites"] + (com_o - com_s - mic(com_o - com_s, S["box"]) * 0)           # translate the lattice onto the O centre of mass
    sp_mic = spring_energy(S, how="mic"); sp_plain = spring_energy(S, how="plain")
    sp_shift = spring_energy(S, sites=S["sites"] + mic(disp.mean(0)[None, :], S["box"]), how="mic")
    print(f"species {S['names']} counts {S['counts']}  box {S['box'] * BOHR} A  T {S['T']:g} K  k {S['k'].max() * HA_EV / BOHR ** 2:.6f} eV/A^2")
    print(f"mean O displacement from the REFCAR sites (A): {disp.mean(0) * BOHR}   |.| = {np.linalg.norm(disp.mean(0)) * BOHR:.4f} A; rms {np.sqrt((disp ** 2).sum(1).mean()) * BOHR:.4f} A")
    ep_lin = pair_energy_total(S, mode="linear")
    print(f"pair energy (own spline of the AU tables)     {ep:14.8f} Ha")
    print(f"pair energy (linear between the table nodes)  {ep_lin:14.8f} Ha")
    print(f"spring energy, minimum image, REFCAR sites    {sp_mic:14.8f} Ha")
    print(f"spring energy, minimum image, COM-aligned     {sp_shift:14.8f} Ha   (sites shifted by the mean O displacement)")
    print(f"spring energy, plain difference (no MIC)      {sp_plain:14.8f} Ha")
    print(f"V_eff candidates: pair + spring(MIC) = {ep + sp_mic:.8f}   pair + spring(COM-aligned) = {ep + sp_shift:.8f}   pair + spring(plain) = {ep + sp_plain:.8f} Ha")
    cmc_log = a.cmc_log or os.path.join(S["case"], "cmc", "cmc_1.00")
    if os.path.exists(cmc_log):
        m = re.search(r"Initial potential energy: V\[Ha/cell\]=\s*([-\d.eE+]+)", open(cmc_log).read())
        if m: print(f"reference log initial potential energy V           {float(m.group(1)):14.8f} Ha")
    l0 = os.path.join(S["case"], "lambda_0.000")
    files = outcar_files(l0)
    if files:
        tm, te, _ = parse_outcar(files[0]); lam = float(read_incar(os.path.join(l0, "INCAR"))["MIX_EFFPOT"].split()[0])
        if len(tm):
            veff = (te[0] - lam * tm[0]) / HA_EV
            print(f"VASP step 1 ({os.path.relpath(files[0], S['case'])}): TOTEN = {te[0]:.6f} eV, totenminuseffpot = {tm[0]:.6f} eV, lambda = {lam}")
            print(f"VASP V_eff = TOTEN - lambda*(E_DFT-V_eff)    {veff:14.8f} Ha   (assumes printed TOTEN is the mixed energy)")
            print(f"  E_DFT(step 1) = {(te[0] - lam * tm[0] + tm[0]) / HA_EV:.6f} Ha = {(te[0] - lam * tm[0] + tm[0]) / len(S['typ']):.4f} eV/atom")


def cmd_ref(a):
    S = load_case(a.case, a.poscar, a.refcar)
    F, parts = F0_reference(S)
    print(f"F0 = {F:.8f} Ha  " + "  ".join(f"{k}: {v:.8f}" for k, v in parts.items()))
    json.dump(dict(F0=F, parts=parts, T=S["T"], V_bohr3=float(np.prod(S["box"])), counts=S["counts"], names=S["names"]), open(a.out, "w"), indent=1) if a.out else None


def cmd_cmc(a):
    if not HAVE_NUMBA: raise SystemExit("needs numba:  pip install --user numba")
    S = load_case(a.case, a.poscar, a.refcar, sites=a.sites)
    print(f"tether sites: {a.sites}")
    print(f"{S['names']} {S['counts']}  T {S['T']:g} K  {a.nodes} Gauss-Legendre nodes in lambda = x^{a.m}, {a.sweeps} sweeps (+{a.eq} equilibration), {a.procs} processes", flush=True)
    t0 = time.time(); r = cmc_integral(S, a.nodes, a.m, a.sweeps, a.eq, a.procs, a.seed)
    print(f"{'lambda':>10s} {'<V_pair> Ha':>14s} {'err':>9s} {'half1':>11s} {'half2':>11s} {'acc':>12s} {'<U_spring>':>11s}")
    for x in r["results"]:
        print(f"{x['lam']:10.6f} {x['mean']:14.6f} {x['err']:9.6f} {x['mean_first_half']:11.4f} {x['mean_second_half']:11.4f} {' '.join('%.2f' % q for q in x['acc']):>12s} {x['spring']:11.4f}")
    print(f"dF_cmc = int <V_pair> dlambda = {r['dF_cmc']:.5f} +- {r['dF_cmc_err']:.5f} Ha   ({(time.time() - t0) / 60:.1f} min)")
    if a.ref_table:
        rows = [l.split() for l in open(a.ref_table) if "lambda=" in l]
        lam = np.array([float(x[-4]) for x in rows]); f = np.array([float(x[-2]) for x in rows])
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(lam, f); print("comparison with the reference table (spline):")
        for x in r["results"]:
            if x["lam"] >= lam[0]: print(f"  lambda {x['lam']:9.5f}: own {x['mean']:11.4f} +- {x['err']:.4f}   ref {float(cs(x['lam'])):11.4f}   diff {x['mean'] - float(cs(x['lam'])):+.4f}")
        print(f"trapezoid of the reference table {np.trapezoid(f, lam):.5f}")
    if a.out: json.dump(r, open(a.out, "w"), indent=1)


def cmd_dft(a):
    TAG_FIELD[0], TAG_FIELD[1] = a.tag, a.field
    r = dft_stage(a.case, a.discard, tag=a.tag, com=a.com, stride=a.stride, field=a.field, end=a.end)
    f = r["fdt"]
    print("  fluctuation-dissipation consistency (d<dV>/dlambda = -beta var): interval, observed change, predicted from the variances, difference (Ha)")
    for i in range(len(r["lambdas"]) - 1):
        print(f"    [{r['lambdas'][i]:.2f},{r['lambdas'][i + 1]:.2f}] {f['interval_obs'][i]:+.4f} {f['interval_pred'][i]:+.4f} {f['interval_obs'][i] - f['interval_pred'][i]:+.4f}")
    print(f"    std of dV per window (Ha): {['%.3f' % x for x in f['std_Ha']]}  (a window whose std is far above its neighbours is not stationary)")
    for k in ("boole", "trapezoid", "simpson3", "spline", "mbar", "hermite"):
        if r.get(k) is not None: print(f"  dF_dft [{k:9s}] = {r[k]:.6f} Ha")
    if r.get("com"): print(f"  COM-corrected: dV' = E_DFT - V_pair - (1/2)k sum|u_i - u_mean|^2 (translation invariant); unprimed means {['%.4f' % v for v in r['dV_unprimed_means']]}, COM spring energy {['%.4f' % v for v in r['S_com_means']]}")
    P, Pe, V = pressure_gpa(a.case, a.discard, a.tag, a.field)
    r.update(P_GPa=P, P_err_GPa=Pe, V_bohr3=V)
    print(f"  pressure of lambda=1 run (external + N kT/V) = {P:.3f} +- {Pe:.3f} GPa, V = {V * BOHR ** 3:.4f} A^3")
    if a.out: json.dump(r, open(a.out, "w"), indent=1)


def cmd_assemble(a):
    S = load_case(a.case, a.poscar, a.refcar)
    F0, _ = F0_reference(S); c = json.load(open(a.cmc)); d = json.load(open(a.dft))
    V = d["V_bohr3"]; Pt = a.P / GPA_PER_HA_BOHR3
    kT = KB_HA * S["T"]; own = S["k"] > 0; comterms = {}
    if own.any() and d.get("com"):
        # Z_true = V * Z_cons (tdi/refs.py): the O COM mode of the tethered ensemble is replaced by the free translation of the whole system
        F_com0 = -1.5 * kT * log(2 * pi * kT / (own.sum() * S["k"][own][0]))
        comterms = dict(F_com0=F_com0, minus_kT_lnV=-kT * log(V))
        corr = -F_com0 - kT * log(V)
    else:
        corr = 0.0
        if own.any(): print("WARNING: tethered system with a dft json that is not COM-corrected (run `dft --com`): the lambda = 1 average contains an arbitrary COM spring term and F lacks the COM terms", file=sys.stderr)
    F = F0 + c["dF_cmc"] + corr + d["dF_dft"]; Fe = float(np.hypot(c["dF_cmc_err"], d.get("dF_dft_err") or 0.0))
    G = F + Pt * V; N = sum(S["counts"])
    out = dict(case=S["case"], names=S["names"], counts=S["counts"], T=S["T"], P_target=a.P, V_bohr3=V, F0=F0, dF_cmc=c["dF_cmc"], dF_dft=d["dF_dft"], com_terms=comterms, com_corrected=bool(d.get("com")), F=F, G=G, G_err=Fe,
               G_meV_atom=G / N * HA_EV * 1e3, G_err_meV_atom=Fe / N * HA_EV * 1e3, P_meas=d["P_GPa"])
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


def cmd_diff(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    gx, gy = x["G_meV_atom"], y["G_meV_atom"]; e = float(np.hypot(x["G_err_meV_atom"], y["G_err_meV_atom"]))
    print(f"G({os.path.basename(a.a)}) - G({os.path.basename(a.b)}) = {gx - gy:+.1f} +- {e:.1f} meV/atom   ({gx:.1f} vs {gy:.1f})")


def cmd_mix(a):
    ice, iceh, h = (json.load(open(f)) for f in (a.ice, a.iceh, a.pureh))
    n = a.n_add if a.n_add else iceh["counts"][-1] - ice["counts"][-1]
    gH = h["G"] / h["counts"][0]; gHe = h["G_err"] / h["counts"][0]
    dG = (iceh["G"] - ice["G"] - n * gH) / n; e = float(np.sqrt(iceh["G_err"] ** 2 + ice["G_err"] ** 2 + (n * gHe) ** 2)) / n
    print(f"mixing per added H (n_add = {n}) = {dG * HA_EV:+.4f} +- {e * HA_EV:.4f} eV   (G_H = {gH * HA_EV * 1e3:.2f} meV/atom)")


# ----------------------------------------------------------------------------------------------------------------------------------------------- self test
def selftest():
    ok = True
    def check(name, cond, info=""):
        nonlocal ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + " " + info, flush=True)
    lam, w = gl_nodes(8, 4); check("GL nodes integrate x^2.5 on [0,1] (lambda = x^4)", abs((w * lam ** 2.5).sum() - 1 / 3.5) < 1e-6)
    rng = np.random.default_rng(0); x = np.zeros(20000)
    for i in range(1, len(x)): x[i] = 0.9 * x[i - 1] + rng.normal()
    check("block_err sees the autocorrelation (AR(1) 0.9: true SEM ~ 4.36/sqrt(N) *...)", block_err(x) > 2.5 * x.std() / sqrt(len(x)), f"{block_err(x):.4f} vs naive {x.std() / sqrt(len(x)):.4f}")
    # MBAR on Gaussians: F(1)-F(0) = mu - beta sigma^2 / 2
    mu, sg, beta = 3.0, 1.2, 1.0; lams = np.array([0, .25, .5, .75, 1.0]); smp = [rng.normal(mu - beta * l * sg ** 2, sg, 20000) for l in lams]
    f = mbar(np.concatenate(smp), [len(s) for s in smp], beta * lams)
    check("MBAR Gaussian test", abs(f[-1] - (mu - beta * sg ** 2 / 2)) < 0.03, f"{f[-1]:.4f} vs {mu - beta * sg ** 2 / 2:.4f}")
    check("Boole exact for degree 4", abs(boole(np.array([l ** 4 for l in lams])) - 0.2) < 1e-12)
    # F0 against an independent closed-form evaluation (tethered harmonic atoms + ideal gas), made-up parameters
    nO, nH, Vb, Tk, kO = 64, 128, 5000.0, 4000.0, 8.0
    typ = np.repeat([0, 1], [nO, nH]); kT_ = KB_HA * Tk
    S = dict(names=["O", "H"], counts=[nO, nH], typ=typ, box=np.array([1, 1, 1.0]) * Vb ** (1 / 3), T=Tk,
             k=np.where(typ == 0, kO / HA_EV * BOHR ** 2, 0.0), masses=np.where(typ == 0, MASS["O"], MASS["H"]) * AMU)
    F, parts = F0_reference(S)
    w_ = sqrt(kO / HA_EV * BOHR ** 2 / (MASS["O"] * AMU)); lam3_ = (2 * pi / (MASS["H"] * AMU * kT_)) ** 1.5
    ref_O = nO * 3 * kT_ * log(w_ / kT_); ref_H = kT_ * (lgamma(nH + 1) - nH * log(Vb / lam3_))
    check("F0 matches the closed form", abs(F - (ref_O + ref_H)) < 1e-9, f"{F:.8f} vs {ref_O + ref_H:.8f}")
    # MC TI against the exact one-particle integral (stiff O lattice + 1 H)
    if HAVE_NUMBA:
        L = 6.0; sites = np.array([[x, y, z] for x in (0, 3) for y in (0, 3) for z in (0, 3)], float); no = len(sites)
        r = np.linspace(0, 3.0, 62); tOH = 0.6 * np.exp(-r / 0.5) - 0.6 * np.exp(-3.0 / 0.5)
        tab = np.zeros((2, 2, len(dense_table(r, tOH)))); tab[0, 1] = tab[1, 0] = dense_table(r, tOH)
        rcs = np.zeros((2, 2)); rcs[0, 1] = rcs[1, 0] = 3.0
        T = 3000.0; kT = KB_HA * T
        Sx = dict(typ=np.r_[np.zeros(no, int), 1], box=np.array([L] * 3), pos=np.vstack([sites, [[1.0, 1.0, 1.0]]]), sites=np.vstack([sites, [[0, 0, 0]]]), k=np.r_[np.full(no, 1.0e4), 0.0],
                  tab=tab, rcs=rcs, dr=5e-5, T=T, names=["O", "H"], counts=[no, 1], masses=np.r_[np.full(no, MASS["O"]), MASS["H"]] * AMU)
        g = np.linspace(0, L, 49)[:-1] + L / 96; X = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        e = np.zeros(len(X))
        for s in sites:
            d = mic(X - s, np.array([L] * 3)); rr = np.sqrt((d ** 2).sum(1)); m = rr < 3.0; u = rr[m] / 5e-5; k = u.astype(int); fr = u - k
            e[m] += (1 - fr) * tab[0, 1, k] + fr * tab[0, 1, k + 1]
        exact = -kT * np.log(np.mean(np.exp(-e / kT)))
        res = cmc_integral(Sx, nodes=10, m=4, sweeps=30000, eq=3000, procs=1, seed=5)
        check("MC thermodynamic integration vs exact one-H integral", abs(res["dF_cmc"] - exact) < 4 * res["dF_cmc_err"] + 2e-4, f"MC {res['dF_cmc']:.5f} +- {res['dF_cmc_err']:.5f}  exact {exact:.5f} Ha, drift check {max(abs(q['drift']) for q in res['results']):.1e}")
    else:
        print("SKIP MC test (no numba)")
    # OUTCAR parser
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        txt = "".join(f"  external pressure =  {5000 + i}.0 kB  Pullay stress = 0.0 kB\n  free  energy   TOTEN  =   {100 + i}.5 eV\n  totenminuseffpot     =   -{1800 + i}.25 eV\n" for i in range(5))
        p = os.path.join(td, "OUTCAR.gz"); gzip.open(p, "wt").write(txt)
        tm, te, pe = parse_outcar(p); check("OUTCAR parser", len(tm) == 5 and tm[2] == -1802.25 and te[0] == 100.5 and pe[4] == 5004.0)
    print("ALL PASS" if ok else "FAILURES"); return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("case"); p.add_argument("--poscar"); p.add_argument("--refcar"); p.add_argument("--kspring", help='override KSPRING, e.g. "3.9535307 0.0" (eV/A^2 per species)'); p.add_argument("--pots", help="directory with the pair tables (Poten_ab.dat or Poten_1_ab_AU.dat)")
    p = sub.add_parser("vcheck"); common(p); p.add_argument("--cmc-log"); p.set_defaults(f=cmd_vcheck)
    p = sub.add_parser("vscan"); common(p); p.add_argument("--tag", default="totenminuseffpot"); p.add_argument("--field", type=int, default=0); p.add_argument("--steps", type=int, default=40); p.set_defaults(f=cmd_vscan)
    p = sub.add_parser("survey"); p.add_argument("root"); p.add_argument("--csv"); p.add_argument("--fast", action="store_true", help="skip reading the OUTCARs for the tag type"); p.set_defaults(f=cmd_survey)
    p = sub.add_parser("lam0"); common(p); p.add_argument("--samples", type=int, default=100); p.add_argument("--seed", type=int, default=3); p.set_defaults(f=cmd_lam0)
    p = sub.add_parser("vensemble"); common(p); p.add_argument("--tag", default="totenminuseffpot"); p.add_argument("--field", type=int, default=0); p.add_argument("--every", type=int, default=20); p.add_argument("--discard", type=float, default=0.0); p.add_argument("--lam-dir", default="lambda_0.000"); p.set_defaults(f=cmd_vensemble)
    p = sub.add_parser("ref"); common(p); p.add_argument("--out"); p.set_defaults(f=cmd_ref)
    p = sub.add_parser("cmc"); common(p); p.add_argument("--nodes", type=int, default=20); p.add_argument("--m", type=int, default=4)
    p.add_argument("--sweeps", type=int, default=20000); p.add_argument("--eq", type=int, default=4000); p.add_argument("--procs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--sites", choices=["refcar", "poscar"], default="refcar"); p.add_argument("--ref-table", help="reference table of (V1-V0)(lambda) for a point-by-point comparison"); p.add_argument("--out"); p.set_defaults(f=cmd_cmc)
    p = sub.add_parser("dft"); p.add_argument("case"); p.add_argument("--kspring"); p.add_argument("--pots"); p.add_argument("--tag", default="totenminuseffpot", help="OUTCAR tag of E_DFT - V_eff (liquids: V_KS-V_EP)"); p.add_argument("--field", type=int, default=0, help="1-based whitespace field holding the value (liquid-style outputs: 4); 0 = after '='"); p.add_argument("--discard", type=float, default=0.2); p.add_argument("--end", type=float, default=1.0, help="use the run only up to this fraction (with --discard: a window, e.g. --discard 0 --end 0.5 = first half)"); p.add_argument("--com", action="store_true", help="use the translation-invariant dV' (reads all positions)"); p.add_argument("--stride", type=int, default=5); p.add_argument("--out"); p.set_defaults(f=cmd_dft)
    p = sub.add_parser("assemble"); common(p); p.add_argument("--cmc", required=True); p.add_argument("--dft", required=True); p.add_argument("--P", type=float, default=500.0); p.add_argument("--out"); p.set_defaults(f=cmd_assemble)
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b"); p.set_defaults(f=cmd_diff)
    p = sub.add_parser("mix"); p.add_argument("ice"); p.add_argument("iceh"); p.add_argument("pureh"); p.add_argument("--n-add", type=int, default=0); p.set_defaults(f=cmd_mix)
    p = sub.add_parser("selftest"); p.set_defaults(f=lambda a: sys.exit(selftest()))
    a = ap.parse_args()
    if getattr(a, "kspring", None): os.environ["FE_KSPRING"] = a.kspring
    if getattr(a, "pots", None): os.environ["FE_POTS"] = os.path.abspath(os.path.expanduser(a.pots))
    a.f(a)


if __name__ == "__main__":
    main()
