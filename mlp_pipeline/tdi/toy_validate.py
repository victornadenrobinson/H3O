"""End-to-end validation of the TDI / insertion pipeline on the 'toyice' pair potential (tdi/toy_ice.py), against references that do not use the pipeline.
usage (repo root, a python with ASE; CPU):
  python mlp_pipeline/tdi/toy_validate.py eos   --T 5000 --P 450 [--out runs/toy]     volumes at P for H2O / H2.25O crystal (32 O) and pure H (128), bulk moduli
  python mlp_pipeline/tdi/toy_validate.py widom --T 5000 [--out runs/toy]               independent mu_H: own MD + Widom insertion + deletion, Bennett acceptance ratio
  bash   mlp_pipeline/tdi/toy_validate.sh 5000                                          the pipeline runs (run_real.py toyice ...) at the eos volumes
  python mlp_pipeline/tdi/toy_validate.py compare [--out runs/toy]                      the tests below, pass / fail
  python mlp_pipeline/tdi/toy_validate.py bookkeeping --T 5000 [--out runs/toy]         T0: exact, no MD statistics (seconds)
Tests (all at the same P, per T):
  T0 reference bookkeeping, exact: crystal route [H2.25O - H2O in the same box] = insertion of 8 H = kT ln(72!/64!) - 8 kT ln V + 24 kT ln Lambda; pure H and
     insertion into pure H against the ideal-gas formulas (run_real.py --stage0 jsons, no MD statistics)
  T1 insertion TDI integral (n = 1, H2O host) = Widom/BAR excess chemical potential on the same host volume
  T2 the same for insertion into pure H
  T3 METHOD CHECK: pipeline insertion into pure H minus pipeline G_H (fluid route) = finite-size term only (~ -kT ln(2 pi N)/(2N), a few meV)
  T4 pipeline insertion n = 8 (P-corrected, average per added H) = crystal route [G(H2.25O) - G(H2O) - 8 G_H] / 8 (Einstein + ideal-gas bookkeeping vs ghost bookkeeping)
  T5 dg = mu_H(H2O host) - G_H from the pipeline vs from Widom/BAR alone, at two T: the temperature dependence
"""
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
EV_A3_GPA = 160.21766


def systems():
    from importlib import import_module
    eos = import_module("17_eos_md"); rr = import_module("tdi.run_real")
    return {"h2o": lambda v: eos.build("fcc", v, 2, 0, 2.0), "h2.25o": lambda v: eos.build("fcc", v, 2, 0, 2.25), "H": lambda v: rr.build_h_n(v, 128)}


def md(at, T, ps, equil_ps, seed, every=40, dt=0.5, frames=False):
    """Langevin MD with the toy potential (own loop, not the pipeline's engine); returns mean P (GPa, incl. kinetic), its block error, mean U, frames"""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    from tdi.toy_ice import ToyIce
    at = at.copy(); at.calc = ToyIce(); rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
    dyn = Langevin(at, dt * units.fs, temperature_K=T, friction=0.01, rng=rng)
    dyn.run(int(equil_ps * 1000 / dt))
    P, U, fr = [], [], []
    for k in range(int(ps * 1000 / dt / every)):
        dyn.run(every)
        P.append(-at.get_stress()[:3].mean() * EV_A3_GPA + len(at) * units.kB * T / at.get_volume() * EV_A3_GPA); U.append(at.get_potential_energy())
        if frames:
            fr.append(at.positions.copy())
    b = np.array_split(np.array(P), 5); pe = np.std([x.mean() for x in b]) / 2
    return float(np.mean(P)), float(pe), float(np.mean(U)), (np.array(fr), at)


def cmd_eos(a):
    """P(V) by short MD at 3 starting volumes, then secant steps on ln P vs ln V until a point is within 1.5% of the target; final vpa from the local power law
    through the 3 points closest to the target (bracketing it)"""
    S = systems(); f = os.path.join(a.out, f"eos_{a.T:.0f}K.json"); out = json.load(open(f)) if os.path.exists(f) else {}
    guess = {"h2o": [2.05, 2.14, 2.23], "h2.25o": [2.0, 2.09, 2.18], "H": ([1.25, 1.35, 1.45] if os.environ.get("TOYICE_FIT") else [1.6, 1.9, 2.2])}
    for name in (a.systems.split(",") if a.systems else S):
        build = S[name]; rows = []
        def point(v):
            P, pe, _, _ = md(build(v), a.T, 1.0, 0.25, 1); rows.append((v, P, pe)); print(f"  {name:7s} vpa {v:.4f}: P {P:7.1f} +- {pe:.1f} GPa", flush=True)
        for v in guess[name]:
            point(v)
        for _ in range(5):
            r = sorted(rows, key=lambda t: abs(np.log(t[1] / a.P)))
            if abs(r[0][1] / a.P - 1) < 0.015 and min(t[1] for t in rows) < a.P < max(t[1] for t in rows):
                break
            (v1, p1, _), (v2, p2, _) = r[0], r[1]
            point(float(np.exp(np.log(v1) + (np.log(a.P) - np.log(p1)) * (np.log(v2) - np.log(v1)) / (np.log(p2) - np.log(p1)))))
        r = sorted(rows, key=lambda t: abs(np.log(t[1] / a.P)))[:3]
        v, P = np.array([t[0] for t in r]), np.array([t[1] for t in r]); c = np.polyfit(np.log(v), np.log(P), 1)
        vt = float(np.exp((np.log(a.P) - c[1]) / c[0])); B = float(-c[0] * a.P)
        out[name] = dict(vpa=vt, B_GPa=B, scan=rows)
        print(f"{name:7s}: vpa at {a.P:g} GPa = {vt:.4f} A^3/atom, B = {B:.0f} GPa", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(out, T=a.T, P=a.P), open(f, "w"), indent=1); print("wrote", f)


def bar(wf, wr, kT):
    """Bennett acceptance ratio for dF = F(N+1) - F(N) (excess): wf = insertion energies (from N frames), wr = -(interaction energy of a particle) (from N+1 frames)"""
    nf, nr = len(wf), len(wr); M = kT * np.log(nf / nr)
    def g(dF):   # monotone in dF
        a = np.sum(1 / (1 + np.exp(np.clip((wf - dF + M) / kT, -700, 700)))); b = np.sum(1 / (1 + np.exp(np.clip((wr + dF - M) / kT, -700, 700))))
        return a - b
    lo, hi = min(wf.min(), -wr.max()) - 50 * kT, max(wf.max(), -wr.min()) + 50 * kT
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0: hi = mid   # g increases with dF
        else: lo = mid
    return 0.5 * (lo + hi)


def lse_mean(x):
    m = x.max(); return m + np.log(np.mean(np.exp(x - m)))


def widom_one(name, vpa, T, ps, seed, ntrial):
    from tdi.toy_ice import ghost_energy, params
    from ase import units
    p = params(); kT = units.kB * T; S = systems(); host = S[name](vpa)
    cell = np.asarray(host.cell)
    isH = host.numbers == 1; rng = np.random.default_rng(seed)
    P0, pe0, U0, (fr0, _) = md(host, T, ps, 0.5, seed, frames=True)
    wf = []
    for x in fr0:   # insertion: uniform trial points
        g = rng.random((ntrial, 3)) @ cell; wf.append(ghost_energy(g, x, isH, cell, p))
    wf = np.concatenate(wf)
    # N+1: host + one H at a random point, equilibrated; deletion energies of every H (all H identical)
    from tdi.run_real import add_ghosts
    big = add_ghosts(host, 1, seed + 7); isH1 = big.numbers == 1
    P1, pe1, U1, (fr1, _) = md(big, T, ps, 0.5, seed + 1, frames=True)
    wr = []
    for x in fr1:
        for i in np.where(isH1)[0]:
            others = np.ones(len(x), bool); others[i] = False
            wr.append(-ghost_energy(x[i:i + 1], x[others], isH1[others], cell, p)[0])
    wr = np.array(wr)
    mu_w = -kT * lse_mean(-wf / kT)   # Widom insertion: exp(-beta mu_ex) = <exp(-beta wf)>_N
    mu_d = kT * lse_mean(-wr / kT)   # deletion: mu_ex = kT ln <exp(+beta U_int)>_{N+1} = kT ln <exp(-beta wr)>
    mu_b = bar(wf, wr, kT)
    # block errors over frames (5 blocks)
    bw = [-kT * lse_mean(-np.concatenate(np.array_split(wf.reshape(len(fr0), -1), 5)[k]) / kT) for k in range(5)]
    nH1 = isH1.sum(); wrr = wr.reshape(len(fr1), nH1)
    bd = [kT * lse_mean(-np.array_split(wrr, 5)[k].ravel() / kT) for k in range(5)]
    bb = [bar(np.array_split(wf.reshape(len(fr0), -1), 5)[k].ravel(), np.array_split(wrr, 5)[k].ravel(), kT) for k in range(5)]
    err = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
    NH = int(isH.sum()); V = host.get_volume()
    from tdi import refs
    ideal = kT * np.log(NH + 1) - kT * np.log(V) + 3 * kT * np.log(refs.thermal_wavelength(float(host.get_masses()[isH][0]), T))
    r = dict(name=name, vpa=vpa, V=V, N_H=NH, P_host=P0, P_host_err=pe0, P_full=P1, P_full_err=pe1, U_host=U0, U_full=U1,
             mu_ex_widom=mu_w * 1e3, mu_ex_widom_err=err(bw) * 1e3, mu_ex_delete=mu_d * 1e3, mu_ex_delete_err=err(bd) * 1e3, mu_ex_bar=mu_b * 1e3, mu_ex_bar_err=err(bb) * 1e3,
             ideal_meV=ideal * 1e3, mu_fixedV_meV=(mu_b + ideal) * 1e3, n_ins=len(wf), n_del=len(wr))
    print(f"  {name}: P_host {P0:.1f} +- {pe0:.1f}, P(+1 H) {P1:.1f} +- {pe1:.1f} GPa | mu_ex Widom {mu_w * 1e3:.1f} +- {err(bw) * 1e3:.1f}, deletion {mu_d * 1e3:.1f} +- {err(bd) * 1e3:.1f}, "
          f"BAR {mu_b * 1e3:.1f} +- {err(bb) * 1e3:.1f} meV | ideal part {ideal * 1e3:.1f} meV -> mu(fixed V) {(mu_b + ideal) * 1e3:.1f} meV", flush=True)
    return r


def cmd_widom(a):
    e = json.load(open(os.path.join(a.out, f"eos_{a.T:.0f}K.json"))); res = {}
    for name in ("h2o", "H"):
        res[name] = widom_one(name, e[name]["vpa"], a.T, a.ps, a.seed, a.ntrial)
    dg = res["h2o"]["mu_fixedV_meV"] - res["H"]["mu_fixedV_meV"]
    print(f"Widom/BAR dg = mu_H(ice) - mu_H(pure H) = {dg:.1f} +- {np.hypot(res['h2o']['mu_ex_bar_err'], res['H']['mu_ex_bar_err']):.1f} meV (fixed V, both at their eos volume for {e['P']:g} GPa)")
    f = os.path.join(a.out, f"widom_{a.T:.0f}K.json"); json.dump(dict(T=a.T, P=e["P"], dg_meV=dg, **res), open(f, "w"), indent=1); print("wrote", f)


def cmd_bookkeeping(a):
    """T0: the reference (non-MD) pieces of the crystal route and of the insertion route, written by run_real.py --stage0 itself, against each other and
    against the exact ideal-gas formulas. H2O and H2.25O crystals in the SAME box (O Einstein term and -kT ln V identical, so they must cancel), insertion of
    n = 8 and n = 1 into that H2O box, pure H (N) and insertion of 1 H into it. Exact to rounding: no statistics involved (the MD of the insertion stage 0
    only measures pressures for dG_P, which is reported separately and excluded here)."""
    import subprocess
    from ase import units
    from tdi import refs, gcalc
    out = os.path.join(a.out, "bookkeeping"); os.makedirs(out, exist_ok=True)
    T, P, vw, nO = a.T, a.P, 2.15, 32; Vw = vw * 3 * nO
    base = [sys.executable, os.path.join(HERE, "run_real.py"), "toyice", "--T", str(T), "--P", str(P), "--outdir", out, "--stage0", "--device", "cpu", "--nrep", "2",
            "--equil-ps", "0.05", "--ps1", "0.6", "--bulk-gpa", "1500", "--fresh"]
    runs = {"w": ["--phase", "fcc", "--vpa", f"{vw!r}", "--k", "20"],
            "m": ["--phase", "fcc", "--hper", "2.25", "--vpa", repr(Vw / (3 * nO + 8)), "--k", "20"],
            "i8": ["--phase", "insert", "--nghost", "8", "--vpa", f"{vw!r}"],
            "i1": ["--phase", "insert", "--nghost", "1", "--vpa", f"{vw!r}"],
            "h": ["--phase", "H", "--natoms", "128", "--vpa", "2.18"],
            "ih": ["--phase", "insert", "--host-phase", "H", "--natoms", "128", "--nghost", "1", "--vpa", "2.18"]}
    st = {}
    for k, extra in runs.items():
        tag = subprocess.run(base + extra + ["--print-tag"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        r = subprocess.run(base + extra, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"run_real.py stage 0 failed for {k}")
        st[k] = json.load(open(os.path.join(out, f"{tag}_stage0.json")))
        print(f"  {k:3s} {tag:42s} N {st[k]['N']:4d} nO {st[k]['nO']:3d} nH {st[k]['nH']:4d} V {st[k]['V']:.6f}", flush=True)
    ref = lambda s: gcalc.G_total_eV(dict(s, dG_P_eV=0.0), 0.0)   # whole-cell G with dF = 0 and no pressure correction: the reference pieces only
    from ase.data import atomic_masses
    kT = units.kB * T; lnL = np.log(refs.thermal_wavelength(float(atomic_masses[1]), T)); rows = []   # the H mass ASE puts on the atoms (1.008)
    def chk(name, got, exact):
        rows.append((name, got, exact, got - exact)); print(f"  {name:62s} code {got:16.9f}  exact {exact:16.9f}  diff {got - exact:+.2e} eV")
    from math import lgamma
    assert abs(st["w"]["V"] - st["m"]["V"]) < 1e-6 * Vw, "crystal boxes differ in volume"
    V = st["w"]["V"]
    exact8 = kT * (lgamma(73) - lgamma(65)) - 8 * kT * np.log(V) + 24 * kT * lnL
    print(f"T0 at {T:g} K, V = {V:.4f} A^3 (H2O and H2.25O crystals in the same box), all in eV:")
    chk("crystal route  G_ref(H2.25O) - G_ref(H2O)   [72 vs 64 H]", ref(st["m"]) - ref(st["w"]), exact8)
    chk("insertion route n = 8 (reference pieces)", ref(st["i8"]) * 1.0, exact8)
    chk("insertion route n = 1", ref(st["i1"]), kT * np.log(65) - kT * np.log(V) + 3 * kT * lnL)
    Vh = st["h"]["V"]; N = 128
    chk("pure H fluid G_ref (N = 128, whole cell)", ref(st["h"]), kT * lgamma(N + 1) - N * kT * np.log(Vh) + 3 * N * kT * lnL + P / EV_A3_GPA * Vh)
    chk("insertion into pure H, n = 1", ref(st["ih"]), kT * np.log(N + 1) - kT * np.log(Vh) + 3 * kT * lnL)
    fs = kT * (np.log(N + 1) - lgamma(N + 1) / N - 1)   # exact mu(N -> N+1) - G/N of an ideal gas with P V = N kT (Stirling: kT[ln((N+1)/N) - ln(2 pi N)/(2N)])
    gH = ref(st["h"]) / N - P / EV_A3_GPA * Vh / N + kT   # ideal-gas G per atom at the ideal-gas pressure N kT / V
    chk("ideal gas: insertion into H - G/N (P V = N kT), exact", ref(st["ih"]) - gH, fs)
    bad = [r for r in rows if abs(r[3]) > 1e-6]
    print("T0 bookkeeping:", "PASS (all within 1e-6 eV)" if not bad else f"FAIL: {[r[0] for r in bad]}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["eos", "widom", "compare", "bookkeeping"]); ap.add_argument("--T", type=float, default=5000); ap.add_argument("--P", type=float, default=450)
    ap.add_argument("--out", default="runs/toy"); ap.add_argument("--ps", type=float, default=4.0); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--ntrial", type=int, default=2000)
    ap.add_argument("--systems", default=None, help="eos: comma list of h2o,h2.25o,H (default all; results are merged into the existing json)")
    a = ap.parse_args()
    {"eos": cmd_eos, "widom": cmd_widom, "compare": lambda a: cmd_compare(a), "bookkeeping": cmd_bookkeeping}[a.cmd](a)


def dGP(V, B, Ph, Pf, P, n):
    """the pipeline's pressure correction to P (eV per added H): -V/(2B) [(Pf - P)^2 - (Ph - P)^2] / n"""
    return -V / (2 * B) * ((Pf - P) ** 2 - (Ph - P) ** 2) / EV_A3_GPA / n


def cmd_compare(a):
    import pandas as pd
    from ase import units
    summ = os.path.join(a.out, "summary.csv")
    os.system(f"python3 {HERE}/summarize.py {a.out}/tdi -o {summ} > {a.out}/summarize.log 2>&1")
    df = pd.read_csv(summ); tdi = os.path.join(a.out, "tdi"); rows = []
    def row(tag):
        r = df[df.tag == tag]
        return r.iloc[0] if len(r) else None
    def dF_perH(tag):   # TDI integral alone, meV per added H: G minus the reference pieces of the stage0 json
        r = row(tag); st = json.load(open(os.path.join(tdi, f"{tag}_stage0.json")))
        pe = st.get("dG_P_err_eV", 0.0) / st["N"] * 1e3   # G_err = hypot(integral error, pressure-correction error): keep the integral's own error
        return r.G_meV_atom - (st["f_ref_cons_eV"] + st["f_kin_eV"] + st.get("dG_P_eV", 0.0)) / st["N"] * 1e3, float(np.sqrt(max(r.G_err_meV_atom ** 2 - pe ** 2, 0.0)))
    def test(name, got, gerr, ref, rerr, tol=10.0, expect=0.0):
        d = got - ref - expect; e = float(np.hypot(gerr, rerr)); ok = abs(d) <= max(3 * e, tol)
        rows.append(dict(test=name, pipeline=got, ref=ref, expected_diff=expect, diff_minus_expected=d, err=e, verdict="PASS" if ok else "FAIL"))
    Ts = sorted(float(f.split("_")[-1][:-6]) for f in glob.glob(os.path.join(a.out, "widom_*K.json")))
    dg = {}
    for T in Ts:
        W = json.load(open(os.path.join(a.out, f"widom_{T:.0f}K.json"))); E = json.load(open(os.path.join(a.out, f"eos_{T:.0f}K.json"))); P = W["P"]; kT = units.kB * T * 1e3
        t_w, t_m, t_h = f"fcc_{P:.0f}GPa_{T:.0f}K_rep2", f"fcc_{P:.0f}GPa_{T:.0f}K_h2.25o_rep2", f"H_{P:.0f}GPa_{T:.0f}K_N128"
        t_i1, t_i8, t_iH = f"insert_{P:.0f}GPa_{T:.0f}K_n1", f"insert_{P:.0f}GPa_{T:.0f}K_n8", f"insert_{P:.0f}GPa_{T:.0f}K_n1_Hhost_N128"
        if row(t_i1) is not None:
            g, ge = dF_perH(t_i1); test(f"T1 {T:.0f} K: insertion TDI integral vs BAR mu_ex (H2O host)", g, ge, W["h2o"]["mu_ex_bar"], W["h2o"]["mu_ex_bar_err"])
        if row(t_iH) is not None:
            g, ge = dF_perH(t_iH); test(f"T2 {T:.0f} K: insertion TDI integral vs BAR mu_ex (pure-H host)", g, ge, W["H"]["mu_ex_bar"], W["H"]["mu_ex_bar_err"])
        N = W["H"]["N_H"]; fs = kT * (np.log((N + 1) / N) - np.log(2 * np.pi * N) / (2 * N))   # ideal-gas finite-size term of mu(N -> N+1) vs G/N
        if row(t_iH) is not None and row(t_h) is not None:
            test(f"T3 {T:.0f} K: METHOD CHECK insertion into pure H - G_H", row(t_iH).G_meV_atom, row(t_iH).G_err_meV_atom, row(t_h).G_meV_atom, row(t_h).G_err_meV_atom, expect=fs)
        if row(t_i8) is not None and row(t_m) is not None and row(t_w) is not None:
            m, w = row(t_m), row(t_w); cr = (m.N * m.G_meV_atom - w.N * w.G_meV_atom) / 8; ce = np.hypot(m.N * m.G_err_meV_atom, w.N * w.G_err_meV_atom) / 8
            test(f"T4 {T:.0f} K: insertion n=8 (P-corrected) vs crystal route H2.25O - H2O, per added H", row(t_i8).G_meV_atom, row(t_i8).G_err_meV_atom, cr, ce)
        if row(t_i1) is not None and row(t_h) is not None:   # dg from the pipeline vs from Widom/BAR with the same pressure correction and the ideal finite-size term
            gp = row(t_i1).G_meV_atom - row(t_h).G_meV_atom; gpe = np.hypot(row(t_i1).G_err_meV_atom, row(t_h).G_err_meV_atom)
            wi, wh = W["h2o"], W["H"]
            mi = wi["mu_fixedV_meV"] + 1e3 * dGP(wi["V"], E["h2o"]["B_GPa"], wi["P_host"], wi["P_full"], P, 1)
            mh = wh["mu_fixedV_meV"] + 1e3 * dGP(wh["V"], E["H"]["B_GPa"], wh["P_host"], wh["P_full"], P, 1) - fs
            gw = mi - mh; gwe = np.hypot(wi["mu_ex_bar_err"], wh["mu_ex_bar_err"])
            test(f"T5 {T:.0f} K: dg = mu_H(H2O) - G_H, pipeline vs Widom/BAR", gp, gpe, gw, gwe, tol=15.0); dg[T] = (gp, gpe, gw, gwe)
    if len(dg) >= 2:
        T1, T2 = min(dg), max(dg); sp = (dg[T2][0] - dg[T1][0]) / (T2 - T1); sw = (dg[T2][2] - dg[T1][2]) / (T2 - T1)
        spe = np.hypot(dg[T2][1], dg[T1][1]) / (T2 - T1); swe = np.hypot(dg[T2][3], dg[T1][3]) / (T2 - T1)
        test(f"T5 slope d(dg)/dT {T1:.0f}-{T2:.0f} K (meV/K)", sp, spe, sw, swe, tol=0.005)
    R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(R.round(4).to_string(index=False)); R.to_csv(os.path.join(a.out, "toy_tests.csv"), index=False)


if __name__ == "__main__":
    main()
