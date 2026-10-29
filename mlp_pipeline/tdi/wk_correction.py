"""Leading quantum (Wigner-Kirkwood) correction to the pwidom results, from the classical MD frames each scan already saved (<dir>/pw_*_frames.npz): no new MD.
  A_q = (hbar^2 / (24 (kT)^2)) sum_i <|F_i|^2> / m_i      (first term of the expansion in (hbar w / kT)^2; at 5000 K, kT ~ 3500 cm^-1, so the series converges fast)
For the transfer of one H from pure H into a host:   dg_q = [A_q(host + 1 H) - A_q(host)] - [A_q(pure H + 1) - A_q(pure H)]
Two estimates are printed per host:
  own  : the added proton's own term, a_H <F_H^2> per H (identical protons: the mean over all H), host minus pure H. Small statistical error; leaves out how the
         extra H changes the forces on its neighbours.
  whole system: host + 1 H frames minus host frames (two separate MD ensembles): complete (includes how the extra H changes its neighbours' forces) but very
         noisy (+-0.3-0.6 eV with 10 toy frames: it is a difference of sums over ~300 atoms). Removing the H from the SAME frame instead ("paired") is NOT valid:
         the neighbours stay unrelaxed (tested: ~400 meV, far off). The proper low-noise estimator reweights the pwidom insertion structures (Widom weights
         give <X>_(N+1) from host frames): not implemented yet.
Forces are recomputed with the model of the scan (GRACE, batched on the GPU; toy model on CPU) on every --stride-th frame.
Self-test (always run first): an Einstein oscillator, where the exact quantum free energy is known, against the formula and its constant.
usage (GPU session or node, repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/wk_correction.py runs/pwidom_nscan [more dirs] [--stride 2] [--device cuda]
writes <dir>/wk.csv; prints per (P, T, host): dg_q own / sys (meV per H) and the per-H rms force in each environment.
"""
import argparse, glob, json, os, re, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # TensorFlow start-up chatter
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
HBAR2_AMU = 4.18015e-3   # hbar^2 / (1 amu) in eV A^2
KB = 8.617333262e-5
MASS = {1: 1.00794, 8: 15.9994}


def selftest():
    """Einstein oscillator: A_quantum - A_classical (exact) vs the Wigner-Kirkwood term with <F^2> = m w^2 kT per dimension"""
    m, T = 1.00794, 5000.0; kT = KB * T
    for hw in (0.05, 0.2, 0.4):   # eV; hbar w / kT = 0.12 ... 0.93
        exact = kT * np.log(2 * np.sinh(hw / (2 * kT))) - kT * np.log(hw / kT)      # per dimension
        w2 = hw ** 2 / (HBAR2_AMU / m)                                               # m w^2 in eV/A^2 times 1/m ... -> w^2 m = hw^2 m / hbar^2
        F2 = w2 * kT                                                                 # <F^2> = (m w^2) kT per dimension (m w^2 = w2 here, in eV/A^2)
        wk = HBAR2_AMU / (24 * kT ** 2) * F2 / m
        print(f"   self-test: hbar w / kT = {hw / kT:.2f}: exact {exact * 1e3:7.3f} meV, Wigner-Kirkwood {wk * 1e3:7.3f} meV ({100 * (wk - exact) / exact:+.1f} %)")


_CALC = {}


def forces(calc_spec, device, structs):
    """forces of many structures; the calculator (and GraceBatch) is built once per model and reused for every host"""
    from importlib import import_module
    if calc_spec not in _CALC:
        calc = import_module("04_benchmark").get_calc_spec(calc_spec, device); gb = None
        if type(calc).__name__ == "TPCalculator":
            from tdi.batched import GraceBatch
            gb = GraceBatch(calc)
        _CALC[calc_spec] = (calc, gb)
    calc, gb = _CALC[calc_spec]
    if gb is not None:
        out = []
        for i in range(0, len(structs), 8): out += [r[1] for r in gb(None, [(structs[i:i + 8], False)])[0]]
        return out
    res = []
    for at in structs:
        at.calc = calc; res.append(np.array(at.get_forces())); at.calc = None
    return res


def wk_terms(F, numbers, kT):
    """per frame: sum_i a_i |F_i|^2 (eV) and the mean a_H |F_H|^2 over the H atoms (eV), a_i = hbar^2 / (24 (kT)^2 m_i)"""
    a = np.array([HBAR2_AMU / (24 * kT ** 2 * MASS[int(z)]) for z in numbers]); f2 = (F ** 2).sum(-1)
    return float((a * f2).sum()), float((a * f2)[numbers == 1].mean()), float(np.sqrt(f2[numbers == 1].mean()))


def blocks_err(x, nb=5):
    x = np.asarray(x); b = [c.mean() for c in np.array_split(x, min(nb, len(x)))]
    return float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) > 1 else float("nan")


def host_wk(npz, model, device, stride):
    from ase import Atoms
    z = np.load(npz, allow_pickle=True); T = float(z["T"]); kT = KB * T; nums = np.asarray(z["numbers"]); nums1 = np.r_[nums, 1]
    cell = np.asarray(z["cell"]) if "cell" in z.files else np.eye(3) * float(z["V"]) ** (1 / 3)   # older frame files: cubic cells (both host types are cubic)
    out = {}
    for key, nu in (("host", nums), ("full", nums1)):
        X = z[key][::stride]; S = [Atoms(numbers=nu, positions=x, cell=cell, pbc=True) for x in X]
        F = forces(model, device, S); t = [wk_terms(f, nu, kT) for f in F]
        out[key] = dict(sys=np.array([v[0] for v in t]), own=np.array([v[1] for v in t]), frms=np.mean([v[2] for v in t]), n=len(S))
    return T, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=None, help="default: the model recorded in each scan's result json"); a = ap.parse_args()
    print("Wigner-Kirkwood correction (leading quantum term) from saved classical frames"); selftest()
    for d in a.dirs:
        fr = sorted(glob.glob(os.path.join(d, "pw_*_frames.npz")))
        if not fr: print(f"\n{d}: no frame files"); continue
        res, rows = {}, []
        for f in fr:
            tag = os.path.basename(f)[:-len("_frames.npz")]; m = re.match(r"pw_(.+)_([\d.]+)GPa_(\d+)K$", tag)
            if not m: continue
            host, P, Tt = m.group(1), float(m.group(2)), float(m.group(3))
            js = sorted(glob.glob(os.path.join(d, f"{tag}_*_K*.json"))); model = a.model or (json.load(open(js[0]))["model"] if js else None)
            z = np.load(f, allow_pickle=True); model = model or (str(z["model"]) if "model" in z.files else None)
            if not model: print(f"   {tag}: no model recorded (give --model)"); continue
            T, w = host_wk(f, model, a.device, a.stride)
            mu_sys = w["full"]["sys"].mean() - w["host"]["sys"].mean(); e_sys = float(np.hypot(blocks_err(w["full"]["sys"]), blocks_err(w["host"]["sys"])))
            res[(P, Tt, host)] = dict(own=w["full"]["own"].mean(), own_err=blocks_err(w["full"]["own"]), sys=mu_sys, sys_err=e_sys, frms=w["full"]["frms"], n=w["full"]["n"])
            print(f"   {tag:32s} a_H<F_H^2> {1e3 * res[(P, Tt, host)]['own']:7.2f} meV per H (rms force on H {w['full']['frms']:.2f} eV/A), "
                  f"whole system A(+1 H) - A {1e3 * mu_sys:8.2f} +- {1e3 * e_sys:.2f} meV  [{w['full']['n']} frames]", flush=True)
        print(f"\n{d}: quantum correction to dg = mu_H(host) - G_H, meV per H (add to the classical dg)")
        print(f"   {'P/GPa':>6s} {'T/K':>6s} {'host':>10s} {'own':>8s} {'+-':>5s} {'whole system':>13s} {'+-':>6s}")
        for (P, Tt, host), r in sorted(res.items()):
            if host == "pureH" or (P, Tt, "pureH") not in res: continue
            h = res[(P, Tt, "pureH")]
            own = 1e3 * (r["own"] - h["own"]); eo = 1e3 * float(np.hypot(r["own_err"], h["own_err"])); sy = 1e3 * (r["sys"] - h["sys"]); es = 1e3 * float(np.hypot(r["sys_err"], h["sys_err"]))
            rows.append(dict(P_GPa=P, T_K=Tt, host=host, dg_q_own_meV=round(own, 2), err_own=round(eo, 2), dg_q_system_meV=round(sy, 2), err_system=round(es, 2)))
            print(f"   {P:6.0f} {Tt:6.0f} {host:>10s} {own:8.2f} {eo:5.2f} {sy:13.2f} {es:6.2f}")
        if rows:
            import csv
            with open(os.path.join(d, "wk.csv"), "w", newline="") as fh:
                w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
            print(f"   -> {os.path.join(d, 'wk.csv')}")


if __name__ == "__main__":
    main()
