"""Toy-model end-to-end test of the static DFT check (dft_check.py prep-static / analyze) and of the solubility fit (solubility_check.py). CPU only, ~1 min.
h3o_static.py (toyice, one pressure) -> prep-static (fake VASP template) -> a stand-in "VASP" that evaluates a DIFFERENT toyice parameter set and writes OUTCARs -> analyze.
The analysis must reproduce an independent brute-force enthalpy / formation-enthalpy calculation with the stand-in "DFT" to 1.5 meV/atom (the quadratic E(V) fit over +-2 % linear scaling is good to ~0.5)."""
import csv, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "mlp_pipeline"), OMP_NUM_THREADS="1")
PY = sys.executable


def run(*args, env=None, **kw):
    r = subprocess.run([PY, *args], cwd=ROOT, env=env or ENV, capture_output=True, text=True, **kw)
    assert r.returncode == 0, f"{args}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r.stdout


def fake_template(d):
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "INCAR"), "w").write("ENCUT = 1100\nISMEAR = -1\nSIGMA = 0.43\nIBRION = 0\nNSW = 1000\nTEBEG = 5000\nNBANDS = 600\nKSPACING = 0.5\nEDIFF = 1E-5\n")
    open(os.path.join(d, "KPOINTS"), "w").write("gamma only\n0\nGamma\n1 1 1\n0 0 0\n")
    open(os.path.join(d, "POTCAR"), "w").write("".join(f"   PAW_PBE {el} 15Jun2001\n   TITEL  = PAW_PBE {el} 15Jun2001\n   ZVAL   =    {z:.3f}\n End of Dataset\n" for el, z in (("H", 1.0), ("O", 6.0))))
    from ase import Atoms
    from ase.io import write
    write(os.path.join(d, "POSCAR"), Atoms("H4O2", positions=[[0, 0, 0]] * 6, cell=[3, 3, 3], pbc=True), format="vasp")


FAKE = '''
import os, sys
os.environ["TOYICE"] = "D=2.15,B=70,A_OO=1400"
sys.path.insert(0, os.path.join(sys.argv[2], "mlp_pipeline"))
from ase.io import read
from tdi.toy_ice import ToyIce
for d in open(sys.argv[1]).read().split():
    at = read(os.path.join(d, "structure.extxyz")); at.calc = ToyIce(); E = at.get_potential_energy(); P = -at.get_stress(voigt=True)[:3].mean() * 160.21766
    open(os.path.join(d, "OUTCAR"), "w").write(f"  free  energy   TOTEN  =      {E:.8f} eV\\n  aborting loop because EDIFF is reached\\n  external pressure =  {P * 10:10.2f} kB  Pullay stress =      0.00 kB\\n  General timing and accounting informations for this job:\\n")
'''


def brute_enthalpy(path, P):
    """independent: minimum over a fine linear-scale scan of E_fake + P V, per atom"""
    from ase.io import read
    os.environ["TOYICE"] = "D=2.15,B=70,A_OO=1400"
    sys.path.insert(0, os.path.join(ROOT, "mlp_pipeline"))
    from tdi.toy_ice import ToyIce
    at = read(path); best = 1e9
    for sc in np.linspace(0.93, 1.07, 57):
        b = at.copy(); b.set_cell(np.asarray(at.cell) * sc, scale_atoms=True); b.calc = ToyIce()
        best = min(best, (b.get_potential_energy() + P / 160.21766 * b.get_volume()) / len(b))
    del os.environ["TOYICE"]
    return best


def test_static_chain(tmp_path):
    P = 400; st = str(tmp_path / "static"); tpl = str(tmp_path / "tpl"); dft = str(tmp_path / "dft"); fake_template(tpl)
    run("mlp_pipeline/h3o_static.py", "--model", "toyice", "--device", "cpu", "--P", str(P), "--steps", "300", "--fmax", "0.1", "--variants", "3", "-o", os.path.join(st, f"P{P}"))
    out = run("mlp_pipeline/tdi/dft_check.py", "prep-static", st, "--template", tpl, "--P", str(P), "--scales", "0.98", "1.0", "1.02", "--only", "iceX", "Href_H_bcc", "H3O", "-o", dft)
    assert "9 single-point directories" in out, out[-500:]
    inc = open(os.path.join(dft, "static_iceX_400GPa_s1.000", "INCAR")).read()
    assert "KSPACING" not in inc and "ISMEAR = -1" in inc and "IBRION = -1" in inc and "NSW = 0" in inc and "TEBEG" not in inc, inc   # MD tags and the template k-spacing are gone
    assert "Gamma" in open(os.path.join(dft, "static_iceX_400GPa_s1.000", "KPOINTS")).read()
    open(str(tmp_path / "fake.py"), "w").write(FAKE); run(str(tmp_path / "fake.py"), os.path.join(dft, "dirs.txt"), ROOT)
    out = run("mlp_pipeline/tdi/dft_check.py", "analyze", dft, "--model", "toyice", "--device", "cpu")
    assert "9 of 9 DFT single points finished" in out, out[-800:]
    rows = list(csv.DictReader(open(os.path.join(dft, "static_results.csv")))); r = next(x for x in rows if x["structure"] == "H3O")
    d = os.path.join(st, f"P{P}")
    h = {n: brute_enthalpy(os.path.join(d, f), P) for n, f in (("ice", f"iceX_{P}GPa_relaxed.extxyz"), ("H", f"Href_H_bcc_{P}GPa_relaxed.extxyz"), ("h3o", f"H3O_{P}GPa_relaxed.extxyz"))}
    N, nO, extra = 64, 16, 16
    dh = (h["h3o"] * N - nO * 3 * h["ice"] - extra * h["H"]) / N * 1e3
    assert abs(float(r["dH_vasp"]) - dh) < 1.5, (float(r["dH_vasp"]), dh)
    assert abs(float(r["dH_vasp"]) - float(r["dH_grace"])) > 1.0   # the two parameter sets really differ, so the comparison is not trivially zero


def test_solubility_selftest():
    out = run("mlp_pipeline/tdi/solubility_check.py", "--selftest")
    assert "selftest PASSED" in out, out[-1500:]


def test_solubility_mode_a_numbers():
    out = run("mlp_pipeline/tdi/solubility_check.py")
    # 500 GPa 6000 K: n* = 493.3 / (2 * 2157.8) = 0.1143
    line = next(l for l in out.splitlines() if l.strip().startswith("500   6000") and "H2." in l)
    assert line.split()[-1] == "H2.114O", line
