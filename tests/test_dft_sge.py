"""Toy-model test of the SGE VASP wrapper (young_dft_check.sge) and of dft_check.py prep-selftest / prep-widom / status / analyze. CPU only, ~1 min.
A stand-in 'vasp_gam' (toyice energies, writes an OUTCAR), a stand-in mpirun / module / qstat; checks: array chunking (task 2 of CHUNK 4 = directories 5..8), finished points skipped on a rerun,
a failing point reported and the exit code 1, the h_rt guard (stops before a point that cannot finish), status counts, and the widom / selftest analysis."""
import os, stat, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_static_check import ROOT, ENV, PY, fake_template, run

STUB = '''#!/usr/bin/env python3
import os, sys, time
os.environ["TOYICE"] = "D=2.15,B=70,A_OO=1400"
sys.path.insert(0, "%s/mlp_pipeline")
from ase.io import read
from tdi.toy_ice import ToyIce
at = read("structure.extxyz"); at.calc = ToyIce(); E = at.get_potential_energy(); P = -at.get_stress(voigt=True)[:3].mean() * 160.21766
if os.path.exists("FAIL"): print(" VERY BAD NEWS! internal error"); sys.exit(1)
print("DAV:   1    -0.1E+02  -0.2E-01"); sys.stdout.flush(); time.sleep(0.05)
open("OUTCAR", "w").write(" running on   16 total cores\\n NBANDS=    600\\n ENCUT  =  1100.0 eV\\n" + "  LOOP:  cpu time   1.0: real time   2.0\\n" * 3 +
    f"  free  energy   TOTEN  =      {E:.8f} eV\\n  aborting loop because EDIFF is reached\\n  external pressure =  {P * 10:10.2f} kB  Pullay stress =      0.00 kB\\n  General timing and accounting informations for this job:\\n  Elapsed time (sec):        3.000\\n")
''' % ROOT


def sge(env_extra, d, tmp):
    bind = tmp / "bin"; env = dict(ENV, PATH=f"{bind}:{os.environ['PATH']}", BASH_ENV=str(tmp / "env.sh"), NSLOTS="16", PE="smp", JOB_ID="7", HEARTBEAT="1"); env.update(env_extra)
    return subprocess.run(["bash", "mlp_pipeline/young_dft_check.sge", d], cwd=ROOT, env=env, capture_output=True, text=True)


def test_sge_wrapper_and_modes(tmp_path):
    b = tmp_path / "bin"; b.mkdir()
    for n in ("vasp_std", "vasp_gam"): (b / n).write_text(STUB)
    (b / "mpirun").write_text('#!/bin/bash\nif [ "$1" = -np ]; then shift 2; fi\nexec "$@"\n'); (b / "qstat").write_text('#!/bin/bash\n[ "$STUB_HRT" ] && echo "hard resource_list:   h_rt=$STUB_HRT,threads=2"\n')
    for f in b.iterdir(): f.chmod(f.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "env.sh").write_text('module() { :; }; export -f module\n')
    tpl = str(tmp_path / "tpl"); fake_template(tpl)
    from ase import Atoms
    from ase.io import write
    rng = np.random.default_rng(1); write(os.path.join(tpl, "POSCAR"), Atoms("H16O8", positions=rng.random((24, 3)) * 6.0, cell=[6.0] * 3, pbc=True), format="vasp")
    pw = str(tmp_path / "pw")
    run("mlp_pipeline/tdi/pwidom.py", "run", "--model", "toyice", "--device", "cpu", "--phase", "fcc", "--vpa", "2.08", "--T", "6000", "--P", "450", "--bulk-gpa", "1400", "--proxy", "docs/data/toyfit_t4.json",
        "--ps", "0.2", "--equil-ps", "0.05", "--every-fs", "20", "--frames-only", "--out", pw)
    wd = str(tmp_path / "widom")
    out = run("mlp_pipeline/tdi/dft_check.py", "prep-widom", os.path.join(pw, "pw_fcc_450GPa_6000K_frames.npz"), "--template", tpl, "--proxy", "docs/data/toyfit_t4.json", "--frames-n", "3", "--sel", "2",
              "--model", "toyice", "--device", "cpu", "--top", "20", "-o", wd)
    assert "15 single-point directories" in out and "young_dft_check.sge" in out, out[-600:]
    r = sge(dict(SGE_TASK_ID="2", CHUNK="4"), wd, tmp_path); assert r.returncode == 0 and "this task runs 5..8" in r.stdout and r.stdout.count("finished") >= 4, r.stdout[-1500:]
    open(os.path.join(wd, "f2_host", "FAIL"), "w").close()
    r = sge(dict(), wd, tmp_path); assert r.returncode == 1 and "FAILED" in r.stdout and "VERY BAD NEWS" in r.stdout and "done already" in r.stdout, r.stdout[-1500:]
    st = run("mlp_pipeline/tdi/dft_check.py", "status", wd); assert "done 14  running 0  failed 1  pending 0  of 15" in st, st[-400:]
    os.remove(os.path.join(wd, "f2_host", "FAIL")); r = sge(dict(), wd, tmp_path); assert r.returncode == 0 and "1 ran, 14 already done" in r.stdout, r.stdout[-600:]
    out = run("mlp_pipeline/tdi/dft_check.py", "analyze", wd, "--model", "toyice", "--device", "cpu"); assert "15 of 15 DFT single points finished" in out and "mlpmin" in out and "lowphi" in out, out[-900:]
    open(os.path.join(tpl, "INCAR"), "a").write("NCORE = 16\n")
    sd = str(tmp_path / "selftest"); out = run("mlp_pipeline/tdi/dft_check.py", "prep-selftest", "--template", tpl, "-o", sd); assert "2 single-point directories" in out and "selftest_small" in out, out[-700:]
    r = sge(dict(NSLOTS="12"), sd, tmp_path); assert r.returncode == 0 and "NCORE 16 -> 12" in r.stdout, r.stdout[-900:]   # NCORE must divide the ranks
    assert "NCORE = 12" in open(os.path.join(sd, "selftest_small", "INCAR")).read()
    out = run("mlp_pipeline/tdi/dft_check.py", "analyze", sd, "--model", "toyice", "--device", "cpu"); assert "finished; 24 atoms on 16 ranks" in out and "finished; 12 atoms" in out and "MLP - DFT" in out, out[-900:]
    nm = subprocess.run(["bash", "mlp_pipeline/young_dft_check.sge", sd], cwd=ROOT, env=dict(ENV, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", NSLOTS="16"), capture_output=True, text=True)   # no module command, not a login shell
    assert nm.returncode == 2 and "module command not found" in nm.stdout, nm.stdout[-400:]
    wo = os.path.join(sd, "selftest_small", "OUTCAR"); tr = str(tmp_path / "train_OUTCAR"); open(tr, "w").write(open(wo).read().replace("1100.0", "1000.0"))
    out = run("mlp_pipeline/tdi/dft_check.py", "audit", sd, "--template", tpl, "--train-outcar", tr, "--new-outcar", wo)
    assert "2 of 2" in out and "[single-point tag]" in out and "NCORE: template '16' -> '12'   [parallelization" in out and "UNEXPECTED" not in out and "<-- differs" in out, out[-1200:]
    for x in [n for n in sorted(os.listdir(wd)) if os.path.isdir(os.path.join(wd, n))][:6]: os.remove(os.path.join(wd, x, "OUTCAR"))   # PAR: six points at once-ish on one exclusive node, NP split between them
    r = sge(dict(PAR="3", NP="12"), wd, tmp_path); assert r.returncode == 0 and "PAR 3: 3 directories at once, 4 ranks each" in r.stdout and "15 of 15 finished" in r.stdout and r.stdout.count("finished /") == 6, r.stdout[-1800:]
    g = str(tmp_path / "guard"); os.makedirs(g); ds = [os.path.join(wd, x) for x in ("f0_host", "f1_host")]
    for x in ds: os.remove(os.path.join(x, "OUTCAR"))
    open(os.path.join(g, "dirs.txt"), "w").write("\n".join(ds) + "\n")
    r = sge(dict(STUB_HRT="500", MARGIN="600"), g, tmp_path); assert "stopping" in r.stdout and "STOPPED EARLY" in r.stdout, r.stdout[-600:]   # 500 s h_rt < 600 s margin: nothing starts
