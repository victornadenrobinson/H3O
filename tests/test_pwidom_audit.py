"""Synthetic CPU-only EOS, periodic geometry and saved-work regressions."""
import importlib.util
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import pytest
from ase.geometry import find_mic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'mlp_pipeline'))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


eos = load(ROOT/'build_eos_2000.py')
conv = load(ROOT/'mlp_pipeline/plot_bar_convergence.py')
rdf = load(ROOT/'mlp_pipeline/plot_pwidom_diagnostics.py')


def data():
    rows = []
    for t, scale in [(2000, 1), (4000, 1.1)]:
        for v in np.linspace(1, 3, 8):
            rows.append(dict(good=True, Path='fcc/run', T_bin=t, True_P_GPa=400*(v/scale)**-2,
                             V_per_atom=v, comp_snap='h2o', N_H=2, N_O=1, Other='', N_ions=3))
    return pd.DataFrame(rows)


def test_eos_powerlaw_interpolation_and_unrestricted_pressure():
    d = data(); fit = eos.fit_eos(eos.select(d, 'h2o', 2000), 'h2o', 2000)
    for p in (1e-6, 200, 1e10):
        v, b, _ = eos.volume_at_pressure(fit, p)
        assert v == pytest.approx(np.sqrt(400/p), rel=1e-8)
        assert b == pytest.approx(2*p, rel=1e-8)
    state = eos.state_at(d, 'h2o', 100, 3000)
    assert state['vpa'] == pytest.approx(2.1)
    assert state['B_GPa'] == pytest.approx(200)
    assert eos.state_at(d, 'h2o', 100, 6000)['temperature_extrapolated']
    for p in (np.nan, np.inf, 0):
        with pytest.raises(ValueError): eos.volume_at_pressure(fit, p)


def test_foreign_species_and_false_labels_excluded(tmp_path):
    d = data(); bad = d.iloc[:1].copy(); bad['Other'] = 'Fe:32'
    bad2 = d.iloc[:1].copy(); bad2['N_ions'] = 99
    bad3 = d.iloc[:1].copy(); bad3['N_O'] = 0
    p = tmp_path/'inventory.csv'; pd.concat([d,bad,bad2,bad3]).to_csv(p,index=False)
    assert len(eos.load_inventory(p)) == len(d)


def test_bar_unequal_samples_and_reconstruction(tmp_path):
    # Exact constant-work Crooks pair, with eight vs four trials per frame.
    f = np.full((20,8), 1.7); r = np.full((13,4), -1.7)
    assert conv.diagnose(f,r,2000)['mu_ex_bar_meV'] == pytest.approx(1700)
    p = tmp_path/'run_work.npz'; np.savez(p,forward_work_eV=f,reverse_work_eV=r,T=2000)
    p.with_name('run.json').write_text(json.dumps(dict(mu_ex_bar_meV=1700)))
    rows = conv.analyse(p,tmp_path/'plots',4,3)
    assert all(row['mu_ex_bar_meV'] == pytest.approx(1700) for row in rows)
    assert (tmp_path/'plots/diagnostics.csv').exists()
    with pytest.raises(FileExistsError): conv.analyse(p,tmp_path/'plots',4,3)


def test_skew_mic_and_full_coordinates(tmp_path):
    from argparse import Namespace
    cell = np.array([[8,0,0],[6,8,0],[0,0,8.]])
    delta = np.array([[7.8,0,0], [6,7.8,0]])
    _, distances = find_mic(delta,cell)
    np.testing.assert_allclose(distances, [.2,.2],atol=1e-10)
    host = np.array([[[4,4,4],[4,4,5]]],float)
    full = np.array([[[.2,0,0],[0,.2,0],[0,0,0]]])
    rdf.rdf(dict(host=host, full=full, numbers=np.array([8,1]),cell=cell,V=512),
            Namespace(assume_cubic=False,rmax=1,dr=.1,stride=1),tmp_path)
    with np.load(tmp_path/'rdf_data.npz') as z:
        assert z['saved_extraH_O'].sum() > 0
        assert z['g_OH'].sum() == 0  # host separation is exactly cutoff
    with pytest.raises(ValueError): rdf.get_cells(dict(V=512),1,False)


def test_auto_dry_run_missing_inventory_and_argument_support(monkeypatch):
    monkeypatch.setenv("PY", sys.executable)
    r = subprocess.run(['bash','mlp_pipeline/pwidom_auto.sh','--dry-run','--inventory','/missing',
                        '--P','500','--T','2000','4000'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode == 0, r.stderr
    assert '--extra-h 5 11 16 22' in r.stdout
    assert '--reference-run' in r.stdout and '4000K_extra' in r.stdout
    r = subprocess.run([sys.executable,'mlp_pipeline/tdi/perlmutter_pwidom.py','run','--help'],
                       cwd=ROOT,capture_output=True,text=True)
    assert r.returncode == 0 and '--extra-h' in r.stdout and '--reference-run' in r.stdout


def test_eos_json_compositions_and_force(tmp_path):
    d = data()
    pure = d.copy(); pure['comp_snap']='h'; pure['N_H']=3; pure['N_O']=0
    rich = d.copy(); rich['comp_snap']='h2.25o'; rich['N_H']=9; rich['N_O']=4; rich['N_ions']=13
    inv=tmp_path/'inventory.csv'; pd.concat([d,pure,rich]).to_csv(inv,index=False)
    out=tmp_path/'envs'
    cmd=[sys.executable,str(ROOT/'build_eos_2000.py'),'--inventory',str(inv),'--outdir',str(out),
         '--P','100','200','--T','2000','3000']
    subprocess.run(cmd,check=True,capture_output=True)
    assert len(list(out.glob('*.json'))) == 4
    p=out/'pwidom_100GPa_2000K.json'; state=json.loads(p.read_text())
    assert state['ice']['nrep']==3 and state['h225']['nrep']==3 and state['H']['natoms']==250
    p.write_text('sentinel')
    subprocess.run(cmd,check=True,capture_output=True); assert p.read_text()=='sentinel'
    subprocess.run(cmd+['--force'],check=True,capture_output=True); assert json.loads(p.read_text())['P']==100


def test_bar_crooks_gaussians_unequal_counts():
    rng=np.random.default_rng(451); kt=conv.KB*3000; mu=.7; sigma=.3
    f=rng.normal(mu+sigma**2/(2*kt),sigma,(40000,2))
    r=-rng.normal(mu-sigma**2/(2*kt),sigma,(10000,4))
    assert conv.diagnose(f,r,3000)['mu_ex_bar_meV']==pytest.approx(700,abs=5)


def test_rosenbluth_extended_space_identity():
    # Enumerate all K=2 pools for a two-site insertion model. Target deletion
    # samples its occupied site with Boltzmann probability and one uniform site.
    from itertools import product
    kt=.4; u=np.array([.2,1.]); phi=np.array([-.1,.7]); z=np.exp(-u/kt).mean()
    forward=reverse=0.
    for pool in product(range(2),repeat=2):
        weights=np.exp(-phi[list(pool)]/kt); w=weights.mean()
        for j,site in enumerate(pool):
            work=u[site]-phi[site]-kt*np.log(w)
            forward+=.25*weights[j]/weights.sum()*np.exp(-work/kt)
    for site,other in product(range(2),repeat=2):
        w=np.exp(-phi[[site,other]]/kt).mean()
        work=u[site]-phi[site]-kt*np.log(w)
        reverse+=np.exp(-u[site]/kt)/(2*z)*.5*np.exp(work/kt)
    assert forward==pytest.approx(z)
    assert reverse==pytest.approx(1/z)


def test_auto_respects_explicit_bad_python(monkeypatch):
    monkeypatch.setenv("PWIDOM_PYTHON", "/missing/python")
    r = subprocess.run(['bash','mlp_pipeline/pwidom_auto.sh','--dry-run'],
                       cwd=ROOT,capture_output=True,text=True)
    assert r.returncode != 0 and "PY is explicit" in r.stderr


def test_auto_validates_existing_eos_in_dry_run(tmp_path, monkeypatch):
    import shutil
    # Isolated checkout-shaped fixture; never alter real envs/runs.
    shutil.copytree(ROOT/'mlp_pipeline',tmp_path/'mlp_pipeline')
    shutil.copy2(ROOT/'build_eos_2000.py',tmp_path/'build_eos_2000.py')
    (tmp_path/'envs').mkdir()
    state=dict(P=500,T=2000,
        ice=dict(vpa=float('inf'),B_GPa=1000,nrep=3,source='synthetic'),
        h225=dict(vpa=2,B_GPa=1000,nrep=3,source='synthetic'),
        H=dict(vpa=2,B_GPa=1000,natoms=250,source='synthetic'))
    (tmp_path/'envs/pwidom_500GPa_2000K.json').write_text(json.dumps(state))
    monkeypatch.setenv("PY",sys.executable)
    r=subprocess.run(['bash',str(tmp_path/'mlp_pipeline/pwidom_auto.sh'),'--dry-run'],
                     capture_output=True,text=True)
    assert r.returncode != 0 and 'must be finite and positive' in r.stderr
    assert not (tmp_path/'runs').exists()
