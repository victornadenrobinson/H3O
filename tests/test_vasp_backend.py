"""DFT force-engine plumbing using a mock executable, never real VASP."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from tdi import vasp_backend as backend
from tdi import dft_pwidom
from tdi import pwidom
from test_static_check import fake_template


@pytest.fixture
def configured(tmp_path,monkeypatch):
    template=tmp_path/'template';fake_template(str(template))
    exe=tmp_path/'fake_vasp'
    exe.write_text(f'''#!{sys.executable}
from pathlib import Path
Path('OUTCAR').write_text('free energy TOTEN = -1.25 eV\\n aborting loop because EDIFF is reached\\n General timing\\n')
Path('WAVECAR').write_text('mock wave')
''');exe.chmod(0o755)
    path=tmp_path/'config.json'
    backend.vasp_config(Namespace(out=str(path),template=str(template),exe=str(exe),T=3000.,ranks=32,work=str(tmp_path/'work'),timeout_seconds=10))
    config=json.loads(path.read_text());config['command']=[str(exe)];config['allow_local_test']=True;path.write_text(json.dumps(config))
    from ase.io import read
    def target(path,**kwargs):
        at=read(Path(path).parent/'POSCAR');at.calc=SinglePointCalculator(at,energy=-99.,forces=np.arange(len(at)*3).reshape(-1,3)*.1,stress=np.arange(6)*.01)
        return at
    monkeypatch.setattr(backend,'read',target)
    return path,exe,template


def test_force_order_totEN_cache_and_warm_start(configured):
    path,exe,_=configured;calc=backend.VaspEngine(path)
    at=Atoms('OH2',positions=[[0,0,0],[1,0,0],[0,1,0]],cell=[5,5,5],pbc=True);at.calc=calc
    assert at.get_potential_energy()==-1.25
    sorted_forces=np.arange(9).reshape(3,3)*.1
    np.testing.assert_allclose(at.get_forces(),sorted_forces[[2,0,1]])
    assert calc.ncalls==1
    calc.reset();at.get_forces();assert calc.ncalls==1 # persistent geometry cache
    at.positions[0,2]+=.01;at.get_forces();assert calc.ncalls==2
    inc=(Path(calc.directory)/'INCAR').read_text()
    assert 'ISTART = 1' in inc
    assert inc.count('LWAVE =')==1 and inc.count('ISTART =')==1
    assert 'LWAVE = .TRUE.' in inc
    assert len(list(calc.cache.glob('*.npz')))==2
    assert 'vasp' not in str(pwidom.model_fingerprint('vasp:'+str(path)))


def test_failure_never_cached(configured):
    path,exe,_=configured
    exe.write_text(f'#!{sys.executable}\nfrom pathlib import Path\nPath("OUTCAR").write_text("free energy TOTEN = -1.0 eV")\n');exe.chmod(0o755)
    calc=backend.VaspEngine(path);at=Atoms('H',positions=[[0,0,0]],cell=[5,5,5],pbc=True);at.calc=calc
    with pytest.raises(RuntimeError,match='unconverged'):at.get_forces()
    assert not list(calc.cache.glob('*.npz'))


def test_template_change_rejected(configured):
    path,_,template=configured
    (template/'INCAR').write_text((template/'INCAR').read_text()+'ENCUT=1200\n')
    with pytest.raises(ValueError,match='changed'):backend.VaspEngine(path)


def test_plan_and_smoke_isolation(configured,tmp_path,monkeypatch):
    path,exe,template=configured
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(P=500.,T=5000.,ice=dict(vpa=2.,B_GPa=1000.,nrep=3,source='test'),
        h225=dict(vpa=2.1,B_GPa=1100.,nrep=3,source='test'),H=dict(vpa=1.5,B_GPa=1200.,natoms=250,source='test'))))
    out=tmp_path/'plan'
    args=Namespace(state_json=str(state),out=str(out),replicates=2,ps=.5,equil_ps=1.,every_fs=20.,sel=4,ndel=4,
        template=str(template),exe=str(exe),ice_proxy='none',h_proxy='none')
    dft_pwidom.dft_plan(args)
    manifest=json.loads((out/'dft_plan.json').read_text());assert len(manifest['workers'])==8
    hydrogen=manifest['workers'][3]['args']
    assert hydrogen[hydrogen.index('--natoms')+1]=='729'
    assert len({json.loads((Path(w['directory'])/'vasp.json').read_text())['work'] for w in manifest['workers']})==8
    calls=[];monkeypatch.setattr(pwidom,'main',lambda:calls.append(sys.argv.copy()))
    dft_pwidom.dft_worker(Namespace(directory=str(out),index=0,smoke=True))
    assert calls[0][calls[0].index('--ps')+1]=='.01'
    assert calls[0][calls[0].index('--out')+1].endswith('/smoke')
    assert not (Path(manifest['workers'][0]['directory'])/'pw.json').exists()


def test_same_pwidom_pipeline_runs_with_vasp_force_engine(configured,tmp_path,monkeypatch):
    path,_,_=configured;out=tmp_path/'integration'
    monkeypatch.setattr(sys,'argv',['pwidom.py','run','--model','vasp:'+str(path),'--device','cpu',
        '--phase','fcc','--nrep','1','--vpa','5','--T','3000','--P','500','--ps','.01',
        '--equil-ps','.002','--every-fs','2','--dt','.5','--sel','1','--ndel','1','--batch','1',
        '--proxy','none','--out',str(out),'--tag','smoke'])
    pwidom.main()
    result=json.loads((out/'smoke_none_K1.json').read_text())
    assert result['model'].startswith('vasp:')
    assert result['n_ins']==5 and result['n_del']==5
    assert abs(result['mu_ex_bar_meV'])<1e-5
    assert (out/'smoke_host_md.npz').exists() and (out/'smoke_full_md.npz').exists()


def test_resize_h_preserves_old_workers_and_density(configured,tmp_path):
    _,exe,template=configured
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(P=500.,T=5000.,ice=dict(vpa=2.,B_GPa=1000.,nrep=3,source='test'),
        h225=dict(vpa=2.1,B_GPa=1100.,nrep=3,source='test'),H=dict(vpa=1.5,B_GPa=1200.,natoms=250,source='test'))))
    root=tmp_path/'plan'
    dft_pwidom.dft_plan(Namespace(state_json=str(state),out=str(root),replicates=1,ps=.5,equil_ps=1.,every_fs=20.,sel=4,ndel=4,
        template=str(template),exe=str(exe),ice_proxy='none',h_proxy='none',h_natoms=250))
    before=json.loads((root/'dft_plan.json').read_text());old=Path(before['workers'][3]['directory'])
    sentinel=old/'pw_host_md.npz';sentinel.write_bytes(b'old-checkpoint')
    dft_pwidom.dft_h_size(Namespace(directory=str(root),natoms=729))
    after=json.loads((root/'dft_plan.json').read_text())
    assert after['workers'][:3]==before['workers'][:3]
    assert sentinel.read_bytes()==b'old-checkpoint'
    assert after['state']['H']['vpa']==1.5
    new=after['workers'][3];assert 'pureH729' in new['directory']
    assert new['args'][new['args'].index('--natoms')+1]=='729'
    from tdi.run_real import build_h_n
    assert build_h_n(1.5,729).get_volume()==pytest.approx(729*1.5)
