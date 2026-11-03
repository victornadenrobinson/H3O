"""Paired VASP preparation and resumable execution with a fake executable only."""
from argparse import Namespace
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from ase.io import read

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
sys.path.insert(0,str(ROOT/'tests'))
from tdi import perlmutter_vasp_check as check
from tdi import pwidom as pw
from test_static_check import fake_template


@pytest.fixture
def prepared(tmp_path):
    template=tmp_path/'template';fake_template(str(template))
    root=tmp_path/'run';root.mkdir();jobs=[]
    cell=np.array([[5.,0,0],[1,5,0],[.5,1,5]])
    rng=np.random.default_rng(1)
    for name,numbers in [('H2O',np.array([8,1,1])),('pureH',np.array([1,1,1,1]))]:
        tag='pw_'+name
        host=rng.random((8,len(numbers),3))@cell
        full=rng.random((8,len(numbers)+1,3))@cell
        np.savez(root/(tag+'_frames.npz'),host=host,full=full,numbers=numbers,
            cell=cell,V=abs(np.linalg.det(cell)),T=3000.,model_fp=pw.model_fingerprint('toyice'))
        result=dict(tag=tag,T=3000.,P_target=300.,model='toyice',model_fp=pw.model_fingerprint('toyice'))
        path=root/(tag+'.json');path.write_text(json.dumps(result))
        jobs.append(dict(name=name,result=str(path),command=['--proxy','none']))
    (root/'manifest.json').write_text(json.dumps(dict(status='complete',jobs=jobs)))
    out=tmp_path/'check'
    args=Namespace(run=str(root),template=str(template),out=str(out),pairs=8,hosts=['H2O','pureH'],K=2000,discard_fraction=.5,seed=17)
    check.prep(args)
    return out,args,cell


def test_balanced_pairs_actual_cell_and_template(prepared):
    out,args,cell=prepared
    manifest=json.loads((out/'manifest.json').read_text())
    assert len(manifest['pairs'])==8 and len(manifest['points'])==16
    for pair in manifest['pairs']:
        a,b=[read(out/i/'POSCAR') for i in pair['points']]
        np.testing.assert_allclose(a.cell.array,cell)
        assert len(b)==len(a)+1 and sum(b.numbers==1)==sum(a.numbers==1)+1
        assert pair['frame']>=4
        for i in pair['points']:
            inc=(out/i/'INCAR').read_text()
            assert 'NSW = 0' in inc and 'IBRION = -1' in inc and 'TEBEG' not in inc
            assert 'ENCUT = 1100' in inc and 'SIGMA = 0.258520' in inc
            pot=(out/i/'POTCAR').read_text()
            if pair['host']=='pureH':assert 'PAW_PBE O' not in pot
    with pytest.raises(ValueError,match='fresh'):check.prep(args)


def runner_args(out,exe,**overrides):
    values=dict(directory=str(out),exe=str(exe),ranks=1,minutes=5.,launcher='',margin_seconds=0.,max_points=None,index=None,dry_run=False)
    values.update(overrides);return Namespace(**values)


def fake_vasp(tmp_path):
    exe=tmp_path/'fake_vasp'
    exe.write_text(f'''#!{sys.executable}
from pathlib import Path
from ase.io import read
point=Path.cwd().parent.parent
energy=-sum(read('structure.extxyz').numbers)/10
text=f"vasp.5.4.4 test\\n free energy TOTEN = {{energy:.9f}} eV\\n General timing and accounting\\n Elapsed time (sec): 2.0\\n"
if not (point/'FAIL').exists():text+=' aborting loop because EDIFF is reached\\n'
Path('OUTCAR').write_text(text)
''')
    exe.chmod(0o700)
    return exe


def test_execute_skip_and_preserve_attempts(prepared,tmp_path,monkeypatch):
    out,_,_=prepared;exe=fake_vasp(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','test')
    check.run(runner_args(out,exe,max_points=2))
    first=out/'p000_without_H';record=check.done(first);old=(first/record['attempt']/'OUTCAR').read_bytes()
    check.run(runner_args(out,exe,max_points=2))
    assert (first/record['attempt']/'OUTCAR').read_bytes()==old
    assert len(list((first/'attempts').iterdir()))==1
    manifest=json.loads((out/'manifest.json').read_text())
    assert sum(check.done(out/p['path']) is not None for p in manifest['points'])==4
    # First complete pair has ΔU=-0.1 eV by the independent stand-in energy.
    analysis=tmp_path/'analysis'
    check.analyze(Namespace(directory=str(out),out=str(analysis),compare_mlp=False,device='cpu',batch=1))
    import csv
    rows=list(csv.DictReader((analysis/'paired_energies.csv').open()))
    assert float(rows[0]['dU_DFT_eV'])==pytest.approx(-.1)


def test_unconverged_footer_not_accepted_and_retry_is_nondestructive(prepared,tmp_path,monkeypatch):
    out,_,_=prepared;exe=fake_vasp(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','test')
    point=out/'p000_without_H';(point/'FAIL').touch()
    with pytest.raises(SystemExit):check.run(runner_args(out,exe,max_points=1))
    assert check.done(point) is None
    old_attempt=next((point/'attempts').iterdir());old=(old_attempt/'OUTCAR').read_bytes()
    assert check.converged(old_attempt/'OUTCAR') is None
    (point/'FAIL').unlink()
    check.run(runner_args(out,exe,max_points=1))
    assert check.done(point) is not None
    assert len(list((point/'attempts').iterdir()))==2
    assert (old_attempt/'OUTCAR').read_bytes()==old


def test_no_login_node_vasp_execution(prepared,tmp_path,monkeypatch):
    out,_,_=prepared;exe=fake_vasp(tmp_path);monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='Slurm allocation'):check.run(runner_args(out,exe))


def test_exactly_one_cached_snapshot(prepared,tmp_path):
    _,args,cell=prepared
    out=tmp_path/'one_snapshot'
    cache=Path(args.run)/'pw_H2O_frames.npz'
    check.snapshot(Namespace(frames=str(cache),template=args.template,out=str(out),frame=-1,ensemble='host'))
    manifest=json.loads((out/'manifest.json').read_text())
    assert len(manifest['points'])==1 and manifest['points'][0]['natoms']==3
    at=read(out/'snapshot/POSCAR')
    np.testing.assert_allclose(at.cell.array,cell)
    with np.load(cache) as z:
        np.testing.assert_allclose(np.sort(at.get_scaled_positions()%1,axis=0),np.sort((z['host'][-1]@np.linalg.inv(cell))%1,axis=0))


def make_queue(prepared,tmp_path):
    from ase import Atoms
    from tdi.pwidom_candidates import enqueue
    _,args,cell=prepared
    queue=tmp_path/'queue'
    with np.load(Path(args.run)/'pw_H2O_frames.npz') as z:
        host=Atoms(numbers=z['numbers'],positions=z['host'][4],cell=cell,pbc=True)
    full=host+Atoms('H',positions=[[.5,.5,.5]])
    metadata=dict(run=str(Path(args.run).resolve()),tag='pw_H2O',direction='insertion',
        geometry_order=['host_without_selected_H','host_with_selected_H'],dU_mlp_eV=-.05,
        model='toyice',model_fp=pw.model_fingerprint('toyice'),T_K=3000.,P_GPa=300.,
        K=2000,frame=4,trial=0,phi_eV=.1,proxy='none',log_weight=1.,
        source_frames=str(Path(args.run)/'pw_H2O_frames.npz'),reasons=['representative'])
    enqueue(queue,[host,full],metadata)
    enqueue(queue,[host,full],dict(metadata,run=str(tmp_path/'another_run')))
    return queue,Namespace(queue=str(queue),run=args.run,hosts=['H2O'],pairs=1,template=args.template,out=str(tmp_path/'queued'))


def test_saved_queue_filtered_and_analyzed_without_model(prepared,tmp_path,monkeypatch):
    queue,args=make_queue(prepared,tmp_path)
    check.prep_queue(args)
    out=Path(args.out);manifest=json.loads((out/'manifest.json').read_text())
    assert len(manifest['pairs'])==1 and len(manifest['points'])==2
    assert manifest['pairs'][0]['dU_mlp_eV']==-.05
    assert manifest['pairs'][0]['request_file'].startswith(str(queue))
    monkeypatch.setenv('SLURM_JOB_ID','test');exe=fake_vasp(tmp_path)
    check.run(runner_args(out,exe))
    def no_model(*a,**k):raise AssertionError('Saved GRACE difference must not trigger model calls')
    monkeypatch.setattr(pw,'Model',no_model)
    analysis=tmp_path/'queued_analysis'
    check.analyze(Namespace(directory=str(out),out=str(analysis),compare_mlp=False,device='cpu',batch=1))
    import csv
    row=next(csv.DictReader((analysis/'paired_energies.csv').open()))
    assert float(row['dU_DFT_eV'])==pytest.approx(-.1)
    assert float(row['dU_MLP_eV'])==pytest.approx(-.05)
    assert float(row['DFT_minus_MLP_eV'])==pytest.approx(-.05)


def test_queue_geometry_changes_refused(prepared,tmp_path):
    from ase.io import write
    queue,args=make_queue(prepared,tmp_path)
    file=next(queue.glob('*.extxyz'));at=read(file);at.positions+=.01;write(file,at,format='extxyz')
    with pytest.raises(ValueError,match='hash changed'):check.prep_queue(args)
    assert not Path(args.out).exists()


def test_four_concurrent_points_and_allocation_limit(prepared,tmp_path,monkeypatch):
    out,_,_=prepared;exe=fake_vasp(tmp_path)
    text=exe.read_text();lines=text.splitlines(keepends=True)
    lines.insert(1,"import os,time\nfrom pathlib import Path\ntrace=Path(os.environ['FAKE_VASP_TRACE'])\nwith trace.open('a') as f:f.write(f'S {os.getpid()} {time.monotonic_ns()}\\n')\ntime.sleep(.8)\n")
    lines.append("with trace.open('a') as f:f.write(f'E {os.getpid()} {time.monotonic_ns()}\\n')\n")
    exe.write_text(''.join(lines));trace=tmp_path/'trace'
    monkeypatch.setenv('SLURM_JOB_ID','test');monkeypatch.setenv('SLURM_NTASKS','4');monkeypatch.setenv('FAKE_VASP_TRACE',str(trace))
    check.run(runner_args(out,exe,parallel=4,max_points=4))
    events=sorted((int(line.split()[2]),line.split()[0]) for line in trace.read_text().splitlines())
    active=peak=0
    for _,event in events:
        active+=1 if event=='S' else -1;peak=max(peak,active)
    assert peak==4 and active==0
    manifest=json.loads((out/'manifest.json').read_text())
    assert sum(check.done(out/p['path']) is not None for p in manifest['points'])==4
    monkeypatch.setenv('SLURM_NTASKS','3')
    with pytest.raises(ValueError,match='Packing needs'):check.run(runner_args(out,exe,parallel=4,max_points=4))


def test_parallel_srun_is_exclusive_and_exact(prepared,tmp_path,capsys):
    out,_,_=prepared;exe=fake_vasp(tmp_path)
    check.run(runner_args(out,exe,launcher='srun',parallel=4,max_points=4,dry_run=True))
    output=capsys.readouterr().out
    assert '--exclusive --exact --nodes=1 --ntasks 1' in output


def test_gamma_eligibility_and_live_saved_pair(prepared,tmp_path,monkeypatch,capsys):
    out,_,_=prepared
    point=out/'p000_without_H'
    assert check.gamma_only(point)
    kp=point/'KPOINTS';original=kp.read_text()
    kp.write_text('mesh\n0\nGamma\n2 2 2\n0 0 0\n')
    assert not check.gamma_only(point)
    kp.write_text('mesh\n0\nGamma\n1 1 1\n.5 0 0\n')
    assert not check.gamma_only(point)
    kp.write_text(original)
    queue,args=make_queue(prepared,tmp_path)
    check.prep_queue(args);root=Path(args.out)
    monkeypatch.setenv('SLURM_JOB_ID','test');exe=fake_vasp(tmp_path)
    check.run(runner_args(root,exe))
    assert 'VASP−GRACE=-50.00 meV/H' in capsys.readouterr().out
    check.watch(Namespace(directory=str(root),interval=1,once=True))
    assert 'ΔU_GRACE=-0.050000 eV' in capsys.readouterr().out


def test_mesh_check_preserves_source_and_pair_geometry(prepared,tmp_path,monkeypatch):
    out,_,_=prepared;exe=fake_vasp(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','test')
    check.run(runner_args(out,exe,max_points=2))
    old=(out/'p000_without_H/KPOINTS').read_bytes()
    mesh=tmp_path/'mesh'
    check.mesh_check(Namespace(directory=str(out),hosts=['H2O'],pairs_per_host=1,mesh=[2,2,2],out=str(mesh)))
    manifest=json.loads((mesh/'manifest.json').read_text())
    assert len(manifest['points'])==2 and manifest['pairs'][0]['baseline_dU_DFT_eV']==pytest.approx(-.1)
    assert (out/'p000_without_H/KPOINTS').read_bytes()==old
    assert '2 2 2' in (mesh/'p000_without_H/KPOINTS').read_text()
    assert not check.gamma_only(mesh/'p000_without_H')
    assert (mesh/'p000_without_H/structure.extxyz').read_bytes()==(out/'p000_without_H/structure.extxyz').read_bytes()


def test_training_label_order_and_reproduction(tmp_path,monkeypatch,capsys):
    from ase import Atoms
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    template=tmp_path/'tdi'/'5000K'/'volume';fake_template(str(template))
    inc=template/'INCAR';inc.write_text(inc.read_text().replace('SIGMA = 0.43','SIGMA = 0.43086652'))
    at=Atoms('OH2',positions=[[0,0,0],[1,0,0],[0,1,0]],cell=[5,5,5],pbc=True)
    forces=np.arange(9).reshape(3,3)/100
    at.calc=SinglePointCalculator(at,energy=-1.,free_energy=-1.,forces=forces,stress=np.zeros(6))
    at.info.update(source='tdi/5000K/volume/out10/OUTCAR.gz',frame=42,T_K=5000.,P_GPa=600.)
    data=tmp_path/'train.extxyz';write(data,at)
    out=tmp_path/'labels'
    args=Namespace(data=[str(data)],out=str(out),count=20,source_root=str(tmp_path),template=None,T=5000.,P=600.,pressure_window=100.)
    check.training(args)
    manifest=json.loads((out/'manifest.json').read_text());point=manifest['points'][0]
    with np.load(out/point['id']/'reference.npz') as ref:np.testing.assert_allclose(ref['forces'],forces[[1,2,0]])
    assert point['frame']==42 and len(manifest['points'])==1
    monkeypatch.setenv('SLURM_JOB_ID','fake')
    check.run(runner_args(out,fake_vasp(tmp_path)))
    result=json.loads((out/point['id']/'label_comparison.json').read_text())
    assert result['energy_difference_eV']==pytest.approx(0.)
    assert 'force_stress_error' in result # synthetic OUTCAR intentionally contains energy only
    assert 'LABEL train_000' in capsys.readouterr().out
    check.run(runner_args(out,tmp_path/'fake_vasp')) # completion remains resumable
    with pytest.raises(ValueError,match='fresh'):check.training(args)


def test_training_rejects_wrong_temperature_before_writing(tmp_path):
    from ase import Atoms
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    template=tmp_path/'template';fake_template(str(template))
    at=Atoms('H2O',positions=np.zeros((3,3)),cell=[5,5,5],pbc=True)
    at.calc=SinglePointCalculator(at,energy=-1.,forces=np.zeros((3,3)))
    at.info.update(source='tdi/OUTCAR.gz',frame=0,T_K=3000.)
    data=tmp_path/'train.extxyz';write(data,at);out=tmp_path/'bad'
    with pytest.raises(ValueError,match='SIGMA'):
        check.training(Namespace(data=[str(data)],out=str(out),count=1,source_root=str(tmp_path),template=str(template),T=None,P=None,pressure_window=100.))
    assert not out.exists()


def test_replay_recomputes_saved_delta(prepared,tmp_path,monkeypatch):
    import csv
    _,args=make_queue(prepared,tmp_path);check.prep_queue(args);out=Path(args.out)
    monkeypatch.setenv('SLURM_JOB_ID','test');check.run(runner_args(out,fake_vasp(tmp_path)))
    class DirectModel:
        def __init__(self,model,device,batch):assert batch==1
        def energies(self,atoms):return np.array([-10.,-10.5])
    monkeypatch.setattr(pw,'Model',DirectModel)
    analysis=tmp_path/'replay'
    check.analyze(Namespace(directory=str(out),out=str(analysis),compare_mlp=True,device='cpu',batch=1))
    row=next(csv.DictReader((analysis/'paired_energies.csv').open()))
    assert float(row['dU_MLP_saved_eV'])==pytest.approx(-.05)
    assert float(row['dU_MLP_recomputed_eV'])==pytest.approx(-.5)
    assert float(row['DFT_minus_MLP_eV'])==pytest.approx(.4)
    assert json.loads((analysis/'summary.json').read_text())['mlp_replay_audit']['mismatched_pairs']==1
