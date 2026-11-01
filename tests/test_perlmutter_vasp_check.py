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
