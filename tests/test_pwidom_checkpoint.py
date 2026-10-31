"""Short CPU harmonic MD verifies exact checkpoint continuation; no GRACE/GPU."""
from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'mlp_pipeline'))
from tdi import pwidom_checkpoint as cp
from tdi.pwidom import md_frames


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = dict(energy=float((self.atoms.positions**2).sum()/2),
                            forces=-self.atoms.positions, stress=np.zeros(6))


def setup(tmp_path):
    atoms=Atoms('H2O',positions=[[1,1,1],[2,1,1],[1,2,1]],cell=[8]*3,pbc=True)
    args=Namespace(T=300,dt=.5,equil_ps=.002,ps=.01,every_fs=2,
                   model='toyice',checkpoint_seconds=1e-12,resume_from=None,
                   _checkpoint_prefix=str(tmp_path/'first'))
    return SimpleNamespace(calc=Harmonic()),atoms,args


@pytest.mark.parametrize('cut', [2, 4, 7, 24])
def test_interrupted_md_matches_uninterrupted(tmp_path,monkeypatch,cut):
    model, atoms, args=setup(tmp_path)
    expected=cp.md_frames(model,atoms,args,3,'host')
    prefix=args._checkpoint_prefix
    args._checkpoint_prefix=str(tmp_path/'interrupted')
    original=cp.atomic_npz
    def interrupt(path, **payload):
        original(path,**payload)
        if payload['step']==cut:
            raise SystemExit(143)
    monkeypatch.setattr(cp,'atomic_npz',interrupt)
    with pytest.raises(SystemExit): cp.md_frames(model,atoms,args,3,'host')
    saved=Path(args._checkpoint_prefix+'_host_md.npz'); before=saved.read_bytes()
    args.resume_from=args._checkpoint_prefix; args._checkpoint_prefix=str(tmp_path/'resumed')
    monkeypatch.setattr(cp,'atomic_npz',original)
    actual=cp.md_frames(model,atoms,args,3,'host')
    for x,y in zip(actual,expected): np.testing.assert_array_equal(x,y)
    assert saved.read_bytes()==before
    with np.load(prefix+'_host_md.npz') as a, np.load(args._checkpoint_prefix+'_host_md.npz') as b:
        for key in ('positions','momenta','rng_state','step'):
            np.testing.assert_array_equal(a[key],b[key])


def test_mismatch_and_existing_destination_refused(tmp_path):
    model,atoms,args=setup(tmp_path)
    cp.md_frames(model,atoms,args,3,'host')
    with pytest.raises(ValueError,match='Checkpoint exists'): cp.md_frames(model,atoms,args,3,'host')
    args.resume_from=args._checkpoint_prefix; args._checkpoint_prefix=str(tmp_path/'bad'); args.T=400
    with pytest.raises(ValueError,match='mismatch'): cp.md_frames(model,atoms,args,3,'host')
    assert not Path(args._checkpoint_prefix+'_host_md.npz').exists()


def test_checkpoint_does_not_change_trajectory(tmp_path):
    model,atoms,args=setup(tmp_path)
    with_checkpoints=cp.md_frames(model,atoms,args,3,'host')
    del args._checkpoint_prefix
    without=md_frames(model,atoms,args,3,'host')
    for x,y in zip(with_checkpoints,without): np.testing.assert_array_equal(x,y)


def test_energy_batches_resume_and_validate(tmp_path,monkeypatch):
    from tdi.pwidom import Model
    model,atoms,_=setup(tmp_path)
    evaluator=Model.__new__(Model)
    evaluator.calc=model.calc; evaluator.batch=2; evaluator.gb=None; evaluator.ncalls=0
    evaluator.checkpoint_model_fp='test'
    evaluator.energy_checkpoint=str(tmp_path/'energies.npz')
    structs=[]
    for shift in range(5):
        at=atoms.copy(); at.positions+=shift*.01; structs.append(at)
    original=cp.atomic_npz
    def interrupt(path,**payload):
        original(path,**payload)
        if len(payload['energies'])==2: raise SystemExit(143)
    monkeypatch.setattr(cp,'atomic_npz',interrupt)
    with pytest.raises(SystemExit): evaluator.energies(structs)
    before=Path(evaluator.energy_checkpoint).read_bytes()
    evaluator.energy_resume=evaluator.energy_checkpoint
    evaluator.energy_checkpoint=str(tmp_path/'resumed-energies.npz')
    evaluator.ncalls=0
    monkeypatch.setattr(cp,'atomic_npz',original)
    actual=evaluator.energies(structs)
    np.testing.assert_allclose(actual,[(at.positions**2).sum()/2 for at in structs])
    assert evaluator.ncalls==2
    assert Path(evaluator.energy_resume).read_bytes()==before
    evaluator.energy_checkpoint=str(tmp_path/'bad-energies.npz')
    structs[0].positions+=.1
    with pytest.raises(ValueError,match='mismatch'): evaluator.energies(structs)


def test_term_signal_flushes_at_complete_step(tmp_path,monkeypatch):
    import signal
    model,atoms,args=setup(tmp_path)
    expected=cp.md_frames(model,atoms,args,3,'host')
    args._checkpoint_prefix=str(tmp_path/'signal')
    original=Harmonic.calculate
    calls=[0]
    def terminate(self,*a,**kw):
        original(self,*a,**kw); calls[0]+=1
        if calls[0]==7: signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(Harmonic,'calculate',terminate)
    model.calc=Harmonic()
    with pytest.raises(SystemExit) as stopped: cp.md_frames(model,atoms,args,3,'host')
    assert stopped.value.code==143
    with np.load(args._checkpoint_prefix+'_host_md.npz') as z:
        assert 0<int(z['step'])<24
    monkeypatch.setattr(Harmonic,'calculate',original)
    args.resume_from=args._checkpoint_prefix; args._checkpoint_prefix=str(tmp_path/'after-signal')
    model.calc=Harmonic()
    actual=cp.md_frames(model,atoms,args,3,'host')
    for x,y in zip(actual,expected): np.testing.assert_array_equal(x,y)


def test_failed_atomic_write_retains_previous_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'checkpoint.npz'
    cp.atomic_npz(path,step=1)
    original=path.read_bytes()
    def fail(*a,**kw): raise OSError('disk write failure')
    monkeypatch.setattr(cp.np,'savez',fail)
    with pytest.raises(OSError): cp.atomic_npz(path,step=2)
    assert path.read_bytes()==original
    assert len(list(tmp_path.iterdir()))==1
