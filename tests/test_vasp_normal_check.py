"""Ordinary host sampling and original-label baseline without real DFT/GRACE."""
from argparse import Namespace
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from tdi import perlmutter_vasp_check as check
from tdi import vasp_normal_check as normal
from test_perlmutter_vasp_check import prepared, fake_template


def test_normal_uses_unmodified_host_frames(prepared,tmp_path):
    _,args,cell=prepared;out=tmp_path/'ordinary'
    normal.normal(Namespace(run=args.run,template=args.template,hosts=['H2O','pureH'],per_host=2,
        discard_fraction=.5,device='cpu',out=str(out)))
    manifest=json.loads((out/'manifest.json').read_text())
    assert len(manifest['points'])==4
    from ase.io import read
    for point in manifest['points']:
        assert point['frame'] in [4,7] and point['reference_kind']=='GRACE'
        at=read(out/point['path']/'structure.extxyz')
        assert len(at)==(3 if point['host']=='H2O' else 4)
        np.testing.assert_allclose(at.cell,cell)
        with np.load(out/point['path']/'reference.npz') as labels:
            assert labels['forces'].shape==(len(at),3)
            assert labels['stress'].shape==(6,)
    with pytest.raises(ValueError,match='fresh'):
        normal.normal(Namespace(out=str(out)))


def test_original_baseline_energy_and_forces(tmp_path,monkeypatch):
    template=tmp_path/'template';fake_template(str(template))
    inc=template/'INCAR';inc.write_text(inc.read_text().replace('SIGMA = 0.43','SIGMA = 0.43086652'))
    at=Atoms('OH2',positions=[[0,0,0],[1,0,0],[0,1,0]],cell=[5,5,5],pbc=True)
    at.calc=SinglePointCalculator(at,energy=-1.,forces=np.zeros((3,3)),stress=np.zeros(6))
    at.info.update(source='source/OUTCAR.gz',frame=0,T_K=5000.)
    data=tmp_path/'train.extxyz';write(data,at);root=tmp_path/'labels'
    check.training(Namespace(data=[str(data)],out=str(root),count=1,source_root='.',template=str(template),T=None,P=None,pressure_window=100.))
    class Model:
        def __init__(self,*args):
            from ase.calculators.calculator import Calculator,all_changes
            class Calc(Calculator):
                implemented_properties=['energy','forces','stress']
                def calculate(self,atoms=None,properties=None,system_changes=all_changes):
                    super().calculate(atoms,properties,system_changes)
                    self.results=dict(energy=-.97,forces=np.ones((len(atoms),3))*.2,stress=np.zeros(6))
            self.calc=Calc()
    monkeypatch.setattr(normal.pw,'Model',Model)
    out=tmp_path/'baseline';normal.baseline(Namespace(directory=str(root),out=str(out),model='toyice',device='cpu'))
    row=next(csv.DictReader((out/'baseline.csv').open()))
    assert float(row['GRACE_minus_DFT_meV_atom'])==pytest.approx(10.)
    assert float(row['force_component_RMSE_eV_A'])==pytest.approx(.2)


def test_fitprep_keeps_test_out_of_training(tmp_path):
    def frame(x):
        at=Atoms('H',positions=[[x,0,0]],cell=[5,5,5],pbc=True)
        at.calc=SinglePointCalculator(at,energy=x,forces=np.zeros((1,3)),stress=np.zeros(6))
        return at
    data=tmp_path/'new';data.mkdir()
    for name,x in [('train',1.),('valid',2.),('test',3.)]:write(data/f'{name}.extxyz',frame(x))
    original_train=tmp_path/'train.extxyz';original_valid=tmp_path/'valid.extxyz'
    write(original_train,[frame(0.),frame(3.)]);write(original_valid,frame(4.))
    out=tmp_path/'fit'
    normal.fitprep(Namespace(dataset=str(data),base_train=str(original_train),base_valid=str(original_valid),out=str(out)))
    info=json.loads((out/'fit_manifest.json').read_text())
    assert info['train']==2 and info['test']==1 and info['removed_training_overlaps']==1


def test_dataset_keeps_pair_members_together(tmp_path,monkeypatch):
    import ase.io
    root=tmp_path/'queue';root.mkdir();points=[];pairs=[]
    for i in range(6):
        ids=[f'p{i}_without',f'p{i}_with']
        points.extend(dict(id=identity,path=identity) for identity in ids)
        pairs.append(dict(points=ids,source_frames='cache.npz',frame=i))
    (root/'manifest.json').write_text(json.dumps(dict(mode='paired_pwidom_energy_check',points=points,pairs=pairs)))
    def atoms_for(path,**kwargs):
        path=Path(path);identity=next(p for p in path.parts if p.startswith('p') and ('_with' in p or '_without' in p))
        index=int(identity.split('_')[0][1:]);n=1 if identity.endswith('_without') else 2
        at=Atoms('H'*n,positions=[[index*.2+j,0,0] for j in range(n)],cell=[8,8,8],pbc=True)
        at.calc=SinglePointCalculator(at,energy=float(index),forces=np.zeros((n,3)),stress=np.zeros(6))
        return at
    monkeypatch.setattr(normal,'read',atoms_for)
    monkeypatch.setattr(check,'done',lambda directory:dict(attempt='attempt',TOTEN_eV=float(Path(directory).name.split('_')[0][1:])))
    out=tmp_path/'dataset';normal.dataset(Namespace(directories=[str(root)],out=str(out),seed=17))
    occupied={}
    for split in ['train','valid','test']:
        for at in ase.io.read(out/f'{split}.extxyz',index=':'):
            source=at.info['source']
            assert source not in occupied or occupied[source]==split
            occupied[source]=split
            assert at.get_potential_energy()==at.calc.results['free_energy']
    assert len(occupied)==6


def test_geometry_identity_ignores_species_order():
    at=Atoms('OH2',positions=[[0,0,0],[1,0,0],[0,1,0]],cell=[5,5,5],pbc=True)
    assert normal.canonical_geometry_id(at)==normal.canonical_geometry_id(at[[1,2,0]])
