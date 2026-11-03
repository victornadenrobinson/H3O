"""Force/stress units and atom-order validation, without VASP or GRACE."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
from tdi import vasp_force_check as check
from tdi import pwidom as pw


def test_stress_sign_units_and_species_metrics():
    m=check.metrics(np.array([1,8,1]),np.ones((3,3)),np.zeros((3,3)),np.array([-2,-2,-2,0,0,0]),np.array([-1,-1,-1,0,0,0]))
    assert m['force_component_RMSE_eV_A']==1
    assert m['force_vector_RMSE_eV_A']==pytest.approx(np.sqrt(3))
    assert m['P_DFT_config_GPa']==pytest.approx(2*pw.EV_A3_GPA)
    assert m['P_MLP_config_GPa']==pytest.approx(pw.EV_A3_GPA)
    assert m['P_DFT_minus_MLP_GPa']==pytest.approx(pw.EV_A3_GPA)


def test_matched_geometry_force_comparison_no_vasp_call(tmp_path,monkeypatch):
    root=tmp_path/'run';point=root/'p0';point.mkdir(parents=True)
    out=tmp_path/'analysis';out.mkdir()
    at=Atoms('H2O',positions=[[1,1,1],[2,1,1],[1,2,1]],cell=[5]*3,pbc=True)
    write(point/'structure.extxyz',at)
    target=at.copy();target.calc=SinglePointCalculator(target,energy=0.,forces=np.ones((3,3)),stress=np.full(6,-2.))
    calculator=SinglePointCalculator(at,energy=0.,forces=np.full((3,3),.5),stress=np.full(6,-1.))
    original=check.read
    def read(path,*args,**kwargs):
        return target if Path(path).name=='OUTCAR' else original(path,*args,**kwargs)
    monkeypatch.setattr(check,'read',read)
    monkeypatch.setattr(importlib.import_module('04_benchmark'),'get_calc_spec',lambda *a,**k:calculator)
    pair=dict(points=['p0','p1'],host='H2O',direction='insertion',P=300.,T=3000.,model='toyice',model_fp=pw.model_fingerprint('toyice'))
    manifest=dict(pairs=[pair],points=[dict(id='p0'),dict(id='p1')])
    rows=check.compare(root,manifest,dict(p0=dict(attempt='attempts/a'),p1=None),out)
    assert len(rows)==1 and rows[0]['force_component_RMSE_eV_A']==.5
    assert (out/'force_stress.csv').exists()
    with np.load(out/'force_stress_arrays.npz') as z:
        np.testing.assert_array_equal(z['offsets'],[0,3])
        np.testing.assert_allclose(z['stress_DFT_GPa'],-2*pw.EV_A3_GPA)
