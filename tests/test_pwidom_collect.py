"""Synthetic completed-run collection and provisional hull regression tests."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
spec=importlib.util.spec_from_file_location('collector',ROOT/'mlp_pipeline/collect_pwidom_results.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


def result(phase='fcc',n=0,mu=100):
    no=108; nh=round((2+n)*no) if phase!='H' else 250
    return dict(phase=phase,T=3000,P_target=500,N_H=nh,natoms=nh+no if phase!='H' else nh,
        model='test',model_fp='abc123',mu_ex_bar_meV=50,mu_ex_bar_err_meV=2,
        mu_fixedV_meV=mu,mu_P_meV=mu,mu_P_err_meV=3,
        mu_ex_widom_meV=50,mu_ex_delete_meV=50,kish_frac_ins=.5,pressure_corrected=True,
        n_frames_host=10,n_frames_full=10,melted=False,tag='pw_host')


def payload():
    h=result('H',mu=20)
    return {'/runs/pwidom_a/H.json':h,
            '/runs/pwidom_a/zero.json':result(mu=10),
            '/runs/pwidom_a/rich.json':result(n=.25,mu=30)}


def test_report_references_hull_and_replot(tmp_path):
    source=payload()
    rows,curves=collector.report(source,tmp_path/'first')
    assert len(rows)==3 and len(curves)==2
    row=next(r for r in rows if r['n']==0)
    assert row['dg_meV']==pytest.approx(10-row['reservoir_G_H_meV'])
    assert row['dg_err_meV']==pytest.approx(np.hypot(3,3))
    assert all(r['selected_for_integral'] for r in rows if r['phase']=='fcc')
    bundle=json.loads((tmp_path/'first/results_bundle.json').read_text())
    again,againcurves=collector.report(bundle['records'],tmp_path/'replot')
    assert again==rows and againcurves==curves
    assert len(list((tmp_path/'first').glob('*.pdf')))==2
    with pytest.raises(FileExistsError): collector.report(source,tmp_path/'first')


def test_lower_hull_composition_normalization():
    x=np.array([2/3, .7,.8,1.]); y=np.array([0,-10,5,0])
    assert collector.lower_hull(x,y)==[0,1,3]
    with pytest.raises(ValueError):collector.lower_hull(np.array([0,0]),np.array([0,1]))


def test_extensions_not_pooled_and_flags_explicit(tmp_path):
    source=payload()
    child=result(n=.25,mu=40);child['n_frames_host']=20;child['n_frames_full']=20
    source['/runs/pwidom_extension/rich.json']=child
    source['/runs/pwidom_extension/manifest.json']=dict(status='running',reference=dict(snapshot=source['/runs/pwidom_a/H.json'],result='/runs/pwidom_a/H.json'))
    rows,_=collector.rows_from(source)
    group=list(collector.selected_groups(rows).values())[0]
    assert group[.25]['source']=='/runs/pwidom_extension/rich.json'
    assert group[.25]['launch_status']=='running' # Finished individual result survives sibling failure.
    child['melted']=True
    rows,_=collector.rows_from(source)
    assert all('extension' not in r['source'] for g in collector.selected_groups(rows,True).values() for r in g.values())
    source['/runs/pwidom_a/zero.json']['kish_frac_ins']=.001
    rows,_=collector.rows_from(source)
    assert 0 not in next(iter(collector.selected_groups(rows).values()))
    assert 0 in next(iter(collector.selected_groups(rows,True).values()))


def test_inventory_excludes_archives_and_partial_json(tmp_path):
    run=tmp_path/'pwidom_case';run.mkdir()
    (run/'result.json').write_text(json.dumps(result()))
    archive=run/'_resume_history/old';archive.mkdir(parents=True)
    (archive/'result.json').write_text(json.dumps(result(mu=999)))
    (run/'partial.json').write_text('{')
    source,errors=collector.inventory(tmp_path)
    assert len(source)==1 and len(errors)==1


def test_no_reference_not_silently_guessed(tmp_path):
    source=payload();source['/runs/pwidom_a/H2.json']=result('H',mu=25)
    rows,_=collector.report(source,tmp_path/'ambiguous')
    assert all(r['dg_meV']=='' for r in rows if r['phase']=='fcc')
    assert list((tmp_path/'ambiguous').glob('all_*.pdf'))
    assert not (tmp_path/'ambiguous/formation_hull.csv').exists()
