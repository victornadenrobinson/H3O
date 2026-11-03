"""Synthetic cumulative-prefix tests; no GRACE, trajectories or MD."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
import plot_pwidom_time_hulls as plot
from tdi.pwidom_scan import mu_P


def bundle():
    records=[]
    for phase,n,mu in [('fcc',0.,50.),('fcc',.25,150.),('H',0.,100.)]:
        nh=round((2+n)*108) if phase!='H' else 250
        r=dict(phase=phase,P_target=300.,T=3000.,model_fp='test',N_H=nh,
            natoms=nh+108 if phase!='H' else nh,V=300.,B_GPa=1000.,
            mu_fixedV_meV=mu,mu_ex_bar_err_meV=1.,P_host_GPa=300.,P_host_err_GPa=0.,melted=False)
        final=mu_P(r,'robust')[0]
        prefixes=[dict(production_ps=t,mu_bar_meV=final+(1-t/.3)*(10 if phase=='H' else 5)) for t in [.1,.2,.3]]
        records.append(dict(source=phase+str(n),result=r,prefixes=prefixes,
                            ensemble_frame_ids_match=True,frame_interval_fs=20.))
    return dict(schema='pwidom-prefixes-v1',records=records)


def test_time_hulls_use_same_time_reservoir_and_correct_atomic_x(tmp_path):
    b=bundle()
    rows,hulls=plot.analyse(b,tmp_path/'out',.1)
    assert len(rows)==6 and len(hulls)==6
    for row in rows:
        assert row['dg_meV']==pytest.approx(row['mu_H_meV']-row['G_H_meV'])
    for t in [.1,.2,.3]:
        group=[r for r in rows if np.isclose(r['time_ps'],t)]
        h=[r for r in hulls if np.isclose(r['time_ps'],t)]
        assert h[0]['formation_meV_atom']==0
        assert h[1]['formation_meV_atom']==pytest.approx(.25*(group[0]['dg_meV']+group[1]['dg_meV'])/2/3.25)
        assert h[1]['x_H']==pytest.approx(2.25/3.25)
    assert (tmp_path/'out/hulls_coloured_by_time.pdf').stat().st_size>100
    assert (tmp_path/'out/time_series.csv').exists()
    with pytest.raises(FileExistsError):plot.analyse(b,tmp_path/'out',.1)


def test_fixed_reference_and_missing_exact_snapshot(tmp_path):
    b=bundle()
    rows,_=plot.analyse(b,tmp_path/'fixed',.1,'fixed')
    assert len(set(row['G_H_meV'] for row in rows))==1
    b['records'][0]['prefixes'].pop(1)
    rows,_=plot.analyse(b,tmp_path/'gap',.1)
    assert len(rows)==4
    assert (tmp_path/'gap/skipped_times.csv').exists()
    assert not any(np.isclose(r['time_ps'],.2) for r in rows)


def test_incompatible_or_missing_baseline_refused():
    b=bundle();b['records'][1]['result']['T']=4000
    with pytest.raises(ValueError,match='identical'):plot.prepare(b)
    b=bundle();b['records'][0]['result']['N_H']+=1
    with pytest.raises(ValueError,match='including H2O'):plot.prepare(b)
    b=bundle();b['records'][0]['prefixes'][-1]['mu_bar_meV']+=10
    with pytest.raises(ValueError,match='does not reproduce'):plot.prepare(b)


def test_shared_reference_sampling_se_is_propagated(tmp_path):
    b=bundle()
    for entry in b['records']:
        for row in entry['prefixes']:
            row['mu_bar_se_meV']=4. if entry['result']['phase']=='H' else 3.
    rows,_=plot.analyse(b,tmp_path/'errors',.1)
    for row in rows:
        assert row['mu_H_se_meV']==3
        assert row['G_H_se_meV']==4
        assert row['dg_se_meV']==5
