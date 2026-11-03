"""Frame-block sampling errors retain correlated trials and pressure coupling."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
from tdi.pwidom_uncertainty import prefix_sampling_se


def state():
    return dict(T=3000.,P_target=300.,phase='fcc',N_H=216,natoms=324,V=300.,B_GPa=1000.,
                ideal_meV=100.,mu_fixedV_meV=300.,mu_ex_bar_meV=200.,
                mu_ex_bar_err_meV=0.,P_host_GPa=300.,P_host_err_GPa=0.)


def test_constant_work_has_zero_se_and_early_prefix_is_unshaded():
    n=20;ids=np.arange(n);f=np.full((n,8),.2);r=np.full((n,4),-.2);p=np.full(n,300.)
    assert prefix_sampling_se(state(),f,r,ids,ids,p,20)['mu_bar_se_meV']==pytest.approx(0,abs=1e-9)
    assert prefix_sampling_se(state(),f,r,ids,ids,p,10)['mu_bar_se_meV'] is None
    assert prefix_sampling_se(state(),f,r,ids,ids,p,17)['mu_bar_se_meV'] is None


def test_frame_correlations_and_pressure_are_preserved():
    n=20;ids=np.arange(n);f=np.full((n,8),.2);r=np.full((n,4),-.2)
    f=np.repeat((.2+np.repeat([0.,.1,-.1,.05],5))[:,None],8,axis=1)
    r=np.repeat((-.2+.03*np.cos(np.arange(n)))[:,None],4,axis=1)
    p=np.repeat([300.,320.,280.,310.],5)
    se=prefix_sampling_se(state(),f,r,ids,ids,p,n)['mu_bar_se_meV']
    assert se>0
    # Duplicating perfectly correlated trials must not produce an artificial
    # sqrt(2) precision gain. Blocks remove the entire frame and its pressure.
    repeated=prefix_sampling_se(state(),np.repeat(f,2,axis=1),np.repeat(r,2,axis=1),ids,ids,p,n)['mu_bar_se_meV']
    assert repeated==pytest.approx(se,rel=1e-10)


def test_work_only_sampling_variance_is_finite():
    rng=np.random.default_rng(5);n=20;ids=np.arange(n)
    f=rng.normal(.2,.1,(n,8));r=rng.normal(-.2,.1,(n,4))
    answer=prefix_sampling_se(state(),f,r,ids,ids,np.full(n,300.),n)
    assert answer['uncertainty_status']=='block_jackknife'
    assert answer['uncertainty_blocks']==4
    assert answer['mu_bar_se_meV']>0
