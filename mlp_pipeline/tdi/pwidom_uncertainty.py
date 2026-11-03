"""Stratified delete-one-block sampling SE for frame-grouped PWidom BAR.

Each NVT ensemble is resampled independently. A deleted block removes all trials
from its MD frames; deleting host frames also updates their mean pressure.
These are sampling errors conditional on the observed trajectories, not a
correction for missing tails, nonstationarity, proxy bias or model error.
"""
import numpy as np
from tdi.pwidom import bar,KB
from tdi.pwidom_scan import mu_P


def prefix_sampling_se(result,forward,reverse,host_ids,full_ids,pressures,limit,
                       block_frames=5,min_blocks=3):
    """Return a SE only for complete equal-length contiguous block partitions."""
    if block_frames<1 or min_blocks<3:raise ValueError('Need positive block size and at least three blocks')
    count=limit//block_frames
    base=dict(mu_bar_se_meV=None,uncertainty_blocks=int(count),uncertainty_status='too_few_blocks')
    if count<min_blocks:return base
    if limit%block_frames:
        base['uncertainty_status']='not_a_complete_block_prefix';return base
    ih=np.asarray(host_ids);ip=np.asarray(full_ids)
    f=np.asarray(forward)[ih<limit];r=np.asarray(reverse)[ip<limit]
    ih=ih[ih<limit];ip=ip[ip<limit];p=np.asarray(pressures)[:limit]
    if len(p)!=limit or not len(f) or not len(r):raise ValueError('Prefix outside saved pressure/work arrays')
    kt=KB*result['T']
    ideal=result['ideal_meV'] if 'ideal_meV' in result else result['mu_fixedV_meV']-result['mu_ex_bar_meV']
    def estimate(ff,rr,pressure):
        if not len(ff) or not len(rr):raise ValueError('Deleting a block leaves an empty ensemble')
        mb=bar(ff.ravel(),rr.ravel(),kt)*1000
        state=dict(result,mu_fixedV_meV=mb+ideal,P_host_GPa=float(pressure))
        return mu_P(state,'robust')[0]
    host_values=[];full_values=[]
    for block in range(count):
        lo=block*block_frames;hi=lo+block_frames
        keep_h=(ih<lo)|(ih>=hi);keep_f=(ip<lo)|(ip>=hi)
        keep_p=np.ones(limit,dtype=bool);keep_p[lo:hi]=False
        host_values.append(estimate(f[keep_h],r,p[keep_p].mean()))
        full_values.append(estimate(f,r[keep_f],p.mean()))
    def variance(values):
        values=np.asarray(values)
        return (count-1)/count*float(np.sum((values-values.mean())**2))
    base.update(mu_bar_se_meV=float(np.sqrt(variance(host_values)+variance(full_values))),
                uncertainty_status='block_jackknife')
    return base
