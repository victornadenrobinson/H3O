#!/usr/bin/env python3
"""Plot cumulative PWidom chemical potentials and colour formation hulls by time.

Consumes the small prefix export made by the existing Perlmutter convergence
module. No trajectories, model calls, MD, or independent-trial uncertainty bands.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from collect_pwidom_results import lower_hull
from tdi import pwidom_scan as scan
from tdi.pwidom_formation import integrate


def prepare(bundle):
    if bundle.get('schema') != 'pwidom-prefixes-v1':
        raise ValueError('Unsupported prefix export schema')
    records = bundle['records']
    if len(records) < 3:
        raise ValueError('Need two host compositions and a pure-H reservoir')
    state = None
    hosts, reservoirs = [], []
    for entry in records:
        result, rows = entry['result'], entry['prefixes']
        key = (result['P_target'], result['T'], result.get('model_fp'))
        if state is None: state = key
        if key != state or not key[2]:
            raise ValueError('All prefixes must have identical P/T/verified Hamiltonian')
        if not entry.get('ensemble_frame_ids_match'):
            raise ValueError('Export must verify matching host/full sampled frame IDs')
        spacing = float(entry['frame_interval_fs'])
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('Need confirmed positive sampling interval')
        times = np.array([r['production_ps'] for r in rows],float)
        mu = np.array([r['mu_bar_meV'] for r in rows],float)
        if (not len(times) or not np.isfinite(times).all() or not np.isfinite(mu).all()
                or np.any(times <= 0) or np.any(np.diff(times) <= 0)):
            raise ValueError('Need finite, strictly increasing prefix times and finite mu')
        if not np.isclose(mu[-1],scan.mu_P(result,'robust')[0],atol=1e-5,rtol=1e-8):
            raise ValueError(f"{entry['source']}: final prefix does not reproduce robust mu")
        entry = dict(entry,times=times,mu=mu)
        if result['phase']=='H':
            n = result['N_H']; kt = scan.KB*result['T']*1000
            fs = kt*(np.log(n+1)-__import__('math').lgamma(n+1)/n-1)
            entry['G_H'] = mu-fs
            reservoirs.append(entry)
        else:
            if result['phase']!='fcc' or result.get('melted') or scan.o_growing(result):
                raise ValueError('Hull input must be non-melted FCC; remove other phases explicitly')
            no = result['natoms']-result['N_H']
            if no <= 0: raise ValueError('Host has no oxygen')
            entry['n'] = result['N_H']/no-2
            entry['extra_H_108'] = entry['n']*108
            hosts.append(entry)
    if len(reservoirs)!=1:
        raise ValueError('Choose exactly one pure-H result; never pool overlapping references')
    hosts.sort(key=lambda e:e['n'])
    n = np.array([h['n'] for h in hosts])
    if len(n)<2 or not np.isclose(n[0],0) or np.any(np.diff(n)<=0):
        raise ValueError('Need distinct sorted compositions, including H2O; choose one baseline/extension')
    return state,hosts,reservoirs[0]


def at_time(entry,t,field='mu'):
    # Exact sampling prefixes only: never interpolate work-derived estimators or
    # use future samples to manufacture a point at an earlier requested time.
    indices=np.flatnonzero(np.isclose(entry['times'],t,atol=1e-9,rtol=0))
    if len(indices)!=1: raise ValueError(f'No exact {t:g} ps prefix for {entry["source"]}')
    return float(entry[field][indices[0]])


def analyse(bundle,out,step_ps=.1,reference_mode='cumulative'):
    if not np.isfinite(step_ps) or step_ps<=0: raise ValueError('step-ps must be finite and positive')
    state,hosts,reservoir=prepare(bundle)
    all_entries=hosts+[reservoir]
    limit=min(e['times'][-1] for e in all_entries)
    earliest=max(e['times'][0] for e in all_entries)
    candidates=np.arange(1,int(np.floor((limit+1e-10)/step_ps))+1)*step_ps
    times=[float(t) for t in candidates if t+1e-9>=earliest]
    if not times:raise ValueError('No common prefix at requested interval')
    n=np.array([h['n'] for h in hosts]);n[0]=0
    time_rows,hull_rows,skipped=[],[],[]
    for t in times:
        try:
            mu=np.array([at_time(h,t) for h in hosts])
            gh=at_time(reservoir,t,'G_H') if reference_mode=='cumulative' else float(reservoir['G_H'][-1])
        except ValueError as exc:
            skipped.append(dict(time_ps=t,reason=str(exc)));continue
        dg=mu-gh
        values,_,_=integrate(n,dg)
        x=np.r_[(2+n)/(3+n),1.];g=np.r_[values/(3+n),0.]
        hull=lower_hull(x,g)
        for i,h in enumerate(hosts):
            time_rows.append(dict(time_ps=t,n=n[i],extra_H_per_108_O=n[i]*108,
                mu_H_meV=mu[i],G_H_meV=gh,dg_meV=dg[i],
                dg_fixed_final_reference_meV=mu[i]-float(reservoir['G_H'][-1]),source=h['source']))
            hull_rows.append(dict(time_ps=t,n=n[i],x_H=x[i],formation_meV_atom=g[i],
                on_discrete_hull=i in hull,above_hull_meV_atom=max(0.,g[i]-np.interp(x[i],x[hull],g[hull]))))
    if not time_rows:raise ValueError('No exact shared prefixes; inspect intervals and frame IDs')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name,rows in [('time_series.csv',time_rows),('time_hulls.csv',hull_rows),('skipped_times.csv',skipped)]:
        if rows:
            with (out/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'prefix_input.json').write_text(json.dumps(bundle,indent=1))
    p,temp,fp=state
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for h in hosts:
        label=f"+{h['extra_H_108']:g} H / 108 O"
        axes[0].plot(h['times'],h['mu'],label=label)
        rows=[r for r in time_rows if np.isclose(r['n'],h['n'])]
        axes[2].plot([r['time_ps'] for r in rows],[r['dg_meV'] for r in rows],label=label)
    axes[1].plot(reservoir['times'],reservoir['G_H'],color='black')
    axes[0].set_ylabel('Cumulative μH (meV/H)');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Cumulative pure-H G reference (meV/H)')
    axes[2].set_ylabel('Cumulative transfer dG (meV/H)');axes[2].axhline(0,color='.5',lw=.8)
    for ax in axes:ax.set_xlabel('Cumulative production sampled per ensemble (ps)');ax.grid(alpha=.2)
    fig.suptitle(f'{p:g} GPa / {temp:g} K — robust pressure correction; {reference_mode} reservoir for dG')
    fig.text(.5,.015,'Overlapping cumulative prefixes; no independent-sample uncertainty inferred. Time excludes equilibration and extension re-equilibration gaps.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ('pdf','png'):fig.savefig(out/f'mu_G_dg_vs_time.{ext}',dpi=180)
    plt.close(fig)
    plotted=sorted(set(r['time_ps'] for r in hull_rows))
    norm=Normalize(min(plotted),max(plotted) if len(plotted)>1 else min(plotted)+step_ps)
    cmap=plt.get_cmap('viridis')
    fig,axes=plt.subplots(1,2,figsize=(11.5,5),layout='constrained')
    for t in plotted:
        rows=[r for r in hull_rows if r['time_ps']==t]
        x=np.r_[[r['x_H'] for r in rows],1.]
        g=np.r_[[r['formation_meV_atom'] for r in rows],0.]
        ix=lower_hull(x,g);color=cmap(norm(t))
        axes[0].plot(x[:-1],g[:-1],'o-',color=color,ms=3,alpha=.85)
        axes[0].plot(x[ix],g[ix],'--',color=color,alpha=.8)
        axes[1].plot(x[ix],g[ix],'-',color=color,alpha=.9)
    axes[0].set_xlim(x[0]-.001,x[-2]+.001)
    axes[0].set_title('Sampled formation nodes + dashed hull (zoom)')
    axes[1].set_title('Discrete lower hull including pure H')
    for ax in axes:
        ax.set_xlabel('Atomic H fraction');ax.set_ylabel('Provisional formation ΔG (meV/atom)')
        ax.axhline(0,color='.6',lw=.7);ax.grid(alpha=.2)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='Cumulative production sampled per ensemble (ps)')
    fig.suptitle(f'{p:g} GPa / {temp:g} K — hull evolution every {step_ps:g} ps\nIntegrated μH − G_H; {reference_mode} reservoir; convergence not established')
    for ext in ('pdf','png'):fig.savefig(out/f'hulls_coloured_by_time.{ext}',dpi=180)
    plt.close(fig)
    (out/'README.txt').write_text('No absolute crystal G(t) is inferred. Formation ΔG is reconstructed from chemical-potential differences.\nSame compositions, model and reference are used at every snapshot; no pooling of parent/extension datasets.\nCumulative means prefix [0,t], not an independent 0.1-ps block or MD wall time.\nHost and N+1 are independent ensembles; t is sampling effort per ensemble.\nLegacy frame timing comes from the confirmed exporter interval; extension equilibration gaps are excluded.\nPrefix curves and hulls are correlated. No independent-trial confidence intervals or equilibrium stability claims.\nFrames with no exact requested time are skipped as complete snapshots, never filled with later/final values.\nRobust pressure correction uses cumulative host pressure and existing partial-volume assumptions.\n')
    print(f'{len(hosts)} host compositions; {len(plotted)} common snapshots through {max(plotted):g} ps; output {out}')
    return time_rows,hull_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix_bundle',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--step-ps',type=float,default=.1)
    p.add_argument('--reference-mode',choices=['cumulative','fixed'],default='cumulative')
    a=p.parse_args()
    analyse(json.loads(a.prefix_bundle.read_text()),a.out,a.step_ps,a.reference_mode)


if __name__=='__main__':main()
