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
        se=np.array([row.get('mu_bar_se_meV') if row.get('mu_bar_se_meV') is not None else np.nan for row in rows],float)
        if np.any(np.isfinite(se)&(se<0)):raise ValueError('Negative sampling SE')
        entry = dict(entry,times=times,mu=mu,mu_se=se)
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
        mu_se=np.array([at_time(h,t,'mu_se') for h in hosts])
        gh_se=at_time(reservoir,t,'mu_se') if reference_mode=='cumulative' else float(reservoir['mu_se'][-1])
        dg_se=np.sqrt(mu_se**2+gh_se**2)
        dg=mu-gh
        optional=lambda v:float(v) if np.isfinite(v) else None
        values,_,_=integrate(n,dg)
        x=np.r_[(2+n)/(3+n),1.];g=np.r_[values/(3+n),0.]
        hull=lower_hull(x,g)
        for i,h in enumerate(hosts):
            time_rows.append(dict(time_ps=t,n=n[i],extra_H_per_108_O=n[i]*108,
                mu_H_meV=mu[i],G_H_meV=gh,dg_meV=dg[i],
                mu_H_se_meV=optional(mu_se[i]),G_H_se_meV=optional(gh_se),dg_se_meV=optional(dg_se[i]),
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
    # Keep central estimates unchanged; uncertainty metadata is optional.
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    style={'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
           'axes.spines.top':False,'axes.spines.right':False,
           'axes.edgecolor':'#b8bdc5','grid.color':'#dfe3e8',
           'grid.linewidth':.6,'figure.facecolor':'white','savefig.facecolor':'white',
           'pdf.fonttype':42}
    def save(fig,name):
        for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def band(ax,t,y,se,color):
        t,y,se=np.asarray(t),np.asarray(y),np.asarray(se,dtype=float)
        good=np.isfinite(se)
        if np.count_nonzero(good)>=2:
            ax.fill_between(t[good],y[good]-se[good],y[good]+se[good],color=color,alpha=.11,lw=0,zorder=0)
    def clean(ax):
        ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True)
        ax.tick_params(length=3,color='#aab0b8')
    with plt.rc_context(style):
        palette=plt.get_cmap('tab10')
        fig=plt.figure(figsize=(10.5,7.2))
        grid=fig.add_gridspec(2,2,height_ratios=[1.3,.8],hspace=.44,wspace=.3)
        axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
        handles=[];labels=[]
        for i,h in enumerate(hosts):
            color=palette(i%10)
            line,=axes[0].plot(h['times'],h['mu'],color=color,lw=1.8)
            handles.append(line);labels.append(f"{h['extra_H_108']:g}")
            band(axes[0],h['times'],h['mu'],h['mu_se'],color)
            rows=[r for r in time_rows if np.isclose(r['n'],h['n'])]
            axes[1].plot([r['time_ps'] for r in rows],[r['dg_meV'] for r in rows],color=color,lw=1.8)
            band(axes[1],[r['time_ps'] for r in rows],[r['dg_meV'] for r in rows],
                 [r['dg_se_meV'] if r['dg_se_meV'] is not None else np.nan for r in rows],color)
        axes[2].plot(reservoir['times'],reservoir['G_H'],color='#34445c',lw=1.8)
        band(axes[2],reservoir['times'],reservoir['G_H'],reservoir['mu_se'],'#34445c')
        for ax,title in zip(axes,['Hydrogen chemical potential','Transfer free energy','Pure-hydrogen reference']):
            ax.set_title(title,loc='left',pad=10,fontweight='medium');clean(ax)
            ax.set_xlabel('Production sampled per ensemble (ps)')
            ax.xaxis.set_major_locator(MaxNLocator(6))
        axes[0].set_ylabel(r'$\mu_\mathrm{H}$ (meV/H)')
        axes[1].set_ylabel(r'$\Delta\mu_\mathrm{H}$ (meV/H)');axes[1].axhline(0,color='#888f99',lw=.8,zorder=0)
        axes[2].set_ylabel(r'$G_\mathrm{H}$ (meV/H)')
        fig.legend(handles,labels,title='Extra H / 108 O',loc='upper left',
                   bbox_to_anchor=(.86,.88),frameon=False,labelspacing=.8,handlelength=1.8)
        fig.subplots_adjust(left=.09,right=.85,bottom=.12,top=.86)
        fig.suptitle(f'{p:g} GPa  ·  {temp:g} K',x=.09,y=.975,ha='left',fontsize=17)
        fig.text(.09,.925,f'Cumulative estimates · robust pressure correction · {reference_mode} reservoir',fontsize=10,color='#616974')
        fig.text(.09,.025,'Shading: ±1 contiguous-block jackknife SE where available. Correlated prefixes; missing-state/model bias excluded.',fontsize=8.5,color='#616974')
        save(fig,'mu_G_dg_vs_time')

        if all(all('mu_ex_bar_meV' in row and 'pressure_correction_meV' in row for row in e['prefixes']) for e in all_entries):
            drift,da=plt.subplots(1,2,figsize=(10.5,4.2),sharex=True)
            for i,h in enumerate(hosts):
                for ax,field in zip(da,['mu_ex_bar_meV','pressure_correction_meV']):
                    values=np.array([row[field] for row in h['prefixes']])
                    ax.plot(h['times'],values-values[0],color=palette(i%10),lw=1.6)
            for ax,field,title in zip(da,['mu_ex_bar_meV','pressure_correction_meV'],
                    ['Excess BAR contribution','Pressure-correction contribution']):
                values=np.array([row[field] for row in reservoir['prefixes']])
                ax.plot(reservoir['times'],values-values[0],'--',color='#34445c',lw=1.8)
                ax.set(title=title,xlabel='Production sampled per ensemble (ps)',ylabel='Change since first prefix (meV/H)')
                ax.axhline(0,color='#969da6',lw=.7);clean(ax)
            drift.legend(handles+[plt.Line2D([],[],ls='--',color='#34445c')],labels+['Pure H'],
                         title='Extra H / 108 O',loc='upper left',bbox_to_anchor=(.86,.95),frameon=False,fontsize=8)
            drift.subplots_adjust(left=.09,right=.85,bottom=.17,top=.77,wspace=.3)
            drift.suptitle(f'{p:g} GPa · {temp:g} K — what drives the drift?',x=.09,ha='left',fontsize=15)
            drift.text(.09,.02,'Ideal term is constant. Both contributions can drift; this decomposition does not establish equilibrium.',fontsize=8.5,color='#616974')
            save(drift,'drift_components_vs_time')

        plotted=sorted(set(r['time_ps'] for r in hull_rows))
        norm=Normalize(min(plotted),max(plotted) if len(plotted)>1 else min(plotted)+step_ps)
        cmap=LinearSegmentedColormap.from_list('sampling_time',plt.get_cmap('viridis')(np.linspace(.06,.88,256)))
        fig,axes=plt.subplots(2,1,figsize=(8.2,7.8),sharex=True,
                              gridspec_kw={'height_ratios':[1.35,1]},layout='constrained')
        full,fullax=plt.subplots(figsize=(7.3,4.7),layout='constrained')
        for j,t in enumerate(plotted):
            rows=[r for r in hull_rows if r['time_ps']==t]
            x=np.r_[[r['x_H'] for r in rows],1.]
            g=np.r_[[r['formation_meV_atom'] for r in rows],0.]
            ix=lower_hull(x,g);color=cmap(norm(t))
            last=j==len(plotted)-1;lw=2.1 if last else 1.15;alpha=1. if last else .65
            axes[0].plot(x[:-1],g[:-1],color=color,lw=lw,alpha=alpha)
            if last:axes[0].plot(x[:-1],g[:-1],'o',ms=4.2,mfc='white',mec=color,mew=1.3,zorder=5)
            axes[1].plot(x[ix],g[ix],color=color,lw=lw,alpha=alpha)
            fullax.plot(x[ix],g[ix],color=color,lw=lw,alpha=alpha)
        axes[0].set_title('Formation free energy',loc='left',pad=9)
        axes[1].set_title('Lower convex envelope',loc='left',pad=9)
        span=x[-2]-x[0];axes[1].set_xlim(x[0]-.035*span,x[-2]+.035*span)
        for ax in axes:
            clean(ax);ax.axhline(0,color='#969da6',lw=.7,zorder=0)
            ax.set_ylabel('ΔG (meV/atom)');ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axes[1].set_xlabel('Atomic H fraction')
        def x_to_extra(v):return 108*(3*np.asarray(v)-2)/(1-np.asarray(v))
        def extra_to_x(v):return (2+np.asarray(v)/108)/(3+np.asarray(v)/108)
        top=axes[0].secondary_xaxis('top',functions=(x_to_extra,extra_to_x))
        top.set_xlabel('Extra H per 108 O',labelpad=7)
        top.set_xticks([0,9,16,27] if hosts[-1]['extra_H_108']>=27 else [h['extra_H_108'] for h in hosts])
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,pad=.035,shrink=.86,
                     label='Production sampled per ensemble (ps)',ticks=np.linspace(min(plotted),max(plotted),min(6,len(plotted))))
        fig.suptitle(f'{p:g} GPa  ·  {temp:g} K\nProvisional formation and hull evolution',fontsize=15)
        save(fig,'hulls_coloured_by_time')
        clean(fullax);fullax.axhline(0,color='#969da6',lw=.7,zorder=0)
        fullax.set(xlabel='Atomic H fraction',ylabel='ΔG (meV/atom)',title=f'{p:g} GPa · {temp:g} K — full composition range')
        fullax.annotate('Pure H',(1,0),xytext=(-8,9),textcoords='offset points',ha='right',fontsize=9)
        full.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=fullax,pad=.035,label='Production sampled per ensemble (ps)')
        save(full,'hull_full_composition')
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
