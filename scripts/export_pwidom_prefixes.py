#!/usr/bin/env python3
"""Small numerical time-series export using Perlmutter's published convergence code.

Run from the H3O root. No trajectories are copied, no MD/model calls are made.
Requires mlp_pipeline/tdi/pwidom_convergence.py on gpt/perlmutter-pwidom.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline/tdi'))


def export_runs(directories):
    from pwidom_convergence import prefixes
    from pwidom_scan import mu_P
    entries=[]
    for root in directories:
        root=Path(root)
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest.get('status')!='complete':
            raise ValueError(f'{root}: choose a completed run')
        for job in manifest['jobs']:
            source=Path(job['result'])
            if not source.is_file():source=root/source.name
            result=json.loads(source.read_text())
            work=Path(result.get('work_file') or source.with_name(source.stem+'_work.npz'))
            if not work.is_file():work=root/work.name
            if not work.is_file():
                raise ValueError(f'{source}: saved work arrays missing; final mu alone cannot recover time history')
            result=dict(result,work_file=str(work))
            with np.load(work,allow_pickle=False) as z:
                f,r=z['forward_work_eV'],z['reverse_work_eV']
                ih,ip=z['host_frame_ids'],z['full_frame_ids']
                if (f.ndim!=2 or r.ndim!=2 or len(f)!=len(ih) or len(r)!=len(ip)
                        or not np.array_equal(ih,ip) or len(f)<5
                        or np.any(np.diff(ih)<=0) or np.any(ih<0)):
                    raise ValueError(f'{work}: need at least five grouped frames with matching increasing host/full IDs')
                if not np.isfinite(f).all() or not np.isfinite(r).all():raise ValueError(f'{work}: nonfinite work')
                if str(z['model_fp'])!=result.get('model_fp'):raise ValueError('Work/result fingerprint mismatch')
                command=job['command']
                if 'frame_interval_fs'in z:
                    interval=float(z['frame_interval_fs']);basis='saved work metadata'
                else:
                    dt=float(command[command.index('--dt')+1]) if '--dt'in command else .5
                    every=float(command[command.index('--every-fs')+1]) if '--every-fs'in command else 20.
                    actual=max(1,round(every/dt))*dt
                    if not np.isclose(actual,every):raise ValueError('Legacy cadence rounding needs explicit timing metadata')
                    interval=every;basis='saved command, with published defaults dt=0.5 fs / every=20 fs when omitted'
                if not np.isfinite(interval) or interval<=0:raise ValueError('Invalid frame cadence')
                # Existing prefixes opens this path internally; never mutate the cache.
                framepath=Path(str(z['source_frames']))
                if not framepath.is_file():raise ValueError(f'Frame cache not available at recorded path: {framepath}')
                with np.load(framepath,allow_pickle=False) as frames:
                    if len(frames['P_host'])!=len(frames['P_full']):raise ValueError('Unequal ensemble lengths need separate timing reconstruction')
                    if ih[-1]>=len(frames['P_host']):raise ValueError('Frame ID outside pressure cache')
            rows=prefixes(result,command,reference=None,points=len(f))
            if not np.isclose(rows[-1]['mu_bar_meV'],mu_P(result,'robust')[0],atol=1e-5,rtol=1e-8):
                raise ValueError(f'{source}: full prefix does not reconstruct stored robust mu')
            clean=[{k:v for k,v in row.items() if k!='dg_meV'} for row in rows]
            entries.append(dict(source=str(source),result=result,prefixes=clean,
                                ensemble_frame_ids_match=True,frame_interval_fs=interval,timing_basis=basis))
    return dict(schema='pwidom-prefixes-v1',records=entries,
                note='Cumulative production sampling per independent ensemble; equilibration excluded; no pooling')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh export path')
    data=export_runs(a.runs)
    with a.out.open('x') as f:json.dump(data,f,allow_nan=False)
    print(f'Exported {len(data["records"])} series to {a.out}; no MD/model calls')


if __name__=='__main__':main()
