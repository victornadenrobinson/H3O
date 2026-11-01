"""CPU Slurm plan/worker for the SAME PWidom MD and BAR estimator with VASP."""
from argparse import Namespace
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time


def dft_plan(a):
    from tdi.perlmutter_vasp_check import save
    from tdi.vasp_backend import vasp_config
    from tdi import pwidom
    pwidom.proxy_params(a.ice_proxy);pwidom.proxy_params(a.h_proxy)
    state=json.loads(Path(a.state_json).read_text());out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use fresh plan directory; existing plans are resumed with dft-worker')
    if a.replicates<1 or a.replicates>2:raise ValueError('Use one or two independent replicas of four hosts')
    if a.ps<=0 or a.equil_ps<0 or a.every_fs<=0 or a.sel<1 or a.ndel<1:raise ValueError('Invalid simulation budget')
    if int(a.ps*1000/a.every_fs)<5:raise ValueError('Need at least five sampled frames for existing estimator')
    for key in ('ice','h225','H'):
        if key not in state or not state[key].get('source') or state[key]['vpa']<=0 or state[key]['B_GPa']<=0:raise ValueError('Need complete reviewed EOS state JSON')
    if state['ice']['nrep']!=state['h225']['nrep']:raise ValueError('Ice host sizes differ')
    out.mkdir(parents=True);workers=[]
    for replica in range(a.replicates):
        for host,hper,key in [('H2O',2.,'ice'),('H2p083O',2+9/108,'mid'),('H2p25O',2.25,'h225'),('pureH',None,'H')]:
            directory=out/f'{host}_rep{replica}';directory.mkdir();config=directory/'vasp.json'
            vasp_config(Namespace(template=a.template,T=state['T'],ranks=32,exe=a.exe,work=str(directory/'vasp_work'),out=str(config),timeout_seconds=1800))
            ice,rich=state['ice'],state['h225']
            if key=='mid':
                # Same volume interpolation as the existing four-host launcher.
                vpa=(3*ice['vpa']+(hper-2)/.25*(3.25*rich['vpa']-3*ice['vpa']))/(hper+1)
                bulk=(ice['B_GPa']+rich['B_GPa'])/2
            else:vpa=state[key]['vpa'];bulk=state[key]['B_GPa']
            args=['run','--model','vasp:'+str(config),'--device','cpu','--batch','1',
                '--phase','H' if host=='pureH' else 'fcc','--P',str(state['P']),'--T',str(state['T']),
                '--vpa',str(vpa),'--bulk-gpa',str(bulk),'--ps',str(a.ps),'--equil-ps',str(a.equil_ps),
                '--dt','.5','--every-fs',str(a.every_fs),'--sel',str(a.sel),'--ndel',str(a.ndel),'--K','2000',
                '--proxy',a.h_proxy if host=='pureH' else a.ice_proxy,'--seed',str(3+100*replica),
                '--checkpoint-seconds','300','--label','vasp','--out',str(directory),'--tag','pw']
            if host=='pureH':args+=['--natoms',str(state['H']['natoms'])]
            else:args+=['--hper',str(hper),'--nrep',str(ice['nrep'])]
            workers.append(dict(host=host,replica=replica,directory=str(directory),args=args))
    save(out/'dft_plan.json',dict(state=state,workers=workers,ps=a.ps,equil_ps=a.equil_ps,
        note='Same ASE Langevin MD, forward/reverse proposals and BAR as GRACE PWidom. VASP supplies all target energies/forces. Check phase, pressure correction and sampling convergence.'))
    print(f'Prepared {len(workers)} isolated 32-rank workers. No MD/VASP executed. Plan: {out}')


def dft_worker(a):
    from tdi import pwidom
    from tdi.perlmutter_vasp_check import save
    root=Path(a.directory).resolve();plan=json.loads((root/'dft_plan.json').read_text())
    worker=plan['workers'][a.index];directory=Path(worker['directory']);args=worker['args'].copy()
    if a.smoke:
        directory=directory/'smoke';directory.mkdir(exist_ok=True)
        config=json.loads(Path(args[args.index('--model')+1].split(':',1)[1]).read_text())
        config['work']=str(directory/'vasp_work');configpath=directory/'vasp.json'
        if not configpath.exists():save(configpath,config)
        for key,value in [('--model','vasp:'+str(configpath)),('--out',str(directory)),('--ps','.01'),('--equil-ps','.002'),('--every-fs','2'),('--sel','1'),('--ndel','1')]:args[args.index(key)+1]=value
    with (directory/'.worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('This worker is already running')
        import hashlib
        args_hash=hashlib.sha256(json.dumps(args).encode()).hexdigest()
        if (directory/'DONE.json').exists():
            done=json.loads((directory/'DONE.json').read_text())
            if done['args_sha256']!=args_hash:raise ValueError('Worker settings changed after completion')
            complete=json.loads((directory/done['result']).read_text())
            if 'mu_ex_bar_meV' in complete and complete.get('model_fp'):
                from tdi.vasp_backend import fingerprint
                if complete['model_fp']!=fingerprint(args[args.index('--model')+1].split(':',1)[1]):raise ValueError('Completed worker Hamiltonian changed')
                print(f'Already complete: {directory/done["result"]}');return
        prefix=str(directory/'pw')
        if any(Path(prefix+suffix).exists() for suffix in ('_host_md.npz','_full_md.npz','_frames.npz')):
            args+=['--resume-from',prefix,'--resume-in-place']
        previous=sys.argv;sys.argv=['pwidom.py',*args]
        try:pwidom.main()
        finally:sys.argv=previous
        results=list(directory.glob('pw_vasp_K*.json'))
        if len(results)==1:
            save(directory/'DONE.json',dict(result=results[0].name,args_sha256=args_hash))
