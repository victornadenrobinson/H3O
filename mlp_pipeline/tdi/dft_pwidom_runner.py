"""Bounded multi-system VASP PWidom runner, with checkpointed stop and cleanup."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def rank_requirement(workers):
    ranks=[]
    for worker in workers:
        args=worker['args'];path=Path(args[args.index('--model')+1].split(':',1)[1])
        command=json.loads(path.read_text())['command']
        ranks.append(int(command[command.index('--ntasks')+1]))
    return sum(ranks)


def kill_active_vasp(worker,smoke,owner_pid):
    directory=Path(worker['directory'])/('smoke' if smoke else '')
    path=directory/'vasp_work'/'active_process.json'
    try:
        record=json.loads(path.read_text());pid=int(record['pid'])
        if record.get('owner_pid')!=owner_pid:return
        if record['job']!=os.environ.get('SLURM_JOB_ID'):return
        if Path(f'/proc/{pid}/stat').read_text().split()[21]!=record['proc_start']:return
        os.killpg(pid,signal.SIGKILL)
    except (OSError,ValueError,KeyError,IndexError):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--minutes',type=float,default=1380)
    parser.add_argument('--stop-grace-seconds',type=float,default=120);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    if not math.isfinite(a.minutes) or not math.isfinite(a.stop_grace_seconds) or a.minutes<=0 or a.stop_grace_seconds<=0 or a.stop_grace_seconds>=a.minutes*60:parser.error('Need positive budget longer than stop grace')
    root=Path(a.directory).resolve();plan=json.loads((root/'dft_plan.json').read_text());needed=rank_requirement(plan['workers'])
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_NTASKS',0))<needed:
        raise SystemExit(f'Need an allocation with at least {needed} MPI tasks')
    jobs=[];stop=[False];stop_started=None;timed=False
    def request_stop(signum,frame):stop[0]=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,request_stop)
    started=time.monotonic();deadline=started+a.minutes*60
    for index,worker in enumerate(plan['workers']):
        directory=Path(worker['directory']);name='smoke64.log' if a.smoke else 'worker.log'
        log=(directory/name).open('a',buffering=1)
        command=[sys.executable,'-u',str(Path(__file__).with_name('perlmutter_vasp_check.py')),
            'dft-worker',str(root),'--index',str(index)]
        if a.smoke:command.append('--smoke')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((process,log,worker));print(f"START {worker['host']}: {directory/name}",flush=True)
    try:
        while any(p.poll() is None for p,_,_ in jobs):
            now=time.monotonic()
            if now>=deadline-a.stop_grace_seconds:stop[0]=True;timed=True
            if stop[0] and stop_started is None:
                stop_started=now
                for process,_,_ in jobs:
                    if process.poll() is None:process.terminate()
                print('Checkpointed stop requested; allowing current integration steps to finish.',flush=True)
            if stop_started is not None and now>=min(deadline,stop_started+a.stop_grace_seconds):
                for process,_,worker in jobs:
                    if process.poll() is None:
                        kill_active_vasp(worker,a.smoke,process.pid);process.kill()
                break
            time.sleep(1)
    finally:
        for process,log,worker in jobs:
            kill_active_vasp(worker,a.smoke,process.pid)
            if process.poll() is None:
                kill_active_vasp(worker,a.smoke,process.pid);process.kill()
            process.wait();log.close()
    codes={worker['host']:process.returncode for process,_,worker in jobs}
    print(json.dumps(dict(elapsed_seconds=time.monotonic()-started,budget_stop=timed,exit_codes=codes)),flush=True)
    failed=any(code not in (0,-signal.SIGTERM,-signal.SIGKILL) for code in codes.values())
    raise SystemExit(0 if (timed and not failed) or all(code==0 for code in codes.values()) else 1)


if __name__=='__main__':main()
