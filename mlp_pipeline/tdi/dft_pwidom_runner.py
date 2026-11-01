"""Run up to eight node-local VASP-backed PWidom workers; preserve restart state."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root=Path(sys.argv[1]).resolve();plan=json.loads((root/'dft_plan.json').read_text())
    count=len(plan['workers'])
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_NTASKS',0))<32*count:
        raise SystemExit(f'Need an allocation with at least {count*32} MPI tasks')
    jobs=[];stop=[False]
    def request_stop(signum,frame):stop[0]=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,request_stop)
    start=time.monotonic();sent=False
    for index,worker in enumerate(plan['workers']):
        directory=Path(worker['directory'])
        log=(directory/'worker.log').open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('perlmutter_vasp_check.py')),
            'dft-worker',str(root),'--index',str(index)],stdout=log,stderr=subprocess.STDOUT)
        jobs.append((process,log,worker))
        print(f"START {worker['host']} replica {worker['replica']}: {directory}/worker.log",flush=True)
    try:
        while any(p.poll() is None for p,_,_ in jobs):
            if time.monotonic()-start>23*3600:stop[0]=True
            if stop[0] and not sent:
                for process,_,_ in jobs:
                    if process.poll() is None:process.terminate()
                sent=True
                print('Stop requested: MD workers save after completed steps; cached force results remain reusable.',flush=True)
            time.sleep(2)
    finally:
        for process,log,worker in jobs:
            if process.poll() is None:process.terminate()
            log.close()
    for process,_,worker in jobs:print(f"EXIT {worker['host']} replica {worker['replica']}: {process.returncode}",flush=True)
    raise SystemExit(0 if all(p.returncode==0 for p,_,_ in jobs) else 1)


if __name__=='__main__':main()
