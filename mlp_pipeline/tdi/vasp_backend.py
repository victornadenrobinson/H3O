"""ASE VASP force engine for existing PWidom/TDI interfaces.

Original licensed template POTCAR blocks, force-consistent TOTEN, isolated worker
scratch, atomic geometry-result cache. MD and trial energies use the same DFT.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from ase.io import read
from ase.geometry import find_mic
from tdi import dft_check as dft
from tdi.pwidom_checkpoint import atomic_npz


def template_hash(directory):
    h=hashlib.sha256()
    for name in ('INCAR','KPOINTS','POTCAR','POSCAR'):
        h.update(name.encode());h.update((Path(directory)/name).read_bytes())
    return h.hexdigest()


def config(path):
    path=Path(path).resolve();data=json.loads(path.read_text())
    if template_hash(data['template'])!=data['template_sha256']:raise ValueError('VASP template changed since config preparation')
    return data


def fingerprint(path):
    data=config(path)
    identity={k:data[k] for k in ('template_sha256','T','executable')}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]


class VaspEngine(Calculator):
    implemented_properties=['energy','free_energy','forces','stress']

    def __init__(self,path):
        super().__init__();self.settings=config(path);self.temperature=float(self.settings['T'])
        self.tp=dft.template(self.settings['template']);self.fp=fingerprint(path)
        self.root=Path(self.settings['work']).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.directory=str(self.root/'active');Path(self.directory).mkdir(exist_ok=True)
        self.cache=self.root/'cache';self.cache.mkdir(exist_ok=True)
        self.ncalls=0;self.wall_seconds=0.

    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        if not os.environ.get('SLURM_JOB_ID') and not self.settings.get('allow_local_test',False):
            raise RuntimeError('VASP force engine requires a Slurm allocation')
        digest=hashlib.sha256(self.fp.encode())
        for array in (atoms.numbers,atoms.positions,atoms.cell.array,atoms.pbc):digest.update(np.ascontiguousarray(array).tobytes())
        resultpath=self.cache/(digest.hexdigest()+'.npz')
        with (self.root/'.calculator.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('VASP working directory already in use: give each worker its own config/work directory')
            if resultpath.exists():
                with np.load(resultpath,allow_pickle=False) as z:self.results={k:z[k].copy() if z[k].ndim else float(z[k]) for k in self.implemented_properties}
                return
            directory=Path(self.directory)
            signature=hashlib.sha256(atoms.numbers.tobytes()+atoms.cell.array.tobytes()).hexdigest()
            state=directory/'restart.json'
            previous=json.loads(state.read_text()) if state.exists() else {}
            wave=directory/'WAVECAR';warm=previous.get('signature')==signature and wave.exists() and wave.stat().st_size>0
            if not warm:wave.unlink(missing_ok=True)
            dft.write_dft(str(directory),atoms,self.temperature,self.tp,'ASE VASP force engine')
            inc=dft.parse_incar(directory/'INCAR')
            inc.update(ISIF='2',ISTART=str(1 if warm else 0),ICHARG='2',LWAVE='.TRUE.')
            (directory/'INCAR').write_text(''.join(f'{key} = {value}\n' for key,value in inc.items()))
            # Overwritten active files are scratch only. Completed results are atomic.
            for name in ('OUTCAR','vasprun.xml','CONTCAR'): (directory/name).unlink(missing_ok=True)
            command=self.settings['command'];started=time.perf_counter()
            with (directory/'vasp.out').open('w') as log:
                process=subprocess.Popen(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),start_new_session=True)
                try:code=process.wait(timeout=self.settings.get('timeout_seconds',1800))
                except BaseException:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    wave.unlink(missing_ok=True);raise
            from tdi.perlmutter_vasp_check import converged,save
            energy=converged(directory/'OUTCAR')
            if code or energy is None:
                wave.unlink(missing_ok=True)
                raise RuntimeError(f'VASP failed/unconverged (exit {code}); inspect {directory}/vasp.out and OUTCAR; no result cached')
            target=read(directory/'OUTCAR',index=-1,format='vasp-out')
            order=[i for s in self.tp['species'] for i,symbol in enumerate(atoms.get_chemical_symbols()) if symbol==s]
            expected=atoms[order]
            if not np.array_equal(target.numbers,expected.numbers) or not np.allclose(target.cell,expected.cell,atol=1e-4,rtol=0):raise ValueError('VASP output order/cell mismatch')
            if find_mic(target.positions-expected.positions,expected.cell,pbc=expected.pbc)[1].max()>1e-4:raise ValueError('VASP moved single-point atoms')
            forces=np.empty((len(atoms),3));forces[order]=target.get_forces(apply_constraint=False)
            self.results=dict(energy=energy['TOTEN_eV'],free_energy=energy['TOTEN_eV'],forces=forces,
                stress=target.get_stress(voigt=True,apply_constraint=False,include_ideal_gas=False))
            if not all(np.isfinite(value).all() for value in self.results.values()):raise ValueError('Nonfinite VASP results')
            atomic_npz(resultpath,**self.results);save(state,dict(signature=signature))
            self.ncalls+=1;elapsed=time.perf_counter()-started;self.wall_seconds+=elapsed
            with (self.root/'timings.jsonl').open('a') as f:f.write(json.dumps(dict(call=self.ncalls,natoms=len(atoms),wall_seconds=elapsed,warm_start=warm,cache=resultpath.name))+'\n')
            print(f'VASP force call {self.ncalls}: {len(atoms)} atoms, {elapsed:.1f}s, warm={warm}, E={energy["TOTEN_eV"]:.8f} eV',flush=True)


def vasp_config(a):
    import shutil
    from tdi.perlmutter_vasp_check import validated_template,gamma_only,save
    path=Path(a.out).resolve()
    if path.exists():raise ValueError('Config exists: reuse it for resume or choose fresh output')
    tp=validated_template(a.template)
    ncore=int(tp['incar'].get('NCORE','1'));kpar=int(tp['incar'].get('KPAR','1'))
    if ncore<1 or kpar<1 or a.ranks%(ncore*kpar):raise ValueError('VASP ranks must be divisible by NCORE*KPAR')
    if int(float(tp['incar']['ISMEAR']))!=-1:raise ValueError('Pure-DFT finite-temperature backend requires ISMEAR=-1')
    executable=shutil.which(a.exe)
    if not executable:raise ValueError(f'{a.exe} unavailable; load VASP module first')
    if 'vasp_gam' in Path(executable).name and not gamma_only(a.template):raise ValueError('vasp_gam requires Gamma-only template')
    if not np.isfinite(a.T) or not np.isfinite(a.timeout_seconds) or a.timeout_seconds<=0 or a.T<=0 or a.ranks<1:raise ValueError('Positive temperature/ranks required')
    path.parent.mkdir(parents=True,exist_ok=True)
    save(path,dict(template=str(Path(a.template).resolve()),template_sha256=template_hash(a.template),T=a.T,
        work=str(Path(a.work).resolve()),executable=executable,timeout_seconds=a.timeout_seconds,
        command=['srun','--exclusive','--exact','--nodes=1','--ntasks',str(a.ranks),'--cpus-per-task=1','--cpu-bind=cores',executable]))
    print(f'VASP model spec: vasp:{path}; same ASE MD and PWidom estimators, batch=1 recommended')
