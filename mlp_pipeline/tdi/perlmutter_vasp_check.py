"""Paired VASP energy validation on cached GRACE PWidom ensembles; no MD.

This is an energy/model check, not a pure-DFT chemical-potential calculation.
VASP free energy TOTEN is the label; unconverged electronic results are rejected.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tdi import dft_check as dft
from tdi import pwidom as pw
from tdi.pwidom_checkpoint import atomic_npz


def save(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def input_hash(directory):
    h=hashlib.sha256()
    for name in ('INCAR','KPOINTS','POSCAR','POTCAR'):
        h.update(name.encode());h.update((Path(directory)/name).read_bytes())
    return h.hexdigest()


def converged(path):
    """A timing footer alone is insufficient: require finite TOTEN and EDIFF."""
    path=Path(path)
    if not path.is_file():return None
    text=path.read_text(errors='replace')
    matches=re.findall(r'free\s+energy\s+TOTEN\s*=\s*([-+\d.eEdD]+)',text)
    if (not matches or 'General timing' not in text
            or 'aborting loop because EDIFF is reached' not in text):return None
    value=float(matches[-1].replace('D','E').replace('d','e'))
    if not np.isfinite(value):return None
    timing=re.findall(r'Elapsed time \(sec\):\s*([\d.]+)',text)
    version=re.search(r'vasp[.\s]+[\w.]+',text,re.I)
    return dict(TOTEN_eV=value,elapsed_seconds=float(timing[-1]) if timing else None,
                vasp_version=version.group(0) if version else 'not recorded')


def done(point):
    point=Path(point)
    try:
        record=json.loads((point/'DONE.json').read_text())
        if record['input_sha256']!=input_hash(point):raise ValueError(f'{point}: inputs changed after completion')
        energy=converged(point/record['attempt']/'OUTCAR')
        if energy is None or not np.isclose(energy['TOTEN_eV'],record['TOTEN_eV'],atol=1e-8,rtol=0):
            raise ValueError(f'{point}: completion record does not match a converged OUTCAR')
        return record
    except FileNotFoundError:return None


def gamma_only(directory):
    """Conservative check of prepared k-points and real-wavefunction eligibility."""
    directory=Path(directory);inc=dft.parse_incar(directory/'INCAR')
    if any(inc.get(k,'F').strip('.').upper() in ('T','TRUE') for k in ('LSORBIT','LNONCOLLINEAR')):return False
    lines=[line.split('!')[0].strip() for line in (directory/'KPOINTS').read_text().splitlines() if line.strip()]
    try:
        count=int(lines[1].split()[0])
        if count==0 and lines[2].lower().startswith(('g','m')):
            mesh=[int(v) for v in lines[3].split()[:3]]
            shift=[float(v) for v in lines[4].split()[:3]] if len(lines)>4 else [0.,0.,0.]
            return mesh==[1,1,1] and len(shift)==3 and np.allclose(shift,0,atol=1e-12,rtol=0)
        if count==1 and lines[2].lower().startswith(('r','c','k')):
            return np.allclose([float(v) for v in lines[3].split()[:3]],[0,0,0],atol=1e-12,rtol=0)
    except (ValueError,IndexError):pass
    return False


def report_pairs(root,manifest,reported=None,changed=None):
    reported=reported if reported is not None else set();rows=[]
    for pair in manifest.get('pairs',[]):
        if changed is not None and changed not in pair['points']:continue
        if pair['pair'] in reported:continue
        records=[done(Path(root)/identity) for identity in pair['points']]
        if not all(records):continue
        delta=records[1]['TOTEN_eV']-records[0]['TOTEN_eV']
        grace=pair.get('dU_mlp_eV')
        message=f'PAIR {pair["pair"]:03d} {pair["host"]} {pair["direction"]}: ΔU_VASP={delta:.6f} eV'
        if grace is not None:message+=f'; ΔU_GRACE={grace:.6f} eV; VASP−GRACE={(delta-grace)*1000:+.2f} meV/H'
        else:message+='; saved GRACE ΔU unavailable'
        if pair.get('baseline_dU_DFT_eV') is not None:message+=f'; k-mesh shift={(delta-pair["baseline_dU_DFT_eV"])*1000:+.2f} meV/H'
        print(message,flush=True);reported.add(pair['pair'])
        rows.append(dict(pair=pair['pair'],dU_DFT_eV=delta,dU_MLP_eV=grace,
                         DFT_minus_MLP_eV=delta-grace if grace is not None else None))
    return rows


def watch(a):
    root=Path(a.directory).resolve();manifest=json.loads((root/'manifest.json').read_text());reported=set()
    if not manifest.get('pairs'):raise ValueError('This check has no saved pairs to watch')
    if not np.isfinite(a.interval) or a.interval<=0:raise ValueError('Positive watch interval required')
    try:
        while True:
            report_pairs(root,manifest,reported)
            if a.once or len(reported)==len(manifest['pairs']):return
            time.sleep(a.interval)
    except KeyboardInterrupt:return


def validated_template(path):
    path=Path(path).resolve()
    missing=[n for n in ('INCAR','KPOINTS','POTCAR','POSCAR') if not (path/n).is_file()]
    if missing:raise ValueError(f'{path}: missing {missing}')
    tp=dft.template(str(path))
    if not {'H','O'}.issubset(tp['species']):raise ValueError('Use a matching training H/O template; pure-H POTCAR is selected from its H block')
    inc=tp['incar']
    if any(k in inc for k in ('NELECT','MAGMOM','M_CONSTR')) or inc.get('LDAU','F').strip('.').upper() in ('T','TRUE'):
        raise ValueError('Template has charge/atom-specific/magnetic settings needing explicit composition-dependent review')
    if any('LAMBDA' in key for key in inc):
        raise ValueError('Template contains coupling parameters; select/review a pure-DFT training template')
    for tag in ('ENCUT','EDIFF','ISMEAR'):
        if tag not in inc:raise ValueError(f'Template must explicitly specify {tag}')
    return tp


def discover(a):
    import pandas as pd
    path=Path(a.inventory)
    if not path.is_file():raise ValueError(f'Inventory missing: {path}')
    data=pd.read_csv(path,usecols=lambda c:c in ('Path','comp_snap','N_H','N_O','Other'))
    candidates=[];seen=set()
    for _,row in data.iterrows():
        if 'Other' in row and str(row['Other']) not in ('','nan'):continue
        source=Path(str(row['Path'])).expanduser()
        directory=source if source.is_dir() else source.parent
        if str(directory) in seen:continue
        seen.add(str(directory))
        if all((directory/n).is_file() for n in ('INCAR','KPOINTS','POTCAR','POSCAR')):
            try:
                tp=validated_template(directory)
            except (ValueError,OSError,SystemExit):continue
            candidates.append(dict(path=str(directory.resolve()),ENCUT=tp['incar'].get('ENCUT'),
                EDIFF=tp['incar'].get('EDIFF'),ISMEAR=tp['incar'].get('ISMEAR'),SIGMA=tp['incar'].get('SIGMA'),species=tp['species']))
            if len(candidates)>=a.limit:break
    print(json.dumps(dict(template_candidates=candidates,note='No files copied; no VASP/model calls. Select a template matching the training labels.'),indent=2))


def preflight(a):
    tp=validated_template(a.template)
    exe=shutil.which(a.exe)
    training=None
    if a.train_outcar:
        import gzip,itertools
        opener=gzip.open if str(a.train_outcar).endswith('.gz') else open
        with opener(a.train_outcar,'rt',errors='replace') as f:text=''.join(itertools.islice(f,5000))
        version=re.search(r'vasp[.\s]+[\w.]+',text,re.I)
        training=dict(path=str(a.train_outcar),version=version.group(0) if version else 'not found in header')
        for tag in ('ENCUT','EDIFF','ISMEAR','SIGMA','PREC','LREAL'):
            match=re.search(r'\b'+tag+r'\s*=\s*([-+\w.]+)',text)
            if match:training[tag]=match.group(1)
    print(json.dumps(dict(template=str(Path(a.template).resolve()),species=tp['species'],training_header=training,
        gamma_only=gamma_only(a.template),KPOINTS=tp['kpoints'],
        settings={k:tp['incar'].get(k) for k in ('ENCUT','EDIFF','ISMEAR','SIGMA','PREC','GGA','LASPH','ISPIN','LREAL','ALGO','NELM','NCORE','NPAR','KPAR')},
        executable=exe,allocation=os.environ.get('SLURM_JOB_ID'),
        note='No VASP execution. POTCAR contents are not printed or committed.'),indent=2))
    if not exe:raise ValueError(f'{a.exe} unavailable; load the selected Perlmutter VASP module first')


def prep(a):
    if not a.hosts or len(set(a.hosts))!=len(a.hosts):raise ValueError('Choose distinct host names')
    if a.pairs<4 or a.pairs%(2*len(a.hosts)):
        raise ValueError('pairs must be divisible by 2 × number of hosts, with both directions represented')
    if a.K<1 or not 0<=a.discard_fraction<1:raise ValueError('Invalid K/discard fraction')
    if Path(a.out).exists():raise ValueError('Use a fresh output directory; existing inputs/results are never overwritten')
    tp=validated_template(a.template)
    root=Path(a.run).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='complete':raise ValueError('Choose a completed PWidom run')
    jobs={j['name']:j for j in manifest['jobs']}
    if not all(h in jobs for h in a.hosts):raise ValueError(f'Run does not contain requested hosts {a.hosts}')
    per_direction=a.pairs//(2*len(a.hosts));rng=np.random.default_rng(a.seed)
    trials=[]
    for name in a.hosts:
        job=jobs[name];path=Path(job['result'])
        if not path.is_file():path=root/path.name
        result=json.loads(path.read_text());framepath=root/(result['tag']+'_frames.npz')
        with np.load(framepath,allow_pickle=False) as z:
            if 'cell' not in z.files:raise ValueError(f'{framepath}: missing actual cell; no cubic reconstruction is assumed')
            cell=np.asarray(z['cell'],float);numbers=z['numbers'].copy()
            if cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<=0:raise ValueError('Invalid fixed NVT cell')
            if not np.isclose(abs(np.linalg.det(cell)),float(z['V']),rtol=1e-7):raise ValueError('Cell/volume mismatch')
            if str(z['model_fp'])!=result.get('model_fp') or float(z['T'])!=result['T']:raise ValueError('Frame/result T or Hamiltonian mismatch')
            positions={k:z[k].copy() for k in ('host','full')}
        if (positions['host'].shape[1:]!=(len(numbers),3) or positions['full'].shape[1:]!=(len(numbers)+1,3)
                or not all(np.isfinite(x).all() for x in positions.values())):raise ValueError('Invalid host/full positions')
        proxy=job['command'][job['command'].index('--proxy')+1]
        params=pw.proxy_params(proxy);K=1 if params is None else a.K;kt=pw.KB*result['T']
        for direction,key in (('insertion','host'),('deletion','full')):
            frames=positions[key];start=int(np.ceil(len(frames)*a.discard_fraction))
            if len(frames)-start<per_direction:raise ValueError(f'{name}/{direction}: insufficient distinct post-discard frames')
            selected=np.linspace(start,len(frames)-1,per_direction).round().astype(int)
            for frame in selected:
                x=frames[frame]
                if direction=='insertion':
                    without=Atoms(numbers=numbers,positions=x,cell=cell,pbc=True)
                    points=rng.random((K,3))@cell
                    phi=pw.proxy_energy(params,points,x,numbers==1,cell)
                    logsum=pw.lse(-phi/kt);pick=int(rng.choice(K,p=np.exp(-phi/kt-logsum)))
                    with_h=Atoms(numbers=np.r_[numbers,1],positions=np.vstack([x,points[pick]]),cell=cell,pbc=True)
                    ph=float(phi[pick]);lnw=float(logsum-np.log(K));deleted=None
                else:
                    full_numbers=np.r_[numbers,1];deleted=int(rng.choice(np.flatnonzero(full_numbers==1)))
                    keep=np.arange(len(x))!=deleted
                    without=Atoms(numbers=full_numbers[keep],positions=x[keep],cell=cell,pbc=True)
                    with_h=Atoms(numbers=full_numbers,positions=x,cell=cell,pbc=True)
                    ph=float(pw.proxy_energy(params,x[deleted:deleted+1],x[keep],full_numbers[keep]==1,cell)[0])
                    candidates=np.r_[ph,pw.proxy_energy(params,rng.random((K-1,3))@cell,x[keep],full_numbers[keep]==1,cell)] if K>1 else np.array([ph])
                    lnw=float(pw.lse(-candidates/kt)-np.log(K))
                trials.append(dict(host=name,direction=direction,frame=int(frame),T=result['T'],P=result['P_target'],
                    proxy=proxy,K=K,phi_eV=ph,lnW=lnw,deleted_index=deleted,model=result['model'],model_fp=result['model_fp'],
                    source_frames=str(framepath),geometries=[without,with_h]))
    write_pairs(a,tp,trials,dict(run=str(root),seed=a.seed,discard_fraction=a.discard_fraction,
        note='NEW reproducible proxy trials on cached GRACE ensembles, not a replay of the saved production work arrays. No MD/model/VASP calls during preparation.'))


def write_pairs(a,tp,trials,metadata):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    points=[];pairs=[]
    for i,trial in enumerate(trials):
        structures=trial.pop('geometries');ids=[]
        for role,at in zip(('without_H','with_H'),structures):
            identity=f'p{i:03d}_{role}';directory=out/identity
            dft.write_dft(str(directory),at,trial['T'],tp,f'PWidom validation pair {i}, {trial["direction"]}, {role}')
            ids.append(identity);points.append(dict(id=identity,path=identity,input_sha256=input_hash(directory),natoms=len(at)))
        pairs.append(dict(pair=i,points=ids,**trial))
    save(out/'manifest.json',dict(schema=1,mode='paired_pwidom_energy_check',
        template=str(Path(a.template).resolve()),pairs=pairs,points=points,**metadata))
    (out/'dirs.txt').write_text(''.join(str(out/p['path'])+'\n' for p in points))
    print(f'Prepared {len(pairs)} paired differences / {len(points)} VASP single points in {out}')
    print('Diagnostic geometries only; these data alone are not pure-VASP free energies.')


def queue_entries(a):
    root=Path(a.queue).resolve()
    manifest=None;allowed={};runpath=None
    if a.run:
        runpath=Path(a.run).resolve();manifest=json.loads((runpath/'manifest.json').read_text())
        for job in manifest['jobs']:
            if a.hosts and job['name'] not in a.hosts:continue
            path=Path(job['result'])
            if not path.is_file():path=runpath/path.name
            result=json.loads(path.read_text());allowed[result['tag']]=(job['name'],result)
        if not allowed:raise ValueError('No matching host tags in the selected run')
    entries=[]
    for file in sorted(root.glob('request_*.json')):
        request=json.loads(file.read_text())
        if runpath:
            source=Path(request.get('run','')).resolve()
            if source!=runpath:continue
            if request.get('tag') not in allowed:continue
            host,result=allowed[request['tag']]
            if (request.get('model_fp')!=result.get('model_fp') or request.get('T_K')!=result['T']
                    or request.get('P_GPa')!=result['P_target']):
                raise ValueError(f'{file}: queue/result state or Hamiltonian mismatch')
        else:host=request.get('tag','unknown')
        if request.get('direction') not in ('insertion','deletion') or 'dU_mlp_eV' not in request:continue
        if request.get('geometry_order')!=['host_without_selected_H','host_with_selected_H']:
            raise ValueError(f'{file}: geometry order missing or unsupported')
        if len(request.get('structures',[]))!=2 or not np.isfinite(float(request['dU_mlp_eV'])):
            raise ValueError(f'{file}: invalid paired request')
        entries.append((file,host,request))
    return root,entries


def queue(a):
    _,entries=queue_entries(a)
    from collections import Counter
    counts=Counter((r.get('run'),host,r['direction']) for _,host,r in entries)
    print(json.dumps(dict(groups=[dict(run=run,host=host,direction=direction,pairs=n)
        for (run,host,direction),n in sorted(counts.items())],total_pairs=len(entries),
        note='Saved production ΔU and paired geometries. Nothing prepared or evaluated.'),indent=2))


def prep_queue(a):
    if a.pairs<1:raise ValueError('pairs must be positive')
    if Path(a.out).exists():raise ValueError('Use a fresh output directory')
    tp=validated_template(a.template);root,entries=queue_entries(a)
    groups={};seen=set()
    for file,host,request in entries:
        identity=tuple(request['structures'])
        if identity in seen:continue
        seen.add(identity)
        groups.setdefault((host,request['direction']),[]).append((file,host,request))
    if sum(map(len,groups.values()))<a.pairs:
        raise ValueError(f'Only {sum(map(len,groups.values()))} unique saved pairs match; choose fewer or list queue groups')
    chosen=[]
    while len(chosen)<a.pairs:
        for key in sorted(groups):
            if groups[key] and len(chosen)<a.pairs:chosen.append(groups[key].pop(0))
    from tdi.pwidom_candidates import geometry_id
    from collections import Counter
    trials=[]
    for file,host,r in chosen:
        atoms=[]
        for name in r['structures']:
            path=(root/name).resolve()
            if path.parent!=root or path.suffix!='.extxyz':raise ValueError('Queue geometry must be a direct extxyz child')
            at=read(path)
            if geometry_id(at)!=path.stem:raise ValueError(f'{path}: geometry hash changed; saved GRACE ΔU cannot be trusted')
            atoms.append(at)
        without,with_h=atoms
        if len(with_h)!=len(without)+1 or not np.allclose(with_h.cell,without.cell) or not np.array_equal(with_h.pbc,without.pbc):
            raise ValueError(f'{file}: invalid paired cell/species')
        def sites(at):return Counter((int(z),*np.round(pos,7)) for z,pos in zip(at.numbers,at.positions))
        extra=sites(with_h)-sites(without);missing=sites(without)-sites(with_h)
        if missing or sum(extra.values())!=1 or next(iter(extra))[0]!=1:raise ValueError(f'{file}: pair must differ by exactly one H')
        trials.append(dict(host=host,direction=r['direction'],frame=r['frame'],T=r['T_K'],P=r['P_GPa'],
            K=r['K'],phi_eV=r['phi_eV'],lnW=None,proxy=r['proxy'],model=r['model'],model_fp=r['model_fp'],
            source_frames=r['source_frames'],tag=r['tag'],trial=r.get('trial'),request_file=str(file),
            dU_mlp_eV=float(r['dU_mlp_eV']),log_weight=r['log_weight'],reasons=r.get('reasons',[]),geometries=atoms))
    write_pairs(a,tp,trials,dict(run=str(Path(a.run).resolve()),queue=str(root),
        note='EXACT saved production pairs and GRACE ΔU; round-robin across available host/direction groups. Targeted/weight-selected diagnostics, not unbiased DFT ensemble samples.'))


def mesh_check(a):
    source=Path(a.directory).resolve();manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('mode')!='paired_pwidom_energy_check':raise ValueError('Mesh check requires prepared paired energies')
    if Path(a.out).exists():raise ValueError('Use a fresh mesh-check directory')
    if len(a.mesh)!=3 or min(a.mesh)<1 or a.pairs_per_host<1:raise ValueError('Positive mesh and pair count required')
    chosen=[]
    for host in a.hosts:
        eligible=[]
        for pair in manifest['pairs']:
            if pair['host']!=host:continue
            records=[done(source/i) for i in pair['points']]
            if all(records):eligible.append((pair,records))
        if len(eligible)<a.pairs_per_host:raise ValueError(f'{host}: need {a.pairs_per_host} completed pairs in the source')
        chosen.extend(eligible[:a.pairs_per_host])
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    points=[];pairs=[]
    for i,(pair,records) in enumerate(chosen):
        identities=[]
        for role,old in zip(('without_H','with_H'),pair['points']):
            identity=f'p{i:03d}_{role}';directory=out/identity;directory.mkdir()
            for name in ('INCAR','KPOINTS','POSCAR','POTCAR','structure.extxyz'):
                shutil.copy2(source/old/name,directory/name)
            # Explicit KPOINTS controls the new mesh; remove unused spacing tags
            # rather than leaving contradictory intent in the prepared INCAR.
            lines=(directory/'INCAR').read_text().splitlines()
            (directory/'INCAR').write_text('\n'.join(line for line in lines if line.split('=')[0].strip().upper() not in ('KSPACING','KGAMMA'))+'\n')
            (directory/'KPOINTS').write_text('paired k-point convergence check\n0\nGamma\n'+' '.join(map(str,a.mesh))+'\n0 0 0\n')
            identities.append(identity)
            points.append(dict(id=identity,path=identity,input_sha256=input_hash(directory),natoms=len(read(directory/'structure.extxyz'))))
        metadata={k:v for k,v in pair.items() if k not in ('pair','points')}
        pairs.append(dict(metadata,pair=i,points=identities,baseline_pair=pair['pair'],baseline_directory=str(source),
                          baseline_dU_DFT_eV=records[1]['TOTEN_eV']-records[0]['TOTEN_eV']))
    save(out/'manifest.json',dict(schema=1,mode='paired_pwidom_energy_check',template=manifest['template'],
        run=manifest.get('run'),pairs=pairs,points=points,mesh=list(a.mesh),
        note='Exact source geometries; new explicit Gamma-centered k mesh. Baseline inputs/results untouched.'))
    print(f'Prepared {len(pairs)} exact matched pairs / {len(points)} structures at mesh {a.mesh}; run with vasp_std')


def training(a):
    """Prepare labelled original snapshots without running MD or GRACE."""
    from ase.io import iread
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a fresh training check directory')
    if a.count<1:raise ValueError('count must be positive')
    candidates=[];seen=set()
    for filename in a.data:
        for index,at in enumerate(iread(filename,index=':')):
            source=at.info.get('source');temperature=at.info.get('T_K')
            if not source or temperature is None:continue
            if source in seen:continue
            if a.T is not None and abs(float(temperature)-a.T)>1:continue
            if a.P is not None and abs(float(at.info.get('P_GPa',float('inf')))-a.P)>a.pressure_window:continue
            if at.calc is None or 'forces' not in at.calc.results:continue
            energy=at.calc.results.get('free_energy',at.calc.results.get('energy'))
            if energy is None or not np.isfinite(energy):continue
            candidates.append((filename,index,at,float(energy)));seen.add(source)
    if not candidates:raise ValueError('No labelled snapshots match the filters')
    indices=np.linspace(0,len(candidates)-1,min(a.count,len(candidates))).round().astype(int)
    selected=[]
    for i in indices:
        filename,index,at,energy=candidates[i]
        source=(Path(a.source_root)/at.info['source']).resolve()
        template_dir=Path(a.template).resolve() if a.template else None
        if template_dir is None:
            # Restart subdirectories often inherit inputs from the volume directory.
            for parent in [source.parent,*list(source.parents)[1:]]:
                if all((parent/n).is_file() for n in ('INCAR','KPOINTS','POTCAR')):
                    template_dir=parent;break
        if template_dir is None:raise ValueError(f'No original input template found for {source}; supply --template only after checking shared settings')
        tp=dft.template(str(template_dir))
        if any(k in tp['incar'] for k in ('NELECT','MAGMOM','M_CONSTR')) or any('LAMBDA' in k for k in tp['incar']):
            raise ValueError(f'{template_dir}: atom-specific or coupling settings require review')
        if int(float(tp['incar'].get('ISMEAR','0')))!=-1:raise ValueError(f'{template_dir}: expected Fermi smearing ISMEAR=-1')
        sigma=float(tp['incar'].get('SIGMA','nan'))
        if not np.isfinite(sigma) or abs(sigma-dft.KB*float(at.info['T_K']))>2e-5:
            raise ValueError(f'{template_dir}: SIGMA does not match snapshot T_K; select its original inputs')
        selected.append((filename,index,at,energy,tp,template_dir))
    out.mkdir(parents=True);points=[]
    for i,(filename,index,at,energy,tp,template_dir) in enumerate(selected):
        identity=f'train_{i:03d}';point=out/identity
        order=[j for species in tp['species'] for j,symbol in enumerate(at.get_chemical_symbols()) if symbol==species]
        dft.write_dft(str(point),at,float(at.info['T_K']),tp,'Original labelled DFT snapshot')
        # Keep the original smearing precision and request an explicit stress tensor.
        with (point/'INCAR').open('a') as f:f.write(f"ISIF = 2\nSIGMA = {tp['incar']['SIGMA']}\n")
        results=at.calc.results
        arrays=dict(forces=np.asarray(results['forces'])[order],energy=np.array(energy))
        if 'stress' in results:arrays['stress']=at.get_stress()
        np.savez(point/'reference.npz',**arrays)
        points.append(dict(id=identity,path=identity,natoms=len(at),input_sha256=input_hash(point),
            reference_sha256=hashlib.sha256((point/'reference.npz').read_bytes()).hexdigest(),
            source=at.info['source'],frame=int(at.info['frame']),data=str(Path(filename).resolve()),data_index=index,
            T_K=float(at.info['T_K']),P_GPa=at.info.get('P_GPa'),template=str(template_dir)))
    save(out/'manifest.json',dict(schema=1,mode='training_label_check',pairs=[],points=points))
    print(f'Prepared {len(points)} labelled snapshots in {out}; original inputs selected per source, no VASP launched',flush=True)


def report_training(root,point):
    if 'reference_sha256' not in point:return
    directory=Path(root)/point['path'];record=done(directory)
    reference=directory/'reference.npz'
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=point['reference_sha256']:
        raise ValueError('Original reference labels changed after preparation')
    with np.load(reference,allow_pickle=False) as labels:
        delta=record['TOTEN_eV']-float(labels['energy'])
        result=dict(id=point['id'],energy_difference_eV=delta,energy_difference_meV_atom=delta*1000/point['natoms'])
        message=f"LABEL {point['id']}: VASP−{point.get('reference_kind','original')}={delta:+.6f} eV ({result['energy_difference_meV_atom']:+.3f} meV/atom)"
        try:
            at=read(directory/record['attempt']/'OUTCAR',format='vasp-out',index=-1)
            expected=read(directory/'POSCAR')
            if not np.array_equal(at.numbers,expected.numbers) or not np.allclose(at.cell,expected.cell,atol=1e-5,rtol=0):raise ValueError('OUTCAR geometry/order mismatch')
            displacement=at.positions-expected.positions
            fractional=displacement@np.linalg.inv(expected.cell.array);fractional-=np.rint(fractional)
            if np.max(np.abs(fractional@expected.cell.array))>1e-4:raise ValueError('OUTCAR positions mismatch')
            force_error=at.get_forces()-labels['forces']
            result['force_component_rmse_eV_A']=float(np.sqrt(np.mean(force_error**2)))
            result['force_max_component_error_eV_A']=float(np.max(np.abs(force_error)))
            message+=f"; force RMSE={result['force_component_rmse_eV_A']:.6f} eV/Å"
            if 'stress' in labels:
                error=(at.get_stress()-labels['stress'])*160.21766208
                result['stress_rmse_GPa']=float(np.sqrt(np.mean(error**2)))
                message+=f"; stress RMSE={result['stress_rmse_GPa']:.4f} GPa"
        except Exception as exc:
            result['force_stress_error']=str(exc);message+=f'; force/stress comparison unavailable: {exc}'
    save(directory/'label_comparison.json',result);print(message,flush=True)


def snapshot(a):
    if Path(a.out).exists():raise ValueError('Use a fresh snapshot directory')
    tp=validated_template(a.template)
    with np.load(a.frames,allow_pickle=False) as z:
        if 'cell' not in z.files:raise ValueError('Snapshot cache must record the actual cell')
        cell=np.asarray(z['cell'],float)
        if cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<=0:raise ValueError('Invalid cell')
        numbers=z['numbers'].copy()
        if a.ensemble=='full':numbers=np.r_[numbers,1]
        positions=np.array(z[a.ensemble][a.frame]);temperature=float(z['T'])
        if positions.shape!=(len(numbers),3) or not np.isfinite(positions).all():raise ValueError('Invalid snapshot')
        at=Atoms(numbers=numbers,positions=positions,cell=cell,pbc=True)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);point=out/'snapshot'
    dft.write_dft(str(point),at,temperature,tp,'Single cached PWidom snapshot; no new MD')
    save(out/'manifest.json',dict(schema=1,mode='vasp_smoke',template=str(Path(a.template).resolve()),
        source_frames=str(Path(a.frames).resolve()),ensemble=a.ensemble,frame=a.frame,T=temperature,pairs=[],
        points=[dict(id='snapshot',path='snapshot',input_sha256=input_hash(point),natoms=len(at))]))
    print(f'Prepared exactly one {len(at)}-atom {a.ensemble} snapshot at {temperature:g} K in {out}')


def smoke(a):
    if Path(a.out).exists():raise ValueError('Use a fresh smoke directory')
    tp=validated_template(a.template)
    from importlib import import_module
    at=import_module('17_eos_md').build('fcc',a.vpa,1,0,2.)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);point=out/'small_12atom'
    dft.write_dft(str(point),at,a.T,tp,'Perlmutter VASP chain check: 12-atom FCC H2O')
    save(out/'manifest.json',dict(schema=1,mode='vasp_smoke',template=str(Path(a.template).resolve()),pairs=[],
        points=[dict(id=point.name,path=point.name,input_sha256=input_hash(point),natoms=len(at))]))
    print(f'Prepared one {len(at)}-atom chain check in {out}; no VASP launched')


def run(a):
    root=Path(a.directory).resolve();manifest=json.loads((root/'manifest.json').read_text())
    points=manifest['points'];parallel=getattr(a,'parallel',1)
    if a.index is not None:
        if not 0<=a.index<len(points):raise ValueError('Task index outside prepared point list')
        points=[points[a.index]]
    if (a.ranks<1 or parallel<1 or not np.isfinite(a.minutes) or a.minutes<=0
            or not np.isfinite(a.margin_seconds) or a.margin_seconds<0
            or (a.max_points is not None and a.max_points<1)):
        raise ValueError('Positive ranks/parallel/minutes/max-points and finite nonnegative margin required')
    exe=shutil.which(a.exe)
    if not exe:raise ValueError(f'{a.exe} not found; load the selected VASP module')
    if not a.dry_run and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run VASP inside a Slurm allocation, not on the login node')
    launcher=shlex.split(a.launcher)
    is_srun=bool(launcher and Path(launcher[0]).name=='srun')
    command=launcher+(['--exclusive','--exact','--nodes=1','--ntasks',str(a.ranks),'--cpus-per-task=1','--cpu-bind=cores'] if is_srun else [])+[exe]
    pending=[];completed=0
    for point in points:
        directory=root/point['path']
        if input_hash(directory)!=point['input_sha256']:raise ValueError(f'{directory}: inputs changed since preparation')
        if done(directory):
            completed+=1;print(f'SKIP converged {point["id"]}',flush=True);report_training(root,point);continue
        if 'vasp_gam' in Path(exe).name and not gamma_only(directory):
            raise ValueError(f'{point["id"]}: vasp_gam requires confirmed Γ-only k-points and no SOC/noncollinear flags')
        inc=dft.parse_incar(directory/'INCAR');ncore=int(inc.get('NCORE','1'));kpar=int(inc.get('KPAR','1'))
        if ncore<1 or kpar<1 or a.ranks%(ncore*kpar):
            raise ValueError(f'{point["id"]}: ranks={a.ranks} incompatible with NCORE={ncore}, KPAR={kpar}; review parallel settings')
        pending.append(point)
    if a.max_points is not None:pending=pending[:a.max_points]
    allocated=int(os.environ.get('SLURM_NTASKS','0'))
    needed=min(parallel,len(pending))*a.ranks
    if not a.dry_run and parallel>1 and (allocated<needed):
        raise ValueError(f'Packing needs {needed} allocated MPI ranks; SLURM_NTASKS={allocated}. Request enough ranks with salloc -n.')
    print(f'Packing up to {parallel} independent VASP points × {a.ranks} MPI ranks; OMP_NUM_THREADS=1',flush=True)
    if a.dry_run:
        for point in pending:print(f'{point["id"]}: {shlex.join(command)}',flush=True)
        return
    deadline=time.monotonic()+a.minutes*60;stopped=[None];previous={};active=[];ran=0;failed=0;budget_stop=False
    reported=set();report_pairs(root,manifest,reported)
    def stop(signum,frame):stopped[0]=signum
    def launch(point):
        directory=root/point['path'];lock=(directory/'.run.lock').open('a')
        log=None
        try:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError(f'{point["id"]}: another runner holds this point lock')
            if done(directory):lock.close();return None
            attempt=directory/'attempts'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            attempt.mkdir(parents=True,exist_ok=False)
            for name in ('INCAR','KPOINTS','POSCAR','structure.extxyz'):
                if (directory/name).exists():shutil.copy2(directory/name,attempt/name)
            (attempt/'POTCAR').symlink_to(directory/'POTCAR')
            save(attempt/'attempt.json',dict(command=command,input_sha256=point['input_sha256'],allocation=os.environ.get('SLURM_JOB_ID'),
                started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),parallel=parallel,note='New isolated attempt; prior outputs preserved'))
            log=(attempt/'vasp.out').open('x')
            process=subprocess.Popen(command,cwd=attempt,env=dict(os.environ,OMP_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(f'START {point["id"]}: {shlex.join(command)}',flush=True)
            return dict(point=point,directory=directory,attempt=attempt,lock=lock,log=log,process=process,started=time.monotonic())
        except BaseException:
            if log:log.close()
            lock.close();raise
    def finish(job):
        nonlocal completed,failed
        process=job['process'];code=process.wait();job['log'].close()
        energy=converged(job['attempt']/'OUTCAR');timing=time.monotonic()-job['started']
        try:
            save(job['attempt']/'exit.json',dict(exit_code=code,wall_seconds=timing,electronic_converged=energy is not None))
            if code==0 and energy is not None:
                save(job['directory']/'DONE.json',dict(energy,input_sha256=job['point']['input_sha256'],
                    attempt=str(job['attempt'].relative_to(job['directory'])),wall_seconds=timing,ranks=a.ranks))
                completed+=1;print(f'DONE {job["point"]["id"]}: TOTEN={energy["TOTEN_eV"]:.9f} eV, wall={timing:.1f}s',flush=True)
                report_pairs(root,manifest,reported,job['point']['id'])
                report_training(root,job['point'])
            else:
                failed+=1;print(f'INCOMPLETE {job["point"]["id"]}: exit={code}; converged TOTEN not accepted; inspect {job["attempt"]}',flush=True)
        finally:job['lock'].close()
    def terminate():
        for job in active:
            if job['process'].poll() is None:
                try:os.killpg(job['process'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        end=time.monotonic()+5
        while any(j['process'].poll() is None for j in active) and time.monotonic()<end:time.sleep(.1)
        for job in list(active):
            if job['process'].poll() is None:
                try:os.killpg(job['process'].pid,signal.SIGKILL)
                except ProcessLookupError:pass
            finish(job);active.remove(job)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):previous[sig]=signal.signal(sig,stop)
    last_report=time.monotonic()
    try:
        while pending or active:
            if stopped[0] is not None or time.monotonic()>=deadline:
                budget_stop=time.monotonic()>=deadline;terminate();break
            if pending and deadline-time.monotonic()<a.margin_seconds:budget_stop=True
            while pending and len(active)<parallel and not failed and not budget_stop:
                point=pending.pop(0);job=launch(point)
                if job:active.append(job);ran+=1
                else:completed+=1
            for job in list(active):
                if job['process'].poll() is not None:finish(job);active.remove(job)
            if not active and (failed or budget_stop or not pending):break
            if time.monotonic()-last_report>=10:
                print(f'Progress: {completed} complete; {len(active)} running; {len(pending)} queued in this invocation',flush=True)
                last_report=time.monotonic()
            time.sleep(.2)
    finally:
        try:
            if active:terminate()
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(f'{completed} selected points complete, {ran} attempted now, {failed} failed/incomplete; old attempts preserved')
    if stopped[0] is not None:raise SystemExit(128+stopped[0])
    if failed:raise SystemExit(1)
    if budget_stop and pending:raise SystemExit(124)


def status(a):
    root=Path(a.directory).resolve();manifest=json.loads((root/'manifest.json').read_text())
    rows=[]
    for point in manifest['points']:
        record=done(root/point['path'])
        rows.append(dict(id=point['id'],state='converged' if record else 'pending/incomplete',
            TOTEN_eV=record['TOTEN_eV'] if record else None,wall_seconds=record.get('wall_seconds') if record else None,
            vasp_version=record.get('vasp_version') if record else None))
    complete=[r for r in rows if r['state']=='converged']
    median=float(np.median([r['wall_seconds'] for r in complete])) if complete else None
    print(json.dumps(dict(complete=len(complete),total=len(rows),median_seconds_per_point=median,
        estimated_remaining_serial_hours=(len(rows)-len(complete))*median/3600 if median else None,points=rows),indent=2))


def analyze(a):
    root=Path(a.directory).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['mode']!='paired_pwidom_energy_check':raise ValueError('Analysis requires paired production-size probes')
    out=Path(a.out)
    if out.exists():raise ValueError('Use a fresh analysis directory')
    pairs=manifest['pairs'];records={p['id']:done(root/p['path']) for p in manifest['points']}
    model_cache={};fingerprints={};rows=[]
    tolerance=getattr(a,"replay_tol",1e-3)
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError("Nonnegative finite replay tolerance required")
    point_sizes={p["id"]:p["natoms"] for p in manifest["points"]}
    for pair in pairs:
        ids=pair['points'];complete=all(records[i] for i in ids)
        row={k:v for k,v in pair.items() if k not in ('points',)}
        row.update(complete=complete,dU_DFT_eV=None,dU_MLP_eV=None,DFT_minus_MLP_eV=None,kmesh_shift_eV=None,
            E_DFT_without_H_eV=None,E_DFT_with_H_eV=None,E_MLP_without_H_eV=None,E_MLP_with_H_eV=None,
            dU_MLP_saved_eV=pair.get('dU_mlp_eV'),dU_MLP_recomputed_eV=None,GRACE_replay_difference_meV=None,
            GRACE_replay_matches=None,DFT_minus_MLP_without_H_meV_atom=None,DFT_minus_MLP_with_H_meV_atom=None)
        if complete:
            row['E_DFT_without_H_eV']=records[ids[0]]['TOTEN_eV']
            row['E_DFT_with_H_eV']=records[ids[1]]['TOTEN_eV']
            row['dU_DFT_eV']=row['E_DFT_with_H_eV']-row['E_DFT_without_H_eV']
            if pair.get('baseline_dU_DFT_eV') is not None:row['kmesh_shift_eV']=row['dU_DFT_eV']-pair['baseline_dU_DFT_eV']
            if a.compare_mlp:
                model=pair['model']
                if model not in fingerprints:fingerprints[model]=pw.model_fingerprint(model)
                if fingerprints[model]!=pair['model_fp']:raise ValueError('Saved GRACE Hamiltonian changed; refuse comparison')
                if model not in model_cache:model_cache[model]=pw.Model(model,a.device,a.batch)
                atoms=[read(root/i/'structure.extxyz') for i in ids]
                values=model_cache[model].energies(atoms)
                row['E_MLP_without_H_eV'],row['E_MLP_with_H_eV']=map(float,values)
                row['dU_MLP_recomputed_eV']=float(values[1]-values[0])
                row['dU_MLP_eV']=row['dU_MLP_recomputed_eV']
                if pair.get('dU_mlp_eV') is not None:
                    difference=row['dU_MLP_eV']-float(pair['dU_mlp_eV'])
                    row['GRACE_replay_difference_meV']=difference*1000
                    row['GRACE_replay_matches']=abs(difference)<=tolerance
                for identity,role in zip(ids,['without_H','with_H']):
                    row[f'DFT_minus_MLP_{role}_meV_atom']=(row[f'E_DFT_{role}_eV']-row[f'E_MLP_{role}_eV'])*1000/point_sizes[identity]
                print(f'AUDIT pair {pair["pair"]:03d}: E_VASP N/N+H={row["E_DFT_without_H_eV"]:.9f}/{row["E_DFT_with_H_eV"]:.9f}; '
                      f'E_GRACE N/N+H={values[0]:.9f}/{values[1]:.9f}; saved/recomputed ΔU={pair.get("dU_mlp_eV")}/{row["dU_MLP_eV"]:.9f}; '
                      f'match={row["GRACE_replay_matches"]}',flush=True)
            elif pair.get('dU_mlp_eV') is not None:
                row['dU_MLP_eV']=float(pair['dU_mlp_eV'])
            if row['dU_MLP_eV'] is not None:row['DFT_minus_MLP_eV']=row['dU_DFT_eV']-row['dU_MLP_eV']
        rows.append(row)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'paired_energies.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for host in sorted(set(r['host'] for r in rows)):
        for direction in ('insertion','deletion'):
            group=[r for r in rows if r['host']==host and r['direction']==direction and r['complete']]
            residuals=[r['DFT_minus_MLP_eV'] for r in group if r['DFT_minus_MLP_eV'] is not None]
            summary.append(dict(host=host,direction=direction,pairs_complete=len(group),
                mean_DFT_minus_MLP_eV=float(np.mean(residuals)) if residuals else None,
                rms_DFT_minus_MLP_eV=float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None))
            shifts=[r['kmesh_shift_eV'] for r in group if r['kmesh_shift_eV'] is not None]
            if shifts:summary[-1]['mean_kmesh_shift_eV']=float(np.mean(shifts))
    audited=[r for r in rows if r['dU_MLP_recomputed_eV'] is not None]
    differences=[abs(r['GRACE_replay_difference_meV']) for r in audited if r['GRACE_replay_difference_meV'] is not None]
    audit=dict(evaluated_pairs=len(audited),mismatched_pairs=sum(r['GRACE_replay_matches'] is False for r in audited),
               max_saved_recomputed_difference_meV=max(differences) if differences else None,tolerance_eV=tolerance,
               batch=a.batch if a.compare_mlp else None)
    if audit['mismatched_pairs']:print('WARNING: saved GRACE ΔU did not reproduce; inspect replay audit before interpreting model error',flush=True)
    save(out/'summary.json',dict(groups=summary,mlp_replay_audit=audit,note='Paired energy diagnostic on GRACE ensembles; not a DFT free energy or independent-sample uncertainty.'))
    usable=[r for r in rows if r['DFT_minus_MLP_eV'] is not None]
    if usable:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for host in sorted(set(r['host'] for r in usable)):
            for direction,marker in [('insertion','o'),('deletion','s')]:
                group=[r for r in usable if r['host']==host and r['direction']==direction]
                axes[0].scatter([r['dU_MLP_eV'] for r in group],[r['dU_DFT_eV'] for r in group],label=host+' '+direction,marker=marker)
                axes[1].scatter([r['phi_eV'] for r in group],[r['DFT_minus_MLP_eV'] for r in group],marker=marker)
        limits=axes[0].get_xlim();axes[0].plot(limits,limits,'--',color='.5')
        axes[0].set(xlabel='GRACE ΔU (eV)',ylabel='VASP ΔU (eV)');axes[0].legend(fontsize=7)
        axes[1].axhline(0,color='.5',lw=.8);axes[1].set(xlabel='Proxy insertion energy (eV)',ylabel='VASP − GRACE ΔU (eV)')
        fig.suptitle('Matched insertion/deletion energies — sampling/model check')
        for extension in ('png','pdf'):fig.savefig(out/f'paired_energies.{extension}',dpi=180)
        plt.close(fig)
    if getattr(a,'forces_stress',False):
        from tdi.vasp_force_check import compare
        compare(root,manifest,records,out,a.device)
    print(json.dumps(summary,indent=2));print(f'Wrote {out}; no chemical potential reconstructed from biased diagnostic sampling')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('discover');q.add_argument('--inventory',default='runs/inventory_binned.csv');q.add_argument('--limit',type=int,default=5)
    q=sub.add_parser('preflight');q.add_argument('--template',default='vasp_inputs');q.add_argument('--exe',default=os.environ.get('VASP_EXE','vasp_std'));q.add_argument('--train-outcar',type=Path)
    q=sub.add_parser('queue');q.add_argument('--queue',default='runs/dft_candidates');q.add_argument('--run');q.add_argument('--hosts',nargs='+')
    q=sub.add_parser('prep-queue');q.add_argument('--queue',default='runs/dft_candidates');q.add_argument('--run',required=True);q.add_argument('--hosts',nargs='+',default=['H2O','pureH']);q.add_argument('--template',default='vasp_inputs');q.add_argument('--pairs',type=int,default=1);q.add_argument('--out',required=True)
    q=sub.add_parser('mesh-check');q.add_argument('directory');q.add_argument('--hosts',nargs='+',default=['H2O','pureH']);q.add_argument('--pairs-per-host',type=int,default=1);q.add_argument('--mesh',nargs=3,type=int,default=[2,2,2]);q.add_argument('--out',required=True)
    q=sub.add_parser('fitprep');q.add_argument('dataset');q.add_argument('--base-train',default='runs/train_m.extxyz');q.add_argument('--base-valid',default='runs/valid_m.extxyz');q.add_argument('--out',required=True)
    q=sub.add_parser('dataset');q.add_argument('directories',nargs='+');q.add_argument('--seed',type=int,default=17);q.add_argument('--out',required=True)
    q=sub.add_parser('normal');q.add_argument('--run',required=True);q.add_argument('--template',default='vasp_inputs');q.add_argument('--hosts',nargs='+',default=['H2O','H2p083O','H2p25O','pureH']);q.add_argument('--per-host',type=int,default=10);q.add_argument('--discard-fraction',type=float,default=.5);q.add_argument('--device',default='cpu');q.add_argument('--out',required=True)
    q=sub.add_parser('baseline');q.add_argument('directory');q.add_argument('--model',required=True);q.add_argument('--device',default='cpu');q.add_argument('--out',required=True)
    q=sub.add_parser('training');q.add_argument('--data',nargs='+',default=['runs/train_m.extxyz','runs/valid_m.extxyz']);q.add_argument('--count',type=int,default=20);q.add_argument('--source-root',default='..');q.add_argument('--template');q.add_argument('--T',type=float);q.add_argument('--P',type=float);q.add_argument('--pressure-window',type=float,default=100);q.add_argument('--out',required=True)
    q=sub.add_parser('snapshot');q.add_argument('--frames',required=True);q.add_argument('--template',default='vasp_inputs');q.add_argument('--out',required=True);q.add_argument('--frame',type=int,default=-1);q.add_argument('--ensemble',choices=['host','full'],default='host')
    q=sub.add_parser('smoke');q.add_argument('--template',default='vasp_inputs');q.add_argument('--out',required=True);q.add_argument('--T',type=float,default=3000);q.add_argument('--vpa',type=float,default=2.2)
    q=sub.add_parser('prep');q.add_argument('--run',required=True);q.add_argument('--template',default='vasp_inputs');q.add_argument('--out',required=True)
    q.add_argument('--hosts',nargs='+',default=['H2O','pureH']);q.add_argument('--pairs',type=int,default=40);q.add_argument('--K',type=int,default=2000)
    q.add_argument('--discard-fraction',type=float,default=.5);q.add_argument('--seed',type=int,default=17)
    q=sub.add_parser('run');q.add_argument('directory');q.add_argument('--exe',default=os.environ.get('VASP_EXE','vasp_std'))
    q.add_argument('--launcher',default='srun');q.add_argument('--ranks',type=int,default=32);q.add_argument('--parallel',type=int,default=1,help='independent concurrent VASP points; ranks are per point');q.add_argument('--minutes',type=float,default=20)
    q.add_argument('--margin-seconds',type=float,default=60);q.add_argument('--max-points',type=int);q.add_argument('--index',type=int);q.add_argument('--dry-run',action='store_true')
    q=sub.add_parser('status');q.add_argument('directory')
    q=sub.add_parser('watch');q.add_argument('directory');q.add_argument('--interval',type=float,default=2.);q.add_argument('--once',action='store_true')
    q=sub.add_parser('analyze');q.add_argument('directory');q.add_argument('--out',required=True);q.add_argument('--compare-mlp',action='store_true',help='reevaluate both GRACE energies, even for saved pairs; use --batch 1 for an independent nonbatched audit');q.add_argument('--replay-tol',type=float,default=1e-3);q.add_argument('--forces-stress',action='store_true',help='reevaluate GRACE forces/stress against completed OUTCARs; no new VASP calls');q.add_argument('--device',default='cpu');q.add_argument('--batch',type=int,default=8)
    a=p.parse_args()
    if a.action in ('normal','baseline','dataset','fitprep'):
        from tdi import vasp_normal_check
        globals()[a.action]=getattr(vasp_normal_check,a.action)
    try:globals()[a.action.replace('-','_')](a)
    except (ValueError,OSError,KeyError) as exc:raise SystemExit(str(exc))


if __name__=='__main__':main()
