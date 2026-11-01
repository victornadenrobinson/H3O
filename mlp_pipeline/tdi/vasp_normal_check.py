"""Ordinary cached MD configurations and GRACE comparisons to original DFT labels."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from tdi import dft_check as dft
from tdi import pwidom as pw
from tdi.vasp_force_check import metrics


def canonical_geometry_id(at):
    from tdi.pwidom_candidates import geometry_id
    copy=at.copy();copy.wrap()
    positions=np.round(copy.positions,8)
    order=np.lexsort((positions[:,2],positions[:,1],positions[:,0],copy.numbers))
    return geometry_id(copy[order])


def normal(a):
    from tdi import perlmutter_vasp_check as check
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a fresh output directory')
    if a.per_host<1 or not 0<=a.discard_fraction<1:raise ValueError('Invalid count/discard fraction')
    root=Path(a.run).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='complete':raise ValueError('Select a completed PWidom run')
    jobs={j['name']:j for j in manifest['jobs']}
    tp=check.validated_template(a.template);selected=[]
    for host in a.hosts:
        if host not in jobs:raise ValueError(f'{host} absent from run')
        path=Path(jobs[host]['result'])
        if not path.is_file():path=root/path.name
        result=json.loads(path.read_text());cache=root/(result['tag']+'_frames.npz')
        model=result['model']
        if pw.model_fingerprint(model)!=result['model_fp']:raise ValueError('Source GRACE model fingerprint changed')
        with np.load(cache,allow_pickle=False) as z:
            if str(z['model_fp'])!=result['model_fp'] or float(z['T'])!=result['T']:raise ValueError('Cache/result mismatch')
            cell=z['cell'];numbers=z['numbers'];frames=z['host']
            if not np.isclose(abs(np.linalg.det(cell)),float(z['V']),rtol=1e-7):raise ValueError('Invalid cache cell/volume')
            start=int(np.ceil(len(frames)*a.discard_fraction))
            if len(frames)-start<a.per_host:raise ValueError(f'{host}: insufficient distinct retained frames')
            for frame in np.linspace(start,len(frames)-1,a.per_host).round().astype(int):
                at=Atoms(numbers=numbers,positions=frames[frame],cell=cell,pbc=True)
                selected.append((host,int(frame),at,result,cache))
    out.mkdir(parents=True);points=[];calculators={}
    for i,(host,frame,at,result,cache) in enumerate(selected):
        identity=f'normal_{i:03d}_{host}';point=out/identity
        dft.write_dft(str(point),at,result['T'],tp,'Unmodified cached host MD snapshot; no insertion/deletion')
        with (point/'INCAR').open('a') as f:f.write('ISIF = 2\n')
        prepared=read(point/'structure.extxyz');model=result['model']
        if model not in calculators:calculators[model]=pw.Model(model,a.device,1).calc
        prepared.calc=calculators[model]
        energy=float(prepared.get_potential_energy());forces=prepared.get_forces();stress=prepared.get_stress()
        if not all(np.isfinite(v).all() for v in (energy,forces,stress)):raise ValueError('Nonfinite GRACE predictions')
        np.savez(point/'reference.npz',energy=energy,forces=forces,stress=stress)
        points.append(dict(id=identity,path=identity,natoms=len(at),input_sha256=check.input_hash(point),
            reference_sha256=hashlib.sha256((point/'reference.npz').read_bytes()).hexdigest(),reference_kind='GRACE',
            host=host,frame=frame,source_frames=str(cache),T_K=result['T'],P_target_GPa=result['P_target'],
            model=model,model_fp=result['model_fp']))
        print(f'PREPARED {identity}: frame {frame}, T={result["T"]} K, E_GRACE={energy:.9f} eV',flush=True)
    check.save(out/'manifest.json',dict(schema=1,mode='normal_MD_check',run=str(root),template=str(Path(a.template).resolve()),
        discard_fraction=a.discard_fraction,pairs=[],points=points,
        note='Evenly spaced unmodified host frames after discard; equilibration and statistical independence are not established.'))
    print(f'Prepared {len(points)} ordinary host snapshots; no MD or VASP executed',flush=True)


def baseline(a):
    from tdi import perlmutter_vasp_check as check
    root=Path(a.directory).resolve();out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a fresh comparison directory')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('mode')!='training_label_check':raise ValueError('baseline requires original labelled training check directory')
    model=pw.Model(a.model,a.device,1);rows=[]
    out.mkdir(parents=True)
    for point in manifest['points']:
        directory=root/point['path'];reference=directory/'reference.npz'
        if check.input_hash(directory)!=point['input_sha256']:raise ValueError('Prepared inputs changed')
        if hashlib.sha256(reference.read_bytes()).hexdigest()!=point['reference_sha256']:raise ValueError('Reference labels changed')
        at=read(directory/'structure.extxyz');at.calc=model.calc
        energy=float(at.get_potential_energy());forces=at.get_forces();stress=at.get_stress()
        with np.load(reference,allow_pickle=False) as ref:
            delta=energy-float(ref['energy'])
            row=dict(id=point['id'],source=point['source'],frame=point['frame'],T_K=point['T_K'],natoms=len(at),
                E_DFT_eV=float(ref['energy']),E_GRACE_eV=energy,GRACE_minus_DFT_meV_atom=delta*1000/len(at))
            if 'stress' in ref:row.update(metrics(at.numbers,ref['forces'],forces,ref['stress'],stress))
            else:row['force_component_RMSE_eV_A']=float(np.sqrt(np.mean((ref['forces']-forces)**2)))
        rows.append(row)
        print(f'BASELINE {point["id"]}: GRACE−DFT={delta*1000/len(at):+.3f} meV/atom; force RMSE={row["force_component_RMSE_eV_A"]:.4f} eV/Å',flush=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'baseline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    errors=np.array([r['GRACE_minus_DFT_meV_atom'] for r in rows])
    check.save(out/'summary.json',dict(model=a.model,model_fp=pw.model_fingerprint(a.model),structures=len(rows),
        energy_RMSE_meV_atom=float(np.sqrt(np.mean(errors**2))),energy_bias_meV_atom=float(np.mean(errors)),
        note='Selected training and validation sources combined: diagnostic baseline, not held-out model validation.'))


def dataset(a):
    """Export converged VASP labels; keep source-frame pairs in one split."""
    from ase.io import write
    from ase.calculators.singlepoint import SinglePointCalculator
    from tdi import perlmutter_vasp_check as check
    geometry_id=canonical_geometry_id
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a fresh dataset directory')
    groups={};records=[];seen={};parent={}
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key:parent[key]=find(parent[key])
        return parent[key]
    def union(left,right):parent[find(right)]=find(left)
    for name in a.directories:
        root=Path(name).resolve();manifest=json.loads((root/'manifest.json').read_text())
        paired={identity:pair for pair in manifest.get('pairs',[]) for identity in pair['points']}
        for point in manifest['points']:
            directory=root/point['path'];record=check.done(directory)
            if record is None:raise ValueError(f'{directory}: unfinished point; finish queue before freezing dataset')
            pair=paired.get(point['id'])
            if pair:
                if not all(check.done(root/identity) for identity in pair['points']):raise ValueError('Incomplete pair')
                group=f"{pair['source_frames']}:{pair['frame']}"
            else:group=f"{point.get('source_frames',point.get('source',str(root)))}:{point.get('frame',point['id'])}"
            at=read(directory/record['attempt']/'OUTCAR',index=-1,format='vasp-out')
            expected=read(directory/'structure.extxyz')
            from ase.geometry import find_mic
            if not np.array_equal(at.numbers,expected.numbers) or not np.allclose(at.cell,expected.cell,atol=1e-4,rtol=0):raise ValueError('OUTCAR geometry/order mismatch')
            if find_mic(at.positions-expected.positions,expected.cell,pbc=True)[1].max()>1e-4:raise ValueError('OUTCAR position mismatch')
            forces=at.get_forces();stress=at.get_stress();energy=record['TOTEN_eV']
            if not all(np.isfinite(v).all() for v in (energy,forces,stress)):raise ValueError('Nonfinite labels')
            at.calc=SinglePointCalculator(at,energy=energy,free_energy=energy,forces=forces,stress=stress)
            at.info=dict(source=group,config_type=manifest['mode'],label_origin=str(directory/record['attempt']/'OUTCAR'))
            identity=geometry_id(at);find(group)
            if identity in seen:
                previous=seen[identity]
                if abs(previous['energy']-energy)>1e-4:raise ValueError('Duplicate geometry has conflicting energies; check settings/k-mesh')
                union(group,previous['group']);continue
            item=dict(at=at,group=group,energy=energy,identity=identity)
            records.append(item);seen[identity]=item
    for item in records:groups.setdefault(find(item['group']),[]).append(item)
    keys=sorted(groups);np.random.default_rng(a.seed).shuffle(keys)
    if len(keys)<5:raise ValueError('Need at least five source-frame groups for train/validation/test')
    size=max(1,int(round(len(keys)*.2)))
    splits={'test':keys[:size],'valid':keys[size:2*size],'train':keys[2*size:]}
    out.mkdir(parents=True);summary={}
    for split,keys in splits.items():
        items=[item for key in keys for item in groups[key]]
        write(out/f'{split}.extxyz',[item['at'] for item in items])
        summary[split]=dict(groups=len(keys),structures=len(items),geometry_ids=[item['identity'] for item in items])
    check.save(out/'split_manifest.json',dict(seed=a.seed,inputs=a.directories,splits=summary,
        note='New DFT labels only. Linked pairs and duplicate geometries share a split. Adjacent MD frames may remain correlated. Keep test untouched; combine new train/valid with corresponding original datasets before fitting.'))
    print(json.dumps({key:{k:v for k,v in value.items() if k!='geometry_ids'} for key,value in summary.items()},indent=2))


def fitprep(a):
    """Combine original and new labels, retaining validation/test geometries."""
    from ase.io import iread,write
    from tdi import perlmutter_vasp_check as check
    geometry_id=canonical_geometry_id
    out=Path(a.out).resolve();dataset=Path(a.dataset)
    if out.exists():raise ValueError('Use a fresh fit directory')
    test=list(iread(dataset/'test.extxyz',index=':'))
    valid=list(iread(a.base_valid,index=':'))+list(iread(dataset/'valid.extxyz',index=':'))
    test_ids={geometry_id(at) for at in test};valid_ids={geometry_id(at) for at in valid}
    if test_ids&valid_ids:raise ValueError('Original validation overlaps new test geometries; revise split before fitting')
    heldout=test_ids|valid_ids;seen=set();train=[];valid_unique=[]
    for at in valid:
        key=geometry_id(at)
        if key not in seen:valid_unique.append(at);seen.add(key)
    removed=0;seen=set()
    for filename in (a.base_train,dataset/'train.extxyz'):
        for at in iread(filename,index=':'):
            key=geometry_id(at)
            if key in heldout:removed+=1;continue
            if key in seen:continue
            seen.add(key);train.append(at)
    if not train or not valid_unique or not test:raise ValueError('Empty fit/validation/test split')
    out.mkdir(parents=True)
    for name,frames in [('train',train),('valid',valid_unique),('test',test)]:write(out/f'{name}.extxyz',frames)
    check.save(out/'fit_manifest.json',dict(base_train=a.base_train,base_valid=a.base_valid,new_dataset=str(dataset),
        train=len(train),valid=len(valid_unique),test=len(test),removed_training_overlaps=removed,
        note='Geometry-disjoint splits; no new hyperparameter tuning on test. No fitting executed. Source/time correlations require separate assessment.'))
    print(f'Ready for conversion: train={len(train)}, valid={len(valid_unique)}, untouched test={len(test)}; removed train overlaps={removed}')


def timing(a):
    import time
    root=Path(a.directory);manifest=json.loads((root/'manifest.json').read_text())
    if a.repeats<1 or a.count<1:raise ValueError('Positive repeats/count required')
    started=time.perf_counter();model=pw.Model(a.model,a.device,1);startup=time.perf_counter()-started
    selected=manifest['points'][::max(1,len(manifest['points'])//a.count)][:a.count]
    rows=[]
    for point in selected:
        at=read(root/point['path']/'structure.extxyz');durations=[]
        for iteration in range(a.repeats+1):
            fresh=at.copy();model.calc.reset();fresh.calc=model.calc;t0=time.perf_counter()
            energy=float(fresh.get_potential_energy());fresh.get_forces();fresh.get_stress()
            durations.append(time.perf_counter()-t0)
        row=dict(point=point['id'],natoms=len(at),first_seconds=durations[0],warm_median_seconds=float(np.median(durations[1:])))
        rows.append(row);print(f"TIMING {row['point']} {len(at)} atoms: first={durations[0]:.3f}s; warm median E/F/stress={row['warm_median_seconds']:.3f}s",flush=True)
    from tdi import perlmutter_vasp_check as check
    out=Path(a.out)
    if out.exists():raise ValueError('Use a fresh timing output file')
    check.save(out,dict(model=a.model,device=a.device,model_startup_seconds=startup,rows=rows,
        note='One discarded first call per geometry; repeat medians include energy/forces/static stress. Explicit calculator reset prevents ASE result-cache timing.'))
