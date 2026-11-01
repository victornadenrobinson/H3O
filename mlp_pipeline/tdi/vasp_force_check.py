"""Matched static forces/stresses from completed VASP OUTCARs and GRACE.

ASE units/signs throughout: forces eV/A, tensile stress eV/A^3, Voigt order
xx yy zz yz xz xy. Report stress/pressure in GPa; no ionic NkBT/V term.
"""
from pathlib import Path
import csv
import json
import numpy as np
from ase.io import read
from ase.geometry import find_mic

from tdi import pwidom as pw
from tdi.pwidom_checkpoint import atomic_npz


def metrics(numbers,fd,fm,sd,sm):
    fd,fm,sd,sm=map(lambda x:np.asarray(x,float),(fd,fm,sd,sm))
    if (fd.shape!=(len(numbers),3) or fm.shape!=fd.shape or sd.shape!=(6,) or sm.shape!=(6,)
            or not all(np.isfinite(x).all() for x in (fd,fm,sd,sm))):raise ValueError('Invalid force/stress arrays')
    delta=fd-fm;conversion=pw.EV_A3_GPA
    result=dict(force_component_RMSE_eV_A=float(np.sqrt(np.mean(delta**2))),
                force_vector_RMSE_eV_A=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                max_force_vector_error_eV_A=float(np.linalg.norm(delta,axis=1).max()),
                stress_component_RMSE_GPa=float(np.sqrt(np.mean(((sd-sm)*conversion)**2))),
                P_DFT_config_GPa=float(-sd[:3].mean()*conversion),
                P_MLP_config_GPa=float(-sm[:3].mean()*conversion),
                P_DFT_minus_MLP_GPa=float(-(sd[:3]-sm[:3]).mean()*conversion))
    for z,label in ((1,'H'),(8,'O')):
        mask=np.asarray(numbers)==z
        result[f'force_{label}_component_RMSE_eV_A']=float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None
    return result


def compare(root,manifest,records,out,device='cpu'):
    from importlib import import_module
    root,out=Path(root),Path(out);calculators={};rows=[];missing=[]
    all_fd=[];all_fm=[];all_sd=[];all_sm=[];all_numbers=[];ids=[];offsets=[0]
    provenance={identity:pair for pair in manifest['pairs'] for identity in pair['points']}
    for point in manifest['points']:
        identity=point['id'];record=records[identity]
        if record is None:continue
        pair=provenance[identity];model=pair['model']
        if pw.model_fingerprint(model)!=pair['model_fp']:raise ValueError('Saved model fingerprint changed; cannot compare forces')
        at=read(root/identity/'structure.extxyz')
        try:
            target=read(root/identity/record['attempt']/'OUTCAR',index=-1,format='vasp-out')
            fd=target.get_forces(apply_constraint=False)
            sd=target.get_stress(voigt=True,apply_constraint=False,include_ideal_gas=False)
        except Exception as exc:
            missing.append(dict(point=identity,reason=f'{type(exc).__name__}: {exc}'));continue
        if not np.array_equal(at.numbers,target.numbers) or not np.allclose(at.cell.array,target.cell.array,atol=1e-4,rtol=0):
            raise ValueError(f'{identity}: OUTCAR species/order/cell does not match prepared geometry')
        _,distance=find_mic(target.positions-at.positions,at.cell,pbc=at.pbc)
        if distance.max()>1e-4:raise ValueError(f'{identity}: OUTCAR coordinates do not match the prepared single point')
        if model not in calculators:
            calculators[model]=import_module('04_benchmark').get_calc_spec(model,device)
        at.calc=calculators[model]
        fm=at.get_forces(apply_constraint=False)
        sm=at.get_stress(voigt=True,apply_constraint=False,include_ideal_gas=False)
        row=dict(point=identity,host=pair['host'],direction=pair['direction'],P_target_GPa=pair['P'],T_K=pair['T'],
                 natoms=len(at),**metrics(at.numbers,fd,fm,sd,sm))
        rows.append(row);ids.append(identity);offsets.append(offsets[-1]+len(at))
        all_fd.append(fd);all_fm.append(fm);all_sd.append(sd);all_sm.append(sm);all_numbers.append(at.numbers)
    if missing:
        with (out/'force_stress_unavailable.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['point','reason']);writer.writeheader();writer.writerows(missing)
    if not rows:
        print('No completed OUTCAR could supply matched force/stress data; see force_stress_unavailable.csv',flush=True)
        return rows
    with (out/'force_stress.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_npz(out/'force_stress_arrays.npz',point_ids=np.array(ids),offsets=np.array(offsets),
        numbers=np.concatenate(all_numbers),F_DFT_eV_A=np.concatenate(all_fd),F_MLP_eV_A=np.concatenate(all_fm),
        stress_DFT_GPa=np.asarray(all_sd)*pw.EV_A3_GPA,stress_MLP_GPa=np.asarray(all_sm)*pw.EV_A3_GPA,
        convention='ASE tensile stress; xx yy zz yz xz xy; no ionic kinetic contribution')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fd,fm=np.concatenate(all_fd),np.concatenate(all_fm);numbers=np.concatenate(all_numbers)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for z,label in ((1,'H'),(8,'O')):
        mask=numbers==z
        if mask.any():axes[0].scatter(fm[mask].ravel(),fd[mask].ravel(),s=3,alpha=.25,label=label,rasterized=True)
    limits=[min(fd.min(),fm.min()),max(fd.max(),fm.max())]
    axes[0].plot(limits,limits,'--',color='.4');axes[0].set(xlabel='GRACE force component (eV/Å)',ylabel='VASP force component (eV/Å)');axes[0].legend()
    sd,sm=np.asarray(all_sd)*pw.EV_A3_GPA,np.asarray(all_sm)*pw.EV_A3_GPA
    axes[1].scatter(sm.ravel(),sd.ravel(),s=12,alpha=.6)
    limits=[min(sd.min(),sm.min()),max(sd.max(),sm.max())]
    axes[1].plot(limits,limits,'--',color='.4');axes[1].set(xlabel='GRACE stress component (GPa)',ylabel='VASP stress component (GPa)')
    fig.suptitle('Same geometries · configurational forces and stress')
    for extension in ('png','pdf'):fig.savefig(out/f'force_stress.{extension}',dpi=180)
    plt.close(fig)
    summary=dict(structures=len(rows),unavailable=len(missing),
        force_component_RMSE_eV_A=float(np.sqrt(np.mean((fd-fm)**2))),
        note='Targeted queue selection; no independent-sample errors. GRACE reevaluated; no VASP/MD calls. Static stress excludes ionic NkBT/V.')
    (out/'force_stress_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Compared forces/stresses for {len(rows)} completed geometries; {len(missing)} unavailable',flush=True)
    return rows
