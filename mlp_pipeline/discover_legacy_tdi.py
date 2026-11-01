"""Read-only census of original DFT->pair->reference TDI/CMC campaigns.

Use the unfiltered mined inventory: training/EOS inventories exclude lambda_*.
Reads directory names and small text metadata only; never integrates unknown
observables or treats mixed-Hamiltonian OUTCARs as pure DFT training labels.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re

WINDOW=re.compile(r'^(?:lambda|lam)[_=-]([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)$',re.I)
STAGE=re.compile(r'^(?:stage|step)[_-]?[12]$|^cmc(?:[_-].*)?$|^classical(?:[_-].*)?$',re.I)
STATE=re.compile(r'\d+gpa[-_]\d+k',re.I)
TEXT_EXT={'.py','.sh','.job','.sbatch','.sge','.pbs','.f','.f90','.c','.cpp','.json','.yaml','.yml','.txt','.dat'}
EVIDENCE=re.compile(r'lambda|coupling|dudl|dhdl|dU.?d|integrand|trapez|simpson|gauss|spring|einstein|ideal.gas|\bcmc\b|monte.carlo|accept|mass|factorial|lgamma|hbar|planck|\bcom\b|centre.of.mass|center.of.mass',re.I)


def classify(path):
    parts=Path(path).parts
    for i,part in enumerate(parts):
        match=WINDOW.fullmatch(part)
        if match:
            campaign=Path(*parts[:i])
            if STAGE.fullmatch(campaign.name):campaign=campaign.parent
            return str(campaign),float(match[1]),'lambda_window'
    for i,part in enumerate(parts):
        if STAGE.fullmatch(part):return str(Path(*parts[:i])),None,'stage_or_CMC'
    for i,part in enumerate(parts):
        if STATE.search(part):
            # Retain phase/composition folder after the P/T campaign when present.
            end=i+2 if i+1<len(parts)-1 and not re.fullmatch(r'out\d+|[\d.]+',parts[i+1],re.I) else i+1
            return str(Path(*parts[:end])),None,'historical_PT_candidate'
    return None


def writecsv(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def discover(a):
    out=Path(a.out)
    if out.exists():raise ValueError('Use a fresh report directory; original calculations are never modified')
    root=Path(a.root).expanduser().resolve();windows=[];campaigns={};seen=set();used=[]
    for inventory in a.inventory:
        file=Path(inventory)
        if not file.exists():continue
        used.append(str(file.resolve()))
        with file.open(newline='') as f:
            reader=csv.DictReader(f)
            column=next((c for c in reader.fieldnames or [] if c.lower() in ('path','filename','file')),None)
            if column is None:raise ValueError(f'{inventory}: no Path/filename column')
            for row in reader:
                source=row[column];candidate=classify(source)
                if candidate is None:continue
                name,lam,reason=candidate
                path=Path(source).expanduser();path=(path if path.is_absolute() else root/path).resolve()
                if str(path) in seen:continue
                seen.add(str(path));directory=Path(name).expanduser();directory=(directory if directory.is_absolute() else root/directory).resolve()
                campaigns.setdefault(str(directory),dict(path=str(directory),exists=directory.is_dir(),inventory_rows=0,lambda_values=set(),reasons=set()))
                record=campaigns[str(directory)];record['inventory_rows']+=1;record['reasons'].add(reason)
                if lam is not None:record['lambda_values'].add(lam)
                windows.append(dict(campaign=str(directory),path=str(path),exists=path.exists(),lambda_from_path=lam,
                    reason=reason,T_inventory=row.get('True_T_K',row.get('T_bin','')),
                    P_inventory=row.get('True_P_GPa',row.get('P_bin','')),inventory=str(file)))
    if not used:raise ValueError('No inventory found; use the ORIGINAL mined inventory including lambda_* paths')
    out.mkdir(parents=True);files=[];snippets=[];reports=[]
    for directory,record in sorted(campaigns.items()):
        lambdas=sorted(record.pop('lambda_values'));record['lambda_values']=lambdas;record['reasons']=sorted(record['reasons'])
        record['window_count_unique_path_lambdas']=len(lambdas)
        # More than five values may reflect separate stages; do not pool/integrate.
        record['stage1_grid_status']='five values found; stage mapping unverified' if len(lambdas)==5 else 'needs stage mapping / missing or extra windows'
        record['found_CMC_or_stage2']=False;record['scan_truncated']=False
        base=Path(directory);count=0
        if base.is_dir():
            def bounded_files():
                for folder,dirs,names in os.walk(base):
                    dirs.sort()
                    if len(Path(folder).relative_to(base).parts)>=a.depth-1:dirs[:]=[]
                    for name in sorted(names):yield Path(folder)/name
            for path in bounded_files():
                if not path.is_file():continue
                if len(path.relative_to(base).parts)>a.depth:continue
                if path.name.upper().startswith(('POTCAR','WAVECAR','CHGCAR')):continue
                count+=1
                if count>a.max_files:record['scan_truncated']=True;break
                relative=str(path.relative_to(base));stage2=bool(re.search(r'cmc|stage[_-]?2|monte.?carlo',relative,re.I))
                record['found_CMC_or_stage2']|=stage2
                kind='CMC/stage2 candidate' if stage2 else ('code' if path.suffix in {'.py','.f','.f90','.c','.cpp','.sh'} else 'metadata/output')
                files.append(dict(campaign=directory,path=str(path),bytes=path.stat().st_size,kind=kind))
                if path.stat().st_size>a.max_text_bytes:continue
                if path.suffix.lower() not in TEXT_EXT and path.name.upper() not in ('INCAR','README','MAKEFILE','OSZICAR'):continue
                try:text=path.read_text(errors='replace')
                except OSError:continue
                matched=[(i,line) for i,line in enumerate(text.splitlines(),1) if EVIDENCE.search(line)]
                if path.name.upper()=='INCAR':matched=list(enumerate(text.splitlines(),1))
                if matched:
                    snippets.append('\nFILE '+str(path)+'\n'+'\n'.join(f'{i}: {line[:500]}' for i,line in matched[:a.lines_per_file]))
        reports.append(record)
    writecsv(out/'windows.csv',windows,['campaign','path','exists','lambda_from_path','reason','T_inventory','P_inventory','inventory'])
    writecsv(out/'files.csv',files,['campaign','path','bytes','kind'])
    (out/'text_evidence.txt').write_text('\n'.join(snippets)+'\n')
    result=dict(root=str(root),inventories=used,campaigns=reports,
        note='Discovery only. Path lambda is not proof of actual Hamiltonian. Read original scripts/headers and pair energy definitions before integration; CMC has no OUTCAR and may be absent from inventories. No VASP/MD/CMC executed.')
    (out/'discovery.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'README.md').write_text('''# Original DFT two-stage TDI/CMC discovery

Read discovery.json and windows.csv for campaign paths and apparent lambda grids.
files.csv lists sibling inputs, code and outputs; text_evidence.txt contains bounded
small-text excerpts. Licensed POTCAR and large electronic state files are excluded.
A “five values” flag is only a path census, not a valid quadrature/sign check.

Next audit requires original Hamiltonian U_lambda and dU/dlambda definitions,
DFT/pair endpoints, integrand units/normalization, actual lambda grids/weights,
equilibration and correlations, finite-temperature TOTEN, stages/energy-zero match,
CMC proposal/acceptance detailed balance, periodic images/cutoffs, oxygen springs
and COM constraints, analytic reference entropy/kinetic/combinatorial terms,
composition counts and pressure/PV conventions. Re-integrate each stage only once
these are known. Original external CMC implementation is not in this repo.
''')
    print(f'Found {len(reports)} candidate campaigns / {len(windows)} inventory paths. Report: {out}')
    for record in reports:
        print(f"{record['path']}: exists={record['exists']}, lambdas={record['lambda_values']}, CMC/stage2={record['found_CMC_or_stage2']}")
    if not any(r['lambda_values'] for r in reports):print('WARNING: no lambda_* paths found; the selected inventory may already exclude TDI windows. Use the unfiltered mined inventory or original inventory.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',nargs='+',default=['runs/mined_h3o.csv','runs/inventory_binned.csv'])
    parser.add_argument('--root',default='..');parser.add_argument('--out',required=True)
    parser.add_argument('--depth',type=int,default=4);parser.add_argument('--max-files',type=int,default=1500)
    parser.add_argument('--max-text-bytes',type=int,default=262144);parser.add_argument('--lines-per-file',type=int,default=40)
    a=parser.parse_args()
    try:discover(a)
    except (ValueError,OSError) as exc:raise SystemExit(str(exc))


if __name__=='__main__':main()
