from argparse import Namespace
import csv
import importlib.util
import json
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'mlp_pipeline/discover_legacy_tdi.py'
spec=importlib.util.spec_from_file_location('legacy',SCRIPT);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def test_original_inventory_and_cmc_siblings(tmp_path):
    campaign=tmp_path/'tdi/h3o/500gpa-3000K/fcc-h2o-108';rows=[]
    for lam in [0,.25,.5,.75,1]:
        directory=campaign/f'lambda_{lam}'/'out1';directory.mkdir(parents=True)
        outcar=directory/'OUTCAR';outcar.write_text('not read as a trajectory')
        rows.append(dict(Path=str(outcar.relative_to(tmp_path)),True_T_K=3000,True_P_GPa=500))
    cmc=campaign/'stage2';cmc.mkdir();(cmc/'mc.py').write_text('accept = min(1, exp(-beta * deltaU))\n')
    (cmc/'POTCAR').write_text('DO NOT COPY THIS')
    inventory=tmp_path/'mined.csv'
    with inventory.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=tmp_path/'audit'
    legacy.discover(Namespace(out=str(report),root=str(tmp_path),inventory=[str(inventory)],depth=4,max_files=100,max_text_bytes=1024,lines_per_file=40))
    result=json.loads((report/'discovery.json').read_text())
    assert len(result['campaigns'])==1
    record=result['campaigns'][0]
    assert record['lambda_values']==[0,.25,.5,.75,1]
    assert record['found_CMC_or_stage2']
    assert 'accept' in (report/'text_evidence.txt').read_text()
    assert 'DO NOT COPY' not in (report/'text_evidence.txt').read_text()
    assert 'POTCAR' not in (report/'files.csv').read_text()


def test_phase_candidate_and_window():
    assert legacy.classify('tdi/h3o/500gpa-7000K-solid/fcc-h2o/9.1/out1/OUTCAR.gz')[0]=='tdi/h3o/500gpa-7000K-solid/fcc-h2o'
    assert legacy.classify('tdi/h3o/500gpa-3000K/fcc/stage1/lambda_0.25/out1/OUTCAR')[0]=='tdi/h3o/500gpa-3000K/fcc'
    assert legacy.classify('tdi/H3O/5000K/bcc/9.1/OUTCAR') is None
