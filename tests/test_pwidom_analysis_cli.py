"""The README's GRACE-selected shell path works end-to-end on synthetic work."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_grace_selected_time_command_produces_all_outputs(tmp_path):
    root=tmp_path/'run';root.mkdir()
    jobs=[]
    for name,phase,nh,nat,energy in [('H2O','fcc',216,324,.2),('H2p25O','fcc',243,351,.3),('pureH','H',250,250,.4)]:
        frames=root/(name+'_frames.npz')
        np.savez(frames,P_host=np.full(10,500.),P_full=np.full(10,500.))
        work=root/(name+'_work.npz')
        np.savez(work,forward_work_eV=np.full((10,8),energy),reverse_work_eV=np.full((10,4),-energy),
                 host_frame_ids=np.arange(10),full_frame_ids=np.arange(10),source_frames=str(frames),
                 frame_interval_fs=20.,model_fp='test')
        result=dict(work_file=str(work),model_fp='test',T=3000.,P_target=500.,phase=phase,
                    ideal_meV=100.,mu_fixedV_meV=energy*1000+100,mu_ex_bar_err_meV=1.,
                    P_host_GPa=500.,P_host_err_GPa=0.,V=300.,B_GPa=1000.,K=2000,
                    N_H=nh,natoms=nat,melted=False)
        path=root/(name+'.json');path.write_text(json.dumps(result))
        jobs.append(dict(name=name,result=str(path),command=[]))
    (root/'manifest.json').write_text(json.dumps(dict(status='complete',jobs=jobs)))
    env=dict(os.environ,GRACE_ENV=sys.prefix)
    env.pop('PY',None)
    out=tmp_path/'analysis'
    command=['bash',str(ROOT/'mlp_pipeline/perlmutter_pwidom_analysis.sh'),'time',
             '--runs',str(root),'--out',str(out),'--step-ps','.1']
    completed=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert str(Path(sys.prefix)/'bin/python') in completed.stderr
    assert (out/'mu_G_dg_vs_time.pdf').stat().st_size>100
    assert (out/'hulls_coloured_by_time.pdf').stat().st_size>100
    assert (out/'prefix_input.json').exists()
    with (out/'time_series.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==4 and set(float(r['time_ps']) for r in rows)=={.1,.2}
    original=(out/'time_series.csv').read_bytes()
    repeated=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
    assert repeated.returncode!=0 and (out/'time_series.csv').read_bytes()==original
