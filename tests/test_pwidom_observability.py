"""Monitoring parsers and running estimators, with no GPU/model dependencies."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"mlp_pipeline"))
    monkeypatch.syspath_prepend(str(ROOT/"mlp_pipeline/tdi"))


def test_compact_transcript_keeps_real_capacity_and_partial_time(tmp_path):
    from pwidom_telemetry import load_samples, category
    text = tmp_path/"log.txt"
    text.write_text("Elapsed 17:03   State: running\n"
        "Hplus5     single points       █░░░░░░░  13%  [00:38/02:44 ETA] GPU0 4% 26.5/40.0 GiB\n"
        "Elapsed 17:33   State: running\n"
        "Hplus5     DONE                ████████ 100%  -- GPU0 0% 0.0/40.0 GiB\n")
    rows = load_samples(text)
    assert len(rows) == 2 and rows[0]["elapsed_seconds"] == 1023
    assert rows[0]["total_MiB"] == 40960 and rows[0]["memory_MiB"] == 27136
    assert category(rows[0]["stage"]) == "GRACE probes"
    assert category(rows[1]["stage"]) == "Finished"


def test_monitor_records_metrics_without_model_calls(monkeypatch,tmp_path):
    import perlmutter_pwidom as pm
    from pwidom_telemetry import load_samples
    monkeypatch.setattr(pm,"gpu_metrics",lambda jobs,raw=False:{"0":dict(
        utilization=60.,memory_MiB=2560.,total_MiB=40960.)})
    progress = tmp_path/"progress.json"
    progress.write_text(json.dumps(dict(stage="MD host sampling",fraction=.5)))
    manifest = dict(jobs=[dict(name="test",gpu="0",progress=str(progress),log=str(tmp_path/"test.log"))])
    pm.record_telemetry(manifest,12.,{"test":None})
    rows = load_samples(tmp_path)
    assert len(rows)==1 and rows[0]["fraction"] == .5
    assert rows[0]["memory_MiB"]==2560. and rows[0]["elapsed_seconds"]==12.


def test_constant_work_prefixes_have_exact_mu_and_counts(tmp_path):
    from pwidom_convergence import prefixes
    from pwidom_scan import mu_P, dg
    frames = tmp_path/"frames.npz"
    np.savez(frames,P_host=np.full(10,500.))
    work = tmp_path/"work.npz"
    np.savez(work,forward_work_eV=np.full((10,2),.2),reverse_work_eV=np.full((10,1),-.2),
        host_frame_ids=np.arange(10),full_frame_ids=np.arange(10),source_frames=str(frames),
        frame_interval_fs=20.,model_fp="test")
    result = dict(work_file=str(work),model_fp="test",T=3000.,P_target=500.,phase="fcc",
        ideal_meV=100.,mu_fixedV_meV=300.,mu_ex_bar_err_meV=0.,P_host_GPa=500.,
        P_host_err_GPa=0.,V=100.,B_GPa=100.,K=2000,N_H=100,natoms=100)
    reference = dict(result,phase="H")
    rows = prefixes(result,[],reference,points=6)
    assert rows[-1]["production_ps"] == pytest.approx(.2)
    assert rows[-1]["insertion_trials"]==20 and rows[-1]["candidate_sites"]==20000
    assert rows[-1]["kish_fraction"] == pytest.approx(1.)
    assert rows[-1]["kish_effective_count"] == pytest.approx(20.)
    for row in rows:
        assert row["mu_bar_meV"] == pytest.approx(mu_P(result)[0])
        assert row["mu_widom_meV"] == pytest.approx(row["mu_delete_meV"])
        assert row["dg_meV"] == pytest.approx(dg(result,reference)[0])
    with pytest.raises(ValueError,match="Reference P/T/model"):
        prefixes(result,[],dict(reference,T=4000.))
