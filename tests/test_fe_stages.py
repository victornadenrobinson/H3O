import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlp_pipeline", "tdi"))
pytest.importorskip("numba"); pytest.importorskip("scipy")
import fe_stages


def test_selftest(capsys):
    assert fe_stages.selftest() == 0
