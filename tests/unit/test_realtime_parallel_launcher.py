"""The multi-farm launcher isolates paths and native thread budgets."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(monkeypatch):
    """Import the direct script through its normal sibling-script search path."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_realtime_mpc_may_2026_all_farms")


def test_worker_uses_own_repository_from_an_unrelated_directory(launcher, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prepared = object()

    def prepare(config, **kwargs):
        assert Path(config["_project_root"]) == ROOT
        return prepared

    def run(config, *, prepared_inputs, **kwargs):
        assert prepared_inputs is prepared
        assert config["realtime_mpc_comparison"]["check_only"]
        return {"status": "preflight_passed"}

    monkeypatch.setattr(launcher, "prepare_continuous_inputs", prepare)
    monkeypatch.setattr(launcher, "run_realtime_mpc_experiment", run)
    result = launcher._run_comparison_job((901, "continuous", True, str(tmp_path)))
    assert result["status"] == "preflight_passed"


def test_invalid_native_threads_fail_before_creating_outputs(launcher, tmp_path):
    destination = tmp_path / "uncreated"
    with pytest.raises(SystemExit):
        launcher.main(["--check-only", "--native-threads", "0", "--output-dir", str(destination)])
    assert not destination.exists()


def test_native_pools_are_limited_in_a_fresh_process():
    """Inspect actual pools without loading farm data or running inference."""
    code = """
from run_realtime_mpc_may_2026_all_farms import _configure_worker_threads
_configure_worker_threads(1)
import torch
from threadpoolctl import threadpool_info
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
assert all(pool['num_threads'] == 1 for pool in threadpool_info())
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn'),
        initializer=_configure_worker_threads, initargs=(1,)) as pool:
    pending = [pool.submit(torch.get_num_threads) for _ in range(2)]
    assert [future.result() for future in pending] == [1, 1]
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT / "scripts", check=True, timeout=60)


def test_online_crps_rejects_zero_workers_without_starting_a_study():
    """Zero must not silently become the configured 42-worker default."""
    result = subprocess.run([sys.executable, str(ROOT / "scripts/study_pv_online_crps.py"),
        "--config", str(ROOT / "configs/pv_experiments/online_crps_january_august_2026.yaml"),
        "--plan-only", "--workers", "0"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "workers must be positive" in result.stderr
