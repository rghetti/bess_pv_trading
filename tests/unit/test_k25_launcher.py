"""The portable launcher runs both farms concurrently and propagates failures."""
import importlib.util
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """Load the executable as a module and isolate its output directories."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_k25_farms.py"
    spec = importlib.util.spec_from_file_location("k25_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


@pytest.mark.parametrize("failing_farm, expected", [(None, 0), (345, 1)])
def test_two_farms_launch_before_wait_and_failures_propagate(launcher, monkeypatch, failing_farm, expected):
    """Each farm owns a one-worker process, and either failure fails the launcher."""
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = len(calls) + 1
            self.farm = int(command[command.index("--farm") + 1])
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            assert command[command.index("--workers") + 1] == "1"
            assert "--solver-threads" not in command
            calls.append(self)
        def wait(self):
            assert len(calls) == 2
            return int(self.farm == failing_farm)
        def poll(self):
            return 0
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    assert launcher.main(["--study", "january-ensemble", "--run-id", "test", "--check-only"]) == expected
    assert [process.farm for process in calls] == [1, 345]
    assert all("--check-only" in process.command for process in calls)


@pytest.mark.parametrize("filename", [
    "ensemble_k25_january_2026.yaml",
    "deterministic_k25_jan_aug_2026.yaml",
    "deterministic_k25_full_year_2026.yaml",
])
def test_study_sets_five_solver_threads(filename):
    """The launcher uses the per-study setting for both planning and solves."""
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/research" / filename).read_text())
    assert config["solver"]["threads"] == 5
    assert config["energy_research_study"]["solver_threads"] == 5
