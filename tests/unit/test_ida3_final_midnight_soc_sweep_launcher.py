"""Exercise sweep planning and shared-input target isolation without experiments."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future

import pytest


@pytest.fixture
def launcher():
    """Load the standalone study script."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_ida3_final_midnight_soc_sweep.py"
    spec = importlib.util.spec_from_file_location("ida3_soc_sweep_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_enumerates_all_farms_and_targets_without_solving(launcher, capsys):
    """The configured search includes endpoint targets and seven queued workers."""
    assert launcher.main(["--run-id", "soc_sweep_unit_plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["soc_targets_percent"] == [7, *range(10, 91, 5), 93]
    assert plan["farms"] == ["1", "345", "901", "910"]
    assert plan["trajectory_count"] == 76
    assert plan["workers"] == 7
    assert plan["solver_threads_per_job"] == 2
    assert plan["maximum_solver_threads"] == 14
    assert "final next-day midnight equality only" in plan["soc_policy"]
    assert plan["optimization_count_upper_bound"] == 76 * 127
    assert plan["solves_started"] == 0
    assert not Path(plan["output_dir"]).exists()


def test_shared_preflight_gets_independent_target_specific_state(launcher, monkeypatch, tmp_path):
    """Each replay receives its own terminal target and truthful input fingerprint."""
    canonical = {"plant": SimpleNamespace(battery=SimpleNamespace(capacity_mwh=2.0)),
                 "terminal_energy_mwh": 1.0, "provenance": {
                     "asset_input_settings": {"state": {"terminal_soc_percent": 50}}, "weather": "unchanged"}}
    seen = []
    def shared(config, batch, prepare, progress):
        seen.append(config["state"]["terminal_soc_percent"])
        return deepcopy(canonical)

    monkeypatch.setattr(launcher, "prepare_shared", shared)
    monkeypatch.setattr(launcher, "compare_prepared_to_history", lambda *args: {"matched": True})
    monkeypatch.setattr(launcher, "preparation_key", lambda config: "same")
    configs = launcher.make_jobs(["1"], [7, 93], tmp_path, 2)
    assert all(job["config"]["continuous_optimization"]["ida3_enforce_intermediate_soc"] is False
               for job in configs)
    first, _, key1 = launcher.prepare_for_target(configs[0]["config"], tmp_path, tmp_path, None)
    second, _, key2 = launcher.prepare_for_target(configs[1]["config"], tmp_path, tmp_path, None)
    assert seen == [50.0, 50.0]
    assert key1 == key2 == "same"
    assert first["terminal_energy_mwh"] == pytest.approx(0.14)
    assert second["terminal_energy_mwh"] == pytest.approx(1.86)
    assert first["provenance"]["asset_input_settings"]["state"]["terminal_soc_percent"] == 7
    assert second["provenance"]["asset_input_settings"]["state"]["terminal_soc_percent"] == 93
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert canonical["provenance"]["asset_input_settings"]["state"]["terminal_soc_percent"] == 50


def test_targets_outside_battery_limits_are_rejected(launcher, tmp_path):
    """Do not schedule physically invalid SOC constraints."""
    with pytest.raises(ValueError, match="outside farm"):
        launcher.make_jobs(["1"], [6], tmp_path, 2)


def test_all_preflights_complete_before_sweep_and_worker_slots_are_seven(launcher, monkeypatch, tmp_path):
    """The runner never begins optimization until every selected farm passed."""
    events = []
    baseline = {}
    for farm in ("1", "345", "901", "910"):
        root = tmp_path / "baseline" / farm
        root.mkdir(parents=True)
        (root / "input_provenance__track_committed_grid.json").write_text("{}")
        baseline[farm] = root

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 7
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, job, batch, source, **kwargs):
            future = Future()
            stage = "preflight" if kwargs.get("check_only") else "run"
            events.append((stage, job["farm"], job["target_soc_percent"]))
            future.set_result({"farm": job["farm"], "target_soc_percent": job["target_soc_percent"],
                               "status": "preflight_passed" if stage == "preflight" else "complete"})
            return future

    import bess_pv_trading.reporting.ida3_final_midnight_soc_sweep as reporting
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "source_checksums", lambda: {})
    monkeypatch.setattr(launcher, "historical_paths", lambda: baseline)
    monkeypatch.setattr(launcher, "_artifact_requirements", lambda config: [])
    monkeypatch.setattr(launcher, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    monkeypatch.setattr(reporting, "write_soc_sweep_report", lambda batch, *args: batch / "summary.md")
    assert launcher.main(["--run-id", "queue_test", "--run"]) == 0
    assert [event[0] for event in events[:4]] == ["preflight"] * 4
    assert [event[0] for event in events[4:]] == ["run"] * 76


def test_failed_preflight_starts_no_optimization(launcher, monkeypatch, tmp_path):
    """A missing or invalid farm snapshot blocks the entire sweep."""
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "input_provenance__track_committed_grid.json").write_text("{}")
    submitted = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, job, batch, source, **kwargs):
            submitted.append(kwargs.get("check_only", False))
            future = Future()
            future.set_result({"farm": job["farm"], "target_soc_percent": job["target_soc_percent"],
                               "status": "failed", "error": "bad input"})
            return future

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "source_checksums", lambda: {})
    monkeypatch.setattr(launcher, "historical_paths", lambda: {"1": baseline})
    monkeypatch.setattr(launcher, "_artifact_requirements", lambda config: [])
    monkeypatch.setattr(launcher, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    assert launcher.main(["--run-id", "bad_preflight", "--farm-ids", "1", "--run"]) == 1
    assert submitted == [True]
    manifest = json.loads((tmp_path / "outputs/energy_research/bad_preflight/manifest.json").read_text())
    assert manifest["status"] == "failed_preflight"
    assert not manifest["results"]


def test_comparison_can_be_regenerated_without_a_run(launcher, monkeypatch, tmp_path):
    """A later baseline completion can be paired without replaying the sweep."""
    import bess_pv_trading.reporting.ida3_soc_sweep_comparison as reporting

    batch = tmp_path / "outputs/energy_research/new"
    batch.mkdir(parents=True)
    (batch / "manifest.json").write_text(json.dumps({
        "plan": {"farms": ["1"], "soc_targets_percent": [7, 10]},
        "results": [{"farm": "1", "target_soc_percent": 7, "status": "complete"}]}))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text("{}")
    seen = []
    def compare(*args):
        seen.append(args)
        return batch / "comparison_summary.md"

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "write_soc_sweep_comparison", compare)
    assert launcher.main(["--run-id", "new", "--baseline-batch", str(baseline), "--summarize"]) == 0
    assert seen[0][0] == batch
    assert seen[0][1] == baseline
    assert seen[0][3:] == (["1"], [7, 10])


def test_completed_sweep_invokes_matched_comparison(launcher, monkeypatch, tmp_path):
    """A run ending while its baseline exists generates and links the paired report."""
    import bess_pv_trading.reporting.ida3_final_midnight_soc_sweep as report
    import bess_pv_trading.reporting.ida3_soc_sweep_comparison as comparison

    prior = tmp_path / "historical"
    prior.mkdir()
    (prior / "input_provenance__track_committed_grid.json").write_text("{}")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text("{}")
    calls = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, job, batch, source, **kwargs):
            future = Future()
            future.set_result({"farm": job["farm"], "target_soc_percent": job["target_soc_percent"],
                               "status": "preflight_passed" if kwargs.get("check_only") else "complete"})
            return future

    def summarize(batch, *args):
        path = batch / "summary.md"
        path.write_text("# Complete\n")
        return path

    def compare(batch, *args):
        calls.append(args)
        return batch / "baseline_comparison/summary.md"

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "source_checksums", lambda: {})
    monkeypatch.setattr(launcher, "historical_paths", lambda: {"1": prior})
    monkeypatch.setattr(launcher, "_artifact_requirements", lambda config: [])
    monkeypatch.setattr(launcher, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    monkeypatch.setattr(report, "write_soc_sweep_report", summarize)
    monkeypatch.setattr(comparison, "write_soc_sweep_comparison", compare)
    assert launcher.main(["--run-id", "new", "--farm-ids", "1", "--baseline-batch", str(baseline), "--run"]) == 0
    batch = tmp_path / "outputs/energy_research/new"
    manifest = json.loads((batch / "manifest.json").read_text())
    assert calls[0][0] == baseline
    assert manifest["comparison_markdown"].endswith("baseline_comparison/summary.md")
    assert "baseline_comparison/summary.md" in (batch / "summary.md").read_text()
