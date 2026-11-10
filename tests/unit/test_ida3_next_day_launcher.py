"""Test the January study launcher without running a replay or preflight."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    """Load the standalone launcher from the repository scripts directory."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_ida3_next_day_january.py"
    spec = importlib.util.spec_from_file_location("ida3_january_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_tracker_only_and_never_starts_solves(launcher, capsys, tmp_path):
    """Planning all farms has no side effects and shows January plus context."""
    assert launcher.main(["--run-id", "ida3_test_plan_only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["controller"] == "track_committed_grid"
    assert plan["trajectory_count"] == 4
    assert plan["planning_context_through"] == "2026-02-01"
    assert plan["maximum_solver_threads"] == 24
    assert plan["midnight_soc_floor_percent"] == 40
    assert plan["solves_started"] == 0
    assert not Path(plan["output_dir"]).exists()


def test_configuration_selects_only_tracker(launcher, tmp_path):
    """Each farm has its own resolved assets, model, and shared study policy."""
    jobs = launcher.resolved_jobs(["1", "910"], tmp_path, 5)
    assert [job["config"]["farm_id"] for job in jobs] == [1, 910]
    for job in jobs:
        config = job["config"]
        assert config["execution"]["dispatch_policy"] == "track_committed_grid"
        assert config["realtime_mpc_comparison"]["controllers"] == ["track_committed_grid"]
        assert config["continuous_optimization"]["ida3_next_day_horizon"] is True
        assert config["state"]["terminal_soc_percent"] is None
        assert config["state"]["terminal_soc_min_percent"] == 40


def test_completed_batch_generates_and_links_both_reports(launcher, monkeypatch, tmp_path):
    """Successful batches expose the intraday analysis from every summary."""
    seen = []
    def comparison(batch, farms, sources):
        seen.append(("comparison", farms, sources))
        summary = batch / "summary.md"
        summary.write_text("# Summary\n")
        for farm in farms:
            (batch / f"farm_{farm}_summary.md").write_text("# Farm\n")
        return summary

    def intraday(batch, farms, sources):
        seen.append(("intraday", farms, sources))
        return batch / "intraday_analysis/summary.md"

    monkeypatch.setattr(launcher, "write_comparison", comparison)
    monkeypatch.setattr(launcher, "write_ida3_intraday_profiles", intraday)
    baselines = {"1": tmp_path / "old"}
    summary, path = launcher.generate_reports(tmp_path, ["1"], baselines)
    assert [item[0] for item in seen] == ["comparison", "intraday"]
    assert path == tmp_path / "intraday_analysis/summary.md"
    for report in (summary, tmp_path / "farm_1_summary.md"):
        assert "intraday_analysis/summary.md" in report.read_text()
