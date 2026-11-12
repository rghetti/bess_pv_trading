"""Check the independent cycle-counting report on known synthetic trajectories."""
import importlib.util
import json
from pathlib import Path

import pytest
import pandas as pd

from bess_pv_trading.domain import NaumannSettings
from bess_pv_trading.optimization.naumann import cycle_rate_factor, depth_exposure


# Load the standalone report without invoking its command-line entry point.
path = Path(__file__).resolve().parents[2] / "scripts/report_naumann_resolution.py"
spec = importlib.util.spec_from_file_location("resolution_report", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_nested_cycles_and_unfinished_half_cycle():
    """Nested 20% and 60% cycles plus one 10% half-cycle are preserved."""
    settings = NaumannSettings(depth_knots=(0, .05, .2, .5, 1))
    result = report.rainflow_benchmark([.2, .8, .4, .6, .4, .2, .3], 1., settings, .5)
    expected = (depth_exposure(.2, settings) + depth_exposure(.6, settings)
                + .5 * depth_exposure(.1, settings)) * cycle_rate_factor(.5, settings) ** 2
    assert result["equivalent_full_cycles"] == pytest.approx(.85)
    assert result["rainflow_analytic_cycle_exposure"] == pytest.approx(expected)


def test_out_of_capacity_path_is_rejected():
    """An invalid trajectory must not be silently clipped before counting."""
    with pytest.raises(ValueError, match="outside"):
        report.rainflow_benchmark([.5, 1.1], 1., NaumannSettings(), .5)


def test_fine_depth_graph_agrees_with_analytical_stress():
    """Finer post-processing approaches analytical loss on the same cycles."""
    settings = NaumannSettings(depth_knots=(0, .2, 1))
    result = report.rainflow_benchmark([.1, .47, .1], 1., settings, .5, 100)
    assert result["rainflow_fine_segments"] == 100
    assert result["rainflow_fine_cycle_loss_fraction"] == pytest.approx(
        result["rainflow_analytic_cycle_loss_fraction"], abs=1e-12)
    with pytest.raises(ValueError, match="at least one"):
        report.rainflow_benchmark([.1, .47], 1., settings, .5, 0)


def test_local_rate_replay_removes_only_upper_bin_factor():
    """A rate inside a bin is reweighted while an edge is unchanged."""
    settings = NaumannSettings(c_rate_bin_width=.25)
    first = float(cycle_rate_factor(.25, settings) ** 2)
    second = float(cycle_rate_factor(.5, settings) ** 2)
    frame = pd.DataFrame({
        "naumann_cycle_exposure_start": [0., first],
        "naumann_cycle_exposure_end": [first, first + second],
        "charge_c_rate_nominal": [.1, .5],
        "discharge_c_rate_nominal": [0., 0.],
    })
    expected = float(cycle_rate_factor(.1, settings) ** 2) + second
    assert report.unbinned_local_rate_exposure(frame, settings) == pytest.approx(expected)


def test_failed_run_reports_attempts_without_claiming_health(tmp_path):
    """Keep failed solves visible without treating partial dispatch as complete."""
    (tmp_path / "json").mkdir()
    (tmp_path / "json/manifest.json").write_text(json.dumps({"status": "failed", "error": "validation"}))
    attempt_dir = tmp_path / "strategies/integrated/json"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "optimization_attempts.json").write_text(json.dumps([
        {"status": "optimal", "runtime_seconds": 5, "max_violation": 1.5e-6,
         "binary_variables": 30, "validation": {"valid": False, "violations": ["depth dose"]}}
    ]))
    result = report.summarize_run(tmp_path, .5)
    assert result["validation_failures"] == 1
    assert result["solver_seconds_total"] == 5
    assert result["largest_solver_violation"] == 1.5e-6
    assert result["maximum_binary_variables"] == 30
    assert result["last_validation_failure"] == "depth dose"
    assert "final_soh" not in result
    assert report.summarize_run(tmp_path / "missing", .5)["run_status"] == "not_started_or_missing"


def test_incomplete_sequential_run_uses_its_actual_strategy_path(tmp_path):
    """Read the sequential strategy rather than misreporting an empty run."""
    (tmp_path / "json").mkdir()
    (tmp_path / "json/manifest.json").write_text(json.dumps({"status": "failed"}))
    (tmp_path / "json/study_config.json").write_text(json.dumps({
        "continuous_optimization": {"planning_modes": ["current_auction_only"]}}))
    attempt_dir = tmp_path / "strategies/current_auction_only/json"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "optimization_attempts.json").write_text(json.dumps([
        {"status": "optimal", "runtime_seconds": 3}
    ]))
    result = report.summarize_run(tmp_path, .5)
    assert result["planning_mode"] == "current_auction_only"
    assert result["auction_attempts"] == 1


def test_schedule_comparison_aligns_intervals_and_integrates_power(tmp_path):
    """Compare matched positions and compute the energy implied by power differences."""
    paths = [tmp_path / "first", tmp_path / "reference"]
    for index, run in enumerate(paths):
        (run / "json").mkdir(parents=True)
        (run / "json/manifest.json").write_text(json.dumps({"status": "complete"}))
        (run / "json/study_config.json").write_text(json.dumps({
            "continuous_optimization": {"planning_modes": ["current_auction_only"]}}))
        output = run / "strategies/current_auction_only/csv"
        output.mkdir(parents=True)
        pd.DataFrame({"start": ["2026-01-01T00:00Z"], "end": ["2026-01-01T00:15Z"],
                      "duration_hours": [.25], "net_position_mwh": [index * .1],
                      "grid_target_mw": [index * .4], "charge_mw": [index * .2],
                      "discharge_mw": [0.], "energy_end_mwh": [index * .05]}).to_csv(
                          output / "realized_dispatch.csv", index=False)
    result = report.compare_schedules(*paths)
    assert result["mean_absolute_net_position_mwh_difference"] == pytest.approx(.1)
    assert result["total_absolute_grid_target_mw_difference_mwh"] == pytest.approx(.1)
