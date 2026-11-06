"""Check completed and failed farm reports without starting an optimization."""

import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_annual_grid_tracking_farms import write_farm_summary


def test_completed_report_uses_realized_metrics_and_auction_statuses(tmp_path):
    """Financial results and auction diagnostics come from the correct artifacts."""
    batch = tmp_path / "example"
    replay = batch / "runs/farm_1/farm_1_four_cycle_example/continuous"
    controller = replay / "controllers/track_committed_grid/json"
    controller.mkdir(parents=True)
    (replay / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (controller / "controller_kpis.json").write_text(json.dumps({
        "realized_operating_profit_eur": 123.5, "absolute_imbalance_mwh": 2.0,
        "soc_final_percent": 47.0, "mpc_solves": 0}))
    (controller / "solves.json").write_text(json.dumps([
        {"status": "optimal", "fallback_applied": False},
        {"status": "time_limit", "fallback_applied": True}]))
    provenance = {"days": {"2026-01-01": {"pv_provenance": {"curated_realized_pv": {
        "source_counts": {"metered_ac": 90, "pvlib_historical_ghi": 6}}}}}}
    (replay / "input_provenance.json").write_text(json.dumps(provenance))
    plan = {"start_date": "2026-01-01", "end_date": "2026-08-31", "models": {1: "model"}}
    report = write_farm_summary(batch, plan, 1, {"exit_code": 0}, {"exit_code": 0})
    text = Path(report).read_text()
    assert "123.500 EUR" in text
    assert "2.000 MWh" in text
    assert "47.000 %" in text
    assert "time_limit: 1" in text
    assert "Fallbacks: 1" in text
    assert "`pvlib_historical_ghi`: 6 intervals" in text


def test_failed_run_never_reports_partial_metrics(tmp_path):
    """A failed replay cannot be mistaken for a completed financial result."""
    batch = tmp_path / "example"
    replay = batch / "runs/farm_1/farm_1_four_cycle_example/continuous"
    replay.mkdir(parents=True)
    (replay / "manifest.json").write_text(json.dumps({"status": "failed"}))
    plan = {"start_date": "2026-01-01", "end_date": "2026-08-31", "models": {1: "model"}}
    report = write_farm_summary(batch, plan, 1, {"exit_code": 0}, {"exit_code": 1})
    text = Path(report).read_text()
    assert "Replay manifest status: failed" in text
    assert "No complete metrics are reported" in text
    assert "## Realized results" not in text
