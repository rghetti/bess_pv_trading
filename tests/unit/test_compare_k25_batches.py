"""Verify matched-batch reporting without running historical experiments."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def comparison_module():
    """Load the standalone reporting tool without invoking its CLI."""
    path = Path(__file__).resolve().parents[2] / "scripts/compare_k25_batches.py"
    spec = importlib.util.spec_from_file_location("compare_k25_batches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(root, *, difference=0, fingerprint="same", status="complete"):
    """Create one complete, two-controller batch with two matched days."""
    job = {"job_id": "sample", "farm_id": "1", "window_id": "window", "case_id": "original",
           "controllers": ["track_committed_grid", "deterministic_mpc"], "days": ["2026-01-01", "2026-01-02"]}
    path = root / "runs/farm_1/sample/continuous"
    path.mkdir(parents=True)
    (path / "resolved_config.json").write_text(json.dumps({"state": {"terminal_soc_percent": 50}}))
    results = [{"job_id": "sample", "controller": controller, "input_fingerprint": fingerprint,
                "status": "complete"} for controller in job["controllers"]]
    (root / "manifest.json").write_text(json.dumps({"status": status, "plan": {"jobs": [job]}, "results": results}))
    rows = []
    for controller in job["controllers"]:
        for day in job["days"]:
            rows.append({"farm_id": "1", "window_id": "window", "case_id": "original", "controller": controller,
                "delivery_day": day, "absolute_imbalance_mwh": 1 + difference, "pv_curtailment_mwh": 2,
                "realized_operating_profit_eur": 10 + difference, "degradation_eur": 3,
                "mpc_fallbacks": 0, "soc_final_percent": 50 if difference else 7})
    pd.DataFrame(rows).to_csv(root / "daily_metrics.csv", index=False)


def test_comparison_reports_matched_totals_and_daily_soc(comparison_module, tmp_path):
    """Differences and target counts are computed separately for each controller."""
    baseline, candidate = tmp_path / "old", tmp_path / "new"
    make_batch(baseline)
    make_batch(candidate, difference=1)
    old, new, paired, summary = comparison_module.compare_batches(baseline, candidate)
    row = summary[summary.controller == "deterministic_mpc"].iloc[0]
    assert row.absolute_imbalance_mwh_old == pytest.approx(2)
    assert row.absolute_imbalance_mwh_new == pytest.approx(4)
    assert row.absolute_imbalance_mwh_change == pytest.approx(2)
    assert row.days_at_target_old == 0
    assert row.days_at_target_new == 2
    output = tmp_path / "report"
    comparison_module.write_comparison(old, new, paired, summary, output, 0.01)
    assert "full-policy differences" in (output / "comparison.md").read_text()
    assert (output / "farm_1_daily_soc.png").is_file()
    with pytest.raises(FileExistsError):
        comparison_module.write_comparison(old, new, paired, summary, output, 0.01)


def test_comparison_rejects_changed_input_snapshot(comparison_module, tmp_path):
    """A changed input fingerprint cannot produce a misleading matched report."""
    baseline, candidate = tmp_path / "old", tmp_path / "new"
    make_batch(baseline)
    make_batch(candidate, fingerprint="different")
    with pytest.raises(ValueError, match="fingerprint"):
        comparison_module.compare_batches(baseline, candidate)


def test_comparison_rejects_missing_day_and_incomplete_batch(comparison_module, tmp_path):
    """Missing observations or running batches are reported before any output is made."""
    baseline, candidate = tmp_path / "old", tmp_path / "new"
    make_batch(baseline)
    make_batch(candidate)
    metrics = pd.read_csv(candidate / "daily_metrics.csv")
    metrics.iloc[:-1].to_csv(candidate / "daily_metrics.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete delivery days"):
        comparison_module.compare_batches(baseline, candidate)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        comparison_module.compare_batches(baseline, candidate)
