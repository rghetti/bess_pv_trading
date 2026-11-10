"""Check SOC sweep reporting without historical simulations."""
import json

import pandas as pd
import pytest

from bess_pv_trading.reporting import ida3_final_midnight_soc_sweep as report


def test_soc_sweep_partial_results_are_labeled_and_unequal_inputs_rejected(tmp_path, monkeypatch):
    """Reports show incomplete targets and compare input products among successful runs."""
    summaries = {"net_profit_eur": 100.0, "absolute_imbalance_mwh": 2.0,
        "pv_realized_energy_mwh": 100.0, "signed_imbalance_mwh": -2.0,
        "curtailment_mwh": 3.0, "charge_mwh": 4.0, "discharge_mwh": 3.0,
        "auction_revenue_eur": 120.0, "imbalance_cashflow_eur": -10.0,
        "degradation_eur": 10.0, "import_fee_eur": 0.0, "transaction_fee_eur": 0.0,
        "final_soc_percent": 30.0, "mean_midnight_soc_percent": 40.0,
        "auction_fallbacks": 2}
    daily = pd.DataFrame({"initial_soc_percent": [50.0] * 31,
                          "final_soc_percent": [30.0] * 31})
    monkeypatch.setattr(report, "_daily", lambda root: daily)
    monkeypatch.setattr(report, "_metrics", lambda root, values: dict(summaries))
    monkeypatch.setattr(report, "_plot_farm", lambda frame, farm, path: path.write_bytes(b"fixture"))
    seen = []
    monkeypatch.setattr(report, "_check_common_inputs", lambda a, b: seen.append((a, b)) or {"products": 2976})
    jobs = []
    for target in (10, 20):
        directory, root = report._trajectory_root(tmp_path, "1", target)
        root.mkdir(parents=True)
        config = {"state": {"terminal_soc_percent": target, "terminal_soc_min_percent": None},
                  "execution": {"dispatch_policy": "track_committed_grid"},
                  "continuous_optimization": {"ida3_enforce_intermediate_soc": False}}
        (directory / "resolved_config.json").write_text(json.dumps(config))
        summary = {**summaries, "first_delivery_day": "2026-01-01", "last_delivery_day": "2026-01-31",
                   "delivery_days": 31, "components": {key: summaries[key] for key in (
                   "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur", "import_fee_eur",
                   "transaction_fee_eur")}}
        (directory / "summary.json").write_text(json.dumps(summary))
        jobs.append({"farm": "1", "target_soc_percent": target, "status": "complete"})
    path = report.write_soc_sweep_report(tmp_path, jobs, ["1"], [10, 20, 30])
    assert "Completed 2 of 3" in path.read_text()
    assert "Farm 1, 30%: not started" in path.read_text()
    farm = (tmp_path / "farm_1_summary.md").read_text()
    assert "without an intermediate target" in farm
    assert "Incomplete or failed targets (%): 30" in farm
    assert "final-inventory valuation" in farm
    assert "| 10 | 100.00 | 120.00 | -10.00 | 10.00 | 2.00 | 2.00 | 3.00 | 3.00 |" in farm
    assert len(seen) == 1
    result = pd.read_csv(tmp_path / "soc_sweep_results.csv")
    assert set(result.target_soc_percent) == {10, 20}
    assert result.initial_soc_percent.eq(50).all()
    with pytest.raises(ValueError, match="Duplicate"):
        report.write_soc_sweep_report(tmp_path, [jobs[0], jobs[0]], ["1"], [10])
