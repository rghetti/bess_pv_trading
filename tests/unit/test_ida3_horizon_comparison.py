"""Verify that paired reports reject incomparable data and align delivery midnight."""
from pathlib import Path
import json

import pandas as pd
import pytest

from bess_pv_trading.reporting.ida3_horizon_comparison import _all_planned_midnights, _check_common_inputs, _midnight_records


def _fixture(root):
    """Create a complete January product and auction-price fixture."""
    output = root / "csv"
    output.mkdir(parents=True)
    starts = pd.date_range("2025-12-31T23:00:00Z", periods=31 * 96, freq="15min")
    products = [value.isoformat() for value in starts]
    days = starts.tz_convert("Europe/Rome").strftime("%Y-%m-%d")
    pd.DataFrame({"interval_id": products, "delivery_day": days, "pv_ac_mw": 0.5,
                  "imbalance_price_eur_mwh": 100.0}).to_csv(output / "realized_dispatch.csv", index=False)
    pd.DataFrame({"interval_id": products, "delivery_day": days, "auction": "DA",
                  "price_eur_per_mwh": 80.0}).to_csv(output / "accepted_trades.csv", index=False)


def test_common_inputs_validate_products_and_values(tmp_path):
    """Identical input prices pass; one changed settlement price fails."""
    old, new = tmp_path / "old", tmp_path / "new"
    _fixture(old)
    _fixture(new)
    assert _check_common_inputs(old, new) == {"realized_pv_and_settlement_products": 2976, "auction_products": 2976}
    path = new / "csv/realized_dispatch.csv"
    data = pd.read_csv(path)
    data.loc[0, "imbalance_price_eur_mwh"] += 1
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="imbalance_price"):
        _check_common_inputs(old, new)


def test_midnight_plan_ignores_future_day_end(tmp_path):
    """The first midnight of an extended IDA3 plan is matched to the actual day."""
    output = tmp_path / "csv"
    output.mkdir()
    plans = pd.DataFrame([
        {"delivery_day": "2026-01-01", "auction": "IDA3", "interval_end_utc": "2026-01-01T23:00:00Z", "soc_end_mwh": 0.6, "plan_status": "optimized"},
        {"delivery_day": "2026-01-01", "auction": "IDA3", "interval_end_utc": "2026-01-01T23:00:00Z", "soc_end_mwh": 0.8, "plan_status": "provisional_previous_day_ida3"},
        {"delivery_day": "2026-01-01", "auction": "IDA3", "interval_end_utc": "2026-01-02T23:00:00Z", "soc_end_mwh": 0.9, "plan_status": "optimized"},
    ])
    plans.to_csv(output / "expected_dispatch_plans.csv", index=False)
    daily = pd.DataFrame([{"delivery_day": "2026-01-01", "final_soc_percent": 45.0}])
    result = _midnight_records(tmp_path, daily, 1.0, "extended")
    assert result.iloc[0].ida3_planned_end_soc_percent == pytest.approx(60)
    assert result.iloc[0].final_soc_percent == 45


def test_all_horizon_midnights_include_unscored_context_without_actual(tmp_path):
    """Context midnight is reported with no fabricated actual state."""
    output = tmp_path / "json"
    output.mkdir()
    attempts = [{"delivery_day": "2026-01-31", "current_auction": "IDA3", "cutoff_utc": "2026-01-31T08:55:00Z",
        "planned_midnight_energy_mwh": {"2026-01-31T23:00:00+00:00": 0.5, "2026-02-01T23:00:00+00:00": 0.4}}]
    (output / "optimization_attempts.json").write_text(json.dumps(attempts))
    daily = pd.DataFrame([{"delivery_day": "2026-01-31", "final_soc_percent": 42.0}])
    rows = _all_planned_midnights(tmp_path, daily, 1.0, "1")
    assert rows[0]["actual_soc_percent"] == 42
    assert rows[1]["ending_delivery_day"] == "2026-02-01"
    assert rows[1]["actual_soc_percent"] is None
