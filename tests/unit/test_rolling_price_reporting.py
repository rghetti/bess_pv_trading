"""Check matched-product comparisons and the reserved evaluation boundary."""

import json
import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.rolling_price_reporting import rolling_price_report


def test_matched_rows_and_reserved_november_are_explicit(tmp_path):
    """Different feature coverage must not silently change the paired score."""
    rows = [
        ("2026-10-01", "2026-10-01T00:00Z", "core", 10, 11, 8),
        ("2026-10-01", "2026-10-01T00:15Z", "core", 20, 22, 18),
        ("2026-10-01", "2026-10-01T00:00Z", "weather", 10, 10, 8),
        ("2026-11-01", "2026-11-01T00:00Z", "core", 1000, 0, 8),
    ]
    frame = pd.DataFrame(rows, columns=["delivery_day", "product_start", "variant", "actual", "prediction", "reference"])
    frame["auction"] = "IDA2"
    frame["gate"] = "IDA2"
    report = rolling_price_report(frame, tmp_path)
    pair = pd.read_csv(tmp_path / "paired_variants.csv").iloc[0]
    assert report["reserved_rows_not_scored"] == 1
    assert pair["matched_rows"] == 1
    assert pair["excluded_a"] == 1
    assert pair["mae_a"] == 1
    assert pair["mae_b"] == 0
    assert json.loads((tmp_path / "report.json").read_text())["scored_input_rows"] == 3


def test_paired_labels_must_agree(tmp_path):
    """Refuse to compare forecasts made against different actual prices."""
    frame = pd.DataFrame([
        {"delivery_day": "2026-10-01", "product_start": "2026-10-01T00:00Z", "auction": "DA",
         "gate": "DA", "variant": "a", "actual": 10, "prediction": 9, "reference": 8},
        {"delivery_day": "2026-10-01", "product_start": "2026-10-01T00:00Z", "auction": "DA",
         "gate": "DA", "variant": "b", "actual": 11, "prediction": 9, "reference": 8},
    ])
    with pytest.raises(ValueError, match="Actual prices differ"):
        rolling_price_report(frame, tmp_path)


def test_unavailable_feature_blocks_are_not_counted_as_matched(tmp_path):
    """Missing external data cannot masquerade as an observed comparison."""
    frame = pd.DataFrame([
        {"delivery_day": "2026-10-01", "product_start": "2026-10-01T00:00Z", "auction": "DA",
         "gate": "DA", "variant": "core", "actual": 10, "prediction": 9, "reference": 8,
         "feature_inputs_complete": True},
        {"delivery_day": "2026-10-01", "product_start": "2026-10-01T00:00Z", "auction": "DA",
         "gate": "DA", "variant": "swiss", "actual": 10, "prediction": 9, "reference": 8,
         "feature_inputs_complete": False},
    ])
    rolling_price_report(frame, tmp_path, expected_variants=["core", "swiss", "weather"], pair_baseline="core")
    pairs = pd.read_csv(tmp_path / "paired_variants.csv").set_index("variant_b")
    assert pairs.loc["swiss", "matched_rows"] == 0
    assert pairs.loc["swiss", "status"] == "insufficient_common_data"
    assert pairs.loc["weather", "rows_b"] == 0
