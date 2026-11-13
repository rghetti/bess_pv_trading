"""Exercise matched-product and probabilistic price scores with small examples."""

import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import matched_predictions, score


def test_matched_products_reject_inconsistent_truth_and_drop_unpaired():
    """Only compare variants on shared products with the same realized label."""
    rows = pd.DataFrame([
        {"delivery_day": "2026-10-01", "product_start": stamp, "auction": "DA", "gate": "DA",
         "variant": variant, "actual": actual, "prediction": 11.0}
        for stamp, variant, actual in [("a", "history", 10.0), ("a", "covariates", 10.0),
                                       ("b", "history", 20.0)]
    ])
    assert len(matched_predictions(rows, ["history", "covariates"])) == 2
    rows.loc[1, "actual"] = 12.0
    with pytest.raises(ValueError, match="inconsistent"):
        matched_predictions(rows, ["history", "covariates"])


def test_scores_use_reference_specific_common_products():
    """Ratios use identical products for each model and its own reference."""
    frame = pd.DataFrame({"actual": [10.0, 20.0], "prediction": [12.0, 16.0],
                          "weekly_reference": [14.0, float("nan")],
                          "latest_reference": [8.0, 24.0], "q50": [12.0, 16.0]})
    result = score(frame)
    assert result["mae"] == 3.0
    assert result["wape"] == pytest.approx(0.2)
    assert result["weekly_relative_mae"] == pytest.approx(0.5)
    assert result["latest_relative_mae"] == pytest.approx(1.0)
    assert result["q50_pinball"] == pytest.approx(1.5)
    assert result["q50_coverage"] == pytest.approx(0.5)
