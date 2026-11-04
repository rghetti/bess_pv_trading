"""Verify chronological selection and finite convex price-forecast weights."""

import numpy as np
import pytest

from scripts.evaluate_price_ensembles import evaluate_stacking


def test_stacking_never_uses_later_targets_to_fit_or_select_weights():
    """Changing the evaluation targets cannot alter the chosen portfolio."""
    days = np.array([f"2026-01-{day:02d}" for day in range(1, 13)])
    actual = np.full(12, 100.0)
    reference = np.full(12, 110.0)
    predictions = np.column_stack([np.full(12, 99.0), np.full(12, 103.0), np.full(12, 95.0)])
    settings = {"rank_start": "2026-01-01", "rank_end_exclusive": "2026-01-05",
                "tune_start": "2026-01-05", "tune_end_exclusive": "2026-01-09",
                "test_start": "2026-01-09", "test_end_exclusive": "2026-01-13",
                "stacking": {"weight_fit_fraction": 0.5, "ensemble_sizes": [1, 2, 3],
                             "ridge_penalties": [0.0, 0.01]}}
    result, test, forecast = evaluate_stacking(actual, reference, predictions, ["a", "b", "c"], days, settings)
    changed = actual.copy()
    changed[test] = 1000.0
    repeated, _, later_forecast = evaluate_stacking(changed, reference, predictions, ["a", "b", "c"], days, settings)
    assert result["selected_candidates"] == repeated["selected_candidates"]
    assert result["weights"] == pytest.approx(repeated["weights"])
    assert forecast == pytest.approx(later_forecast)
    assert sum(result["weights"]) == pytest.approx(1)
    assert all(weight >= 0 for weight in result["weights"])
    assert result["weight_fit_days"][-1] < result["weight_validation_days"][0]
