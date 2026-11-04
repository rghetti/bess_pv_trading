"""Verify delivery-time ensembles are selected without evaluation leakage."""

import numpy as np
import pytest

from scripts.evaluate_price_ensembles import evaluate_hourwise


def test_hourwise_selection_is_chronological_and_uses_local_blocks():
    """Changing later labels cannot alter rankings or the selected configuration."""
    days = np.repeat([f"2026-01-{day:02d}" for day in range(1, 10)], 2)
    hours = np.tile([2, 14], 9)
    actual = np.full(len(days), 100.0)
    reference = np.full(len(days), 110.0)
    first = np.where(hours == 2, 100.0, 120.0)
    second = np.where(hours == 2, 120.0, 100.0)
    predictions = np.column_stack([first, second])
    settings = {"rank_start": "2026-01-01", "rank_end_exclusive": "2026-01-04",
                "tune_start": "2026-01-04", "tune_end_exclusive": "2026-01-07",
                "test_start": "2026-01-07", "test_end_exclusive": "2026-01-10",
                "hourwise": {"block_hours": [12, 24], "ensemble_sizes": [1, 2]}}
    result, test, forecast = evaluate_hourwise(actual, reference, predictions, ["a", "b"], days, hours, settings)
    changed = actual.copy()
    changed[test] = -1000
    repeated, _, repeated_forecast = evaluate_hourwise(changed, reference, predictions, ["a", "b"], days, hours, settings)
    assert result["block_hours"] == 12
    assert result["selected_candidates_by_block"] == {"0": ["a"], "1": ["b"]}
    assert result["selected_candidates_by_block"] == repeated["selected_candidates_by_block"]
    assert forecast == pytest.approx(repeated_forecast)
    assert result["test"]["mae_eur_per_mwh"] == 0
