"""Verify publication and chronological safeguards for online price correction."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_online_price_bias import online_bias


def test_online_bias_uses_only_prior_published_and_completed_products():
    """An unpublished prior error cannot enter a later decision's correction."""
    starts = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T12:00Z", "2026-01-02T12:00Z",
                                               "2026-01-03T12:00Z"])).as_unit("ns")
    ends = starts + pd.Timedelta(hours=1)
    available = pd.DatetimeIndex(pd.to_datetime(["2026-01-02T11:00Z", "2026-01-04T00:00Z",
                                                  "2026-01-03T13:00Z"])).as_unit("ns")
    decisions = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T10:00Z", "2026-01-02T10:00Z",
                                                  "2026-01-03T10:00Z"])).as_unit("ns")
    actual = np.array([110.0, 120.0, 130.0])
    predicted = np.array([100.0, 100.0, 100.0])
    args = (starts.asi8, ends.asi8, available.asi8, decisions.asi8,
            ["2026-01-01", "2026-01-02", "2026-01-03"], [12, 12, 12])
    correction = online_bias(actual, predicted, *args, window_days=7, block_hours=24, shrinkage=0)
    assert correction == pytest.approx([0, 0, 10])
    changed = actual.copy()
    changed[2] = 10000
    assert online_bias(changed, predicted, *args, window_days=7, block_hours=24, shrinkage=0) == pytest.approx(correction)


def test_bias_configuration_selection_does_not_use_later_targets():
    """Perturbing evaluation labels cannot change tuning-period selection."""
    from scripts.evaluate_online_price_bias import evaluate_bias

    starts = pd.date_range("2026-01-01T12:00Z", periods=8, freq="D").as_unit("ns")
    actual = np.array([100., 102., 104., 106., 108., 110., 112., 114.])
    predicted = np.full(8, 100.)
    ends = (starts + pd.Timedelta(hours=1)).asi8
    decisions = (starts - pd.Timedelta(hours=2)).asi8
    days = starts.strftime("%Y-%m-%d")
    settings = {"tune_start": "2026-01-01", "tune_end_exclusive": "2026-01-05",
                "test_start": "2026-01-05", "test_end_exclusive": "2026-01-09",
                "window_days": [3, 7], "block_hours": [24], "shrinkage": [0, 2]}
    args = (predicted, predicted + 5, starts.asi8, ends, ends, decisions, days, np.full(8, 12), settings)
    first, test, _ = evaluate_bias(actual, *args)
    changed = actual.copy()
    changed[test] += 500
    second, _, _ = evaluate_bias(changed, *args)
    assert first["selected"] == second["selected"]
