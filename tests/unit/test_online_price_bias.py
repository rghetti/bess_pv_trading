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
