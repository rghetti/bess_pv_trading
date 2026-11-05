"""Validate the native daily-curve grouping boundary."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_daily_price_curves import check_joint_cutoffs, curve_target_and_anchor, full_local_days


def test_excludes_partial_and_dst_local_days():
    """Only an exact ordinary local product grid forms a daily target."""
    starts = pd.date_range("2026-03-28", "2026-03-31", freq="15min", inclusive="left", tz="Europe/Rome")
    features = np.arange(len(starts), dtype=float)[:, None]
    targets = np.arange(len(starts), dtype=float)
    x, y, rows, excluded = full_local_days(starts, features, targets, "Europe/Rome", 15)
    assert x.shape == (2, 96)
    assert y.shape == (2, 96)
    assert rows.shape == (2, 96)
    assert excluded == [{"date": "2026-03-29", "rows": 92, "expected_local_rows": 92}]
    with pytest.raises(ValueError, match="chronologically unique"):
        full_local_days(starts[::-1], features, targets, "Europe/Rome", 15)


def test_joint_curve_requires_one_decision_time_per_day():
    """A curve cannot combine forecasts from earlier and later decision times."""
    rows = np.array([[0, 1], [2, 3]])
    check_joint_cutoffs(np.array([1, 1, 2, 2]), rows)
    with pytest.raises(ValueError, match="share one"):
        check_joint_cutoffs(np.array([1, 2, 2, 2]), rows)


def test_residual_curve_uses_only_first_lag_per_product():
    """The residual anchor is the first feature in each product's input row."""
    features = np.array([[10, 99, 20, 88]], dtype=float)
    targets = np.array([[13, 15]], dtype=float)
    residual, anchor = curve_target_and_anchor(features, targets, "first_lag_residual")
    np.testing.assert_array_equal(anchor, [[10, 20]])
    np.testing.assert_array_equal(residual, [[3, -5]])
    absolute, zero = curve_target_and_anchor(features, targets, "absolute")
    np.testing.assert_array_equal(absolute, targets)
    np.testing.assert_array_equal(zero, [[0, 0]])
    with pytest.raises(ValueError, match="Unknown"):
        curve_target_and_anchor(features, targets, "invalid")
