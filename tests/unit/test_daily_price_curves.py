"""Validate the native daily-curve grouping boundary."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_daily_price_curves import full_local_days


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
