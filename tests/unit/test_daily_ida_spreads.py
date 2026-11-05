"""Verify the native IDA delivery-window grouping."""

import numpy as np
import pandas as pd

from scripts.evaluate_daily_ida_spreads import market_days


def test_half_day_window_rejects_dst_and_partial_days():
    """IDA3 accepts only complete ordinary noon-to-midnight windows."""
    times = pd.date_range("2026-03-28", "2026-03-31", freq="15min", inclusive="left", tz="Europe/Rome")
    times = times[times.hour >= 12]
    x = np.arange(len(times), dtype=float)[:, None]
    y = np.arange(len(times), dtype=float)
    features, targets, rows, excluded = market_days(times, x, y, "Europe/Rome", 12, 15)
    assert features.shape == (2, 48, 1)
    assert targets.shape == rows.shape == (2, 48)
    assert excluded == [{"date": "2026-03-29", "rows": 48, "expected_window_rows": 48}]
