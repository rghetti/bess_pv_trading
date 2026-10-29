"""Known-at-issue-time Italian holiday feature contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.data.pv_calendar import italian_national_holiday
from bess_pv_trading.forecasting.training.pv_forecast_search import weather_features


def test_italian_holidays_follow_local_delivery_dates_and_effective_year():
    """Movable, fixed and newly effective holidays use Europe/Rome dates."""
    times = pd.to_datetime([
        '2025-04-19T10:00Z', '2025-04-20T10:00Z', '2025-04-21T10:00Z',
        '2025-10-04T10:00Z', '2026-04-05T10:00Z', '2026-04-06T10:00Z',
        '2026-10-04T10:00Z', '2025-12-31T22:45Z', '2025-12-31T23:00Z',
    ], utc=True)
    expected = [0, 1, 1, 0, 1, 1, 1, 0, 1]
    np.testing.assert_array_equal(italian_national_holiday(times, 'Europe/Rome'), expected)


def test_holiday_channel_is_opt_in_for_existing_forecast_artifacts():
    """Only the new search adds one column to the legacy weather feature set."""
    frame = pd.DataFrame({
        'interval_start_utc': pd.to_datetime(['2025-12-31T23:00Z'], utc=True),
        'forecast_cycle_utc': pd.to_datetime(['2025-12-31T12:00Z'], utc=True),
        'forecast_ghi': [0.0], 'forecast_bhi': [0.0],
        'temperature_c': [3.0], 'wind_u_10m': [0.0], 'wind_v_10m': [1.0],
        'lead_hours': [11.0],
    })
    previous = weather_features(frame, 'core', 'Europe/Rome')
    extended = weather_features(frame, 'core', 'Europe/Rome',
        include_national_holiday=True)
    assert 'italian_national_holiday' not in previous
    assert extended.drop(columns='italian_national_holiday').equals(previous)
    assert extended.italian_national_holiday.iloc[0] == 1.0
