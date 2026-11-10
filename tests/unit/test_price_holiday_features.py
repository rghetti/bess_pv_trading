"""Check date-specific holiday interactions and local calendar boundaries."""

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_holiday_features import build_holiday_features


def test_named_holidays_hour_and_bridge_use_local_dates():
    """Different local hours and holiday types receive distinct indicators."""
    starts = pd.DatetimeIndex([
        "2025-04-25 09:00+00:00",  # 11:00 in Italy on Liberation Day.
        "2025-05-01 12:00+00:00",  # 14:00 in Italy on Labour Day.
        "2025-05-02 12:00+00:00",  # Weekday between the holiday and Saturday.
        "2025-05-03 12:00+00:00",  # Saturday after the bridge.
    ])
    calendar = {"timezone": "Europe/Rome", "fixed_holidays": ["04-25", "05-01"],
                "easter_holidays": True}
    settings = {"hour_interactions": True, "include_adjacent_days": True,
                "include_bridge_day": True}
    matrix, names = build_holiday_features(starts, calendar, settings)
    position = {name: index for index, name in enumerate(names)}
    assert matrix.shape == (4, 27)
    assert matrix[0, position["holiday_hour_11"]] == 1
    assert matrix[1, position["holiday_hour_14"]] == 1
    assert matrix[2, position["weekday_bridge_between_holiday_and_weekend"]] == 1
    assert matrix[2, position["day_after_holiday"]] == 1
    assert matrix[3, position["weekday_bridge_between_holiday_and_weekend"]] == 0
    assert np.isfinite(matrix).all()


def test_repeated_dst_hour_uses_same_local_hour_feature():
    """Both occurrences of a repeated local hour receive the same calendar feature."""
    starts = pd.DatetimeIndex(["2025-10-26 00:30+00:00", "2025-10-26 01:30+00:00"])
    calendar = {"timezone": "Europe/Rome", "fixed_holidays": ["10-26"], "easter_holidays": False}
    settings = {"hour_interactions": True, "include_adjacent_days": False,
                "include_bridge_day": False}
    matrix, names = build_holiday_features(starts, calendar, settings)
    assert matrix[:, names.index("holiday_hour_02")].tolist() == [1.0, 1.0]


def test_general_configuration_shares_profiles_across_holidays():
    """The default expansion gives different holidays the same hourly feature."""
    starts = pd.DatetimeIndex(["2025-04-25 10:00+00:00", "2025-05-01 10:00+00:00"])
    calendar = {"timezone": "Europe/Rome", "fixed_holidays": ["04-25", "05-01"],
                "easter_holidays": True}
    settings = {"hour_interactions": True, "include_adjacent_days": True,
                "include_bridge_day": True}
    matrix, names = build_holiday_features(starts, calendar, settings)
    assert matrix.shape == (2, 27)
    assert np.array_equal(matrix[0, :24], matrix[1, :24])
    assert not any(name.startswith("holiday_date_") for name in names)
