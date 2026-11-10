"""Separate, deterministic holiday interactions for price-forecast studies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .price_feature_data import italian_holidays


def build_holiday_features(starts, calendar_settings, expansion_settings):
    """Return local calendar indicators and names, without using prices or targets.

    Hour indicators apply to all holidays. All date calculations use local
    calendar days, including on daylight-saving transition days.
    """
    times = pd.DatetimeIndex(starts)
    if times.tz is None:
        raise ValueError("Product timestamps must be timezone-aware")
    local = times.tz_convert(calendar_settings["timezone"])
    dates = pd.DatetimeIndex(local.date).tz_localize(calendar_settings["timezone"])
    holiday = italian_holidays(dates, calendar_settings).astype(bool)
    columns = []
    names = []
    # Interactions permit a holiday effect to vary over the 24 local clock hours.
    if expansion_settings["hour_interactions"]:
        for hour in range(24):
            columns.append(holiday & (local.hour == hour))
            names.append(f"holiday_hour_{hour:02d}")
    previous = italian_holidays(dates - pd.DateOffset(days=1), calendar_settings).astype(bool)
    following = italian_holidays(dates + pd.DateOffset(days=1), calendar_settings).astype(bool)
    # Adjacent indicators refer to national holidays, regardless of weekend status.
    if expansion_settings["include_adjacent_days"]:
        columns.extend([following, previous])
        names.extend(["day_before_holiday", "day_after_holiday"])
    if expansion_settings["include_bridge_day"]:
        before_weekend = (dates - pd.DateOffset(days=1)).dayofweek >= 5
        after_weekend = (dates + pd.DateOffset(days=1)).dayofweek >= 5
        bridge = ~holiday & (dates.dayofweek < 5) & ((previous & after_weekend) | (before_weekend & following))
        columns.append(bridge)
        names.append("weekday_bridge_between_holiday_and_weekend")
    if not columns:
        raise ValueError("At least one holiday feature must be enabled")
    if len(names) != len(set(names)):
        raise ValueError("Holiday feature names must be unique")
    return np.column_stack(columns).astype(np.float32), names
