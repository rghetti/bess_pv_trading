"""Known-in-advance Italian calendar features for PV forecasting."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _easter_sunday(year: int) -> date:
    """Compute Gregorian Easter without a runtime calendar dependency."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def italian_national_holiday(starts_utc: pd.DatetimeIndex, timezone: str) -> np.ndarray:
    """Flag Italy-wide public holidays on each local delivery date.

    Fixed holidays follow the national public-holiday calendar; Easter and
    Easter Monday are movable. San Francesco on 4 October applies from 2026.
    Local patron days and one-off closures are deliberately excluded.
    """
    local = pd.DatetimeIndex(starts_utc).tz_convert(timezone)
    fixed = ((1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15),
             (11, 1), (12, 8), (12, 25), (12, 26))
    holidays = []
    # Flag Italy-wide public holidays on each local delivery date for each `year` in `np.unique(local.year)`.
    for year in np.unique(local.year):
        year = int(year)
        holidays.extend(date(year, month, day) for month, day in fixed)
        easter = _easter_sunday(year)
        holidays.extend((easter, easter + timedelta(days=1)))
        # Flag Italy-wide public holidays on each local delivery date when `year >= 2026`.
        if year >= 2026:
            holidays.append(date(year, 10, 4))
    local_days = local.normalize().tz_localize(None)
    return np.asarray(local_days.isin(pd.to_datetime(holidays)), dtype=np.float32)
