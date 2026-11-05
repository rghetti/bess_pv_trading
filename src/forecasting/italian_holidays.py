"""Italian national public holidays for electricity-price model features."""

from datetime import date, timedelta

from dateutil.easter import easter
import numpy as np
import pandas as pd


def italian_national_holidays(year):
    """Return national holidays, excluding municipal patron-saint festivals."""
    fixed = ((1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15),
             (11, 1), (12, 8), (12, 25), (12, 26))
    result = {date(year, month, day) for month, day in fixed}
    easter_sunday = easter(year)
    result.update((easter_sunday, easter_sunday + timedelta(days=1)))
    return result


def italian_holiday_flags(starts, timezone):
    """Map product starts to a national holiday on the local delivery date."""
    local = pd.DatetimeIndex(pd.to_datetime(starts, utc=True)).tz_convert(timezone)
    if local.hasnans:
        raise ValueError('Product start timestamps must be valid')
    holidays = set().union(*(italian_national_holidays(int(year)) for year in local.year.unique()))
    return np.asarray([day in holidays for day in local.date], dtype=float)
