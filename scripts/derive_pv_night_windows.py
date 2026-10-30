"""Derive configurable local-hour night windows from a solar-position envelope."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pvlib
import yaml


def derive_windows(latitudes, longitudes, start_year, end_year, timezone, threshold, margin, minutes):
    """Enclose sampled twilight at every supplied point, adding a clock-time margin."""
    # Enforce the input contract: Invalid year range, margin or sampling interval.
    if end_year <= start_year or margin < 0 or minutes <= 0 or 60 % minutes:
        raise ValueError('Invalid year range, margin or sampling interval')
    start = pd.Timestamp(f'{start_year}-01-01', tz=timezone)
    end = pd.Timestamp(f'{end_year}-01-01', tz=timezone)
    times = pd.date_range(start, end, freq=f'{minutes}min', inclusive='left')
    possible = np.zeros(len(times), dtype=bool)
    # Enclose sampled twilight at every supplied point, adding a clock-time margin for each `latitude` in `latitudes`.
    for latitude in latitudes:
        # Enclose sampled twilight at every supplied point, adding a clock-time margin for each `longitude` in `longitudes`.
        for longitude in longitudes:
            altitude = pvlib.solarposition.get_solarposition(times, latitude, longitude).apparent_elevation
            possible |= altitude.to_numpy() > threshold
    minute = times.hour * 60 + times.minute
    windows = {}
    # Enclose sampled twilight at every supplied point, adding a clock-time margin for each `month` in `range(1, 13)`.
    for month in range(1, 13):
        values = minute[(times.month == month) & possible]
        # Enforce the input contract: No sampled twilight in month `month`.
        if not len(values):
            raise ValueError(f'No sampled twilight in month {month}')
        end_night = int(values.min()) - margin
        start_night = int(values.max()) + minutes + margin
        # Enforce the input contract: Cannot construct night window for month `month`.
        if not 0 < end_night < start_night < 1440:
            raise ValueError(f'Cannot construct night window for month {month}')
        fmt = lambda value: f'{value // 60:02d}:{value % 60:02d}'
        windows[f'{month:02d}'] = {'start': fmt(start_night), 'end': fmt(end_night)}
    return {'enabled': True, 'timezone': timezone, 'monthly_windows': windows}


def main():
    """Print the reproducible monthly policy without altering any input data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latitudes', nargs='+', type=float, required=True)
    parser.add_argument('--longitudes', nargs='+', type=float, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year-exclusive', type=int, required=True)
    parser.add_argument('--timezone', required=True)
    parser.add_argument('--elevation-threshold', type=float, required=True)
    parser.add_argument('--margin-minutes', type=int, required=True)
    parser.add_argument('--sampling-minutes', type=int, required=True)
    args = parser.parse_args()
    print(yaml.safe_dump(derive_windows(args.latitudes, args.longitudes, args.start_year,
        args.end_year_exclusive, args.timezone, args.elevation_threshold,
        args.margin_minutes, args.sampling_minutes), sort_keys=False), end='')


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
