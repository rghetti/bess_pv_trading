"""Compare seasonal night windows with the untouched historical GHI and AC data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data import night_window_mask, open_store


def main():
    """Report all raw quarter-hours without changing targets or excluding data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_realized_irradiance_search']
    policy = section['night_policy']
    start = pd.Timestamp(section['training_start'], tz=policy['timezone']).tz_convert('UTC')
    end = pd.Timestamp(section['end_exclusive'], tz=policy['timezone']).tz_convert('UTC')
    rows = []
    with open_store(config) as store:
        # Report all raw quarter-hours without changing targets or excluding data for each `farm_id` in `section['farm_ids']`.
        for farm_id in section['farm_ids']:
            source = store.sel(farm_id=int(farm_id))
            times = pd.DatetimeIndex(pd.to_datetime(source.time.values, utc=True))
            inside = (times >= start) & (times < end)
            local = times.tz_convert(policy['timezone'])
            night = night_window_mask(times, policy) & inside
            ghi = np.asarray(source.ghi.values, dtype=float)
            power = np.asarray(source.power.values, dtype=float)
            # Report all raw quarter-hours without changing targets or excluding data for each `month` in `range(1, 13)`.
            for month in range(1, 13):
                selected = night & (local.month == month)
                finite = np.isfinite(ghi)
                positive_power = np.isfinite(power) & (power > 0)
                rows.append({'farm_id': int(farm_id), 'month': month,
                    'night_start': policy['monthly_windows'][f'{month:02d}']['start'],
                    'night_end': policy['monthly_windows'][f'{month:02d}']['end'],
                    'quarter_hours': int(selected.sum()),
                    'ghi_zero': int((selected & finite & (ghi == 0)).sum()),
                    'ghi_positive': int((selected & finite & (ghi > 0)).sum()),
                    'ghi_negative': int((selected & finite & (ghi < 0)).sum()),
                    'ghi_missing': int((selected & ~finite).sum()),
                    'positive_power': int((selected & positive_power).sum()),
                    'positive_power_and_positive_ghi': int((selected & positive_power & finite & (ghi > 0)).sum()),
                    'positive_power_and_missing_ghi': int((selected & positive_power & ~finite).sum())})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'monthly_raw_ghi_audit.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps({
        'period': f"{section['training_start']} through {section['end_exclusive']} exclusive",
        'source': config['inputs']['timeseries_zarr'] if 'timeseries_zarr' in config['inputs'] else config['inputs'],
        'scope': 'all original Zarr quarter-hours; no eligibility or reviewed-exclusion filtering',
        'policy': policy,
        'note': 'Raw satellite-derived GHI; zero is not proof of no production. Missing and negative values are separate.'
    }, indent=2, default=str) + '\n')
    print(output / 'monthly_raw_ghi_audit.csv')


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
