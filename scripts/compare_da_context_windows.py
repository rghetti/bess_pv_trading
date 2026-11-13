"""Compare DA context windows and covariates on common model/product rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score


def main():
    """Require completed reports and score the same products for all six variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/da_zero_shot_context_windows/comparison')
    args = parser.parse_args()
    sources = {(7, 'history'): 'outputs/da_zero_shot_context_windows/7d/history/run/report',
               (28, 'history'): 'outputs/da_zero_shot_context_windows/28d/history/run/report',
               (60, 'history'): 'outputs/da_zero_shot_context_windows/60d/history/run/report',
               (7, 'nord'): 'outputs/da_zero_shot_context_windows/7d/nord/run/report',
               (28, 'nord'): 'outputs/da_zero_shot_context_windows/28d/nord/run/report',
               (60, 'nord'): 'outputs/da_zero_shot_context_windows/60d/nord/run/report'}
    frames = []
    counts = []
    for (window, variant), directory in sources.items():
        path = Path(directory)
        status = json.loads((path / 'status.json').read_text())
        if not status['complete']:
            raise ValueError(f'Report is incomplete: {path}')
        frame = pd.read_parquet(path / 'matched_forecasts.parquet')
        frame['window_days'] = window
        frame['input_variant'] = variant
        frames.append(frame)
        counts.append({'window_days': window, 'input_variant': variant, 'available_rows': len(frame)})
    all_rows = pd.concat(frames, ignore_index=True)
    keys = ['delivery_day', 'product_start', 'auction', 'gate', 'variant']
    if all_rows.duplicated(keys + ['window_days', 'input_variant']).any():
        raise ValueError('Duplicate forecast in one experimental variant')
    common_count = all_rows.groupby(keys).size()
    common = common_count[common_count == len(sources)].reset_index()[keys]
    matched = all_rows.merge(common, on=keys, validate='many_to_one')
    checks = matched.groupby(keys).actual.agg(['min', 'max'])
    if not np.allclose(checks['min'], checks['max']):
        raise ValueError('Conflicting observed prices across variants')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(destination / 'matched_forecasts.parquet', index=False)
    rows = []
    for group, block in matched.groupby(['variant', 'period', 'window_days', 'input_variant']):
        rows.append(dict(zip(['model', 'period', 'window_days', 'input_variant'], group)) | score(block))
    pd.DataFrame(rows).to_csv(destination / 'scores.csv', index=False)
    for item in counts:
        item['matched_rows'] = len(matched) // len(sources)
    pd.DataFrame(counts).to_csv(destination / 'coverage.csv', index=False)
    print(f'Matched products per model: {len(common)}; outputs: {destination}')


if __name__ == '__main__':
    main()
