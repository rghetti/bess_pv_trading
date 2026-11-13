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
    parser.add_argument('--study-root', default='outputs/da_zero_shot_context_windows')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.study_root)
    sources = {(window, variant): str(root / f'{window}d' / variant / 'run/report')
               for window in (7, 28, 60) for variant in ('history', 'nord')}
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
    destination = Path(args.output) if args.output else root / 'comparison'
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(destination / 'matched_forecasts.parquet', index=False)
    # Retain interpretable time breakdowns on exactly the six-way matched sample.
    matched['month'] = matched.delivery_day.str[:7]
    matched['hour'] = pd.to_datetime(matched.product_start, utc=True).dt.tz_convert('Europe/Rome').dt.hour
    matched['abs_error'] = (matched.prediction - matched.actual).abs()
    for dimension in ('month', 'hour', 'delivery_day'):
        keys = ['variant', 'window_days', 'input_variant', dimension]
        groups = matched.groupby(keys).agg(products=('actual', 'size'), mae=('abs_error', 'mean'),
                                           absolute_actual=('actual', lambda values: values.abs().sum()),
                                           absolute_error=('abs_error', 'sum')).reset_index()
        groups['wape'] = groups.absolute_error / groups.absolute_actual.replace(0, np.nan)
        filename = {'month': 'monthly_scores.csv', 'hour': 'hourly_scores.csv', 'delivery_day': 'daily_scores.csv'}[dimension]
        groups.to_csv(destination / filename, index=False)
    rows = []
    for group, block in matched.groupby(['variant', 'period', 'window_days', 'input_variant']):
        rows.append(dict(zip(['model', 'period', 'window_days', 'input_variant'], group)) | score(block))
    scores = pd.DataFrame(rows)
    scores.to_csv(destination / 'scores.csv', index=False)
    for item in counts:
        item['matched_rows'] = len(matched) // len(sources)
    coverage = pd.DataFrame(counts)
    coverage.to_csv(destination / 'coverage.csv', index=False)
    from scripts.summarize_da_context_windows import build_report
    (destination / 'summary.md').write_text(build_report(scores, coverage))
    print(f'Matched products per model: {len(common)}; outputs: {destination}')


if __name__ == '__main__':
    main()
