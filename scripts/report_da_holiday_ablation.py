"""Compare DA feature variants on their shared products and write a report."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score


def report_ablation(settings, root, actions):
    """Score all requested feature variants on identical model and product rows."""
    selected = [item for item in actions if item['variant'] == 'nord' or 'blocks' in item]
    frames = []
    counts = []
    for item in selected:
        directory = Path(item['study']) / 'run/report'
        if not json.loads((directory / 'status.json').read_text())['complete']:
            raise ValueError(f'Incomplete report: {directory}')
        frame = pd.read_parquet(directory / 'matched_forecasts.parquet')
        frame['input_variant'] = item['variant']
        frames.append(frame)
        counts.append({'input_variant': item['variant'], 'available_rows': len(frame)})
    data = pd.concat(frames, ignore_index=True)
    keys = ['delivery_day', 'product_start', 'auction', 'gate', 'variant']
    if data.duplicated(keys + ['input_variant']).any():
        raise ValueError('Duplicate forecasts in a feature variant')
    sizes = data.groupby(keys).size()
    common = sizes[sizes == len(selected)].reset_index()[keys]
    matched = data.merge(common, on=keys, validate='many_to_one')
    actual = matched.groupby(keys).actual.agg(['min', 'max'])
    if matched.empty or not np.allclose(actual['min'], actual['max']):
        raise ValueError('No matched products or conflicting actual prices')
    output = Path(root) / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output / 'matched_forecasts.parquet', index=False)
    rows = []
    for key, group in matched.groupby(['variant', 'period', 'input_variant']):
        rows.append(dict(zip(['model', 'period', 'input_variant'], key)) | score(group))
    scores = pd.DataFrame(rows)
    scores.to_csv(output / 'scores.csv', index=False)
    for item in counts:
        item['matched_rows'] = len(common)
    pd.DataFrame(counts).to_csv(output / 'coverage.csv', index=False)
    lines = ['# DA TimesFM feature comparison', '',
             'All rows use matched products across the nine variants. November is reserved.',
             'NORD vintages and prior-day external quantity availability are assumed.', '',
             '| Period | Variant | Products | MAE | WAPE | RMSE | Bias |',
             '|---|---|---:|---:|---:|---:|---:|']
    for _, row in scores.sort_values(['period', 'mae', 'input_variant']).iterrows():
        lines.append(f"| {row['period']} | {row['input_variant']} | {row['products']} | {row['mae']:.3f} | {100 * row['wape']:.2f}% | {row['rmse']:.3f} | {row['bias']:.3f} |")
    lines.extend(['', '[Scores](scores.csv) · [Coverage](coverage.csv) · [Matched forecasts](matched_forecasts.parquet)', ''])
    (output / 'summary.md').write_text('\n'.join(lines))
