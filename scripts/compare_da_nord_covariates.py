"""Compare history-only and NORD forecast models on identical delivery products."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score


def main():
    """Join each model's two variants and save matched scores and exclusions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report', default='outputs/da_zero_shot_run/report')
    parser.add_argument('--nord-report', default='outputs/da_zero_shot_nord_run/report')
    parser.add_argument('--output', default='outputs/da_zero_shot_nord_comparison')
    parser.add_argument('--history-label', default='history_only')
    parser.add_argument('--nord-label', default='nord_forecasts')
    args = parser.parse_args()
    history = pd.read_parquet(Path(args.history_report) / 'matched_forecasts.parquet')
    nord = pd.read_parquet(Path(args.nord_report) / 'matched_forecasts.parquet')
    keys = ['delivery_day', 'product_start', 'auction', 'gate', 'variant']
    paired = history.merge(nord, on=keys, suffixes=('_history', '_nord'), validate='one_to_one')
    if not np.allclose(paired.actual_history, paired.actual_nord, equal_nan=True):
        raise ValueError('Matched variants contain different observed prices')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(destination / 'paired_forecasts.parquet', index=False)
    rows = []
    for (model, period), block in paired.groupby(['variant', 'period_history']):
        for label, suffix in [(args.history_label, 'history'), (args.nord_label, 'nord')]:
            data = pd.DataFrame({'actual': block.actual_history, 'prediction': block[f'prediction_{suffix}'],
                'weekly_reference': block.weekly_reference_history, 'latest_reference': block.latest_reference_history})
            rows.append({'model': model, 'period': period, 'input': label, **score(data)})
    results = pd.DataFrame(rows)
    results.to_csv(destination / 'scores.csv', index=False)
    lines = ['# Matched DA forecast comparison', '', 'Values use identical delivery products. Historical covariate vintages may be unverified.', '']
    for (model, period), block in results.groupby(['model', 'period']):
        variants = block.set_index('input')
        old, new = variants.loc[args.history_label], variants.loc[args.nord_label]
        lines.extend([f'## {model} — {period}', '', f'| Metric | {args.history_label} | {args.nord_label} | Difference |', '|---|---:|---:|---:|'])
        for metric in [name for name in results.columns if name not in ('model', 'period', 'input')]:
            left, right = old[metric], new[metric]
            delta = right - left if pd.notna(left) and pd.notna(right) else float('nan')
            lines.append(f'| {metric} | {left:.4f} | {right:.4f} | {delta:+.4f} |')
        lines.append('')
    (destination / 'summary.md').write_text('\n'.join(lines))
    pd.DataFrame([{'history_rows': len(history), 'nord_rows': len(nord), 'matched_rows': len(paired)}]).to_csv(
        destination / 'coverage.csv', index=False)
    print(f'Matched rows across all models: {len(paired)}')


if __name__ == '__main__':
    main()
