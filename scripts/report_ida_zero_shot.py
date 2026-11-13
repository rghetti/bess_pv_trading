"""Summarize completed IDA studies on common model/product observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score


def main():
    """Require completed own-gate runs and generate CSV and Markdown comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_ida.yaml')
    parser.add_argument('--root')
    args = parser.parse_args()
    settings = load_config(args.config)['zero_shot_ida']
    root = Path(args.root or settings['output'])
    for market in settings['auctions']:
        frames, coverage = [], []
        expected = len(settings['windows']) * len(settings['variants'])
        for window in settings['windows']:
            for variant in settings['variants']:
                directory = root / market / f'{window}d' / variant / 'run/report'
                if not json.loads((directory / 'status.json').read_text())['complete']:
                    raise ValueError(f'Incomplete evaluation: {directory}')
                frame = pd.read_parquet(directory / 'matched_forecasts.parquet')
                frame['context_days'] = window
                frame['input_variant'] = variant
                frames.append(frame)
                coverage.append({'context_days': window, 'input_variant': variant, 'available_rows': len(frame)})
        all_rows = pd.concat(frames, ignore_index=True)
        keys = ['delivery_day', 'product_start', 'auction', 'gate', 'variant']
        if all_rows.duplicated(keys + ['context_days', 'input_variant']).any():
            raise ValueError('Duplicate product in one experimental variant')
        sizes = all_rows.groupby(keys).size()
        common = sizes[sizes.eq(expected)].reset_index()[keys]
        matched = all_rows.merge(common, on=keys, validate='many_to_one')
        truth = matched.groupby(keys).actual.agg(['min', 'max'])
        if not np.allclose(truth['min'], truth['max']):
            raise ValueError('Conflicting target prices across studies')
        output = root / market / 'comparison'
        output.mkdir(parents=True, exist_ok=True)
        matched.to_parquet(output / 'matched_forecasts.parquet', index=False)
        rows = []
        for key, group in matched.groupby(['variant', 'period', 'context_days', 'input_variant']):
            rows.append(dict(zip(['model', 'period', 'context_days', 'input_variant'], key)) | score(group))
        scores = pd.DataFrame(rows)
        scores.to_csv(output / 'scores.csv', index=False)
        changes = []
        if not scores.empty:
            for key, block in scores.groupby(['model', 'period', 'context_days']):
                variants = block.set_index('input_variant')
                for before, after in [('history', 'nord'), ('nord', 'nord_spread')]:
                    if before not in variants.index or after not in variants.index:
                        continue
                    old, new = variants.loc[before], variants.loc[after]
                    changes.append(dict(zip(['model', 'period', 'context_days'], key)) | {
                        'comparison': f'{after}_minus_{before}', 'products': int(new['products']),
                        'mae_change': new['mae'] - old['mae'], 'wape_change': new['wape'] - old['wape'],
                        'rmse_change': new['rmse'] - old['rmse']})
        pd.DataFrame(changes).to_csv(output / 'paired_deltas.csv', index=False)
        for item in coverage:
            item['matched_rows'] = len(common)
        pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
        text = [f'# {market} own-gate zero-shot comparison', '',
                'All variants use common products. November 2026 is reported separately, without selection. '
                'NORD forecast issue vintages remain unverified. Spread inputs contain historical values only; '
                'their reference is the newest published non-cancelled earlier auction for the same delivery day.', '',
                f'Matched rows across models per experimental setting: {len(common):,}.', '']
        if changes:
            text.extend(['## Incremental input changes', '',
                         'Negative MAE/RMSE change means lower error. WAPE change is shown in percentage points.', '',
                         '| Model | Period | Days | Comparison | Products | Δ MAE | Δ RMSE | Δ WAPE |',
                         '|---|---|---:|---|---:|---:|---:|---:|'])
            for item in changes:
                text.append(f"| {item['model']} | {item['period']} | {item['context_days']} | {item['comparison']} | {item['products']} | {item['mae_change']:+.3f} | {item['rmse_change']:+.3f} | {item['wape_change'] * 100:+.3f} pp |")
            text.append('')
        for _, row in scores.iterrows():
            text.extend([f"## {row['model']} — {row['period']} — {row['context_days']} days — {row['input_variant']}",
                         '', '| Metric | Value |', '|---|---:|'])
            for metric in scores.columns:
                if metric in ('model', 'period', 'context_days', 'input_variant') or pd.isna(row[metric]):
                    continue
                text.append(f'| {metric} | {row[metric]:.5f} |')
            text.append('')
        text.extend(['[Scores](scores.csv) · [Coverage](coverage.csv) · [Matched forecasts](matched_forecasts.parquet)', ''])
        (output / 'summary.md').write_text('\n'.join(text))
        print(f'Prepared {market} comparison: {output}')


if __name__ == '__main__':
    main()
