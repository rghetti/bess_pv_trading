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
                pairs = [('history', 'nord'), ('nord', 'nord_spread')]
                if 'holiday' in variants.index:
                    pairs = [('nord_spread', 'holiday')] + [
                        ('holiday', name) for name in settings['variants'] if name.startswith('holiday_')]
                for before, after in pairs:
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
        # Select only on development MAE, then draw the extreme daily curves.
        if 'holiday' in settings['variants']:
            candidates = scores[scores['period'] == 'retrospective_development']
            if candidates.empty:
                raise ValueError(f'No common development products for {market}')
            winner = candidates.sort_values(['mae', 'wape', 'input_variant'], kind='stable').iloc[0]
            selected = matched[(matched.period == 'retrospective_development') &
                               (matched.input_variant == winner['input_variant']) &
                               (matched.variant == winner['model'])].copy()
            selected['absolute_error'] = (selected.prediction - selected.actual).abs()
            daily = selected.groupby('delivery_day').absolute_error.mean().sort_values(kind='stable')
            daily.to_csv(output / 'winner_daily_mae.csv', header=['mae'])
            from matplotlib import pyplot as plt
            chosen = [('best', day) for day in daily.index[:3]] + [('worst', day) for day in daily.index[-3:][::-1]]
            for rank, (category, day) in enumerate(chosen):
                frame = selected[selected.delivery_day == day].sort_values('product_start')
                local = frame.product_start.dt.tz_convert(settings['timezone'])
                figure, axis = plt.subplots(figsize=(12, 4))
                axis.plot(local, frame.actual, label='Realized')
                axis.plot(local, frame.prediction, label='TimesFM median')
                axis.set(title=f'{market} {day}: {category}, MAE {daily[day]:.2f} EUR/MWh',
                         ylabel='EUR/MWh', xlabel='Delivery time (local)')
                axis.legend()
                figure.autofmt_xdate()
                figure.tight_layout()
                figure.savefig(output / f'{category}_{rank % 3 + 1}_{day}.png', dpi=140)
                plt.close(figure)
                text.append(f"[{category.title()} {rank % 3 + 1}: {day}]({category}_{rank % 3 + 1}_{day}.png)")
            text.extend(['', f"Development winner: **{winner['input_variant']}**, MAE {winner['mae']:.3f} EUR/MWh.",
                         'Best and worst daily plots use matched development products; November is excluded from selection.', ''])
        text.extend(['[Scores](scores.csv) · [Coverage](coverage.csv) · [Matched forecasts](matched_forecasts.parquet)', ''])
        (output / 'summary.md').write_text('\n'.join(text))
        print(f'Prepared {market} comparison: {output}')


if __name__ == '__main__':
    main()
