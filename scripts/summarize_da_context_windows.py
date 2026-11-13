"""Write a readable, complete matched DA context-window metrics report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PERCENT_FIELDS = {'wape', 'spread_direction_accuracy', 'interval80_coverage'}


def formatted(metric, value, difference=False):
    """Format percentages as percentages, counts as integers and prices as decimals."""
    if pd.isna(value):
        return 'n/a'
    if metric in PERCENT_FIELDS or metric.endswith('_coverage'):
        suffix = ' pp' if difference else '%'
        return f'{value * 100:+.2f}{suffix}' if difference else f'{value * 100:.2f}{suffix}'
    if metric == 'products' or metric.endswith('_products'):
        return f'{value:+,.0f}' if difference else f'{value:,.0f}'
    return f'{value:+.3f}' if difference else f'{value:.3f}'


def build_report(scores, coverage):
    """Show headline comparisons and every score field, with matched covariate deltas."""
    identifiers = ['model', 'period', 'window_days', 'input_variant']
    if scores.duplicated(identifiers).any():
        raise ValueError('Duplicate score row')
    metrics = [column for column in scores.columns if column not in identifiers]
    rows = ['# NORD DA zero-shot context and forecast comparison', '',
        'Each row compares the same delivery products for the same model and window. '
        'Forecast covariates are load, total generation, solar and onshore wind forecasts.', '',
        'Development is retrospective. November 2026 is reported separately and is not used for selection. '
        'Forecast issue times are not verified; archived values are assumed available at the DA gate. '
        'Pretrained-data overlap is possible. TimesFM 3 has non-commercial weight terms.', '',
        'The 60-day covariate input excludes 1–10 October 2025 because historical total-generation forecasts '
        'are missing; all six variants are evaluated on their common products.', '',
        '## Matched coverage', '', '| Window | Input | Available rows across models | Matched rows across models |',
        '|---:|---|---:|---:|']
    for item in coverage.itertuples(index=False):
        rows.append(f'| {item.window_days} | {item.input_variant} | {item.available_rows:,} | {item.matched_rows:,} |')
    rows += ['', '## Headline point-forecast errors', '',
             'Δ MAE is covariates minus history-only; a negative value means an improvement. '
             'Δ MAE % is relative to history-only MAE. WAPE changes are percentage points, not relative percentages.', '',
             '| Period | Model | Days | Products | History MAE | Forecast MAE | Δ MAE | Δ MAE % | History WAPE | Forecast WAPE | Δ WAPE |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (period, model, window), group in scores.groupby(['period', 'model', 'window_days'], sort=True):
        variants = group.set_index('input_variant')
        if set(variants.index) != {'history', 'nord'}:
            raise ValueError(f'Missing paired variant: {period} {model} {window}')
        old, new = variants.loc['history'], variants.loc['nord']
        relative = f"{(new['mae'] / old['mae'] - 1) * 100:+.2f}%" if old['mae'] else 'n/a'
        rows.append(f"| {period} | {model} | {window} | {int(old['products']):,} | {old['mae']:.2f} | {new['mae']:.2f} | {new['mae']-old['mae']:+.2f} | {relative} | {old['wape']*100:.2f}% | {new['wape']*100:.2f}% | {(new['wape']-old['wape'])*100:+.2f} pp |")
    rows += ['', '## Complete metrics and covariate differences', '',
        'Values in price units are €/MWh. Bias is prediction minus actual. Relative MAE is the '
        'model MAE divided by the reference MAE on products where that reference exists; lower is better. '
        'Weekly reference is the same local product seven calendar days earlier. Latest reference is the '
        'previous delivery-day DA curve. Quantile coverage is the proportion of actual prices at or below '
        'the predicted quantile; 80% interval coverage is between q10 and q90. Pinball losses are in €/MWh. '
        'A positive delta always means the numeric value increased; it does not always indicate improvement.', '']
    for (period, model, window), group in scores.groupby(['period', 'model', 'window_days'], sort=True):
        variants = group.set_index('input_variant')
        old, new = variants.loc['history'], variants.loc['nord']
        rows += [f'### {period} — {model} — {window} days', '',
                 '| Metric | History only | With NORD forecasts | Difference |', '|---|---:|---:|---:|']
        for metric in metrics:
            left, right = old[metric], new[metric]
            delta = right - left if pd.notna(left) and pd.notna(right) else float('nan')
            rows.append(f'| {metric} | {formatted(metric, left)} | {formatted(metric, right)} | {formatted(metric, delta, True)} |')
        rows.append('')
    rows += ['## Detailed files', '',
        '- [Complete numeric metrics](scores.csv)', '- [Common-product coverage](coverage.csv)',
        '- [Monthly errors](monthly_scores.csv)', '- [Local-hour errors](hourly_scores.csv)',
        '- [Daily errors](daily_scores.csv)', '- [Matched forecast products](matched_forecasts.parquet)', '']
    return '\n'.join(rows)


def main():
    """Generate a report from completed comparison CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='outputs/da_zero_shot_context_windows/comparison')
    args = parser.parse_args()
    directory = Path(args.directory)
    (directory / 'summary.md').write_text(build_report(pd.read_csv(directory / 'scores.csv'),
                                                       pd.read_csv(directory / 'coverage.csv')))
    print(f'Wrote {directory / "summary.md"}')


if __name__ == '__main__':
    main()
