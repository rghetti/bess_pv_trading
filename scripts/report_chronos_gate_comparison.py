"""Compare a target auction's errors across decision gates on matched products."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_ida import MARKETS
from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score


def comparable_contract(manifest):
    """Select properties that must agree between gate-specific input bundles."""
    fields = ('models', 'calendar', 'realized_variant', 'realized_feature_definitions',
              'availability_policy', 'actual_dataset_selectors', 'include_historical_spread',
              'past_only_unavailable_policy', 'future_forecast_earliest_gate',
              'source_actuals_sha256', 'actuals_manifest_sha256', 'forecast_snapshot_sha256',
              'realized_builder_sha256', 'ida_builder_sha256', 'direct_gate_builder_sha256',
              'direct_contract_sha256')
    settings = manifest['settings']
    shared_settings = ('start', 'end_exclusive', 'context_days', 'minimum_context_days',
                       'seed', 'precision', 'quantiles', 'timezone', 'november_holdout_start')
    return {'manifest': {name: manifest.get(name) for name in fields},
            'settings': {name: settings.get(name) for name in shared_settings}}


def match_gate_forecasts(frames):
    """Match products across all supplied gates and reject inconsistent labels."""
    keys = ['delivery_day', 'product_start', 'auction', 'variant', 'period']
    prepared = []
    for gate, frame in frames.items():
        frame = frame.copy()
        frame['product_start'] = pd.to_datetime(frame['product_start'], utc=True)
        if not frame['gate'].eq(gate).all():
            raise ValueError(f'Incorrect gate in {gate} report')
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate forecast products in {gate} report')
        frame = frame.loc[np.isfinite(frame['actual']) & np.isfinite(frame['prediction'])]
        prepared.append(frame)
    data = pd.concat(prepared, ignore_index=True)
    if data.empty:
        return data
    # Compare the same target product and model, disregarding the decision gate.
    truth = data.groupby(keys, dropna=False)['actual'].agg(['min', 'max'])
    if not np.allclose(truth['min'], truth['max'], equal_nan=False):
        raise ValueError('Target labels disagree across decision gates')
    counts = data.groupby(keys, dropna=False)['gate'].nunique()
    common = counts[counts == len(frames)].reset_index()[keys]
    matched = data.merge(common, on=keys, how='inner', validate='many_to_one')
    matched['error'] = matched['prediction'] - matched['actual']
    matched['absolute_error'] = matched['error'].abs()
    return matched


def compare_target(root, target, gates, timezone, first_days=3, require_complete=False):
    """Write matched per-gate scores and plots; incomplete inputs remain explicit."""
    root = Path(root)
    output = root / 'gate_comparison' / target
    output.mkdir(parents=True, exist_ok=True)
    # Independent reporting workers can finish the same target concurrently.
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        frames, coverage = {}, []
        reference_contract = None
        for gate in gates:
            directory = root / gate / target
            status_path = directory / 'run/report/status.json'
            manifest_path = directory / 'inputs/manifest.json'
            forecasts_path = directory / 'run/report/matched_forecasts.parquet'
            if not all(path.is_file() for path in (status_path, manifest_path, forecasts_path)):
                coverage.append({'gate': gate, 'status': 'missing_report', 'available_products': 0})
                continue
            status = json.loads(status_path.read_text())
            if not status.get('complete'):
                coverage.append({'gate': gate, 'status': 'incomplete_report', 'available_products': 0})
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest['settings'].get('gate') != gate or manifest['settings'].get('market') != target:
                raise ValueError(f'Wrong gate or target in {manifest_path}')
            contract = comparable_contract(manifest)
            if reference_contract is not None and contract != reference_contract:
                raise ValueError(f'Incompatible model, inputs or settings in {manifest_path}')
            reference_contract = contract
            frame = pd.read_parquet(forecasts_path)
            if not frame['auction'].eq(target).all():
                raise ValueError(f'Wrong target in {forecasts_path}')
            frames[gate] = frame
            coverage.append({'gate': gate, 'status': 'complete', 'available_products': len(frame)})
        required = len(gates)
        if len(frames) < 2:
            pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
            (output / 'status.json').write_text(json.dumps({'complete': False,
                'expected_gates': gates, 'available_gates': list(frames), 'reason': 'fewer_than_two_complete_gates'}, indent=2) + '\n')
            return {'target': target, 'status': 'fewer_than_two_complete_gates'}
        if require_complete and len(frames) < required:
            pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
            (output / 'status.json').write_text(json.dumps({'complete': False,
                'expected_gates': gates, 'available_gates': list(frames), 'reason': 'waiting_for_remaining_gates'}, indent=2) + '\n')
            return {'target': target, 'status': 'waiting_for_remaining_gates'}
        matched = match_gate_forecasts(frames)
        if matched.empty:
            raise ValueError(f'No common forecast products across available gates for {target}')
        available = list(frames)
        for row in coverage:
            row['matched_products'] = int(matched.gate.eq(row['gate']).sum()) if row['gate'] in frames else 0
        pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
        matched.to_parquet(output / 'matched_forecasts.parquet', index=False)
        summaries = []
        for (period, variant, gate), group in matched.groupby(['period', 'variant', 'gate'], sort=True):
            summaries.append({'period': period, 'model': variant, 'gate': gate, **score(group)})
        pd.DataFrame(summaries).to_csv(output / 'scores.csv', index=False)
        daily = (matched.groupby(['period', 'variant', 'gate', 'delivery_day'], as_index=False)
                 .agg(mae=('absolute_error', 'mean'), products=('absolute_error', 'size')))
        daily.to_csv(output / 'daily_mae.csv', index=False)
        local = matched['product_start'].dt.tz_convert(timezone)
        matched['delivery_hour'] = local.dt.hour
        hourly = (matched.groupby(['period', 'variant', 'gate', 'delivery_hour'], as_index=False)
                  .agg(mae=('absolute_error', 'mean'), products=('absolute_error', 'size')))
        hourly.to_csv(output / 'hourly_mae.csv', index=False)
        # Plot periods and models separately so the reserved period cannot enter development scores.
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        for (period, variant), group in matched.groupby(['period', 'variant'], sort=True):
            suffix = f'{period}_{variant}'
            # Every bar is based on the same target products at every gate.
            overall = group.groupby('gate')['absolute_error'].mean().reindex(available)
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.bar(overall.index, overall.values)
            axis.set(title=f'{target} MAE by forecast gate — {period}', xlabel='Decision gate', ylabel='MAE (EUR/MWh)')
            figure.tight_layout()
            figure.savefig(output / f'gate_mae_{suffix}.png', dpi=130)
            plt.close(figure)
            subset = daily[(daily['period'] == period) & (daily['variant'] == variant)]
            figure, axis = plt.subplots(figsize=(11, 4))
            for gate in available:
                series = subset[subset.gate == gate].sort_values('delivery_day')
                axis.plot(pd.to_datetime(series.delivery_day), series.mae, marker='o', markersize=3, label=gate)
            axis.set(title=f'{target} daily MAE by forecast gate — {period}', xlabel='Delivery day', ylabel='MAE (EUR/MWh)')
            axis.legend()
            figure.autofmt_xdate()
            figure.tight_layout()
            figure.savefig(output / f'daily_mae_{suffix}.png', dpi=130)
            plt.close(figure)
            figure, axis = plt.subplots(figsize=(10, 4))
            for gate in available:
                series = hourly[(hourly.period == period) & (hourly.variant == variant) & (hourly.gate == gate)]
                axis.plot(series.delivery_hour, series.mae, marker='o', label=gate)
            axis.set(title=f'{target} MAE by delivery hour — {period}', xlabel=f'Delivery hour ({timezone})', ylabel='MAE (EUR/MWh)')
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f'hourly_mae_{suffix}.png', dpi=130)
            plt.close(figure)
            # Keep detailed daily curves to the first configured days of each month.
            for day, day_data in group.groupby('delivery_day', sort=True):
                if int(str(day)[-2:]) > first_days:
                    continue
                figure, axis = plt.subplots(figsize=(12, 4))
                for gate in available:
                    series = day_data[day_data.gate == gate].sort_values('product_start')
                    axis.plot(series.product_start.dt.tz_convert(timezone), series.error, label=gate)
                axis.axhline(0, color='black', linewidth=0.8)
                axis.set(title=f'{target} forecast error by gate — {day}', xlabel=f'Delivery time ({timezone})', ylabel='Forecast − actual (EUR/MWh)')
                axis.legend()
                figure.autofmt_xdate()
                figure.tight_layout()
                daily_dir = output / 'daily_errors' / str(day)[:7]
                daily_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(daily_dir / f'{day}_{suffix}.png', dpi=130)
                plt.close(figure)
        complete = len(frames) == required
        notes = [f'# {target}: forecast errors across auction gates', '',
                 f"Status: {'complete' if complete else 'partial'}; available gates: {', '.join(available)}.",
                 'All plotted gates use the same target products for each model and period.',
                 'Errors are predicted price minus realized price; lower MAE is better.',
                 'Reserved November results are shown separately. See input manifests for retrospective availability assumptions.', '',
                 '| Period | Model | Gate | Products | MAE | WAPE | RMSE | Bias |',
                 '|---|---|---|---:|---:|---:|---:|---:|']
        for item in summaries:
            wape = item.get('wape')
            formatted = f'{wape * 100:.2f}%' if wape is not None else 'n/a'
            notes.append(f"| {item['period']} | {item['model']} | {item['gate']} | {item['products']} | {item['mae']:.3f} | {formatted} | {item['rmse']:.3f} | {item['bias']:.3f} |")
        (output / 'summary.md').write_text('\n'.join(notes) + '\n')
        (output / 'status.json').write_text(json.dumps({'complete': complete,
            'expected_gates': gates, 'available_gates': available,
            'matched_rows': len(matched), 'coverage_file': 'coverage.csv'}, indent=2) + '\n')
        return {'target': target, 'status': 'complete' if complete else 'partial', 'available_gates': available}


def main():
    """Regenerate gate comparisons from complete saved per-gate reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/chronos2_direct_ida_gates.yaml')
    parser.add_argument('--output', help='Study output directory; defaults to the config')
    parser.add_argument('--target', choices=MARKETS, action='append')
    args = parser.parse_args()
    settings = load_config(args.config)['direct_ida_gate_study']
    root = Path(args.output or settings['output'])
    targets = args.target or [target for target in MARKETS if any(target in targets for targets in settings['gates'].values())]
    for target in targets:
        gates = [gate for gate in MARKETS if target in settings['gates'].get(gate, [])]
        print(json.dumps(compare_target(root, target, gates, settings['timezone'], settings.get('plot_first_days_per_month', 3))), flush=True)


if __name__ == '__main__':
    main()
