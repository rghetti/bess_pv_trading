"""Compare a validation-selected TCN with frozen neural and pvlib PV forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.training.pv_fitting import balanced_weights


def _period_edges(value: str) -> tuple[list[int], list[str]]:
    """Validate four configurable half-open local-clock periods."""
    labels = value.split(',')
    # Enforce the input contract: Exactly five comma-separated period edges are required.
    if len(labels) != 5:
        raise ValueError('Exactly five comma-separated period edges are required')
    minutes = []
    # Validate four configurable half-open local-clock periods for each `label` in `labels`.
    for label in labels:
        hour, minute = map(int, label.split(':'))
        # Enforce the input contract: Invalid local-clock period edge: `label`.
        if not 0 <= minute < 60 or not 0 <= hour <= 24 or (hour == 24 and minute):
            raise ValueError(f'Invalid local-clock period edge: {label}')
        minutes.append(60 * hour + minute)
    # Enforce the input contract: Period edges must increase from 00:00 through 24:00.
    if minutes[0] != 0 or minutes[-1] != 1440 or any(a >= b for a, b in zip(minutes, minutes[1:])):
        raise ValueError('Period edges must increase from 00:00 through 24:00')
    return minutes, [f'{a}–{b}' for a, b in zip(labels, labels[1:])]


def _hash(path: Path) -> str:
    """Fingerprint each frozen result used in the comparison."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, rating: float,
             positions: np.ndarray) -> dict:
    """Balance each observed day and auction within the selected period."""
    group = frame.iloc[positions]
    error = (prediction[positions] - group.measured_ac_mw.to_numpy(dtype=float)) / rating
    # Enforce the input contract: Nonfinite forecast residual on an eligible interval.
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite forecast residual on an eligible interval')
    weight = balanced_weights(group)
    return {'rows': len(group), 'days': group.delivery_day.nunique(),
        'nrmse_percent': 100 * float(np.sqrt(weight @ (error ** 2))),
        'nmae_percent': 100 * float(weight @ np.abs(error)),
        'bias_percent': 100 * float(weight @ error)}


def _plot_day(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
              farm_id: int, rating: float, day: str, timezone: str, destination: Path) -> None:
    """Show all three AC models and weather on one local-time axis."""
    views = frame.loc[frame.delivery_day.eq(day)].copy()
    # Show all three AC models and weather on one local-time axis when `views.empty`.
    if views.empty:
        return
    physical = views.groupby('interval_start_utc', sort=True).agg(
        measured_ac_mw=('measured_ac_mw', 'first'),
        satellite_ghi_w_m2=('satellite_ghi_w_m2', 'first')).reset_index()
    boundary = (pd.Timestamp(day).tz_localize(timezone) + pd.DateOffset(days=1)).tz_convert('UTC')
    next_point = frame.loc[frame.interval_start_utc.eq(boundary)]
    # Update `physical` when this condition is met.
    if not next_point.empty:
        physical = pd.concat([physical, pd.DataFrame([{
            'interval_start_utc': boundary,
            'measured_ac_mw': next_point.measured_ac_mw.iloc[0],
            'satellite_ghi_w_m2': next_point.satellite_ghi_w_m2.iloc[0]}])], ignore_index=True)
    figure, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True, layout='constrained')
    upper = max(rating, float(physical.measured_ac_mw.max()),
        *(float(prediction[views.index].max()) for prediction in predictions.values()))
    # Show all three AC models and weather on one local-time axis for each `(axis, (model, prediction))` in `zip(axes[:3], predictions.items())`.
    for axis, (model, prediction) in zip(axes[:3], predictions.items()):
        axis.plot(physical.interval_start_utc, physical.measured_ac_mw,
            color='black', lw=2, label='Measured AC')
        # Show all three AC models and weather on one local-time axis for each `(auction, positions)` in `views.groupby('auction', sort=True).indices.items()`.
        for auction, positions in views.groupby('auction', sort=True).indices.items():
            subset = views.iloc[positions]
            axis.plot(subset.interval_start_utc, prediction[subset.index], lw=1,
                label=f'{auction} {model}')
        axis.axhline(rating, color='gray', ls='--', lw=1, label='Farm AC rating')
        axis.set_ylim(-.03 * upper, 1.05 * upper)
        axis.set_ylabel('AC power (MW)')
    axes[-1].plot(physical.interval_start_utc, physical.satellite_ghi_w_m2,
        color='black', lw=2, label='Satellite GHI (diagnostic)')
    # Show all three AC models and weather on one local-time axis for each `(auction, group)` in `views.groupby('auction', sort=True)`.
    for auction, group in views.groupby('auction', sort=True):
        axes[-1].plot(group.interval_start_utc, group.forecast_ghi,
            lw=1, label=f'{auction} GHI forecast')
    axes[-1].set_ylabel('GHI (W/m²)')
    axes[-1].set_xlabel(f'Delivery time, {timezone}')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo(timezone)))
    # Show all three AC models and weather on one local-time axis for each `axis` in `axes`.
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(ncol=3, fontsize=7)
    figure.suptitle(f'Farm {farm_id}, {day}: TCN, frozen neural, pvlib and weather')
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run(search_run: Path, farm_id: int, *, periods: str, timezone: str,
        good_days_csv: Path, quality_plots: Path, output_root: Path | None) -> Path:
    """Save four-period errors, dawn jumps and four-panel daily evidence."""
    farm = search_run / f'farm_{farm_id}'
    corpus_path = farm / 'parquet/corpus.parquet'
    prediction_path = farm / 'parquet/predictions.parquet'
    selection_path = farm / 'json/selection.json'
    provenance_path = farm / 'json/input_provenance.json'
    edges, labels = _period_edges(periods)
    selection = json.loads(selection_path.read_text())
    selected = selection['selected_new_model']
    rating = float(json.loads(provenance_path.read_text())['rating_mw'])
    frame = pd.read_parquet(corpus_path)
    saved = pd.read_parquet(prediction_path)
    keys = ['interval_start_utc', 'delivery_day', 'auction', 'measured_ac_mw']
    # Enforce the input contract: Search predictions are not aligned to raw measured AC.
    if not frame[keys].equals(saved[keys]):
        raise ValueError('Search predictions are not aligned to raw measured AC')
    models = {'Selected TCN': saved[f'prediction_{selected}_mw'].to_numpy(dtype=float),
        'Frozen neural': saved.prediction_selected_mw.to_numpy(dtype=float),
        'pvlib': saved.prediction_pvlib_mw.to_numpy(dtype=float)}
    # Enforce the input contract: A saved AC prediction is nonfinite.
    if not all(np.isfinite(prediction).all() for prediction in models.values()):
        raise ValueError('A saved AC prediction is nonfinite')
    local = pd.DatetimeIndex(frame.interval_start_utc).tz_convert(timezone)
    minute = local.hour * 60 + local.minute
    period_index = np.searchsorted(edges, minute, side='right') - 1
    # Enforce the input contract: A physical interval lies outside the configured local periods.
    if np.any((period_index < 0) | (period_index >= 4)):
        raise ValueError('A physical interval lies outside the configured local periods')
    frame['local_period'] = np.asarray(labels)[period_index]
    root = output_root or farm / 'diurnal_diagnostics'
    # Save four-period errors, dawn jumps and four-panel daily evidence for each `name` in `('csv', 'json', 'plots')`.
    for name in ('csv', 'json', 'plots'):
        (root / name).mkdir(parents=True, exist_ok=True)
    metrics = []
    # Save four-period errors, dawn jumps and four-panel daily evidence for each `split` in `('train', 'validation', 'test')`.
    for split in ('train', 'validation', 'test'):
        # Save four-period errors, dawn jumps and four-panel daily evidence for each `period` in `labels`.
        for period in labels:
            # Save four-period errors, dawn jumps and four-panel daily evidence for each `scope` in `('all_eligible', 'daylight_only')`.
            for scope in ('all_eligible', 'daylight_only'):
                population = (frame.split.eq(split) & frame.eligible & frame.local_period.eq(period))
                # When `scope == 'daylight_only'`, set `population` using `frame.daylight`.
                if scope == 'daylight_only':
                    population &= frame.daylight
                # Save four-period errors, dawn jumps and four-panel daily evidence for each `auction` in `('ALL', 'DA', 'IDA1', 'IDA2', 'IDA3')`.
                for auction in ('ALL', 'DA', 'IDA1', 'IDA2', 'IDA3'):
                    mask = population if auction == 'ALL' else population & frame.auction.eq(auction)
                    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
                    # Save four-period errors, dawn jumps and four-panel daily evidence for each `(model, prediction)` in `models.items()`.
                    for model, prediction in models.items():
                        row = {'split': split, 'period': period, 'scope': scope,
                            'auction': auction, 'model': model}
                        row.update(_metrics(frame, prediction, rating, positions) if len(positions)
                            else {'rows': 0, 'days': 0, 'nrmse_percent': np.nan,
                                'nmae_percent': np.nan, 'bias_percent': np.nan})
                        metrics.append(row)
    pd.DataFrame(metrics).to_csv(root / 'csv/period_metrics.csv', index=False)
    overall_source = pd.read_csv(farm / 'csv/metrics.csv')
    comparison_ids = {selected: 'Selected TCN', 'selected': 'Frozen neural', 'pvlib': 'pvlib'}
    overall = overall_source.loc[overall_source.model.isin(comparison_ids)].copy()
    overall['model_label'] = overall.model.map(comparison_ids)
    overall.to_csv(root / 'csv/overall_metrics.csv', index=False)
    dawn = []
    # Save four-period errors, dawn jumps and four-panel daily evidence for each `((day, auction), positions)` in `frame.groupby(['delivery_day', 'auction'], sort=True).indices.items()`.
    for (day, auction), positions in frame.groupby(['delivery_day', 'auction'], sort=True).indices.items():
        # Skip the current item when `auction == 'IDA3'`.
        if auction == 'IDA3':
            continue
        group = frame.iloc[positions].sort_values('interval_start_utc')
        daylight = np.flatnonzero(group.daylight.to_numpy(dtype=bool))
        # Skip the current item when `len(daylight) == 0 or daylight[0] == 0`.
        if len(daylight) == 0 or daylight[0] == 0:
            continue
        first, previous = group.iloc[daylight[0]], group.iloc[daylight[0] - 1]
        # Skip the current item when `first.interval_start_utc - previous.interval_start_utc != pd.Timedelta(minutes=15)`.
        if first.interval_start_utc - previous.interval_start_utc != pd.Timedelta(minutes=15):
            continue
        # Save four-period errors, dawn jumps and four-panel daily evidence for each `(model, prediction)` in `models.items()`.
        for model, prediction in models.items():
            dawn.append({'delivery_day': day, 'auction': auction, 'split': first.split,
                'model': model, 'first_daylight_utc': str(first.interval_start_utc),
                'first_daylight_local': str(first.interval_start_utc.tz_convert(timezone)),
                'previous_mw': float(prediction[previous.name]),
                'first_mw': float(prediction[first.name]),
                'step_mw': float(prediction[first.name] - prediction[previous.name]),
                'first_measured_mw': float(first.measured_ac_mw),
                'forecast_ghi_w_m2': float(first.forecast_ghi),
                'satellite_ghi_w_m2': float(first.satellite_ghi_w_m2)})
    pd.DataFrame(dawn).to_csv(root / 'csv/dawn_transitions.csv', index=False)
    days = set(pd.read_csv(good_days_csv).delivery_day.astype(str))
    days.update(path.stem.removeprefix('quality_day_') for path in quality_plots.glob('quality_day_*.png'))
    # Save four-period errors, dawn jumps and four-panel daily evidence for each `day` in `sorted(days)`.
    for day in sorted(days):
        _plot_day(frame, models, farm_id, rating, day, timezone,
            root / 'plots' / f'daily_models_{day}.png')
    test = pd.DataFrame(metrics)
    test = test.loc[test.split.eq('test') & test.scope.eq('daylight_only') & test.auction.eq('ALL')]
    figure, axis = plt.subplots(figsize=(11, 5), layout='constrained')
    width = .24
    # Save four-period errors, dawn jumps and four-panel daily evidence for each `(number, model)` in `enumerate(models)`.
    for number, model in enumerate(models):
        values = test.loc[test.model.eq(model)].set_index('period').reindex(labels)
        axis.bar(np.arange(4) + (number - 1) * width,
            values.nrmse_percent, width, label=model)
    axis.set_xticks(np.arange(4), labels)
    axis.set_ylabel('Daylight RMSE / AC rating (%)')
    axis.set_xlabel(f'Local delivery period, {timezone}')
    axis.set_title(f'Farm {farm_id}: held-out test error by period')
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    figure.savefig(root / 'plots/period_errors_test.png', dpi=150)
    plt.close(figure)
    manifest = {'search_run': str(search_run), 'farm_id': farm_id,
        'selected_tcn': selected, 'rating_mw': rating, 'timezone': timezone,
        'period_edges_local': periods.split(','), 'models': list(models),
        'plotted_days': sorted(days),
        'input_sha256': {str(path): _hash(path) for path in
            (corpus_path, prediction_path, selection_path, provenance_path, good_days_csv)},
        'selection_policy': selection['selection'],
        'test_interpretation': 'Test period was previously inspected; no test labels selected this TCN.'}
    (root / 'json/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return root


def main() -> None:
    """Run the reproducible offline comparison from explicit input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-run', required=True, type=Path)
    parser.add_argument('--farm-id', required=True, type=int)
    parser.add_argument('--period-edges', required=True)
    parser.add_argument('--timezone', required=True)
    parser.add_argument('--good-days-csv', required=True, type=Path)
    parser.add_argument('--quality-plots', required=True, type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    print(run(args.search_run, args.farm_id, periods=args.period_edges,
        timezone=args.timezone, good_days_csv=args.good_days_csv,
        quality_plots=args.quality_plots, output_root=args.output_root))


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
