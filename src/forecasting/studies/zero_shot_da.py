"""Prepare target-only DA requests and score frozen foundation-model forecasts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .price_feature_data import _price_sources
from .rolling_price_data import PriceIndex, delivery_grid, pad_curve, slot_numbers
from .zero_shot_price_scoring import matched_predictions, score


def digest(path):
    """Hash portable artifacts without loading large files into memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    """Replace a JSON artifact atomically after a complete durable write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, destination)


def reference_curve(index, day, starts, lag, cutoff, timezone):
    """Use the same prior local product; repeat first 02:xx if no second exists."""
    previous = delivery_grid(pd.Timestamp(day) - pd.Timedelta(days=lag), 15, timezone)
    curve = pad_curve(previous, index.lookup(previous, cutoff), timezone)
    for position in range(4):
        if not np.isfinite(curve[96 + position]):
            curve[96 + position] = curve[8 + position]
    return curve[slot_numbers(starts, timezone)]


def prepare_day(index, source, day, settings, cancelled=False):
    """Separate the published historical input from delivery-day evaluation labels."""
    timezone = settings['timezone']
    day = pd.Timestamp(day)
    starts = delivery_grid(day, 15, timezone)
    first = (day - pd.Timedelta(days=settings['context_days'])).tz_localize(timezone)
    history = pd.date_range(first, day.tz_localize(timezone), freq='15min', inclusive='left').tz_convert('UTC')
    decision_day = day + pd.Timedelta(days=settings['decision_day_offset'])
    cutoff = pd.Timestamp(f"{decision_day.date()} {settings['decision_time']}", tz=timezone).tz_convert('UTC')
    context = index.lookup(history, cutoff)
    actual = index.lookup(starts)
    source_starts = pd.DatetimeIndex(pd.to_datetime(source.product_start, utc=True)).as_unit('ns').asi8
    positions = np.searchsorted(source_starts, history.as_unit('ns').asi8, side='right') - 1
    durations = (pd.to_datetime(source.product_end, utc=True) - pd.to_datetime(source.product_start, utc=True)).dt.total_seconds().to_numpy() / 60
    native_minutes = np.where(positions >= 0, durations[np.maximum(positions, 0)], np.nan)
    # Reject target products that are not genuinely quarter-hour labels.
    target_positions = np.searchsorted(source_starts, starts.as_unit('ns').asi8, side='right') - 1
    target_native = (target_positions >= 0) & (durations[np.maximum(target_positions, 0)] == 15)
    actual[~target_native] = np.nan
    published = pd.DatetimeIndex(pd.to_datetime(source.available_at, utc=True)).as_unit('ns').asi8
    active = np.isfinite(context)
    coverage = {'delivery_day': str(day.date()), 'products': len(starts),
                'context_products': len(history), 'context_missing': int((~active).sum()),
                'hourly_context_products': int(((native_minutes == 60) & active).sum()),
                'native_context_products': int(((native_minutes == 15) & active).sum()),
                'observed_products': int(np.isfinite(actual).sum()), 'cancelled': bool(cancelled),
                'eligible': bool(active.all() and not cancelled and np.isfinite(actual).any())}
    # Unexpected resolutions must not masquerade as approved history.
    if np.any(active & ~np.isin(native_minutes, [15, 60])):
        raise ValueError(f'Unexpected native history resolution for {day.date()}')
    if active.any() and (published[positions[active]] > cutoff.value).any():
        raise AssertionError('Unpublished prices escaped the history mask')
    request = None
    if coverage['eligible']:
        request = {'delivery_day': str(day.date()), 'decision_at': cutoff.isoformat(),
                   'context_start': history[0].isoformat(), 'context_end_exclusive': starts[0].isoformat(),
                   'context': context.tolist(), 'context_native_minutes': native_minutes.astype(int).tolist(),
                   'product_starts_utc': [stamp.isoformat() for stamp in starts],
                   'horizon': len(starts), 'feature_names': ['past_NORD_DA_price'],
                   'max_context_publication_at': pd.Timestamp(int(published[positions[active]].max()), tz='UTC').isoformat()}
    labels = pd.DataFrame({'delivery_day': str(day.date()), 'product_start': starts,
                           'actual': actual, 'latest_reference': reference_curve(index, day, starts, 1, cutoff, timezone),
                           'weekly_reference': reference_curve(index, day, starts, 7, cutoff, timezone),
                           'target_cancelled': bool(cancelled)})
    if cancelled:
        labels['actual'] = np.nan
    return request, labels, coverage


def prepare_bundle(snapshot, output, settings, model_specs):
    """Export a small DA-only bundle, preserving the original snapshot lineage."""
    snapshot, output = Path(snapshot), Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new immutable bundle directory: {output}')
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    entry = manifest['tables']['nord_auctions']
    source_path = snapshot / entry['path']
    if digest(source_path) != entry['sha256']:
        raise ValueError('Pinned auction table checksum mismatch')
    prices = pd.read_parquet(source_path)
    prices = prices.loc[prices.dataset.eq('day_ahead_prices') & prices.zone.eq('NORD')].copy()
    if not prices.unit.eq('EUR/MWh').all():
        raise ValueError('Unexpected DA price unit')
    source = _price_sources(prices, {'timezone': settings['timezone'], 'auctions': {'DA': {
        'dataset': 'day_ahead_prices', 'publication_time': settings['publication_time'],
        'publication_day_offset': settings['publication_day_offset']}}})['DA']
    index = PriceIndex(source)
    local_days = pd.DatetimeIndex(pd.to_datetime(source.product_start, utc=True)).tz_convert(settings['timezone']).strftime('%Y-%m-%d')
    # Infer the endpoint only from a complete native delivery day, never row max alone.
    complete_days = []
    for text in sorted(set(local_days)):
        subset = source.loc[local_days == text]
        grid = delivery_grid(text, 15, settings['timezone'])
        durations = pd.to_datetime(subset.product_end, utc=True) - pd.to_datetime(subset.product_start, utc=True)
        if len(subset) == len(grid) and (durations == pd.Timedelta(minutes=15)).all() and np.isfinite(index.lookup(grid)).all():
            complete_days.append(text)
    if not complete_days:
        raise ValueError('No complete native quarter-hour DA day in snapshot')
    end = settings.get('end_exclusive') or str((pd.Timestamp(complete_days[-1]) + pd.Timedelta(days=1)).date())
    days = pd.date_range(settings['start'], end, inclusive='left')
    if days.empty:
        raise ValueError('Empty probe period')
    output.mkdir(parents=True)
    notices = {(item['dataset'], item['delivery_day']) for item in manifest.get('gme_cancellation_notices', [])}
    files, records, labels = {}, [], []
    for day in days:
        text = str(day.date())
        request, label, coverage = prepare_day(index, source, day, settings, ('day_ahead_prices', text) in notices)
        records.append(coverage)
        labels.append(label)
        if request is not None:
            path = output / 'requests' / f'{text}.json'
            write_json(path, request)
            files[str(path.relative_to(output))] = digest(path)
    pd.concat(labels, ignore_index=True).to_parquet(output / 'labels.parquet', index=False)
    pd.DataFrame(records).to_csv(output / 'coverage.csv', index=False)
    # Retain the native source records and source checksums in the portable bundle.
    source_day = pd.to_datetime(source.product_start, utc=True).dt.tz_convert(settings['timezone']).dt.date
    first = (pd.Timestamp(settings['start']) - pd.Timedelta(days=settings['context_days'])).date()
    source.loc[(source_day >= first) & (source_day < pd.Timestamp(end).date())].to_parquet(output / 'source_da.parquet', index=False)
    write_json(output / 'source_manifest.json', manifest)
    for name in ('labels.parquet', 'coverage.csv', 'source_da.parquet', 'source_manifest.json'):
        files[name] = digest(output / name)
    result = {'schema': 1, 'settings': {**settings, 'end_exclusive': end}, 'models': model_specs,
              'files': files, 'eligible_days': sum(row['eligible'] for row in records),
              'requested_days': len(records), 'source_manifest_sha256': digest(snapshot / 'manifest.json'),
              'source_auction_sha256': entry['sha256'], 'features': ['past_NORD_DA_price'],
              'resolution_policy': 'Repeat native hourly prices within their interval; retain native 15-minute labels.',
              'baseline_dst_policy': 'Previous local clock; second autumn occurrence uses first if absent.',
              'publication_assumption': 'Recorded published_at, otherwise configured previous-day publication clock.',
              'pretrained_overlap': 'Possible; checkpoint releases postdate part of this retrospective evaluation.',
              'november_policy': 'Report November 2026 separately, with no tuning or selection.'}
    write_json(output / 'manifest.json', result)
    return result


def report(bundle, run, output):
    """Score common products and produce daily price plots without loading weights."""
    from matplotlib import pyplot as plt

    bundle, run, output = Path(bundle), Path(run), Path(output)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(bundle / relative) != expected:
            raise ValueError(f'Bundle checksum mismatch: {relative}')
    labels = pd.read_parquet(bundle / 'labels.parquet')
    labels['product_start'] = pd.to_datetime(labels.product_start, utc=True)
    rows, status, runtimes = [], [], []
    for model in manifest['models']:
        for relative, expected in manifest['files'].items():
            if not relative.startswith('requests/'):
                continue
            day = Path(relative).stem
            paths = list((run / model).glob(f'shard-*/predictions/{day}.json'))
            if len(paths) > 1:
                raise ValueError(f'Duplicate delivery-day forecast across shards: {model}/{day}')
            if not paths:
                status.append({'model': model, 'day': day, 'status': 'missing_prediction'})
                continue
            path = paths[0]
            result = json.loads(path.read_text())
            payload = {key: value for key, value in result.items() if key != 'payload_sha256'}
            payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
            identity_path = path.parent.parent / 'identity.json'
            identity = json.loads(identity_path.read_text())
            identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if result.get('payload_sha256') != payload_hash or result.get('run_identity') != identity_hash:
                raise ValueError(f'Prediction integrity mismatch: {path}')
            if result['request_sha256'] != expected or result['bundle_sha256'] != digest(bundle / 'manifest.json'):
                raise ValueError(f'Prediction provenance mismatch: {path}')
            prediction = np.asarray(result['quantiles'], dtype=float)
            starts = pd.to_datetime(result['product_starts_utc'], utc=True)
            if prediction.shape != (len(starts), 9) or not np.isfinite(prediction).all():
                raise ValueError(f'Invalid saved forecast: {path}')
            data = pd.DataFrame({'delivery_day': day, 'product_start': starts, 'auction': 'DA', 'gate': 'DA',
                                 'variant': model, 'prediction': prediction[:, 4]})
            for column, values in zip([f'q{q:02}' for q in range(10, 100, 10)], prediction.T):
                data[column] = values
            rows.append(data)
            runtimes.append({'model': model, 'day': day, 'inference_seconds': result.get('elapsed_seconds'),
                             'quantile_crossing_products': result.get('quantile_crossing_products')})
            status.append({'model': model, 'day': day, 'status': 'complete'})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runtimes).to_csv(output / 'runtime.csv', index=False)
    pd.DataFrame(status).to_csv(output / 'completion.csv', index=False)
    if not rows:
        raise ValueError('No saved predictions; completion.csv records missing work')
    forecasts = pd.concat(rows, ignore_index=True).merge(labels, on=['delivery_day', 'product_start'], validate='many_to_one')
    forecasts['period'] = np.where(forecasts.delivery_day >= manifest['settings']['november_holdout_start'], 'november_reserved', 'retrospective_development')
    forecasts['month'] = forecasts.delivery_day.str[:7]
    forecasts['hour'] = forecasts.product_start.dt.tz_convert(manifest['settings']['timezone']).dt.hour
    forecasts.to_parquet(output / 'all_forecasts.parquet', index=False)
    common = matched_predictions(forecasts, list(manifest['models']))
    common.to_parquet(output / 'matched_forecasts.parquet', index=False)
    summaries = []
    for dimensions in (['period'], ['period', 'month'], ['period', 'hour'], ['period', 'delivery_day']):
        for key, group in common.groupby(['variant', *dimensions], sort=True):
            summaries.append({'breakdown': dimensions[-1], **dict(zip(['variant', *dimensions], key)), **score(group)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'scores.csv', index=False)
    for breakdown, column in [('month', 'month'), ('hour', 'hour')]:
        figure, axis = plt.subplots(figsize=(11, 4))
        selected = summary.loc[summary.breakdown.eq(breakdown)] if not summary.empty else summary
        for model, group in selected.groupby('variant'):
            group = group.sort_values(column)
            axis.plot(group[column].astype(str), group.mae, marker='o', label=model)
        axis.set(title=f'NORD DA mean absolute error by {column}', ylabel='EUR/MWh', xlabel=column)
        if not selected.empty:
            axis.legend()
        axis.tick_params(axis='x', rotation=45)
        figure.tight_layout()
        figure.savefig(output / f'{column}_mae.png', dpi=130)
        plt.close(figure)
    # Baselines are scored on exactly the same common products as the models.
    baseline_rows = []
    products = common.drop_duplicates(['delivery_day', 'product_start'])
    for name in ('weekly_reference', 'latest_reference'):
        for period, group in products.groupby('period'):
            baseline = group.drop(columns=[f'q{q:02}' for q in range(10, 100, 10)], errors='ignore')
            baseline_rows.append({'variant': name, 'period': period, **score(baseline.assign(prediction=baseline[name]))})
    pd.DataFrame(baseline_rows).to_csv(output / 'baseline_scores.csv', index=False)
    # Emit an interpretable Markdown companion automatically for every run.
    text = ['# DA zero-shot run summary', '',
            'Scores use products matched across all configured models. November 2026 is reported separately.', '',
            'Archived covariate issue times can be unverified; consult the pinned input manifest.', '']
    for _, row in summary.loc[summary.breakdown.eq('period')].iterrows():
        text.extend([f"## {row['period']} — {row['variant']}", '', '| Metric | Value |', '|---|---:|'])
        for name, value in row.items():
            if name in ('breakdown', 'variant', 'period', 'month', 'hour', 'delivery_day') or pd.isna(value):
                continue
            text.append(f'| {name} | {value:.4f} |' if isinstance(value, (float, np.floating)) else f'| {name} | {value} |')
        text.append('')
    text.extend(['[Complete metrics](scores.csv) · [Baseline metrics](baseline_scores.csv) · '
                 '[Daily metrics](daily_scores.csv) · [Completion](completion.csv)', ''])
    (output / 'summary.md').write_text('\n'.join(text))
    daily = []
    for (model, day), group in common.groupby(['variant', 'delivery_day']):
        daily.append({'variant': model, 'delivery_day': day, **score(group)})
    pd.DataFrame(daily).to_csv(output / 'daily_scores.csv', index=False)
    for day, group in common.groupby('delivery_day'):
        # Limit daily figures to the configured first calendar days each month.
        if int(day[-2:]) > int(manifest['settings'].get('plot_first_days_per_month', 3)):
            continue
        figure, axis = plt.subplots(figsize=(12, 4))
        actual = group.drop_duplicates('product_start').sort_values('product_start')
        axis.plot(actual.product_start, actual.actual, color='black', label='realized', linewidth=1.8)
        axis.plot(actual.product_start, actual.weekly_reference, color='gray', label='weekly naive', alpha=0.6)
        for model, values in group.groupby('variant'):
            values = values.sort_values('product_start')
            axis.plot(values.product_start, values.prediction, label=model, linewidth=1)
        axis.set(title=f'NORD DA {day} — timestamps UTC', ylabel='EUR/MWh')
        axis.legend(ncol=3)
        figure.autofmt_xdate()
        figure.tight_layout()
        directory = output / 'plots' / day[:7]
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f'{day}.png', dpi=130)
        plt.close(figure)
    expected = manifest['eligible_days'] * len(manifest['models'])
    write_json(output / 'status.json', {'expected_daily_predictions': expected,
        'saved_daily_predictions': len(rows), 'complete': len(rows) == expected,
        'matched_products_per_model': len(common) // len(manifest['models']),
        'scoring': 'Incomplete runs are interim; only products common to all configured models are compared.'})
