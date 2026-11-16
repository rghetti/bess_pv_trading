"""Create TimesFM inputs joining realized historical quantities to forecast horizons."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.adaptive_price_context import adapt_request
from bess_pv_trading.forecasting.studies.price_feature_data import interval_values
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, NORD_FEATURES
from bess_pv_trading.config import load_config


def fill_history(actual, historical_forecast=None, timestamps=None, max_gap_minutes=120):
    """Prefer forecast for gaps; interpolate runs of <= two hours, otherwise zero."""
    result = np.asarray(actual, dtype=float).copy()
    missing = ~np.isfinite(result)
    used_forecast = np.zeros(len(result), dtype=bool)
    if historical_forecast is not None:
        forecast = np.asarray(historical_forecast, dtype=float)
        if forecast.shape != result.shape:
            raise ValueError('Historical forecast and actual lengths differ')
        used_forecast = missing & np.isfinite(forecast)
        result[used_forecast] = forecast[used_forecast]
    clock = pd.DatetimeIndex(timestamps) if timestamps is not None else None
    interpolated = 0
    zero_filled = 0
    i = 0
    while i < len(result):
        if np.isfinite(result[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(result) and not np.isfinite(result[i]) and (clock is None or clock[i] - clock[i - 1] == pd.Timedelta(minutes=15)):
            i += 1
        count = i - start
        left = start - 1 if start > 0 and np.isfinite(result[start - 1]) and (clock is None or clock[start] - clock[start - 1] == pd.Timedelta(minutes=15)) else None
        right = i if i < len(result) and np.isfinite(result[i]) and (clock is None or clock[i] - clock[i - 1] == pd.Timedelta(minutes=15)) else None
        if count * 15 <= max_gap_minutes and (left is not None or right is not None):
            if left is None:
                result[start:i] = result[right]
            elif right is None:
                result[start:i] = result[left]
            else:
                result[start:i] = np.linspace(result[left], result[right], count + 2)[1:-1]
            interpolated += count
        else:
            result[start:i] = 0.0
            zero_filled += count
    return result, int(used_forecast.sum()), interpolated, zero_filled


def prepare(base_dir, snapshot_dir, output_dir, variant, settings):
    """Create one immutable, complete-case bundle from pinned price and actuals snapshots."""
    base_dir, snapshot_dir, output_dir = map(Path, (base_dir, snapshot_dir, output_dir))
    base = json.loads((base_dir / 'manifest.json').read_text())
    snapshot = json.loads((snapshot_dir / 'manifest.json').read_text())
    entry = snapshot['tables']['nord_actuals']
    source = snapshot_dir / entry['path']
    if digest(source) != entry['sha256']:
        raise ValueError('NORD actuals snapshot checksum mismatch')
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frame = pd.read_parquet(source)
    definitions = settings['variants'][variant]
    replacements = definitions.get('replace_forecast_history', {})
    past = definitions.get('past_only', {})
    forecast_root = Path(settings['forecast_snapshot'])
    forecast_manifest = json.loads((forecast_root / 'manifest.json').read_text())
    if digest(forecast_root / 'manifest.json') != base['source_manifest_sha256']:
        raise ValueError('Historical price and forecast snapshots must match')
    forecast_entry = forecast_manifest['tables']['nord_forecasts']
    forecast_path = forecast_root / forecast_entry['path']
    if digest(forecast_path) != forecast_entry['sha256']:
        raise ValueError('Forecast snapshot checksum mismatch')
    forecast_frame = pd.read_parquet(forecast_path)
    excluded = set(definitions.get('exclude_forecast', []))
    forecast_profiles = {name: forecast_frame[(forecast_frame.dataset == name) & (forecast_frame.zone == 'NORD')].sort_values('product_start') for name in NORD_FEATURES if name not in excluded}
    extra_forecasts = set(replacements) - set(NORD_FEATURES)
    if extra_forecasts:
        extra_entry = snapshot['tables']['nord_forecasts']
        extra_path = snapshot_dir / extra_entry['path']
        if digest(extra_path) != extra_entry['sha256']:
            raise ValueError('Additional forecast snapshot checksum mismatch')
        extra_frame = pd.read_parquet(extra_path)
        for name in sorted(extra_forecasts):
            forecast_profiles[name] = extra_frame[(extra_frame.dataset == name) & (extra_frame.zone == 'NORD')].sort_values('product_start')
    features = [base['features'][0], *forecast_profiles]
    past_features = [*base.get('past_features', [])]
    market = base['settings'].get('market', 'DA')
    ida_inputs = None
    if market != 'DA':
        ida_settings = dict(base['settings'])
        ida_inputs = IDAInputs(forecast_root, ida_settings, load_config(settings['auction_config'])['price_feature_search'])
        if 'own_minus_latest_cleared' not in past_features:
            past_features.append('own_minus_latest_cleared')
    profiles = {}
    for dataset in set(replacements.values()) | set(past.values()):
        subset = frame[(frame.dataset == dataset) & (frame.zone == 'NORD')].copy()
        if dataset == 'actual_load':
            subset = subset[subset.direction == 'load']
        else:
            subset = subset[subset.direction == 'production']
        subset = subset.sort_values('product_start')
        if not subset.empty and not subset.unit.eq('MW').all():
            raise ValueError(f'Invalid units for {dataset}')
        profiles[dataset] = subset
    for name in past:
        if name in past_features:
            raise ValueError(f'Duplicate past feature {name}')
        past_features.append(name)
    output_dir.mkdir(parents=True)
    files = {}
    for relative, checksum in base['files'].items():
        original = base_dir / relative
        if digest(original) != checksum:
            raise ValueError(f'Changed base bundle: {original}')
        if not relative.startswith('requests/') and relative != 'coverage.csv':
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            files[relative] = digest(destination)
    records = []
    for relative in sorted(base['files']):
        if not relative.startswith('requests/'):
            continue
        request = json.loads((base_dir / relative).read_text())
        history = request.get('context_product_starts_utc') or request.get('context_starts_utc')
        if history:
            grid = pd.DatetimeIndex(pd.to_datetime(history, format='ISO8601', utc=True))
        else:
            grid = pd.date_range(pd.Timestamp(request['context_start']), periods=len(request['context']), freq='15min').tz_convert('UTC')
        local_days = grid.tz_convert(base['settings']['timezone']).date
        delivery_day = pd.Timestamp(request['delivery_day']).date()
        if any(day >= delivery_day for day in local_days):
            raise ValueError('A realized covariate would include delivery-day observations')
        horizon_grid = pd.DatetimeIndex(pd.to_datetime(request['product_starts_utc'], format='ISO8601', utc=True))
        forecast_values = {}
        for name, subset in forecast_profiles.items():
            forecast_values[name] = interval_values(subset, grid.append(horizon_grid), pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
        spread_counts = {}
        if ida_inputs is not None:
            anchor = np.full(len(grid), np.nan)
            for day in sorted(set(local_days)):
                positions = np.flatnonzero(local_days == day)
                item = ida_inputs.day(day, market)
                indexes = item['starts'].get_indexer(grid[positions])
                valid = indexes >= 0
                anchor[positions[valid]] = item['anchor'][indexes[valid]]
            spread = np.asarray(request['context'], dtype=float) - anchor
            spread, _, spread_interpolated, spread_zero = fill_history(spread, timestamps=grid)
            spread_counts = {'spread_interpolated': spread_interpolated, 'spread_zero_filled': spread_zero}
            request.setdefault('past_covariates', {})['own_minus_latest_cleared'] = spread.tolist()
        request['forecast_covariates'] = {}
        missing = {}
        values_by_dataset = {}
        for dataset, subset in profiles.items():
            values = interval_values(subset, grid, pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
            values_by_dataset[dataset] = values
            missing[dataset] = int((~np.isfinite(values)).sum())
        fill_counts = dict(spread_counts)
        for name, values in forecast_values.items():
            dataset = replacements.get(name)
            historical = values_by_dataset[dataset] if dataset else values[:len(grid)]
            filled, forecast_count, interpolated_count, zero_count = fill_history(historical, values[:len(grid)] if dataset else None, grid)
            label = dataset or name
            fill_counts[f'{label}_forecast_filled'] = forecast_count
            fill_counts[f'{label}_interpolated'] = interpolated_count
            fill_counts[f'{label}_zero_filled'] = zero_count
            future = values[len(grid):]
            if np.isfinite(future).all():
                request['forecast_covariates'][name] = filled.tolist() + future.tolist()
                fill_counts[f'{name}_past_only'] = False
            else:
                request.setdefault('past_covariates', {})[name] = filled.tolist()
                fill_counts[f'{name}_past_only'] = True
        for name, dataset in past.items():
            filled, _, interpolated_count, zero_count = fill_history(values_by_dataset[dataset], timestamps=grid)
            fill_counts[f'{dataset}_interpolated'] = interpolated_count
            fill_counts[f'{dataset}_zero_filled'] = zero_count
            request.setdefault('past_covariates', {})[name] = filled.tolist()
        request['feature_names'] = [features[0], *sorted(request['forecast_covariates'])]
        request['realized_fill_counts'] = fill_counts
        request, info = adapt_request(request, settings['minimum_context_days'], base['settings']['timezone'], 'adaptive')
        records.append({'delivery_day': Path(relative).stem, 'eligible': request is not None, **missing, **fill_counts, **info})
        if request is not None:
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, request)
            files[relative] = digest(path)
    pd.DataFrame(records).to_csv(output_dir / 'coverage.csv', index=False)
    files['coverage.csv'] = digest(output_dir / 'coverage.csv')
    manifest = dict(base)
    manifest.update(files=files, models={'timesfm3': base['models']['timesfm3']},
                    features=features, past_features=past_features,
                    eligible_days=sum(row['eligible'] for row in records), requested_days=len(records),
                    source_actuals_sha256=entry['sha256'], actuals_manifest_sha256=digest(snapshot_dir / 'manifest.json'),
                    base_manifest_sha256=digest(base_dir / 'manifest.json'), realized_variant=variant,
                    realized_builder_sha256=digest(Path(__file__)),
                    realized_feature_definitions=definitions, dynamic_covariates=True,
                    forecast_snapshot_sha256=digest(forecast_root / 'manifest.json'),
                    realized_availability='Past delivery dates assumed available at each gate; historical publication times and revisions unverified.',
                    realized_policy='No delivery-day actuals; forecast first, then historical interpolation for gaps up to two hours, otherwise zero.')
    write_json(output_dir / 'manifest.json', manifest)
    return manifest
