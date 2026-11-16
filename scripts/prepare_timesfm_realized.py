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


def fill_history(actual, historical_forecast=None, local_days=None):
    """Fill completely missing local days with zero, then remaining historical gaps."""
    result = np.asarray(actual, dtype=float).copy()
    zero_filled = np.zeros(len(result), dtype=bool)
    if local_days is not None:
        days = np.asarray(local_days)
        if days.shape != result.shape:
            raise ValueError('Historical days and actual lengths differ')
        for day in np.unique(days):
            positions = days == day
            if not np.isfinite(result[positions]).any():
                result[positions] = 0.0
                zero_filled[positions] = True
    missing = ~np.isfinite(result)
    used_forecast = np.zeros(len(result), dtype=bool)
    if historical_forecast is not None:
        forecast = np.asarray(historical_forecast, dtype=float)
        if forecast.shape != result.shape:
            raise ValueError('Historical forecast and actual lengths differ')
        used_forecast = missing & np.isfinite(forecast)
        result[used_forecast] = forecast[used_forecast]
    remaining = ~np.isfinite(result)
    interpolated = np.zeros(len(result), dtype=bool)
    if remaining.any() and (~remaining).any():
        indices = np.arange(len(result))
        result[remaining] = np.interp(indices[remaining], indices[~remaining], result[~remaining])
        interpolated = remaining
    return result, int(used_forecast.sum()), int(interpolated.sum()), int(zero_filled.sum())


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
    features = [*base['features']]
    past_features = [*base.get('past_features', [])]
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
        missing = {}
        values_by_dataset = {}
        for dataset, subset in profiles.items():
            values = interval_values(subset, grid, pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
            values_by_dataset[dataset] = values
            missing[dataset] = int((~np.isfinite(values)).sum())
        fill_counts = {}
        for name, dataset in replacements.items():
            if name not in request['forecast_covariates']:
                raise ValueError(f'Forecast feature missing in base bundle: {name}')
            old = request['forecast_covariates'][name]
            filled, forecast_count, interpolated_count, zero_count = fill_history(values_by_dataset[dataset], old[:len(grid)], local_days)
            fill_counts[f'{dataset}_forecast_filled'] = forecast_count
            fill_counts[f'{dataset}_interpolated'] = interpolated_count
            fill_counts[f'{dataset}_missing_day_zero_values'] = zero_count
            request['forecast_covariates'][name] = filled.tolist() + old[len(grid):]
        for name, dataset in past.items():
            filled, _, interpolated_count, zero_count = fill_history(values_by_dataset[dataset], local_days=local_days)
            fill_counts[f'{dataset}_interpolated'] = interpolated_count
            fill_counts[f'{dataset}_missing_day_zero_values'] = zero_count
            request.setdefault('past_covariates', {})[name] = filled.tolist()
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
                    realized_feature_definitions=definitions,
                    realized_availability='Past delivery dates assumed available at each gate; historical publication times and revisions unverified.',
                    realized_policy='No delivery-day actuals; entirely missing historical local days use zero; other gaps use matching historical forecasts, then linear interpolation within history.')
    write_json(output_dir / 'manifest.json', manifest)
    return manifest
