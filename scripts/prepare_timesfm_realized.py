"""Create TimesFM inputs joining realized historical quantities to forecast horizons."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.adaptive_price_context import adapt_request
from bess_pv_trading.forecasting.data.pv_calendar import italian_national_holiday
from bess_pv_trading.forecasting.studies.price_feature_data import interval_values
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, NORD_FEATURES
from bess_pv_trading.config import load_config


def available_at_gate(frame, cutoff, kind, policy):
    """Mask source rows using recorded timestamps or explicit publication assumptions.

    A document creation time is only a necessary condition, never proof of
    publication. Deadline-based filtering remains an explicitly retrospective
    assumption when the historical publication timestamp is unavailable.
    """
    if frame.empty:
        return frame
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError('Decision timestamp must have a timezone')
    cutoff = cutoff.tz_convert('UTC')
    known = pd.to_datetime(frame.get('historical_available_at', pd.Series(pd.NaT, index=frame.index)), utc=True)
    created = pd.to_datetime(frame.get('source_document_time', pd.Series(pd.NaT, index=frame.index)), utc=True)
    if policy.get('revision_policy') == 'latest_retrospective':
        created_ok = pd.Series(True, index=frame.index)
    elif policy.get('require_document_time', True):
        created_ok = created.notna() & (created <= cutoff)
    else:
        created_ok = created.isna() | (created <= cutoff)
    if kind == 'actual':
        assumed = pd.to_datetime(frame.product_end, utc=True) + pd.Timedelta(minutes=policy['actual_lag_minutes'])
    else:
        dataset = str(frame.dataset.iloc[0])
        if dataset not in policy['forecast_deadline_local']:
            raise ValueError(f'No forecast publication assumption for {dataset}')
        local_day = pd.to_datetime(frame.product_start, utc=True).dt.tz_convert(policy['timezone']).dt.tz_localize(None).dt.normalize()
        days_before = int(policy.get('forecast_days_before', 1))
        hour, minute = map(int, policy['forecast_deadline_local'][dataset].split(':'))
        local_deadline = local_day - pd.Timedelta(days=days_before) + pd.Timedelta(hours=hour, minutes=minute)
        assumed = local_deadline.dt.tz_localize(policy['timezone']).dt.tz_convert('UTC')
    if policy['mode'] == 'recorded_only':
        effective = known
    elif policy['mode'] == 'deadline_assumption':
        effective = known.fillna(assumed)
    else:
        raise ValueError('Unknown publication availability mode')
    # Even erroneous metadata cannot make actuals eligible before their interval ends.
    if kind == 'actual':
        effective = effective.where(effective >= pd.to_datetime(frame.product_end, utc=True))
    return frame.loc[created_ok & effective.notna() & (effective <= cutoff)].copy()


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
    forecast_source_root = snapshot_dir if settings.get('use_actual_snapshot_forecasts') else forecast_root
    forecast_source_manifest = snapshot if settings.get('use_actual_snapshot_forecasts') else forecast_manifest
    forecast_entry = forecast_source_manifest['tables']['nord_forecasts']
    forecast_path = forecast_source_root / forecast_entry['path']
    if digest(forecast_path) != forecast_entry['sha256']:
        raise ValueError('Forecast snapshot checksum mismatch')
    forecast_frame = pd.read_parquet(forecast_path)
    excluded = set(definitions.get('exclude_forecast', []))
    forecast_profiles = {name: forecast_frame[(forecast_frame.dataset == name) & (forecast_frame.zone == 'NORD')].sort_values('product_start') for name in NORD_FEATURES if name not in excluded}
    extra_forecasts = set(replacements) - set(NORD_FEATURES) - set(forecast_profiles)
    if extra_forecasts:
        extra_entry = snapshot['tables']['nord_forecasts']
        extra_path = snapshot_dir / extra_entry['path']
        if digest(extra_path) != extra_entry['sha256']:
            raise ValueError('Additional forecast snapshot checksum mismatch')
        extra_frame = pd.read_parquet(extra_path)
        for name in sorted(extra_forecasts):
            forecast_profiles[name] = extra_frame[(extra_frame.dataset == name) & (extra_frame.zone == 'NORD')].sort_values('product_start')
    features = [base['features'][0], *forecast_profiles]
    calendar = settings.get('calendar', {})
    if calendar.get('weekday_one_hot'):
        features.extend(f'weekday_{day}' for day in range(7))
    if calendar.get('italian_holiday'):
        features.append('italian_national_holiday')
    past_features = [*base.get('past_features', [])]
    market = base['settings'].get('market', 'DA')
    ida_inputs = None
    include_spread = settings.get('include_historical_spread', True)
    if market != 'DA' and include_spread:
        ida_settings = dict(base['settings'])
        ida_inputs = IDAInputs(forecast_root, ida_settings, load_config(settings['auction_config'])['price_feature_search'])
        if 'own_minus_latest_cleared' not in past_features:
            past_features.append('own_minus_latest_cleared')
    profiles = {}
    for dataset in set(replacements.values()) | set(past.values()):
        selector = settings.get('actual_dataset_selectors', {}).get(dataset, {'dataset': dataset})
        subset = frame[(frame.dataset == selector['dataset']) & (frame.zone == 'NORD')].copy()
        if 'psr_type' in selector:
            subset = subset[subset.psr_type == selector['psr_type']]
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
        policy = settings.get('availability_policy')
        cutoff = request.get('decision_at')
        if policy and cutoff is None:
            raise ValueError('Publication-aware input requires decision_at in the base request')
        forecast_values = {}
        availability_counts = {}
        for name, subset in forecast_profiles.items():
            if policy:
                eligible_subset = available_at_gate(subset, cutoff, 'forecast', policy)
                before = interval_values(subset, grid.append(horizon_grid), pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
                after = interval_values(eligible_subset, grid.append(horizon_grid), pd.to_numeric(eligible_subset.value, errors='coerce').to_numpy(float))
                availability_counts[f'{name}_unavailable_at_gate'] = int((np.isfinite(before) & ~np.isfinite(after)).sum())
                subset = eligible_subset
            forecast_values[name] = interval_values(subset, grid.append(horizon_grid), pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
            earliest = settings.get('future_forecast_earliest_gate', {}).get(name)
            if earliest is not None:
                gate_order = ('DA', 'IDA1', 'IDA2', 'IDA3')
                gate = base['settings'].get('gate', market)
                if gate not in gate_order or earliest not in gate_order:
                    raise ValueError('Unknown decision gate in future forecast policy')
                if gate_order.index(gate) < gate_order.index(earliest):
                    forecast_values[name][len(grid):] = np.nan
        spread_counts = {}
        if not include_spread:
            request.get('past_covariates', {}).pop('own_minus_latest_cleared', None)
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
            if policy:
                eligible_subset = available_at_gate(subset, cutoff, 'actual', policy)
                before = interval_values(subset, grid, pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
                after = interval_values(eligible_subset, grid, pd.to_numeric(eligible_subset.value, errors='coerce').to_numpy(float))
                availability_counts[f'{dataset}_unavailable_at_gate'] = int((np.isfinite(before) & ~np.isfinite(after)).sum())
                subset = eligible_subset
            values = interval_values(subset, grid, pd.to_numeric(subset.value, errors='coerce').to_numpy(float))
            values_by_dataset[dataset] = values
            missing[dataset] = int((~np.isfinite(values)).sum())
        fill_counts = {**spread_counts, **availability_counts}
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
        # Calendar values are known for the entire history and prediction horizon.
        calendar_grid = grid.append(horizon_grid)
        if calendar.get('weekday_one_hot'):
            weekdays = calendar_grid.tz_convert(base['settings']['timezone']).dayofweek
            for day in range(7):
                request['forecast_covariates'][f'weekday_{day}'] = (weekdays == day).astype(float).tolist()
        if calendar.get('italian_holiday'):
            request['forecast_covariates']['italian_national_holiday'] = italian_national_holiday(calendar_grid, base['settings']['timezone']).astype(float).tolist()
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
    models = base['models']
    if settings.get('model_config'):
        specs = load_config(settings['model_config'])['zero_shot_price_study']['models']
        models = {name: specs[name] for name in settings['models']}
    else:
        models = {'timesfm3': base['models']['timesfm3']}
    manifest.update(files=files, models=models,
                    features=features, past_features=past_features,
                    eligible_days=sum(row['eligible'] for row in records), requested_days=len(records),
                    source_actuals_sha256=entry['sha256'], actuals_manifest_sha256=digest(snapshot_dir / 'manifest.json'),
                    base_manifest_sha256=digest(base_dir / 'manifest.json'), realized_variant=variant,
                    realized_builder_sha256=digest(Path(__file__)),
                    realized_feature_definitions=definitions, dynamic_covariates=True,
                    forecast_snapshot_sha256=digest(forecast_root / 'manifest.json'),
                    availability_policy=settings.get('availability_policy'),
                    actual_dataset_selectors=settings.get('actual_dataset_selectors', {}),
                    calendar=settings.get('calendar', {}),
                    include_historical_spread=include_spread,
                    future_forecast_earliest_gate=settings.get('future_forecast_earliest_gate', {}),
                    use_actual_snapshot_forecasts=settings.get('use_actual_snapshot_forecasts', False),
                    realized_availability=('Gate-filtered using recorded publication where present, otherwise configured deadline assumptions; latest retrospective revisions are used when configured and document creation is not publication.' if settings.get('availability_policy') else 'Past delivery dates assumed available at each gate; historical publication times and revisions unverified.'),
                    realized_policy='No delivery-day actuals; forecast first, then historical interpolation for gaps up to two hours, otherwise zero.')
    write_json(output_dir / 'manifest.json', manifest)
    return manifest
