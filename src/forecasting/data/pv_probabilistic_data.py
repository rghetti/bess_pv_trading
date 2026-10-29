"""Read and align fixed-cycle ensemble and deterministic PV study inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ...data import ForecastBundle, align_forecast, config_path, load_metadata
from ..models.pv import forecast_pv, local_delivery_starts
from ..data.pv_outages import apply_episode_exclusions
from ..data.pv_study_inputs import farm_settings


def file_hash(path):
    """Hash a source or saved artifact using bounded reads."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def frame_hash(frame):
    """Hash actual rows, including their order, rather than file timestamps."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def write_json(path, value):
    """Publish a complete JSON artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def study_root(config):
    section = config['pv_probabilistic_study']
    return config_path(config, section['output_root']) / section['run_id']


def _cycle(day):
    """Return the preceding local delivery day's 00 UTC forecast cycle."""
    return (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize('UTC')


def day_vintage_segments(day, starts, strategy):
    """Select one preceding vintage or a retrospective same-day vintage with midnight fill."""
    previous = _cycle(day)
    if strategy == 'preceding_day':
        return [(previous, starts)]
    if strategy != 'same_day_retrospective':
        raise ValueError(f'Unsupported forecast vintage strategy: {strategy}')
    current = pd.Timestamp(day).tz_localize('UTC')
    segments = [(previous, starts[starts < current]), (current, starts[starts >= current])]
    return [(cycle, grid) for cycle, grid in segments if len(grid)]


def study_periods(section):
    """Return explicitly configured, non-overlapping delivery-day archive windows."""
    if section.get('validation_end_day'):
        raise ValueError('A held-out test stage is not implemented; leave validation_end_day null')
    if section.get('vintage_strategy', 'preceding_day') not in {'preceding_day', 'same_day_retrospective'}:
        raise ValueError('Unsupported forecast vintage strategy')
    periods = section.get('dataset_periods') or [
        {'ensemble_zarr': section['ensemble_zarr'], 'start_day': section['start_day'], 'end_day': section['end_day']}]
    periods = sorted(periods, key=lambda p: p['start_day'])
    previous = None
    for period in periods:
        start, end = pd.Timestamp(period['start_day']), pd.Timestamp(period['end_day'])
        if start > end or start < pd.Timestamp(section['start_day']) or end > pd.Timestamp(section['end_day']):
            raise ValueError('Invalid or out-of-range ensemble delivery window')
        if previous is not None and start <= previous:
            raise ValueError('Ensemble delivery windows overlap')
        previous = end
    return periods


def study_days(section):
    """List only declared forecast-covered delivery days, preserving chronological gaps."""
    return [day for period in study_periods(section)
            for day in pd.date_range(period['start_day'], period['end_day'], freq='D')]


def validate_study(section):
    """Reject overlapping, future-leaking or unrepresented day-level calibration folds."""
    days = study_days(section)
    day_strings = {d.strftime('%Y-%m-%d') for d in days}
    if section['train_end_day'] not in day_strings:
        raise ValueError('The configured training endpoint is not a covered delivery day')
    if not any(delivery_split(d, section) == 'validation' for d in days):
        raise ValueError('The study requires at least one validation delivery day')
    predicted = set()
    for fold in section['calibration_folds']:
        if not fold['fit_end'] < fold['predict_start'] <= fold['predict_end'] <= section['train_end_day']:
            raise ValueError('Calibration folds must use past training days and predict only later training days')
        if fold['fit_end'] not in day_strings:
            raise ValueError('Calibration fit endpoint is not a covered delivery day')
        fold_days = {d.strftime('%Y-%m-%d') for d in pd.date_range(fold['predict_start'], fold['predict_end'])}
        if not fold_days <= day_strings or predicted & fold_days:
            raise ValueError('Calibration prediction windows must be covered and non-overlapping')
        predicted |= fold_days
    if not predicted: raise ValueError('At least one calibration prediction window is required')
    return days


def delivery_split(day, section):
    """Assign whole local delivery days to chronological training, validation or test."""
    value = pd.Timestamp(day).strftime('%Y-%m-%d')
    if value <= section['train_end_day']: return 'train'
    if not section.get('validation_end_day') or value <= section['validation_end_day']: return 'validation'
    return 'test'


def _bundle(dataset, farm_id, cycle, config, ensemble):
    """Construct one explicit member bundle without substituting forecast vintages."""
    selected = dataset.sel(farm_id=farm_id, cycle_time=cycle.tz_localize(None).to_datetime64())
    leads = pd.to_timedelta(selected.lead_time.values)
    index = pd.DatetimeIndex(cycle + leads, name='valid_time')
    if 'valid_time' in selected and not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(selected.valid_time.values, utc=True)).asi8, index.asi8
    ):
        raise ValueError('Stored forecast valid times differ from cycle plus lead')
    mapping = config['forecast_mapping']
    units = {name: selected[raw].attrs.get('units', '') for name, raw in mapping.items()}
    ids = list(map(str, selected.member.values)) if ensemble else ['deterministic']
    frames = []
    for member in ids:
        values = {}
        for canonical, raw in mapping.items():
            array = selected[raw].sel(member=int(member)) if ensemble else selected[raw]
            if array.dims != ('lead_time',):
                raise ValueError(f'Unexpected dimensions for {raw}: {array.dims}')
            values[canonical] = np.asarray(array.values, dtype=float)
        values['member'] = member
        frames.append(pd.DataFrame(values, index=index))
    latency = float(config['source']['availability_lag_hours'])
    return ForecastBundle(farm_id, cycle, cycle + pd.Timedelta(hours=latency),
        pd.concat(frames), units, tuple(ids),
        'explicit_weather_ensemble' if ensemble else 'deterministic_weather_forecast',
        {'availability_policy': 'assumed_cycle_plus_latency', 'assumed_latency_hours': latency,
         'decision_cutoff_utc': (cycle + pd.Timedelta(hours=latency)).isoformat()})


def align_cycle(dataset, farm_id, cycle, starts, config, ensemble):
    """Vectorize the existing hourly-mean and midpoint interpolation contract over members."""
    selected = dataset.sel(farm_id=farm_id, cycle_time=cycle.tz_localize(None).to_datetime64())
    leads = pd.to_timedelta(selected.lead_time.values)
    source = pd.DatetimeIndex(cycle + leads).as_unit('ns')
    if 'valid_time' in selected and not np.array_equal(
        pd.DatetimeIndex(pd.to_datetime(selected.valid_time.values, utc=True)).as_unit('ns').asi8,
        source.asi8,
    ):
        raise ValueError('Stored forecast valid times differ from cycle plus lead')
    ids = list(map(str, selected.member.values)) if ensemble else ['deterministic']
    nmember, ntime = len(ids), len(starts)
    start_ns = starts.as_unit('ns').asi8
    center_ns = start_ns + pd.Timedelta(minutes=7.5).value
    hourly_ends = starts.floor('h') + pd.Timedelta(hours=1)
    end_ns = hourly_ends.as_unit('ns').asi8
    source_ns = source.asi8
    rows = {'member': np.repeat(ids, ntime),
            'forecast_vintage': np.tile(np.repeat(cycle, ntime), nmember),
            'available_at': np.tile(np.repeat(cycle + pd.Timedelta(hours=float(config['source']['availability_lag_hours'])), ntime), nmember),
            'lead_hours': np.tile((center_ns-cycle.value)/pd.Timedelta(hours=1).value, nmember)}
    flatten = lambda array: np.asarray(array).T.reshape(-1)
    for canonical, raw in config['forecast_mapping'].items():
        array = selected[raw]
        if ensemble:
            if set(array.dims) != {'lead_time', 'member'}: raise ValueError(f'Invalid dimensions for {raw}')
            raw_values = np.asarray(array.transpose('lead_time', 'member').values, dtype=float)
        else:
            if array.dims != ('lead_time',): raise ValueError(f'Invalid dimensions for {raw}')
            raw_values = np.asarray(array.values, dtype=float)[:, None]
        neg = np.isfinite(raw_values) & (raw_values < 0) if canonical in config['policy']['nonnegative_forecasts'] else np.zeros_like(raw_values, dtype=bool)
        values = np.where(neg, 0.0, raw_values)
        kind = config['forecast_semantics'].get(canonical, config['forecast_semantics'].get('default', 'instantaneous'))
        data = np.full((ntime, nmember), np.nan); raw_data = data.copy()
        changed = np.zeros((ntime, nmember), dtype=bool)
        if kind in {'hourly_mean', 'hourly_accumulation'}:
            pos = np.searchsorted(source_ns, end_ns)
            clamped = np.clip(pos, 0, len(source_ns)-1)
            valid = (pos < len(source_ns)) & (source_ns[clamped] == end_ns)
            data[valid] = values[clamped[valid]]; raw_data[valid] = raw_values[clamped[valid]]
            changed[valid] = neg[clamped[valid]]
            if kind == 'hourly_accumulation': data /= 4.0; raw_data /= 4.0
            rows[f'{canonical}_source_start'] = np.tile(hourly_ends - pd.Timedelta(hours=1), nmember)
            rows[f'{canonical}_source_end'] = np.tile(hourly_ends, nmember)
        elif kind == 'instantaneous':
            right = np.searchsorted(source_ns, center_ns, side='left')
            left = np.clip(right-1, 0, len(source_ns)-1); right_clip = np.clip(right, 0, len(source_ns)-1)
            exact = (right < len(source_ns)) & (source_ns[right_clip] == center_ns)
            inside = (right > 0) & (right < len(source_ns)) & ~exact
            data[exact] = values[right_clip[exact]]; raw_data[exact] = raw_values[right_clip[exact]]
            fraction = ((center_ns[inside]-source_ns[left[inside]])/(source_ns[right_clip[inside]]-source_ns[left[inside]]))[:, None]
            data[inside] = values[left[inside]] + fraction*(values[right_clip[inside]]-values[left[inside]])
            raw_data[inside] = raw_values[left[inside]] + fraction*(raw_values[right_clip[inside]]-raw_values[left[inside]])
            changed = neg[left] | neg[right_clip]
        else:
            raise ValueError(f'Unknown forecast temporal semantics for {canonical}: {kind}')
        rows[canonical] = flatten(data)
        rows[f'{canonical}_raw_aligned'] = flatten(raw_data)
        rows[f'{canonical}_negative_forecast_zero'] = flatten(changed)
    index = pd.DatetimeIndex(np.tile(starts, nmember), name='time')
    frame = pd.DataFrame(rows, index=index)
    units = {name: selected[raw].attrs.get('units', '') for name, raw in config['forecast_mapping'].items()}
    return ForecastBundle(int(farm_id), cycle, cycle + pd.Timedelta(hours=float(config['source']['availability_lag_hours'])),
        frame, units, tuple(ids), 'explicit_weather_ensemble' if ensemble else 'deterministic_weather_forecast',
        {'availability_policy': 'assumed_cycle_plus_latency', 'decision_cutoff_utc': (cycle + pd.Timedelta(hours=float(config['source']['availability_lag_hours']))).isoformat(),
         'alignment': 'hourly_interval_end_and_instantaneous_midpoint_within_selected_vintage'})


def _target_frame(dataset, farm_id, starts, config):
    """Join raw measured AC to the grid and apply only approved episode exclusions."""
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values, utc=True))
    series = pd.Series(dataset.power.sel(farm_id=farm_id).values, index=times)
    frame = pd.DataFrame({'interval_start_utc': starts,
                          'measured_ac_mw': series.reindex(starts).to_numpy() * float(config['forecast']['realized_power_scale_to_mw'])})
    frame['eligible'] = np.isfinite(frame.measured_ac_mw) & (frame.measured_ac_mw >= 0)
    return apply_episode_exclusions(frame, config, farm_id=farm_id)


def source_fingerprint(config, farm_id):
    """Hash all selected forecast and target values plus static and review inputs."""
    section = config['pv_probabilistic_study']
    start, end = pd.Timestamp(section['start_day']), pd.Timestamp(section['end_day'])
    periods = study_periods(section)
    expected_cycles = np.array([cycle.tz_localize(None).to_datetime64()
        for period in periods for day in pd.date_range(period['start_day'], period['end_day'], freq='D')
        for cycle, _ in day_vintage_segments(day, local_delivery_starts(day.strftime('%Y-%m-%d')),
                                              section.get('vintage_strategy', 'preceding_day'))])
    hasher = hashlib.sha256()
    paths = list(dict.fromkeys([p['ensemble_zarr'] for p in periods] + [config['inputs']['zarr']]))
    for source_path in paths:
        path = config_path(config, source_path)
        ds = xr.open_zarr(path, consolidated=True, chunks=None)
        hasher.update(str(source_path).encode())
        hasher.update(file_hash(path / '.zmetadata').encode())
        cycles = pd.DatetimeIndex(ds.cycle_time.values)
        mask = np.isin(cycles.to_numpy(dtype='datetime64[ns]'), expected_cycles.astype('datetime64[ns]'))
        selection = ds.sel(farm_id=farm_id).isel(cycle_time=np.flatnonzero(mask))
        for name in sorted(config['forecast_mapping'].values()):
            arr = np.ascontiguousarray(selection[name].values)
            hasher.update(name.encode()); hasher.update(arr.tobytes())
        if 'power' in ds:
            lo = start.tz_localize('Europe/Rome').tz_convert('UTC').tz_localize(None)
            hi = (end + pd.Timedelta(days=1)).tz_localize('Europe/Rome').tz_convert('UTC').tz_localize(None)
            target = ds.power.sel(farm_id=farm_id, time=slice(lo, hi)).values
            hasher.update(np.ascontiguousarray(target).tobytes())
        ds.close()
    for path in (config['inputs']['metadata_parquet'], 'data/farms/metadata.parquet', config['pv_data_quality']['exclusions_csv']):
        hasher.update(file_hash(config_path(config, path)).encode())
    return hasher.hexdigest()


def prepare_farm(config, farm_id, *, force=False):
    """Save one farm's matched 00 UTC weather views and curated raw AC targets."""
    section = config['pv_probabilistic_study']
    validate_study(section)
    root = study_root(config) / f'farm_{farm_id}'
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'input_manifest.json'
    fingerprint = source_fingerprint(config, farm_id)
    input_settings = {key: section.get(key) for key in ('ensemble_zarr', 'dataset_periods', 'start_day', 'train_end_day', 'validation_end_day', 'end_day')}
    if section.get('vintage_strategy', 'preceding_day') != 'preceding_day':
        input_settings['vintage_strategy'] = section['vintage_strategy']
    settings_identity = hashlib.sha256(json.dumps({'section': input_settings, 'source': config['source'],
        'mapping': config['forecast_mapping'], 'semantics': config['forecast_semantics'],
        'policy': config['policy'], 'pv': farm_settings(config, load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id])}, sort_keys=True, default=str).encode()).hexdigest()
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old['source_fingerprint'] != fingerprint or old['settings_fingerprint'] != settings_identity:
            raise ValueError(f'Farm {farm_id} source or settings changed; choose a new run ID')
        for label in ('ensemble', 'deterministic'):
            if file_hash(root / f'{label}_inputs.parquet') != old['files'][label]:
                raise ValueError(f'Farm {farm_id} {label} input checksum changed')
        print(f'Farm {farm_id}: reusing matched inputs', flush=True)
        return old
    periods = study_periods(section)
    ensemble_stores = {p['ensemble_zarr']: xr.open_zarr(config_path(config, p['ensemble_zarr']), consolidated=True, chunks=None) for p in periods}
    det = xr.open_zarr(config_path(config, config['inputs']['zarr']), consolidated=True, chunks=None)
    a = pd.read_parquet(config_path(config, 'data/farms/metadata.parquet')).sort_values('farm_id').reset_index(drop=True)
    b = pd.read_parquet(config_path(config, config['inputs']['metadata_parquet'])).sort_values('farm_id').reset_index(drop=True)
    if not a.equals(b):
        raise ValueError('Ensemble and deterministic farm metadata differ')
    reference = next(iter(ensemble_stores.values()))
    for path, ens in ensemble_stores.items():
        if len(ens.member) != 50 or not np.array_equal(ens.member.values, reference.member.values):
            raise ValueError(f'Expected the same 50 explicit member labels in {path}')
        for raw in config['forecast_mapping'].values():
            for attr in ('units', 'temporal_semantics'):
                if ens[raw].attrs.get(attr) != reference[raw].attrs.get(attr):
                    raise ValueError(f'Ensemble source {path} has incompatible {raw} {attr}')
    for period in periods:
        cycles = set(pd.DatetimeIndex(ensemble_stores[period['ensemble_zarr']].cycle_time.values))
        missing = [str(cycle.tz_localize(None)) for day in pd.date_range(period['start_day'], period['end_day'])
                   for cycle, _ in day_vintage_segments(day, local_delivery_starts(day.strftime('%Y-%m-%d')),
                                                        section.get('vintage_strategy', 'preceding_day'))
                   if cycle.tz_localize(None) not in cycles]
        if missing: raise ValueError(f"Missing preceding 00 UTC cycles in {period['ensemble_zarr']}: {missing}")
    all_frames = {'ensemble': [], 'deterministic': []}
    for period in periods:
        ens = ensemble_stores[period['ensemble_zarr']]
        for day in pd.date_range(period['start_day'], period['end_day'], freq='D'):
            starts = local_delivery_starts(day.strftime('%Y-%m-%d'))
            target = _target_frame(det, farm_id, starts, config)
            for label, dataset, ensemble in (('ensemble', ens, True), ('deterministic', det, False)):
                for cycle, segment_starts in day_vintage_segments(day, starts, section.get('vintage_strategy', 'preceding_day')):
                    aligned = align_cycle(dataset, farm_id, cycle, segment_starts, config, ensemble)
                    frame = aligned.values.copy()
                    frame['interval_start_utc'] = frame.index
                    frame = frame.reset_index(drop=True).merge(target, on='interval_start_utc', how='left', validate='many_to_one')
                    frame['farm_id'] = farm_id
                    frame['delivery_day'] = day.strftime('%Y-%m-%d')
                    frame['forecast_cycle_utc'] = cycle
                    frame['forecast_available_at_utc'] = aligned.available_at
                    frame['split'] = delivery_split(day, section)
                    all_frames[label].append(frame)
            if day.day % 5 == 0 or day.strftime('%Y-%m-%d') == period['end_day']:
                print(f'Farm {farm_id}: aligned through {day.date()}', flush=True)
    for store in ensemble_stores.values(): store.close()
    det.close()
    ensemble_frame = pd.concat(all_frames['ensemble'], ignore_index=True)
    deterministic_frame = pd.concat(all_frames['deterministic'], ignore_index=True)
    if ensemble_frame.duplicated(['interval_start_utc', 'member']).any() or deterministic_frame.interval_start_utc.duplicated().any():
        raise ValueError('Duplicate ensemble or deterministic forecast interval')
    grouped = ensemble_frame.groupby('interval_start_utc', sort=True).agg(
        members=('member', 'nunique'), cycle_count=('forecast_cycle_utc', 'nunique'),
        forecast_cycle_utc=('forecast_cycle_utc', 'first'),
        forecast_available_at_utc=('forecast_available_at_utc', 'first'))
    if not grouped.members.eq(50).all() or not grouped.cycle_count.eq(1).all():
        raise ValueError('Each interval must have exactly 50 members from one selected cycle')
    paired = grouped.join(deterministic_frame.set_index('interval_start_utc')[['forecast_cycle_utc', 'forecast_available_at_utc']],
                          how='outer', lsuffix='_ensemble', rsuffix='_deterministic', validate='one_to_one')
    for name in ('forecast_cycle_utc', 'forecast_available_at_utc'):
        if not paired[f'{name}_ensemble'].equals(paired[f'{name}_deterministic']):
            raise ValueError(f'Ensemble and deterministic {name} mismatch')
    files = {}
    for label, frame in (('ensemble', ensemble_frame), ('deterministic', deterministic_frame)):
        path = root / f'{label}_inputs.parquet'
        frame.to_parquet(path, index=False)
        files[label] = file_hash(path)
    manifest = {'farm_id': farm_id, 'source_fingerprint': fingerprint, 'settings_fingerprint': settings_identity,
        'files': files, 'ensemble_members': 50, 'target': 'raw measured AC MW; approved exclusions only',
        'vintage_strategy': section.get('vintage_strategy', 'preceding_day'),
        'cycle': ('same-day 00 UTC retrospectively; preceding-day cycle before 00 UTC' if section.get('vintage_strategy') == 'same_day_retrospective' else '00 UTC on prior delivery day'),
        'availability': 'cycle plus assumed six hours',
        'training': f"{section['start_day']} through {section['train_end_day']}",
        'validation': f"after {section['train_end_day']} through {section.get('validation_end_day') or section['end_day']}",
        'test': f"after {section['validation_end_day']} through {section['end_day']}" if section.get('validation_end_day') else None,
        'dataset_periods': periods, 'matching_cycle_and_availability_verified': True,
        'limitation': ('Same-day cycles are used retrospectively before availability; stitching member labels across cycles does not establish trajectory identity. ' if section.get('vintage_strategy') == 'same_day_retrospective' else '') +
            'The first validation day uses completed previous-day targets after its historical DA gate; selection uses the full validation period.',
        'source_modified': False}
    write_json(manifest_path, manifest)
    print(f'Farm {farm_id}: saved matched inputs ({ensemble_frame.delivery_day.nunique()} delivery days)', flush=True)
    return manifest


def pvlib_farm(config, farm_id, *, force=False):
    """Apply identical configured physical conversion to all members and the deterministic forecast."""
    root = study_root(config) / f'farm_{farm_id}'
    manifest = json.loads((root / 'input_manifest.json').read_text())
    marker = root / 'pvlib_manifest.json'
    if marker.exists() and not force:
        old = json.loads(marker.read_text())
        if old['input_files'] != manifest['files']:
            raise ValueError('pvlib input fingerprints changed')
        for label in ('ensemble', 'deterministic'):
            if file_hash(root / f'{label}_pvlib.parquet') != old['files'][label]:
                raise ValueError(f'pvlib {label} output checksum mismatch')
        print(f'Farm {farm_id}: reusing pvlib outputs', flush=True)
        return old
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = farm_settings(config, farm)
    output_files = {}
    mapping = config['forecast_mapping']
    for label in ('ensemble', 'deterministic'):
        frame = pd.read_parquet(root / f'{label}_inputs.parquet')
        result = []
        for (day, cycle_value), part in frame.groupby(['delivery_day', 'forecast_cycle_utc'], sort=True):
            cycle = pd.Timestamp(cycle_value)
            part = part.copy().set_index('interval_start_utc', drop=False)
            members = tuple(map(str, part.member.drop_duplicates()))
            bundle = ForecastBundle(farm_id, cycle, pd.Timestamp(part.forecast_available_at_utc.iloc[0]), part,
                {name: '' for name in mapping}, members, label,
                {'decision_cutoff_utc': pd.Timestamp(part.forecast_available_at_utc.iloc[0]).isoformat()})
            converted = forecast_pv(bundle, farm, settings, config['policy']['night']).values.reset_index(drop=True)
            keep = ['interval_start_utc', 'delivery_day', 'member', 'pv_ac_mw', 'pv_input_valid',
                    'bhi_exceeds_ghi', 'low_sun_direct_limited', 'dni_limited', 'nighttime_zero']
            result.append(converted[keep])
        output = pd.concat(result, ignore_index=True)
        path = root / f'{label}_pvlib.parquet'
        output.to_parquet(path, index=False)
        output_files[label] = file_hash(path)
    marker_data = {'input_files': manifest['files'], 'files': output_files,
        'pv_parameters': settings, 'parameter_interpretation': '2025 fitted farm parameters treated as prior metadata'}
    write_json(marker, marker_data)
    print(f'Farm {farm_id}: pvlib conversion complete', flush=True)
    return marker_data
