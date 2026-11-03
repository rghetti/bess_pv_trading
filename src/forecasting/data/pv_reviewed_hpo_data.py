"""Leakage-aware features and masks for the reviewed historical-GHI PV corpus."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...data import config_path, open_store
from .pv_neural_data import fit_preprocessor, transform_features
from .pv_curated_review import sha256


@dataclass
class PreparedFarm:
    """One physical-quarter corpus with checked input, target, and split contracts."""

    farm_id: int
    rating_mw: float
    frame: pd.DataFrame
    features: dict[str, pd.DataFrame]
    training_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    provenance: dict


def _fingerprint_array(hasher, name, values):
    """Hash real source arrays with their names and dtypes, not only metadata."""
    array = np.ascontiguousarray(values)
    hasher.update(name.encode())
    hasher.update(str(array.dtype).encode())
    hasher.update(array.tobytes())


def align_extended_weather(selected, frame, config, canonical_names):
    """Align extra forecast variables to each recorded D-1 00 UTC vintage.

    Instantaneous values are interpolated at quarter-hour midpoints and
    accumulations are uniformly distributed over their labeled source hour.
    Missing endpoints remain missing; no other forecast vintage is consulted.
    """
    starts = pd.DatetimeIndex(frame.interval_start_utc)
    vintages = pd.DatetimeIndex(frame.forecast_cycle_utc)
    cycles = pd.DatetimeIndex(pd.to_datetime(selected.cycle_time.values, utc=True))
    cycle_index = cycles.get_indexer(vintages)
    lead_hours = pd.TimedeltaIndex(selected.lead_time.values) / pd.Timedelta(hours=1)
    elapsed = np.asarray((starts + pd.Timedelta(minutes=7.5) - vintages) / pd.Timedelta(hours=1))
    result = pd.DataFrame(index=frame.index)
    hasher = hashlib.sha256()
    _fingerprint_array(hasher, 'cycle_time', selected.cycle_time.values)
    _fingerprint_array(hasher, 'lead_time', selected.lead_time.values)
    for name in canonical_names:
        source_name = config['forecast_mapping'][name]
        raw = selected[source_name].transpose('cycle_time', 'lead_time').values
        _fingerprint_array(hasher, source_name, raw)
        kind = config['forecast_semantics'].get(name, config['forecast_semantics']['default'])
        if kind in {'hourly_mean', 'hourly_accumulation'}:
            lower = np.ceil(elapsed)
            upper = lower
            fraction = np.zeros(len(frame))
        elif kind == 'instantaneous':
            lower = np.floor(elapsed)
            upper = lower + 1
            fraction = elapsed - lower
        else:
            raise ValueError(f'Unsupported forecast temporal semantics for {name}: {kind}')
        left = pd.Index(lead_hours).get_indexer(lower)
        right = pd.Index(lead_hours).get_indexer(upper)
        valid = (cycle_index >= 0) & (left >= 0) & (right >= 0)
        aligned = np.full(len(frame), np.nan)
        aligned[valid] = (raw[cycle_index[valid], left[valid]] * (1 - fraction[valid])
                          + raw[cycle_index[valid], right[valid]] * fraction[valid])
        if kind == 'hourly_accumulation':
            aligned /= 4.0
        if name in config['policy']['nonnegative_forecasts']:
            aligned = np.maximum(aligned, 0)
        result[name] = aligned
    return result, hasher.hexdigest()


def calendar_features(times, timezone):
    """Encode local interval midpoint, season and month without proxy geometry."""
    local = (pd.DatetimeIndex(times) + pd.Timedelta(minutes=7.5)).tz_convert(timezone)
    hour = local.hour + local.minute / 60 + local.second / 3600
    annual = (local.dayofyear - 1 + hour / 24) / np.where(local.is_leap_year, 366, 365)
    angles = {'hour': 2 * np.pi * hour / 24, 'annual': 2 * np.pi * annual,
              'month': 2 * np.pi * (local.month - 1) / 12}
    columns = {}
    for name, angle in angles.items():
        columns[f'{name}_sin'] = np.sin(angle)
        columns[f'{name}_cos'] = np.cos(angle)
    return pd.DataFrame(columns)


def training_mask(frame, section):
    """Keep reviewed exclusions and allow explicitly zero-filled nighttime GHI."""
    policy = section['missing_night_ghi']
    exclusions = section['nighttime_exclusions']
    if policy not in {'zero_at_night', 'exclude'}:
        raise ValueError('Resolve pv_reviewed_hpo.missing_night_ghi before preparing the run')
    if exclusions not in {'keep_curated', 'include_excluded'}:
        raise ValueError('Resolve pv_reviewed_hpo.nighttime_exclusions before preparing the run')
    # Missing targets cannot be supervised and prior exclusions remain excluded by default.
    target = frame.model_target_mw.to_numpy(dtype=float)
    base = (frame['split'].eq('train') & ~frame.embargoed & np.isfinite(target) & (target >= 0))
    prior = frame.retained_prior_exclusion.to_numpy(dtype=bool)
    if exclusions == 'include_excluded':
        prior &= ~frame.night.to_numpy(dtype=bool)
    base &= ~prior
    # Reproduce the original eligibility except for explicitly requested nighttime changes.
    weather_valid = (np.isfinite(frame.temperature_c) & np.isfinite(frame.wind_u_10m)
                     & np.isfinite(frame.wind_v_10m))
    base &= weather_valid
    ghi_valid = np.isfinite(frame.ghi_w_m2) & frame.ghi_w_m2.ge(0)
    if policy == 'zero_at_night':
        ghi_valid |= frame.night & frame.ghi_w_m2.isna()
    base &= ghi_valid
    return base.to_numpy(dtype=bool)


def evenly_sample_training_nights(frame, eligible, requested):
    """Select a fixed number of night targets evenly across local training days."""
    count = int(requested)
    if count < 0:
        raise ValueError('Nighttime sample count must be nonnegative')
    night = frame.night.to_numpy(dtype=bool)
    candidates = np.flatnonzero(eligible & night)
    if len(candidates) < count:
        raise ValueError(f'Only {len(candidates)} eligible nighttime targets for {count} requested')
    training_days = pd.to_datetime(frame.loc[frame['split'].eq('train'), 'delivery_day'])
    days = pd.date_range(training_days.min(), training_days.max(), freq='D').strftime('%Y-%m-%d')
    grouped = pd.Series(candidates).groupby(frame.delivery_day.to_numpy()[candidates]).apply(list)
    positions = [np.asarray(grouped.get(day, []), dtype=int) for day in days]
    capacity = np.array([len(group) for group in positions], dtype=int)
    # Divide the requested total evenly across calendar days, then redistribute
    # any shortfall from embargoed or excluded days to days with spare targets.
    boundaries = np.floor(np.arange(len(days) + 1) * count / len(days)).astype(int)
    allocation = np.minimum(np.diff(boundaries), capacity)
    remaining = count - int(allocation.sum())
    while remaining:
        available_days = np.flatnonzero(allocation < capacity)
        if not len(available_days):
            raise ValueError('Cannot distribute the requested nighttime targets')
        chosen = available_days[np.floor((np.arange(min(remaining, len(available_days))) + .5)
                                        * len(available_days) / min(remaining, len(available_days))).astype(int)]
        allocation[chosen] += 1
        remaining -= len(chosen)
    selected = eligible & ~night
    # Space selected quarters over each day's available night window as well.
    for group, take in zip(positions, allocation):
        if take:
            index = np.floor((np.arange(take) + .5) * len(group) / take).astype(int)
            selected[group[index]] = True
    return selected


def split_weather_inputs(frame, section):
    """Use observations only in training and forecast GHI in both scored splits."""
    if (section['training_ghi_source'] != 'historical'
            or section['validation_ghi_source'] != 'preceding_day_forecast'
            or section['testing_ghi_source'] != 'preceding_day_forecast'):
        raise ValueError('GHI regimes must match the explicitly requested split-specific contract')
    weather = frame[section['core_weather']].copy().reset_index(drop=True)
    training_rows = frame['split'].eq('train').to_numpy()
    scored_rows = frame['split'].isin(['validation', 'test']).to_numpy()
    weather.loc[scored_rows, 'ghi_w_m2'] = frame.loc[scored_rows, 'nwp_ghi_w_m2'].to_numpy()
    missing_train_ghi = (training_rows & frame.night.to_numpy(dtype=bool)
                         & frame.ghi_w_m2.isna().to_numpy())
    if section['missing_night_ghi'] == 'zero_at_night':
        weather.loc[missing_train_ghi, 'ghi_w_m2'] = 0.0
    return weather, missing_train_ghi


def _load_checked_corpus(config, farm_id):
    """Validate the reviewed corpus and its in-period modeling view before use."""
    section = config['pv_reviewed_hpo']
    root = config_path(config, section['dataset_root'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    parent_manifest_path = config_path(config, manifest['configuration']['input_root']) / 'manifest.json'
    if sha256(parent_manifest_path) != manifest['parent_manifest_sha256']:
        raise ValueError(f'Farm {farm_id}: parent curation manifest checksum mismatch')
    parent_manifest = json.loads(parent_manifest_path.read_text())
    farm_root = root / f'farm_{farm_id}'
    corpus_path = farm_root / 'corpus.parquet'
    view_path = farm_root / 'modeling_views.parquet'
    expected = manifest['farms'][str(farm_id)]['corpus_sha256']
    if sha256(corpus_path) != expected:
        raise ValueError(f'Farm {farm_id}: curated corpus checksum mismatch')
    corpus = pd.read_parquet(corpus_path)
    view = pd.read_parquet(view_path).reset_index(drop=True)
    expected_view = corpus.loc[corpus['split'].ne('outside')].reset_index(drop=True)
    # The separate modeling file must be exactly the in-period corpus, including masks and targets.
    if not view.equals(expected_view):
        raise ValueError(f'Farm {farm_id}: modeling view disagrees with the fingerprinted corpus')
    times = pd.DatetimeIndex(view.interval_start_utc)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError(f'Farm {farm_id}: physical-quarter timestamps are not unique and ordered')
    vintage = pd.DatetimeIndex(view.forecast_cycle_utc)
    expected_vintage = pd.DatetimeIndex(pd.to_datetime(view.delivery_day, utc=True)) - pd.Timedelta(days=1)
    if not vintage.equals(expected_vintage):
        raise ValueError(f'Farm {farm_id}: weather cycle is not preceding-day 00 UTC')
    available = pd.DatetimeIndex(view.forecast_available_at_utc)
    expected_available = vintage + pd.Timedelta(hours=config['source']['availability_lag_hours'])
    if not available.equals(expected_available):
        raise ValueError(f'Farm {farm_id}: stored forecast availability disagrees with configured lag')
    return view, {'dataset_manifest_sha256': sha256(manifest_path),
                  'corpus_sha256': expected, 'modeling_view_sha256': sha256(view_path),
                  'parent_manifest_sha256': manifest['parent_manifest_sha256'],
                  'split_configuration': parent_manifest['configuration']}


def prepare_farm(config, farm_id):
    """Use historical training GHI and D-1 forecast GHI for validation/test."""
    section = config['pv_reviewed_hpo']
    frame, provenance = _load_checked_corpus(config, farm_id)
    train = evenly_sample_training_nights(frame, training_mask(frame, section),
                                         section['night_training_samples_per_farm'])
    forecast_ghi = frame.nwp_ghi_w_m2.to_numpy(dtype=float)
    forecast_valid = np.isfinite(forecast_ghi) & (forecast_ghi >= 0)
    # A scored row must have an actual forecast-GHI input, not an imputed substitute.
    validation = frame.validation_eligible.to_numpy(dtype=bool) & forecast_valid
    test = frame.test_eligible.to_numpy(dtype=bool) & forecast_valid
    if not train.any() or not validation.any() or not test.any():
        raise ValueError(f'Farm {farm_id}: one or more modeling splits have no usable quarters')
    if np.any((train & validation) | (train & test) | (validation & test)):
        raise ValueError(f'Farm {farm_id}: training/validation/test masks overlap')
    if (frame.loc[train, 'split'] != 'train').any():
        raise ValueError(f'Farm {farm_id}: a training row is outside the training split')
    if (frame.loc[validation | test, 'night']).any():
        raise ValueError(f'Farm {farm_id}: evaluation mask contains monthly-window night rows')
    # Historic target availability is enforced by the frozen embargo masks.
    split = provenance['split_configuration']
    expected_split = np.select(
        [frame.delivery_day < split['train_end_day_exclusive'],
         frame.delivery_day < split['validation_end_day_exclusive']],
        ['train', 'validation'], default='test')
    if not np.array_equal(frame['split'].to_numpy(), expected_split):
        raise ValueError(f'Farm {farm_id}: reviewed split labels disagree with original boundaries')
    validation_start = pd.Timestamp(split['train_end_day_exclusive'], tz=section['timezone']).tz_convert('UTC')
    test_start = pd.Timestamp(split['validation_end_day_exclusive'], tz=section['timezone']).tz_convert('UTC')
    if frame.loc[train, 'target_available_at_utc'].max() > validation_start:
        raise ValueError(f'Farm {farm_id}: training targets cross the validation boundary')
    if frame.loc[validation, 'target_available_at_utc'].max() > test_start:
        raise ValueError(f'Farm {farm_id}: validation targets cross the test boundary')
    # Extended weather is selected from the very same stored vintage, independently per farm.
    with open_store(config) as store:
        selected = store.sel(farm_id=farm_id)
        extra, weather_sha = align_extended_weather(selected, frame, config, section['extended_weather'])
    time = calendar_features(frame.interval_start_utc, section['timezone'])
    weather, missing_train_ghi = split_weather_inputs(frame, section)
    core = pd.concat([weather, time], axis=1)
    extended = pd.concat([core, extra.reset_index(drop=True)], axis=1)
    features = {'core': core, 'extended': extended}
    for name, value in features.items():
        if any('pvlib' in column.lower() or 'target' in column.lower() for column in value.columns):
            raise ValueError(f'Farm {farm_id}: {name} includes a forbidden PV-model/target input')
        if value.columns.has_duplicates:
            raise ValueError(f'Farm {farm_id}: {name} contains duplicate features')
    provenance['extended_weather_sha256'] = weather_sha
    provenance['forecast_cycle'] = 'preceding delivery day 00 UTC; release lag is an assumption'
    provenance['ghi_source_by_split'] = {'train': 'historical with missing-night zero treatment',
        'validation': 'preceding-day 00 UTC forecast', 'test': 'preceding-day 00 UTC forecast'}
    provenance['zeroed_missing_train_night_ghi'] = int((missing_train_ghi & train).sum())
    provenance['target'] = 'model_target_mw; measured and user-approved synthetic labels'
    provenance['target_source_counts'] = {
        label: frame.loc[mask, 'model_target_source'].value_counts().to_dict()
        for label, mask in [('train', train), ('validation', validation), ('test', test)]}
    provenance['mask_counts'] = {'train': int(train.sum()), 'validation': int(validation.sum()),
        'test': int(test.sum()), 'train_night': int((train & frame.night.to_numpy()).sum()),
        'train_synthetic': int((train & frame.synthetic_target.to_numpy()).sum()),
        'validation_missing_forecast_ghi': int((frame.validation_eligible & ~forecast_valid).sum()),
        'test_missing_forecast_ghi': int((frame.test_eligible & ~forecast_valid).sum())}
    provenance['train_night_by_month'] = (frame.loc[train & frame.night.to_numpy(dtype=bool),
        'delivery_day'].str[:7].value_counts().sort_index().to_dict())
    provenance['missing_features'] = {
        name: {column: int((~np.isfinite(data[column].to_numpy(dtype=float))).sum())
               for column in data.columns}
        for name, data in features.items()}
    return PreparedFarm(farm_id, float(config['farms'][str(farm_id)]['ac_capacity_kw']) / 1000,
                        frame, features, train, validation, test, provenance)


def fit_transforms(prepared, config):
    """Fit median and scale using only training rows, including eligible nights."""
    fitted = {}
    transformed = {}
    section = config['pv_reviewed_hpo']
    for name, features in prepared.features.items():
        finite = np.isfinite(features.loc[prepared.training_mask].to_numpy(dtype=float))
        if not finite.any(axis=0).all():
            bad = features.columns[~finite.any(axis=0)].tolist()
            raise ValueError(f'Farm {prepared.farm_id}: no training values for features {bad}')
        # Legacy helper's fit mask is explicit; validation/test data do not influence statistics.
        preprocessor = fit_preprocessor(features, prepared.training_mask, section,
            include_missing_indicators=config['pv_reviewed_hpo']['include_missing_indicators'])
        transformed[name] = transform_features(features, preprocessor).astype(np.float32)
        fitted[name] = preprocessor
    return fitted, transformed
