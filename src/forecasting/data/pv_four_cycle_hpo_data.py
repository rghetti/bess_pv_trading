"""Four weather-cycle views of reviewed 15-minute PV training targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data import night_window_mask, open_store
from .pv_reviewed_hpo_data import (
    PreparedFarm, _load_checked_corpus, align_extended_weather, calendar_features,
    evenly_sample_training_nights,
)


def prepare_four_cycle_farm(config, farm_id):
    """Align each issue's forecast horizon with curated physical-quarter targets."""
    section = config['pv_reviewed_hpo']
    original, provenance = _load_checked_corpus(config, farm_id)
    original = original.copy()
    night = original.night.to_numpy(dtype=bool)
    retained = original.retained_prior_exclusion.to_numpy(dtype=bool) & ~night
    reference = original.pvlib_historical_ghi_mw.to_numpy(dtype=float)
    if np.any(retained & (~np.isfinite(reference) | (reference < 0))):
        raise ValueError(f'Farm {farm_id}: a retained daylight exclusion lacks a pvlib reference')
    original.loc[retained, 'model_target_mw'] = reference[retained]
    original.loc[retained, 'model_target_source'] = 'pvlib_retained_exclusion'
    original.loc[retained, 'synthetic_target'] = True
    target = original.model_target_mw.to_numpy(dtype=float)
    valid_target = np.isfinite(target) & (target >= 0)
    historical = original.ghi_w_m2.to_numpy(dtype=float).copy()
    historical[night & ~np.isfinite(historical)] = 0.0
    valid_historical = np.isfinite(historical) & (historical >= 0)
    # Select physical nighttime labels before replicating the four cycle views.
    retained_night = original.retained_prior_exclusion.to_numpy(dtype=bool) & night
    train_physical = (original['split'].eq('train').to_numpy() & ~original.embargoed.to_numpy(dtype=bool)
                      & valid_target & valid_historical & ~retained_night)
    train_physical = evenly_sample_training_nights(original, train_physical,
                                                    section['night_training_samples_per_farm'])
    validation_physical = (original['split'].eq('validation').to_numpy()
                           & ~original.embargoed.to_numpy(dtype=bool) & ~night & valid_target)
    test_physical = original['split'].eq('test').to_numpy() & ~night & valid_target
    times = pd.DatetimeIndex(original.interval_start_utc)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Physical target quarters must be unique and ordered')
    with open_store(config) as store:
        selected = store.sel(farm_id=farm_id)
        cycles = pd.DatetimeIndex(pd.to_datetime(selected.cycle_time.values, utc=True))
        if cycles.has_duplicates or not cycles.is_monotonic_increasing:
            raise ValueError('Weather cycles must be unique and ordered')
        allowed_hours = section['cycle_hours_utc']
        cycles = cycles[cycles.hour.isin(allowed_hours) & (cycles.minute == 0)]
        selected = selected.sel(cycle_time=cycles.tz_localize(None).to_numpy())
        quarter = np.arange(int(section['issue_horizon_hours']) * 4)
        start_ns = (cycles.as_unit('ns').asi8[:, None]
                    + quarter[None, :] * pd.Timedelta(minutes=15).value).reshape(-1)
        starts = pd.DatetimeIndex(pd.to_datetime(start_ns, utc=True))
        vintages = pd.DatetimeIndex(np.repeat(cycles.to_numpy(), len(quarter)))
        frame = pd.DataFrame({'interval_start_utc': starts, 'forecast_cycle_utc': vintages})
        source_names = ['forecast_ghi', 'forecast_bhi', 'temperature_c', 'wind_u_10m', 'wind_v_10m']
        source_names += list(section['extended_weather'])
        weather, weather_hash = align_extended_weather(selected, frame, config, source_names)
    positions = times.get_indexer(starts)
    present = positions >= 0
    safe = np.maximum(positions, 0)
    selected_rows = original.iloc[safe].reset_index(drop=True)
    frame['delivery_day'] = starts.tz_convert(section['timezone']).strftime('%Y-%m-%d')
    frame['split'] = np.where(present, selected_rows['split'], 'outside')
    frame['model_target_mw'] = np.where(present, selected_rows.model_target_mw, np.nan)
    frame['model_target_source'] = np.where(present, selected_rows.model_target_source, 'outside')
    frame['synthetic_target'] = present & selected_rows.synthetic_target.to_numpy(dtype=bool)
    frame['night'] = night_window_mask(starts, config['policy']['forecast_pv_night'])
    frame['evaluation_window'] = present & selected_rows.evaluation_window.to_numpy(dtype=bool)
    frame['ghi_interpolated'] = present & selected_rows.ghi_interpolated.to_numpy(dtype=bool)
    frame['lead_hours'] = np.tile(quarter / 4, len(cycles))
    frame['ghi_forecast_w_m2'] = weather.forecast_ghi
    training_context = (present & selected_rows['split'].eq('train').to_numpy()
                        & ~selected_rows.embargoed.to_numpy(dtype=bool))
    historic_context = np.where(present, historical[safe], np.nan)
    input_ghi = np.where(training_context, historic_context, weather.forecast_ghi.to_numpy())
    frame['ghi_input_w_m2'] = input_ghi
    core = weather[['forecast_ghi', 'forecast_bhi', 'temperature_c', 'wind_u_10m', 'wind_v_10m']].copy()
    core.columns = section['core_weather']
    core['ghi_w_m2'] = input_ghi
    core = pd.concat([core.reset_index(drop=True), calendar_features(starts, section['timezone'])], axis=1)
    # Lead and cycle-hour features let a single network distinguish forecast ages.
    core['forecast_lead_hours'] = frame.lead_hours.to_numpy() + 0.125
    core['cycle_hour_sin'] = np.sin(2 * np.pi * vintages.hour / 24)
    core['cycle_hour_cos'] = np.cos(2 * np.pi * vintages.hour / 24)
    extended = pd.concat([core, weather[section['extended_weather']].reset_index(drop=True)], axis=1)
    features = {'core': core, 'extended': extended}
    required_weather = np.isfinite(weather[['temperature_c', 'wind_u_10m', 'wind_v_10m']]).all(axis=1).to_numpy()
    required_train = required_weather & np.isfinite(input_ghi) & (input_ghi >= 0)
    required_forecast = required_weather & np.isfinite(weather.forecast_ghi.to_numpy())
    causal = frame.lead_hours.to_numpy() >= float(config['source']['availability_lag_hours'])
    train = present & train_physical[safe] & required_train
    validation = present & validation_physical[safe] & required_forecast & causal
    test = present & test_physical[safe] & required_forecast & causal
    if not train.any() or not validation.any() or not test.any():
        raise ValueError(f'Farm {farm_id}: an expanded split has no valid forecast-target pairs')
    forecast_feature_sets = {name: data.assign(ghi_w_m2=weather.forecast_ghi.to_numpy())
                             for name, data in features.items()}
    provenance.update({'extended_weather_sha256': weather_hash,
        'forecast_cycle': f"00, 06, 12, 18 UTC; targets from issue through issue plus {section['issue_horizon_hours']} hours",
        'training_weather': 'historical GHI with forecast weather from all four issue cycles',
        'validation_test_availability': 'target starts at least configured release lag after cycle',
        'target': 'reviewed target; retained daylight exclusions reconstructed with historical-GHI pvlib',
        'physical_night_targets_selected': int((train_physical & night).sum()),
        'physical_daylight_exclusions_reconstructed': int(retained.sum()),
        'mask_counts': {'train': int(train.sum()), 'validation': int(validation.sum()), 'test': int(test.sum()),
            'train_night_pairs': int((train & frame.night.to_numpy()).sum()),
            'train_first_six_hours_pairs': int((train & ~causal).sum())},
        'target_source_counts': {name: frame.loc[mask, 'model_target_source'].value_counts().to_dict()
            for name, mask in [('train', train), ('validation', validation), ('test', test)]},
        'missing_features': {name: {column: int((~np.isfinite(data[column].to_numpy(dtype=float))).sum())
            for column in data.columns} for name, data in features.items()},
        'missing_selected_features': {name: {split: {column:
            int((~np.isfinite(data.loc[mask, column].to_numpy(dtype=float))).sum())
            for column in data.columns} for split, mask in
            [('train', train), ('validation', validation), ('test', test)]}
            for name, data in features.items()},
        'missing_context_rows': {name: {split: int((
            (~np.isfinite((data if split == 'train' else forecast_feature_sets[name]).to_numpy(dtype=float)))
            .any(axis=1).reshape(len(cycles), len(quarter))
            & mask.reshape(len(cycles), len(quarter)).any(axis=1)[:, None]).sum())
            for split, mask in [('train', train), ('validation', validation), ('test', test)]}
            for name, data in features.items()}})
    return PreparedFarm(farm_id, float(config['farms'][str(farm_id)]['ac_capacity_kw']) / 1000,
                        frame, features, train, validation, test, provenance)


def build_cycle_sequences(frame, features, masks, rating_mw, horizon_rows, *, forecast_features=None):
    """Pack each complete weather issue separately for each target split."""
    total = len(frame)
    if total % horizon_rows:
        raise ValueError('Cycle rows do not form complete fixed-length issue windows')
    groups = []
    # Split loss masks at target boundaries while keeping the full known weather context.
    for first in range(0, total, horizon_rows):
        positions = np.arange(first, first + horizon_rows)
        if frame.forecast_cycle_utc.iloc[positions].nunique() != 1:
            raise ValueError('Weather sequence crosses a forecast cycle')
        for label in ('train', 'validation', 'test'):
            if masks[label][positions].any():
                groups.append((positions, label))
    if not groups:
        raise ValueError('No forecast sequences have eligible targets')
    indices = np.stack([positions for positions, _ in groups])
    splits = np.asarray([label for _, label in groups])
    targets = frame.model_target_mw.to_numpy(dtype=float)[indices] / rating_mw
    if forecast_features is None:
        forecast_features = features
    packed_x = np.stack([features[position] if label == 'train' else forecast_features[position]
                         for position, label in groups]).astype(np.float32)
    packed = {'x': packed_x,
              'y': np.where(np.isfinite(targets), targets, 0).astype(np.float32),
              'indices': indices.astype(np.int64), 'split': splits,
              'output': frame['split'].to_numpy()[indices] == splits[:, None]}
    for label in ('train', 'validation', 'test'):
        packed[label] = masks[label][indices] & (splits[:, None] == label)
    return packed
