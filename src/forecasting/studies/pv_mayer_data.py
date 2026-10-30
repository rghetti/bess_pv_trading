"""Causal input preparation for Mayer-style hourly PV post-processing."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from ...data import config_path, load_metadata
from ...markets import AuctionCalendar
from ..data.pv_probabilistic_data import _cycle, _target_frame, align_cycle
from ..data.pv_study_inputs import farm_settings
from ..inference.pv_forecast_bundle import predict_bundle
from ..models.pv import local_delivery_starts


def check_coverage(config):
    """Check declared dates, real source cycles, measurement grid and auction validity."""
    s = config['pv_mayer_postprocessing']
    start, train_end, end = (pd.Timestamp(s[k]).normalize() for k in ('start_day', 'train_end_day', 'end_day'))
    if not start <= train_end < end:
        raise ValueError('Expected start_day <= train_end_day < end_day')
    periods = sorted(s['dataset_periods'], key=lambda p: p['start_day'])
    days = pd.date_range(start, end, freq='D')
    routes = {}
    for period in periods:
        a, b = pd.Timestamp(period['start_day']), pd.Timestamp(period['end_day'])
        if a > b:
            raise ValueError('Ensemble archive has an inverted delivery window')
        for day in pd.date_range(max(a, start), min(b, end), freq='D'):
            if day in routes:
                raise ValueError(f'Overlapping ensemble archives on {day.date()}')
            routes[day] = period['ensemble_zarr']
    missing_days = [str(day.date()) for day in days if day not in routes]
    if missing_days and not s['allow_declared_gaps']:
        raise ValueError(f'Incomplete declared ensemble coverage: {len(missing_days)} days; first {missing_days[:8]}')
    if not routes or not any(day <= train_end for day in routes) or not any(day > train_end for day in routes):
        raise ValueError('Declared windows must include both training and evaluation delivery days')
    calendar = AuctionCalendar.from_config(config)
    with ExitStack() as stack:
        det = stack.enter_context(xr.open_zarr(config_path(config, config['inputs']['zarr']), consolidated=True, chunks=None))
        stores = {path: stack.enter_context(xr.open_zarr(config_path(config, path), consolidated=True, chunks=None))
                  for path in set(routes.values())}
        det_cycles = set(pd.DatetimeIndex(pd.to_datetime(det.cycle_time.values, utc=True)))
        observed_times = set(pd.DatetimeIndex(pd.to_datetime(det.time.values, utc=True)))
        expected_ids = None
        for path, store in stores.items():
            ids = tuple(map(str, store.member.values))
            if len(ids) != int(s['member_count']) or len(set(ids)) != len(ids):
                raise ValueError(f'{path} does not contain {s["member_count"]} distinct members')
            if expected_ids is not None and ids != expected_ids:
                raise ValueError('Ensemble member labels/order differ between archives')
            expected_ids = ids
            for field in config['forecast_mapping'].values():
                if field not in store or field not in det:
                    raise ValueError(f'Forecast field {field} missing from {path} or deterministic source')
                for attribute in ('units', 'temporal_semantics'):
                    if store[field].attrs.get(attribute) != det[field].attrs.get(attribute):
                        raise ValueError(f'{path} {field} {attribute} differs from deterministic source')
        for day in sorted(routes):
            cycle = _cycle(day)
            store = stores[routes[day]]
            if cycle not in set(pd.DatetimeIndex(pd.to_datetime(store.cycle_time.values, utc=True))) or cycle not in det_cycles:
                raise ValueError(f'Missing ensemble or deterministic cycle {cycle}')
            grid = local_delivery_starts(str(day.date()), s['timezone'])
            if not set(grid).issubset(observed_times):
                raise ValueError(f'Measurement time grid incomplete on {day.date()}')
            da = next(e for e in calendar.schedule(str(day.date())) if e.auction == 'DA')
            if cycle + pd.Timedelta(hours=float(config['source']['availability_lag_hours'])) > da.submission_at:
                raise ValueError(f'Forecast cycle unavailable at DA cutoff on {day.date()}')
    return routes


def aggregate_hourly(quarters, members, rating, *, latency_minutes=0):
    """Average actual quarter values while retaining missing-target and bounds flags."""
    frame = quarters.copy()
    frame['interval_start_utc'] = pd.to_datetime(frame.interval_start_utc, utc=True)
    if frame.interval_start_utc.duplicated().any():
        raise ValueError('Duplicate quarter-hour in an ensemble delivery day')
    if any(name not in frame for name in members):
        raise ValueError('A requested ensemble member is missing')
    frame['hour_start_utc'] = frame.interval_start_utc.dt.floor('h')
    records = []
    for hour, group in frame.groupby('hour_start_utc', sort=True):
        expected = pd.date_range(hour, periods=4, freq='15min')
        if not pd.DatetimeIndex(group.interval_start_utc).equals(expected):
            raise ValueError(f'Incomplete or misaligned physical hour {hour}')
        target = group.measured_ac_mw.to_numpy(float)
        valid = group.eligible.to_numpy(bool) & np.isfinite(target)
        out_of_bounds = np.isfinite(target) & ((target < 0) | (target > rating))
        prediction = group[['control_mw', *members]].to_numpy(float)
        valid_forecast = np.isfinite(prediction).all() and (prediction >= 0).all() and (prediction <= rating).all()
        first = group.iloc[0]
        row = {'delivery_day': first.delivery_day, 'hour_start_utc': hour,
               'forecast_cycle_utc': first.forecast_cycle_utc, 'forecast_available_at_utc': first.forecast_available_at_utc,
               'decision_cutoff_utc': first.decision_cutoff_utc,
               'lead_hours': float((hour - pd.Timestamp(first.forecast_cycle_utc)) / pd.Timedelta(hours=1)),
               'target_available_at_utc': hour + pd.Timedelta(hours=1, minutes=latency_minutes),
               'valid_quarters': int(valid.sum()), 'excluded_quarters': int(group.episode_excluded.sum()),
               'out_of_bounds_quarters': int(out_of_bounds.sum()), 'forecast_valid': bool(valid_forecast),
               'measured_ac_mw': float(target.mean()) if valid.all() else np.nan,
               'control_mw': float(prediction[:, 0].mean()) if valid_forecast else np.nan}
        row.update({name: float(prediction[:, i + 1].mean()) if valid_forecast else np.nan for i, name in enumerate(members)})
        records.append(row)
    return pd.DataFrame(records)


def prepare_farm(config, farm_id, routes):
    """Predict each preserved member path and assemble unmatched raw hourly observations."""
    s = config['pv_mayer_postprocessing']
    metadata = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = farm_settings(config, metadata)
    location = settings['location']
    model = config_path(config, config['pv']['forecast_replacement']['models_by_farm'][str(farm_id)]['artifact_path']).parent
    manifest = json.loads((model / 'manifest.json').read_text())
    required_weather = [column for column in manifest['preprocessor']['columns'] if column in config['forecast_mapping']]
    calendar = AuctionCalendar.from_config(config)
    # _target_frame must never invoke the other study's night target rewrite.
    target_config = deepcopy(config)
    target_config.setdefault('pv_probabilistic_study', {})['night_target_treatment'] = 'preserve'
    frames = []; quarter_frames = []; source_digest = hashlib.sha256()
    with ExitStack() as stack:
        det = stack.enter_context(xr.open_zarr(config_path(config, config['inputs']['zarr']), consolidated=True, chunks=None))
        stores = {p: stack.enter_context(xr.open_zarr(config_path(config, p), consolidated=True, chunks=None))
                  for p in set(routes.values())}
        for day, path in sorted(routes.items()):
            date = str(day.date()); starts = local_delivery_starts(date, s['timezone']); cycle = _cycle(day)
            cutoff = pd.Timestamp(next(e for e in calendar.schedule(date) if e.auction == 'DA').submission_at)
            target = _target_frame(det, farm_id, starts, target_config).sort_values('interval_start_utc').reset_index(drop=True)
            columns = target[['interval_start_utc', 'measured_ac_mw', 'eligible', 'episode_excluded']].copy()
            columns['delivery_day'] = date; columns['forecast_cycle_utc'] = cycle
            columns['forecast_available_at_utc'] = cycle + pd.Timedelta(hours=float(config['source']['availability_lag_hours']))
            columns['decision_cutoff_utc'] = cutoff
            ensemble = align_cycle(stores[path], farm_id, cycle, starts, config, True)
            deterministic = align_cycle(det, farm_id, cycle, starts, config, False)
            for source_frame in (target, ensemble.values, deterministic.values):
                source_digest.update(pd.util.hash_pandas_object(source_frame, index=True).values.tobytes())
            for bundle, control in ((deterministic, True), (ensemble, False)):
                for member in bundle.members:
                    view = bundle.values.loc[bundle.values.member.eq(member)].copy()
                    view['interval_start_utc'] = view.index; view['delivery_day'] = date; view['auction'] = 'DA'
                    view['forecast_cycle_utc'] = cycle
                    if not pd.DatetimeIndex(view.interval_start_utc).equals(starts):
                        raise ValueError(f'Weather member {member} does not cover the delivery grid')
                    # Predict only contiguous valid subsequences; never fill missing required weather.
                    weather_valid = np.isfinite(view[required_weather].to_numpy(float)).all(axis=1)
                    values = np.full(len(view), np.nan)
                    if weather_valid.any():
                        values[weather_valid] = predict_bundle(view.iloc[np.flatnonzero(weather_valid)].copy(), model, farm_id=farm_id)
                    columns['control_mw' if control else f'member_{member}'] = values
            names = [f'member_{member}' for member in ensemble.members]
            quarter_frames.append(columns)
            hourly = aggregate_hourly(columns, names, metadata.ac_capacity_mw,
                                      latency_minutes=float(config['source'].get('telemetry_latency_minutes', 0)))
            centers = pd.DatetimeIndex(hourly.hour_start_utc) + pd.Timedelta(minutes=30)
            zenith = pvlib.solarposition.get_solarposition(centers, latitude=float(location['latitude']),
                       longitude=float(location['longitude']), altitude=metadata.elevation_m)['zenith'].to_numpy(float)
            hourly['solar_zenith_deg'] = zenith
            hourly['daylight'] = zenith < 90
            frames.append(hourly)
    return (pd.concat(frames, ignore_index=True), pd.concat(quarter_frames, ignore_index=True),
            names, metadata.ac_capacity_mw, location, source_digest.hexdigest())
