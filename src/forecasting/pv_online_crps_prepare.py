"""Prepare auction-specific deterministic predictions for the online CRPS study."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits

from ..data import config_path
from ..markets import AuctionCalendar
from .pv_probabilistic_data import align_cycle, file_hash, write_json
from .pv_probabilistic_reporting import load_farm
from .pv_probabilistic_search import add_time_features, fit_predictor, predict


def choose_cycle(cycles, cutoff, lag_hours):
    """Choose the most recent existing cycle known at the submission timestamp."""
    cycles = pd.DatetimeIndex(pd.to_datetime(cycles, utc=True)).sort_values()
    eligible = cycles[cycles + pd.Timedelta(hours=lag_hours) <= cutoff]
    return eligible[-1] if len(eligible) else None


def last_target_available(day, latency_minutes):
    """Availability of the final interval of a local delivery day."""
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize('Europe/Rome').tz_convert('UTC') + pd.Timedelta(minutes=latency_minutes)


def prepare_farm(config, source_config, farm, output_root):
    """Reconstruct fold predictors and preserve full provenance per auction interval."""
    settings = config['pv_online_crps_study']
    source = source_config['pv_probabilistic_study']
    root, ensemble, deterministic, ids, rating = load_farm(source_config, farm)
    trial_name = settings['farms'][farm]
    trial_root = root / 'trials' / trial_name
    result = json.loads((trial_root / 'result.json').read_text())
    if result['status'] != 'complete': raise ValueError(f'Incomplete source model: {trial_root}')
    for name, sha in result['files'].items():
        if file_hash(trial_root / name) != sha: raise ValueError(f'Source checksum mismatch: {trial_root/name}')
    trial = result['trial']
    destination = Path(output_root) / f'farm_{farm}'
    destination.mkdir(parents=True, exist_ok=True)
    ensemble = add_time_features(ensemble)
    final_model = joblib.load(trial_root / 'model.joblib')
    data = xr.open_zarr(config_path(source_config, source_config['inputs']['zarr']), consolidated=True, chunks=None)
    calendar = AuctionCalendar.from_config(source_config)
    latency = float(source_config['source'].get('telemetry_latency_minutes', 0))
    lag = float(source_config['source']['availability_lag_hours'])
    folds = source['calibration_folds']
    all_days = []
    try:
        with threadpool_limits(limits=1):
            for fold_number, fold in enumerate(folds, 1):
                marker = json.loads((trial_root / f'oof_fold_{fold_number}.json').read_text())
                oof_path = trial_root / f'oof_fold_{fold_number}.parquet'
                if marker['sha256'] != file_hash(oof_path): raise ValueError('Source OOF fold checksum mismatch')
                model_path = destination / f'fold_{fold_number}.joblib'
                identity_path = destination / f'fold_{fold_number}.json'
                identity = {'source_model_identity': result['identity'], 'fold': fold,
                            'source_oof_sha256': marker['sha256']}
                if model_path.exists() and identity_path.exists():
                    saved = json.loads(identity_path.read_text())
                    if saved['identity'] != identity or saved['sha256'] != file_hash(model_path):
                        raise ValueError('Fold model fingerprint mismatch; choose another run ID')
                    model = joblib.load(model_path)
                else:
                    model = fit_predictor(ensemble, trial, fold['fit_end'], rating, source)
                    temporary = model_path.with_suffix('.tmp.joblib')
                    joblib.dump(model, temporary, compress=1); temporary.replace(model_path)
                    write_json(identity_path, {'identity': identity, 'sha256': file_hash(model_path)})
                all_days.append((pd.read_parquet(oof_path), model, fold['fit_end'], f'fold_{fold_number}'))
                print(f'Farm {farm}: reconstructed fold {fold_number}', flush=True)
            validation = pd.read_parquet(trial_root / 'predictions.parquet')
            all_days.append((validation, final_model, source['train_end_day'], 'saved_validation_model'))
            output = []
            for paired, model, fit_end, label in all_days:
                for day, base in paired.groupby('delivery_day', sort=True):
                    base = base.sort_values('interval_start_utc').reset_index(drop=True)
                    base['interval_start_utc'] = pd.to_datetime(base.interval_start_utc, utc=True)
                    grid = calendar.delivery_grid(day)
                    if not pd.DatetimeIndex(base.interval_start_utc).equals(grid.starts):
                        raise ValueError(f'Unexpected source delivery grid on {day}')
                    for event in calendar.schedule(day):
                        if event.auction not in settings['auctions']: continue
                        cycle = choose_cycle(data.cycle_time.values, event.submission_at, lag)
                        if cycle is None: raise ValueError(f'No eligible deterministic cycle for {day} {event.auction}')
                        aligned = align_cycle(data, farm, cycle, grid.starts, source_config, False)
                        frame = aligned.values.copy().reset_index(drop=True)
                        frame['interval_start_utc'] = grid.starts
                        frame['delivery_day'] = day
                        frame = add_time_features(frame)
                        pv = predict(model, frame, source_config)
                        part = base.iloc[list(event.eligible_intervals)].copy()
                        part['deterministic_prediction'] = pv[list(event.eligible_intervals)]
                        part['auction'] = event.auction
                        part['decision_cutoff_utc'] = event.submission_at
                        part['deterministic_cycle_utc'] = cycle
                        part['deterministic_available_utc'] = aligned.available_at
                        part['target_available_utc'] = part.interval_start_utc + pd.Timedelta(minutes=15+latency)
                        part['model_fit_end'] = fit_end
                        part['model_fit_available_utc'] = last_target_available(fit_end, latency)
                        part['model_name'] = label
                        part['retrospective_model'] = part.model_fit_available_utc > event.submission_at
                        part['source_trial'] = trial_name
                        output.append(part)
                print(f'Farm {farm}: prepared auction forecasts for {label}', flush=True)
    finally:
        data.close()
    table = pd.concat(output, ignore_index=True)
    if table.duplicated(['auction', 'interval_start_utc']).any(): raise ValueError('Duplicate issued forecast')
    path = destination / 'prepared.parquet'
    temporary = path.with_suffix('.tmp.parquet')
    table.to_parquet(temporary, index=False); temporary.replace(path)
    manifest = {'farm_id': farm, 'source_trial': trial_name, 'source_identity': result['identity'],
                'source_result_sha256': file_hash(trial_root / 'result.json'),
                'model_sha256': file_hash(trial_root / 'model.joblib'),
                'prepared_sha256': file_hash(path), 'rating_mw': rating, 'member_ids': ids,
                'rows': len(table), 'assumed_forecast_lag_hours': lag,
                'assumed_telemetry_latency_minutes': latency}
    write_json(destination / 'prepared_manifest.json', manifest)
    return manifest
