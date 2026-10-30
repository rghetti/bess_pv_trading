"""Resumable member-wise statistical and neural PV model search."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import warnings
import yaml

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ...data import night_window_mask
from ..data.pv_probabilistic_data import file_hash, study_root, validate_study, write_json, probabilistic_night_policy
from ..evaluation.pv_probabilistic_metrics import fit_calibrator, paired_inputs, score_frame
from ..plotting.pv_probabilistic_reporting import apply_calibrator, comparable, load_farm, scored_windows, _hourly_scores

CORE = ['forecast_ghi', 'forecast_bhi', 'temperature_c', 'wind_u_10m', 'wind_v_10m']
EXTRA = ['dew_point_c', 'cloud_total_pct', 'cloud_low_pct', 'cloud_mid_pct', 'cloud_high_pct',
         'precipitation_mm', 'snow_water_equivalent_m', 'visibility_km', 'wind_u_100m', 'wind_v_100m']


def trial_plan(section):
    """Enumerate all predeclared family/feature/parameter combinations."""
    trials=[]
    # Enumerate all predeclared family/feature/parameter combinations for each `farm` in `section['farm_ids']`.
    for farm in section['farm_ids']:
        # Enumerate all predeclared family/feature/parameter combinations for each `(family, parameters)` in `section['trial_parameters'].items()`.
        for family, parameters in section['trial_parameters'].items():
            # Enumerate all predeclared family/feature/parameter combinations for each `features` in `section['feature_sets']`.
            for features in section['feature_sets']:
                # Enumerate all predeclared family/feature/parameter combinations for each `(number, spec)` in `enumerate(parameters)`.
                for number, spec in enumerate(parameters):
                    trials.append({'farm_id': int(farm), 'family': family, 'features': features,
                        'parameters': dict(spec), 'name': f'{family}_{features}_{number+1:02d}',
                        'seed': int(section['seed']) + int(farm) + number})
    return trials


def add_time_features(frame):
    """Add only calendar information known at the forecast issue time."""
    frame = frame.copy()
    local = pd.DatetimeIndex(frame.interval_start_utc).tz_convert('Europe/Rome') + pd.Timedelta(minutes=7.5)
    hour = local.hour.to_numpy() + local.minute.to_numpy()/60
    frame['hour_sin'] = np.sin(2*np.pi*hour/24)
    frame['hour_cos'] = np.cos(2*np.pi*hour/24)
    return frame


def features_for(trial):
    return CORE + (EXTRA if trial['features'] == 'weather' else []) + ['hour_sin', 'hour_cos']


def interval_weights(times, valid):
    """Give each target interval total weight one, regardless of member count."""
    valid = np.asarray(valid, dtype=bool)
    counts = pd.Series(valid.astype(int)).groupby(pd.DatetimeIndex(times)).transform('sum').to_numpy()
    return np.where(valid, 1/np.maximum(counts, 1), 0.)


def fit_predictor(frame, trial, fit_end, rating, settings):
    """Fit on past delivery days with each member sharing the target's weight."""
    data = frame.loc[frame.delivery_day.le(fit_end)].reset_index(drop=True)
    features = features_for(trial)
    x = data[features].to_numpy(float)
    y = data.measured_ac_mw.to_numpy(float) / rating
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & data.eligible.to_numpy(bool)
    # Enforce the input contract: No complete training member rows.
    if not valid.any(): raise ValueError('No complete training member rows')
    weights = interval_weights(data.interval_start_utc, valid)
    scaler = StandardScaler().fit(x[valid], sample_weight=weights[valid])
    family = trial['family']; params = dict(trial['parameters']); seed=trial['seed']
    # Fit on past delivery days with each member sharing the target's weight when `family in {'mlp', 'tcn'}`.
    if family in {'mlp', 'tcn'}:
        from ..models.pv_probabilistic_neural import fit_neural
        model, training = fit_neural(data, features, scaler, y, weights, trial, settings, seed)
    else:
        # Fit on past delivery days with each member sharing the target's weight when `family == 'ridge'`.
        if family == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(**params)
        # Fit on past delivery days with each member sharing the target's weight when `family == 'huber'`.
        elif family == 'huber':
            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor(**params, max_iter=500)
        # Fit on past delivery days with each member sharing the target's weight when `family == 'extra_trees'`.
        elif family == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(**params, n_estimators=int(settings['tree_estimators']), n_jobs=1, random_state=seed)
        # Fit on past delivery days with each member sharing the target's weight when `family == 'hist_gradient_boosting'`.
        elif family == 'hist_gradient_boosting':
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(**params, max_iter=int(settings['boosting_iterations']),
                learning_rate=float(settings['boosting_learning_rate']), early_stopping=False, random_state=seed)
        # Fit on past delivery days with each member sharing the target's weight when `family == 'lightgbm'`.
        elif family == 'lightgbm':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params, n_estimators=int(settings['boosting_iterations']),
                learning_rate=float(settings['boosting_learning_rate']), n_jobs=1, random_state=seed, verbosity=-1)
        # Fit on past delivery days with each member sharing the target's weight when `family == 'catboost'`.
        elif family == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**params, iterations=int(settings['boosting_iterations']),
                learning_rate=float(settings['boosting_learning_rate']), thread_count=1,
                random_seed=seed, verbose=False, allow_writing_files=False)
        else: raise ValueError(f'Unknown model family: {family}')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(scaler.transform(x[valid]), y[valid], sample_weight=weights[valid])
        training = {'warnings': [str(w.message) for w in captured][:20]}
    training.update({'fit_end': fit_end, 'valid_member_rows': int(valid.sum()),
                     'independent_target_intervals': int(data.loc[valid, 'interval_start_utc'].nunique()),
                     'training_days': int(data.loc[valid, 'delivery_day'].nunique()),
                     'total_target_weight': float(weights.sum())})
    return {'model': model, 'scaler': scaler, 'features': features, 'family': family,
            'rating_mw': rating, 'training': training}


def predict(model, frame, config):
    """Evaluate one fitted predictor on a member or deterministic weather path."""
    features = model['features']; x = frame[features].to_numpy(float)
    valid = np.isfinite(x).all(axis=1)
    pred = np.full(len(frame), np.nan)
    # Evaluate one fitted predictor on a member or deterministic weather path when `model['family'] in {'mlp', 'tcn'}`.
    if model['family'] in {'mlp', 'tcn'}:
        from ..models.pv_probabilistic_neural import predict_neural
        pred = predict_neural(model['model'], frame, features, model['scaler'])
    # When `valid.any()`, set `pred[valid]` using `model['model'].predict(model['scaler'].transform(x[valid]))`.
    elif valid.any():
        pred[valid] = model['model'].predict(model['scaler'].transform(x[valid]))
    pred = np.clip(pred * model['rating_mw'], 0, model['rating_mw'])
    night = night_window_mask(pd.DatetimeIndex(frame.interval_start_utc), probabilistic_night_policy(config))
    pred[np.isfinite(pred) & night] = 0.
    return pred


def _records(frame):
    """Convert tabular results to strict JSON without nonfinite values."""
    return json.loads(frame.to_json(orient='records'))


def _trial_identity(root, trial, section):
    manifest = json.loads((root / 'input_manifest.json').read_text())
    relevant = {'trial': trial, 'inputs': manifest['files'], 'source': manifest['source_fingerprint'],
                'section': {k:v for k,v in section.items() if k not in {'workers', 'farm_ids', 'resume', 'reserved_cpu_cores', 'estimated_worker_memory_gb', 'respect_shared_load'}},
                'code': {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).resolve().parents[1] / 'models/pv_probabilistic_neural.py',
                    Path(__file__).resolve().parents[1] / 'evaluation/pv_probabilistic_metrics.py', Path(__file__).resolve().parents[1] / 'plotting/pv_probabilistic_reporting.py']}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def run_trial(config, trial):
    """Train chronological folds, calibrate out of fold, and score the final model."""
    section = config['pv_probabilistic_study']; farm = trial['farm_id']
    root, ens, det, ids, rating = load_farm(config, farm)
    dest = root / 'trials' / trial['name']; dest.mkdir(parents=True, exist_ok=True)
    identity = _trial_identity(root, trial, section); result_path = dest / 'result.json'
    # When `section['resume'] and result_path.exists()`, set `old` using `json.loads(result_path.read_text())`.
    if section['resume'] and result_path.exists():
        old = json.loads(result_path.read_text())
        # Enforce the input contract: Resume fingerprint changed for `dest`.
        if old.get('identity') != identity: raise ValueError(f'Resume fingerprint changed for {dest}')
        # Train chronological folds, calibrate out of fold, and score the final model for each `(name, expected)` in `old.get('files', {}).items()`.
        for name, expected in old.get('files', {}).items():
            # Enforce the input contract: Resume artifact checksum changed: `dest / name`.
            if file_hash(dest / name) != expected: raise ValueError(f'Resume artifact checksum changed: {dest/name}')
        # Return `{'trial': trial['name'], 'farm_id': farm, 'status': 'reused'}` when `old.get('status') == 'complete'`.
        if old.get('status') == 'complete': return {'trial': trial['name'], 'farm_id': farm, 'status': 'reused'}
    ens = add_time_features(ens); det = add_time_features(det)
    oof = []
    with threadpool_limits(limits=1):
        # Apply predict for `(number, fold)` from `enumerate(section['calibration_folds'], 1)`.
        for number, fold in enumerate(section['calibration_folds'], 1):
            fold_path = dest / f'oof_fold_{number}.parquet'
            fold_marker = dest / f'oof_fold_{number}.json'
            # When `section['resume'] and fold_path.exists() and fold_marker.exists()`, set `saved` using `json.loads(fold_marker.read_text())`.
            if section['resume'] and fold_path.exists() and fold_marker.exists():
                saved = json.loads(fold_marker.read_text())
                # Enforce the input contract: Out-of-fold checkpoint mismatch: `fold_path`.
                if saved['identity'] != identity or saved['sha256'] != file_hash(fold_path):
                    raise ValueError(f'Out-of-fold checkpoint mismatch: {fold_path}')
                frame = pd.read_parquet(fold_path)
            else:
                model = fit_predictor(ens, trial, fold['fit_end'], rating, section)
                sub_e = ens.loc[ens.delivery_day.between(fold['predict_start'], fold['predict_end'])].reset_index(drop=True)
                sub_d = det.loc[det.delivery_day.between(fold['predict_start'], fold['predict_end'])].reset_index(drop=True)
                frame = paired_inputs(sub_e, sub_d, predict(model, sub_e, config), predict(model, sub_d, config), ids)
                temporary = fold_path.with_suffix('.tmp.parquet')
                frame.to_parquet(temporary, index=False); temporary.replace(fold_path)
                write_json(fold_marker, {'identity': identity, 'fold': fold, 'sha256': file_hash(fold_path),
                                         'training': model['training']})
            oof.append(frame)
        calibration = pd.concat(oof, ignore_index=True)
        valid = comparable(calibration, ids)
        levels = list(map(float, section['quantiles']))
        calibrator = fit_calibrator(calibration.loc[valid, ids].to_numpy(float),
            calibration.loc[valid, 'measured_ac_mw'], calibration.loc[valid, 'interval_start_utc'],
            rating, levels, section['calibration_alpha'])
        model_path = dest / 'model.joblib'
        model_marker = dest / 'model_checkpoint.json'
        # When `section['resume'] and model_path.exists() and model_marker.exists()`, set `saved` using `json.loads(model_marker.read_text())`.
        if section['resume'] and model_path.exists() and model_marker.exists():
            saved = json.loads(model_marker.read_text())
            # Enforce the input contract: Model checkpoint mismatch: `model_path`.
            if saved['identity'] != identity or saved['sha256'] != file_hash(model_path):
                raise ValueError(f'Model checkpoint mismatch: {model_path}')
            model = joblib.load(model_path)
        else:
            model = fit_predictor(ens, trial, section['train_end_day'], rating, section)
            temporary = dest / 'model.tmp.joblib'
            joblib.dump(model, temporary, compress=1); temporary.replace(model_path)
            write_json(model_marker, {'identity': identity, 'sha256': file_hash(model_path), 'training': model['training']})
        sub_e = ens.loc[ens.split.eq('validation')].reset_index(drop=True)
        sub_d = det.loc[det.split.eq('validation')].reset_index(drop=True)
        predictions = paired_inputs(sub_e, sub_d, predict(model, sub_e, config), predict(model, sub_d, config), ids)
        predictions, adjustments = apply_calibrator(predictions, ids, calibrator, levels, probabilistic_night_policy(config))
        metrics = scored_windows(predictions, ids, levels, rating)
        hourly, hourly_metrics = _hourly_scores(predictions, ids, levels, rating)
        predictions.to_parquet(dest / 'predictions.parquet', index=False)
        calibration.to_parquet(dest / 'oof_calibration_predictions.parquet', index=False)
        hourly.to_parquet(dest / 'hourly_predictions.parquet', index=False)
        hourly_metrics.to_csv(dest / 'hourly_metrics.csv', index=False)
        pd.concat([score_frame(part, ids, levels, rating, day)
                   for day, part in predictions.groupby('delivery_day')], ignore_index=True).to_csv(dest / 'daily_metrics.csv', index=False)
        joblib.dump(calibrator, dest / 'quantile_calibrator.joblib', compress=1)
    names = ['predictions.parquet', 'oof_calibration_predictions.parquet', 'hourly_predictions.parquet',
             'hourly_metrics.csv', 'daily_metrics.csv', 'model.joblib', 'quantile_calibrator.joblib']
    result = {'status': 'complete', 'identity': identity, 'trial': trial, 'training': model['training'],
        'oof_calibration_rows': int(valid.sum()), 'quantile_adjustments': adjustments,
        'validation_coverage': {'total_intervals': len(predictions),
            'paired_eligible_intervals': int(comparable(predictions, ids).sum()),
            'excluded_target_intervals': int(predictions.episode_excluded.sum()),
            'missing_or_invalid_forecast_or_target': int((~comparable(predictions, ids) & ~predictions.episode_excluded).sum())},
        'metrics': _records(metrics), 'files': {name: file_hash(dest / name) for name in names},
        'limitation': 'Validation is used for selection; the first validation day is retrospective at the historical DA gate.'}
    write_json(result_path, result)
    return {'trial': trial['name'], 'farm_id': farm, 'status': 'complete'}


def _memory_available_gb():
    try:
        # Perform memory available gb for each `line` in `Path('/proc/meminfo').read_text().splitlines()`.
        for line in Path('/proc/meminfo').read_text().splitlines():
            # Return `int(line.split()[1]) / (1024 * 1024)` when `line.startswith('MemAvailable:')`.
            if line.startswith('MemAvailable:'): return int(line.split()[1])/(1024*1024)
    except OSError: pass
    return None


def run_search(config):
    """Dispatch immutable candidates within the configured CPU and memory budget."""
    section = config['pv_probabilistic_study']; root = study_root(config); root.mkdir(parents=True, exist_ok=True)
    validate_study(section)
    plan = trial_plan(section)
    # Spread work across farms and families so resumable progress is informative.
    families = list(section['trial_parameters'])
    plan.sort(key=lambda t: (int(t['name'].rsplit('_', 1)[-1]), t['features'], families.index(t['family']), t['farm_id']))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    memory = _memory_available_gb()
    workers = min(int(section['workers']), max(1, cpus-int(section['reserved_cpu_cores'])))
    # Update `workers` when this condition is met.
    if memory is not None: workers = min(workers, max(1, int(memory/float(section['estimated_worker_memory_gb']))))
    load = os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
    # Update `workers` when this condition is met.
    if section.get('respect_shared_load', True) and load is not None:
        workers = min(workers, max(1, cpus-int(section['reserved_cpu_cores'])-math.ceil(load)))
    # Enforce the input contract: No CPU or memory available for a study worker.
    if workers < 1: raise ValueError('No CPU or memory available for a study worker')
    # Process each `name` in this collection.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / 'trial_plan.json', {'trials': plan, 'workers': workers, 'cpu_available': cpus,
        'memory_available_gb': memory, 'shared_load_at_launch': load, 'training': [section['start_day'], section['train_end_day']],
        'validation': [str((pd.Timestamp(section['train_end_day']) + pd.Timedelta(days=1)).date()), section.get('validation_end_day') or section['end_day']],
        'calibration_folds': section['calibration_folds'],
        'test_set': [str((pd.Timestamp(section['validation_end_day']) + pd.Timedelta(days=1)).date()), section['end_day']] if section.get('validation_end_day') else None})
    print(f"Search: {len(plan)} candidates, {workers} single-thread workers; training through {section['train_end_day']}, validation through {section.get('validation_end_day') or section['end_day']}", flush=True)
    failures=[]; completed=0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(run_trial, config, trial): trial for trial in plan}
        # Dispatch immutable candidates within the configured CPU and memory budget for each `future` in `as_completed(futures)`.
        for future in as_completed(futures):
            trial = futures[future]
            try:
                result = future.result(); completed += 1
                print(f'{completed}/{len(plan)}: farm {trial["farm_id"]} {trial["name"]} {result["status"]}', flush=True)
            except Exception as exc:
                failures.append({'trial': trial, 'error': f'{type(exc).__name__}: {exc}'})
                write_json(root / 'search_failures.json', failures)
                print(f'FAILED farm {trial["farm_id"]} {trial["name"]}: {exc}', flush=True)
    write_json(root / 'search_status.json', {'completed': completed, 'total': len(plan), 'failures': failures,
                                             'status': 'complete' if not failures else 'partial'})
    # Enforce the input contract: `len(failures)` model trials failed; details saved in search_failures.json.
    if failures: raise RuntimeError(f'{len(failures)} model trials failed; details saved in search_failures.json')
