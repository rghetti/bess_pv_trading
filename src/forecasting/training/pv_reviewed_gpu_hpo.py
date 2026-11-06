"""Separate CUDA HPO runner using reviewed AC targets and forecast-GHI scoring.

This runner deliberately has its own scheduling and checkpoint contract so changes
to CUDA execution do not invalidate existing CPU search identities.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
from copy import deepcopy
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from ...data import config_path
from ..data.pv_reviewed_hpo_data import fit_transforms, prepare_farm
from ..models.pv_reviewed_gpu_hpo_models import build_sequences, fit_neural, fit_statistical, predict_neural
from .pv_reviewed_hpo_common import identity_digest, score_predictions, source_masks, trial_specifications


def _sha(path):
    """Identify a saved search dependency by its exact bytes."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, content):
    """Publish a complete JSON record atomically for resumable workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def _dependencies():
    """Check installed CPU libraries without changing the conda environment."""
    versions = {}
    for package, module in [('numpy', 'numpy'), ('pandas', 'pandas'), ('scikit-learn', 'sklearn'),
                            ('torch', 'torch'), ('lightgbm', 'lightgbm'), ('catboost', 'catboost'),
                            ('pyarrow', 'pyarrow')]:
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    return versions


def _available_memory_gib():
    """Read Linux available memory without treating cache as unavailable."""
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / (1024 * 1024)
    return None


def _runtime_details(section, require_free_memory=False):
    """Inspect the selected accelerator and check it can execute a CUDA operation."""
    runtime = section.get('runtime', {})
    device = runtime.get('device', 'cpu')
    if device == 'cpu':
        return {'device': 'cpu'}
    # CUDA linear algebra must use a deterministic workspace before initialization.
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this process; check the NVIDIA driver, PyTorch build and CUDA_VISIBLE_DEVICES')
    index = torch.device(device).index
    if index is None or index >= torch.cuda.device_count():
        raise RuntimeError(f'{device} is not a visible CUDA device')
    properties = torch.cuda.get_device_properties(index)
    # Execute a tiny operation so unsupported architecture binaries fail before training.
    try:
        torch.ones(1, device=device).add_(1).cpu()
        torch.cuda.synchronize(index)
    except Exception as error:
        raise RuntimeError(f'PyTorch cannot execute CUDA kernels on {device}: {error}') from error
    free, total = torch.cuda.mem_get_info(index)
    minimum = float(runtime.get('minimum_free_gpu_memory_gib', 0))
    free_gib = free / 1024 ** 3
    if require_free_memory and free_gib < minimum:
        raise RuntimeError(f'{device} requires {minimum:g} GiB free GPU memory; currently {free_gib:.2f} GiB is free')
    return {'device': device, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'name': properties.name,
            'compute_capability': [properties.major, properties.minor],
            'total_memory_bytes': total, 'free_memory_gib': free_gib,
            'torch_cuda_version': torch.version.cuda, 'compiled_architectures': torch.cuda.get_arch_list(),
            'cublas_workspace_config': workspace}


def validate_configuration(config):
    """Reject incomplete or incompatible search settings before touching output paths."""
    section = config['pv_reviewed_hpo']
    if section['farm_ids'] != [1, 345, 901, 910]:
        raise ValueError('This search is configured for all four reviewed farms')
    if section['features'] != ['core', 'extended']:
        raise ValueError('Both core and extended feature versions are required')
    regime = section.get('data_regime', 'historical_ghi')
    if regime not in {'historical_ghi', 'four_cycle_forecast_ghi'}:
        raise ValueError(f'Unsupported reviewed HPO data regime: {regime}')
    runtime = section.get('runtime', {})
    device = runtime.get('device', 'cpu')
    if float(runtime.get('minimum_free_gpu_memory_gib', 0)) < 0:
        raise ValueError('Minimum free GPU memory must be nonnegative')
    if device != 'cpu' and (regime != 'four_cycle_forecast_ghi' or device != 'cuda:0'):
        raise ValueError('GPU search requires the four-cycle regime and cuda:0; select the physical GPU using CUDA_VISIBLE_DEVICES')
    expected_workers = 42 if regime == 'four_cycle_forecast_ghi' else 30
    configured_cpu_workers = int(runtime.get('cpu_workers', expected_workers))
    workers = int(section['workers'])
    if device == 'cuda:0':
        maximum = int(runtime.get('maximum_gpu_workers', 1))
        if not 1 <= workers <= maximum:
            raise ValueError(f'GPU workers must be between 1 and configured maximum {maximum}')
    elif workers != configured_cpu_workers:
        raise ValueError(f'This CPU run requires its configured {configured_cpu_workers} workers')
    if int(section['threads_per_worker']) != 1:
        raise ValueError('Each worker must use one CPU thread')
    if regime == 'four_cycle_forecast_ghi' and (
            bool(section['statistical_families']) == bool(section['neural_families']) or
            section['cycle_hours_utc'] != [0, 6, 12, 18] or section['issue_horizon_hours'] != 48):
        raise ValueError('The four-cycle study requires one model class and all four 48-hour cycles')
    if regime == 'four_cycle_forecast_ghi' and (
            section['training_ghi_source'] != 'historical'
            or section['validation_ghi_source'] != 'forecast'
            or section['testing_ghi_source'] != 'forecast'
            or section['include_missing_indicators']):
        raise ValueError('Four-cycle study requires historical training GHI, forecast scoring GHI and no missing flags')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if available is not None and available < int(section['workers']):
        raise ValueError(f'Only {available} CPUs are available for the requested {workers} workers')
    if int(section['trials_per_family_per_feature']) < 1:
        raise ValueError('Every model family needs at least one hyperparameter configuration')
    if section['nighttime_exclusions'] != 'keep_curated' or section['missing_night_ghi'] != 'zero_at_night':
        raise ValueError('The nighttime treatment must match the user-confirmed decisions')
    if int(section['night_training_samples_per_farm']) < 0:
        raise ValueError('The nighttime training sample count must be nonnegative')
    if not section['use_synthetic_labels']:
        raise ValueError('User-reviewed synthetic labels must remain available to training')
    if section['score'] != 'validation_day_balanced_normalized_rmse':
        raise ValueError('Select models only with the declared prior-validation score')
    if section['output_clip'] != [0.0, 1.0]:
        raise ValueError('Forecast outputs must be bounded by the farm AC rating')
    return section


def check_only(config, progress=print):
    """Inspect every reviewed farm and the full trial grid without fitting a model."""
    section = validate_configuration(config)
    runtime = _runtime_details(section)
    versions = _dependencies()
    trials = trial_specifications(section)
    prepared = {}
    for farm_id in section['farm_ids']:
        if progress:
            cycle_description = ('four forecast cycles' if section.get('data_regime') == 'four_cycle_forecast_ghi'
                                 else 'D-1 forecast')
            progress(f'Checking farm {farm_id}: reviewed masks, {cycle_description} features and hashes')
        if section.get('data_regime') == 'four_cycle_forecast_ghi':
            from ..data.pv_four_cycle_hpo_data import prepare_four_cycle_farm
            prepared[farm_id] = prepare_four_cycle_farm(config, farm_id)
        else:
            prepared[farm_id] = prepare_farm(config, farm_id)
        # These are deterministic train-only feature statistics, not supervised model fits.
        preprocessors, matrices = fit_transforms(prepared[farm_id], config)
        if (section.get('data_regime') == 'four_cycle_forecast_ghi' and
                section['neural_families']):
            from ..data.pv_four_cycle_hpo_data import build_cycle_sequences
            item = prepared[farm_id]
            masks = {'train': item.training_mask, 'validation': item.validation_mask, 'test': item.test_mask}
            from ..data.pv_neural_data import transform_features
            forecast_features = item.features['extended'].copy()
            forecast_features['ghi_w_m2'] = item.frame.ghi_forecast_w_m2.to_numpy()
            forecast_matrix = transform_features(forecast_features, preprocessors['extended'])
            packed = build_cycle_sequences(item.frame, matrices['extended'], masks, item.rating_mw,
                int(section['issue_horizon_hours']) * 4, forecast_features=forecast_matrix)
            item.provenance['sequence_counts'] = {
                split: int((packed['split'] == split).sum()) for split in ('train', 'validation', 'test')}
            item.provenance['sequence_length'] = int(packed['indices'].shape[1])
    families = list(section['statistical_families']) + list(section['neural_families'])
    expected = (len(section['farm_ids']) * len(section['features']) * len(families)
                * int(section['trials_per_family_per_feature']))
    if len(trials) != expected:
        raise ValueError('Trial grid does not cover every farm, feature set and model family')
    code_files = [Path(__file__), Path(__file__).with_name('pv_reviewed_hpo_common.py'),
        Path(__file__).parents[1] / 'models' / 'pv_reviewed_gpu_hpo_models.py',
        Path(__file__).parents[1] / 'models' / 'pv_neural_models.py',
        Path(__file__).parents[1] / 'data' / 'pv_reviewed_hpo_data.py',
        Path(__file__).parents[1] / 'data' / 'pv_four_cycle_hpo_data.py',
        Path(__file__).parents[1] / 'data' / 'pv_neural_data.py',
        Path(__file__).parents[1] / 'data' / 'pv_curated_review.py',
        Path(__file__).parents[1] / 'inference' / 'pv_reviewed_hpo.py']
    identity_config = deepcopy(section)
    identity_config.pop('resume', None)
    runtime_identity = {key: value for key, value in runtime.items() if key != 'free_memory_gib'}
    contract = {'configuration': identity_config, 'runtime': runtime_identity, 'forecast_mapping': config['forecast_mapping'],
        'forecast_semantics': config['forecast_semantics'], 'source': config['source'],
        'datasets': {str(farm): item.provenance for farm, item in prepared.items()},
        'code_sha256': {path.name: _sha(path) for path in code_files}, 'versions': versions}
    report = {'status': 'checked_no_training', 'training_started': False,
        'identity': identity_digest(contract), 'workers': section['workers'],
        'threads_per_worker': section['threads_per_worker'], 'runtime': runtime,
        'available_memory_gib': _available_memory_gib(),
        'minimum_available_memory_gib': section.get('minimum_available_memory_gib'),
        'statistical_families': list(section['statistical_families']),
        'neural_families': section['neural_families'], 'features': section['features'],
        'trials_per_family_per_feature_per_farm': section['trials_per_family_per_feature'],
        'hpo_fits': len(trials), 'family_winner_refits': len(families) * len(section['features']) * len(section['farm_ids']),
        'top_three_additional_refits_max': 2 * len(section['farm_ids']),
        'selection': section['score'], 'versions': versions,
        'ghi_source': ({'train': 'historical measured/interpolated GHI; four forecast cycles for other weather',
            'validation_and_test': 'forecast GHI after assumed availability'}
            if section.get('data_regime') == 'four_cycle_forecast_ghi' else
            {'train': 'historical, with missing nighttime GHI set to zero',
             'validation_and_test': 'preceding-day 00 UTC hourly forecast means repeated per quarter'}),
        'availability': 'Six-hour weather release lag is a configured assumption; historical publication is unverified.',
        'limitations': 'The 2026 period was previously inspected; some scoring targets are pvlib-derived reconstructions.',
        'farms': {str(farm): {'provenance': item.provenance,
            'features': {name: list(value.columns) for name, value in item.features.items()},
            'rating_mw': item.rating_mw} for farm, item in prepared.items()}}
    return report, prepared, trials, contract


def _prepare_cache(root, item, config):
    """Freeze train-only transforms and day sequences for reproducible trial workers."""
    directory = root / f'farm_{item.farm_id}' / 'cache'
    directory.mkdir(parents=True, exist_ok=True)
    fitted, matrices = fit_transforms(item, config)
    masks = {'train': item.training_mask, 'validation': item.validation_mask, 'test': item.test_mask}
    np.savez(directory / 'masks.npz', **masks)
    columns = ['interval_start_utc', 'forecast_cycle_utc', 'delivery_day', 'split', 'model_target_mw',
        'model_target_source', 'synthetic_target', 'night', 'evaluation_window', 'ghi_interpolated']
    if 'lead_hours' in item.frame:
        columns.append('lead_hours')
    metadata = item.frame[columns].copy()
    metadata['ghi_input_w_m2'] = item.features['core'].ghi_w_m2.to_numpy()
    metadata.to_parquet(directory / 'frame.parquet', index=False)
    for name, matrix in matrices.items():
        # Packed sequences split at retained prior exclusions but retain nighttime labels.
        if config['pv_reviewed_hpo'].get('data_regime') == 'four_cycle_forecast_ghi':
            if config['pv_reviewed_hpo']['neural_families']:
                from ..data.pv_four_cycle_hpo_data import build_cycle_sequences
                from ..data.pv_neural_data import transform_features
                forecast_features = item.features[name].copy()
                forecast_features['ghi_w_m2'] = item.frame.ghi_forecast_w_m2.to_numpy()
                forecast_matrix = transform_features(forecast_features, fitted[name])
                packed = build_cycle_sequences(item.frame, matrix, masks, item.rating_mw,
                    int(config['pv_reviewed_hpo']['issue_horizon_hours']) * 4,
                    forecast_features=forecast_matrix)
            else:
                packed = None
        else:
            packed = build_sequences(item.frame, matrix, masks, item.rating_mw)
        np.save(directory / f'x_{name}.npy', matrix, allow_pickle=False)
        if packed is not None:
            np.savez(directory / f'packed_{name}.npz', **packed)
        _write_json(directory / f'preprocessor_{name}.json', fitted[name])


def _trial_path(root, trial):
    """Give every farm, feature set and family an independent resumable record."""
    return (root / f"farm_{trial['farm_id']}" / 'trials' / trial['feature'] /
            trial['family'] / f"{trial['number']:03d}.json")


def _load_cache(root, trial):
    """Open immutable arrays with read-only memory mapping where supported."""
    directory = root / f"farm_{trial['farm_id']}" / 'cache'
    frame = pd.read_parquet(directory / 'frame.parquet')
    masks = dict(np.load(directory / 'masks.npz', allow_pickle=False))
    matrix = np.load(directory / f"x_{trial['feature']}.npy", mmap_mode='r')
    return directory, frame, masks, matrix


def _fit_candidate(root, trial, config):
    """Fit one candidate on training labels and return the best-validation model."""
    section = config['pv_reviewed_hpo']
    directory, frame, masks, matrix = _load_cache(root, trial)
    rating = float(config['farms'][str(trial['farm_id'])]['ac_capacity_kw']) / 1000
    if trial['family'] in section['statistical_families']:
        y = frame.model_target_mw.to_numpy(dtype=float) / rating
        model = fit_statistical(trial['family'], trial['parameters'], trial['seed'],
            matrix, y, masks['train'], frame.loc[masks['train'], 'delivery_day'])
        info = {'training_loss': trial['parameters'].get('loss',
            trial['parameters'].get('objective', trial['parameters'].get('loss_function', 'squared_or_huber')))}
        packed = None
    else:
        packed = dict(np.load(directory / f"packed_{trial['feature']}.npz", allow_pickle=False))
        training_settings = dict(section['neural_training'])
        training_settings['device'] = section.get('runtime', {}).get('device', 'cpu')
        model, info = fit_neural(trial['family'], trial['parameters'], training_settings,
                                 trial['seed'], packed, frame, masks, rating)
        info['training_loss'] = trial['parameters']['loss']
    return model, info, frame, masks, matrix, packed, rating


def _predict_candidate(model, trial, matrix, packed, frame, groups, rating):
    """Predict only requested rows; neural padding never becomes a physical interval."""
    if packed is None:
        prediction = np.full(len(frame), np.nan)
        selected = np.asarray(groups, dtype=bool)
        prediction[selected] = np.clip(model.predict(np.asarray(matrix[selected], dtype=float)), 0, 1) * rating
        prediction[frame.night.to_numpy(dtype=bool) & np.isfinite(prediction)] = 0.0
        return prediction
    selected_groups = np.flatnonzero(np.isin(packed['split'], groups))
    prediction = predict_neural(model, packed, selected_groups, int(trial['parameters']['batch_size']), len(frame), rating)
    prediction[frame.night.to_numpy(dtype=bool) & np.isfinite(prediction)] = 0.0
    return prediction


def _execute_trial(job):
    """Run one seeded candidate and record validation only; do not inspect test scores."""
    import sklearn  # Initialize the conda C++ runtime before optional PyTorch.
    from threadpoolctl import threadpool_limits

    root = Path(job['root'])
    trial = job['trial']
    path = _trial_path(root, trial)
    if path.exists():
        record = json.loads(path.read_text())
        if record['identity'] != job['identity'] or record['trial'] != trial:
            raise ValueError(f"Resume identity mismatch for {trial['name']}")
        if record.get('status') == 'complete':
            return record
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        model, info, frame, masks, matrix, packed, rating = _fit_candidate(root, trial, job['config'])
        groups = masks['validation'] if packed is None else ['validation']
        prediction = _predict_candidate(model, trial, matrix, packed, frame, groups, rating)
        scores = {name: score_predictions(frame, prediction, mask, rating)
                  for name, mask in source_masks(frame, masks['validation']).items()}
    record = {'status': 'complete', 'identity': job['identity'], 'trial': trial,
        'validation': scores, 'training': info, 'elapsed_seconds': time.perf_counter() - started,
        'warnings': [str(warning.message) for warning in caught], 'test_targets_scored': False}
    _write_json(path, record)
    return record


def _clear_cuda_cache():
    """Release unused worker allocations so other GPU fits can reuse memory."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _trial_worker(job):
    """Run a trial and release unused device memory before taking another job."""
    try:
        return _execute_trial(job)
    finally:
        _clear_cuda_cache()


def _bounded_parallel(jobs, worker, workers, progress):
    """Retry CUDA out-of-memory jobs alone and reduce subsequent concurrency."""
    import torch

    pending = {}
    waiting = deque((job, False) for job in jobs)
    limit = workers
    done_count = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or waiting:
            # An OOM retry waits until every other job completes, then runs alone.
            while waiting and len(pending) < limit:
                job, serial = waiting[0]
                if (serial and pending) or any(item[1] for item in pending.values()):
                    break
                waiting.popleft()
                pending[pool.submit(worker, job)] = (job, serial)
                if serial:
                    break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job, serial = pending.pop(future)
                try:
                    record = future.result()
                except torch.cuda.OutOfMemoryError as error:
                    if serial or workers == 1:
                        raise RuntimeError(f"GPU ran out of memory running {job['trial']['name']} alone") from error
                    limit = max(1, limit - 1)
                    waiting.appendleft((job, True))
                    if progress:
                        progress(f"GPU out of memory for {job['trial']['name']}; retrying alone and reducing concurrency to {limit}")
                    continue
                except Exception as error:
                    for outstanding in pending:
                        outstanding.cancel()
                    raise RuntimeError(f"Failed job {job['trial']['name']} for farm {job['trial']['farm_id']}") from error
                yield record
                done_count += 1
                if progress and (done_count % 20 == 0 or done_count == len(jobs)):
                    progress(f'Completed {done_count}/{len(jobs)} jobs')


def _score_splits(frame, prediction, masks, rating):
    """Score each split in its daylight window and separate synthetic targets."""
    result = []
    for split in ['train', 'validation', 'test']:
        base = masks[split].copy()
        if split == 'train':
            base &= frame.evaluation_window.to_numpy(dtype=bool)
        for source, mask in source_masks(frame, base).items():
            score = score_predictions(frame, prediction, mask, rating)
            if score is not None:
                result.append({'split': split, 'target_source': source, **score})
    return result


def _execute_refit(job):
    """Replay one validation-selected configuration and export its portable weights."""
    import sklearn
    from threadpoolctl import threadpool_limits

    root = Path(job['root'])
    trial = job['trial']
    directory = _model_directory(root, trial)
    result_path = directory / 'result.json'
    if result_path.exists():
        record = json.loads(result_path.read_text())
        if record['identity'] != job['identity'] or record['trial'] != trial:
            raise ValueError(f"Refit resume identity mismatch for {trial['name']}")
        for filename, expected in record['checksums'].items():
            if _sha(directory / filename) != expected:
                raise ValueError(f'Refit artifact checksum mismatch: {filename}')
        return record
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        model, info, frame, masks, matrix, packed, rating = _fit_candidate(root, trial, job['config'])
        groups = np.ones(len(frame), dtype=bool) if packed is None else ['train', 'validation', 'test']
        prediction = _predict_candidate(model, trial, matrix, packed, frame, groups, rating)
        validation = score_predictions(frame, prediction, masks['validation'], rating)
    difference = abs(validation['day_balanced_nrmse_percent'] - job['expected_validation_score'])
    tolerance = float(job['config']['pv_reviewed_hpo']['refit_replay_tolerance_nrmse_percent'])
    if difference > tolerance:
        raise ValueError(f"Refit validation replay differs by {difference:g} percentage points")
    directory.mkdir(parents=True, exist_ok=True)
    if packed is None:
        model_path = directory / 'model.joblib'
        temporary = directory / 'model.joblib.tmp'
        joblib.dump(model, temporary)
        temporary.replace(model_path)
    else:
        import torch
        model_path = directory / 'model.pt'
        temporary = directory / 'model.pt.tmp'
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
        temporary.replace(model_path)
    np.save(directory / 'prediction_mw.npy', prediction.astype(np.float32), allow_pickle=False)
    cache = root / f"farm_{trial['farm_id']}" / 'cache'
    preprocessor = json.loads((cache / f"preprocessor_{trial['feature']}.json").read_text())
    _write_json(directory / 'preprocessor.json', preprocessor)
    checksums = {name: _sha(directory / name) for name in [model_path.name, 'prediction_mw.npy', 'preprocessor.json']}
    four_cycle = job['config']['pv_reviewed_hpo'].get('data_regime') == 'four_cycle_forecast_ghi'
    record = {'status': 'complete', 'identity': job['identity'], 'trial': trial,
        'schema': 'reviewed_ghi_forecast_bundle_v1', 'timezone': job['config']['pv_reviewed_hpo']['timezone'],
        'rating_mw': rating, 'training': info, 'validation_replay_difference_percent': difference,
        'metrics': _score_splits(frame, prediction, masks, rating),
        'elapsed_seconds': time.perf_counter() - started,
        'warnings': [str(warning.message) for warning in caught], 'checksums': checksums,
        'input_contract': {'ghi_training': 'historical measured/interpolated',
            'ghi_validation_test': 'forecast, after assumed release' if four_cycle else 'D-1 00 UTC forecast',
            'hourly_radiation': 'interval mean repeated across four quarters',
            'context': '48-hour weather issue sequence' if four_cycle else 'single frozen forecast cycle per local delivery day',
            **({'issue_horizon_hours': job['config']['pv_reviewed_hpo']['issue_horizon_hours']} if four_cycle else {}),
            'target': 'model_target_mw', 'pvlib_model_input': False,
            'night_training_samples_per_farm': job['config']['pv_reviewed_hpo']['night_training_samples_per_farm'],
            'evaluation': 'outside month-specific night window'},
        'availability': ('four issue cycles with assumed configured release lag' if four_cycle
                         else 'D-1 00 UTC cycle with assumed configured release lag'),
        'limitations': 'Selected on previously inspected 2026 validation; some labels are synthetic.'}
    _write_json(result_path, record)
    return record


def _refit_worker(job):
    """Refit and release unused device memory before the next selected model."""
    try:
        return _execute_refit(job)
    finally:
        _clear_cuda_cache()


def _run_baselines(root, prepared, masks, rating):
    """Score zero and a training-only day-balanced constant after model selection."""
    from .pv_reviewed_hpo_common import day_balanced_weights

    train = masks['train']
    weights = day_balanced_weights(prepared.loc[train, 'delivery_day'])
    mean = float(np.dot(weights, prepared.loc[train, 'model_target_mw']))
    rows = []
    for name, value in [('zero', 0.0), ('training_day_balanced_mean', mean)]:
        prediction = np.full(len(prepared), np.clip(value, 0, rating))
        prediction[prepared.night.to_numpy(dtype=bool)] = 0.0
        rows.extend([{'family': name, **record} for record in _score_splits(prepared, prediction, masks, rating)])
    return rows


def _trial_rows(records):
    """Flatten every HPO configuration without adding test metrics."""
    rows = []
    for record in records:
        trial = record['trial']
        row = {key: trial[key] for key in ['farm_id', 'feature', 'family', 'number', 'name', 'seed']}
        row['parameters_json'] = json.dumps(trial['parameters'], sort_keys=True)
        row['validation_nrmse_percent'] = record['validation']['all']['day_balanced_nrmse_percent']
        for source in ['measured', 'synthetic']:
            score = record['validation'][source]
            row[f'validation_{source}_nrmse_percent'] = None if score is None else score['day_balanced_nrmse_percent']
        row['elapsed_seconds'] = record['elapsed_seconds']
        row['warning_count'] = len(record['warnings'])
        rows.append(row)
    return pd.DataFrame(rows)


def _family_winners(records):
    """Select each family/feature winner and each farm winner using validation only."""
    ordered = sorted(records, key=lambda item: (
        item['validation']['all']['day_balanced_nrmse_percent'],
        item['trial']['family'], item['trial']['feature'], item['trial']['number']))
    families = {}
    farms = {}
    for record in ordered:
        trial = record['trial']
        key = (trial['farm_id'], trial['feature'], trial['family'])
        families.setdefault(key, record)
        farms.setdefault(trial['farm_id'], record)
    return families, farms


def _top_three(records):
    """Rank distinct trial configurations per farm using validation error only."""
    ordered = sorted(records, key=lambda item: (
        item['validation']['all']['day_balanced_nrmse_percent'],
        item['trial']['family'], item['trial']['feature'], item['trial']['number']))
    best = {}
    for record in ordered:
        farm = record['trial']['farm_id']
        if len(best.setdefault(farm, [])) < 3:
            best[farm].append(record)
    return best


def _daily_errors(frame, prediction, mask, rating):
    """Aggregate squared normalized errors per day for block-bootstrap comparisons."""
    selected = np.asarray(mask, dtype=bool)
    residual = (np.asarray(prediction)[selected] - frame.model_target_mw.to_numpy()[selected]) / rating
    values = pd.DataFrame({'delivery_day': frame.delivery_day.to_numpy()[selected],
                           'squared_error': residual ** 2})
    return values.groupby('delivery_day', sort=True).squared_error.mean()


def _paired_bootstrap(reference, candidate, section, seed):
    """Compare day-balanced test RMSE using contiguous local-day block resampling."""
    common = pd.concat([reference.rename('reference'), candidate.rename('candidate')], axis=1).dropna()
    if common.empty:
        return None
    dates = pd.DatetimeIndex(pd.to_datetime(common.index))
    block = min(int(section['block_bootstrap_days']), len(dates))
    starts = [number for number in range(len(dates) - block + 1)
              if (dates[number + block - 1] - dates[number]).days == block - 1]
    if not starts:
        return None
    rng = np.random.default_rng(seed)
    values = common.to_numpy(dtype=float)
    differences = np.empty(int(section['bootstrap_repetitions']))
    for number in range(len(differences)):
        chosen = []
        while len(chosen) < len(values):
            first = starts[int(rng.integers(len(starts)))]
            chosen.extend(range(first, first + block))
        sample = values[np.asarray(chosen[:len(values)])]
        differences[number] = 100 * (np.sqrt(sample[:, 1].mean()) - np.sqrt(sample[:, 0].mean()))
    point = 100 * (np.sqrt(values[:, 1].mean()) - np.sqrt(values[:, 0].mean()))
    return {'common_days': len(common), 'block_days': block, 'repetitions': len(differences),
        'candidate_minus_selected_nrmse_pp': float(point),
        'lower_95_pp': float(np.quantile(differences, 0.025)),
        'upper_95_pp': float(np.quantile(differences, 0.975))}


def _model_directory(root, trial):
    """Keep distinct selected trials, including multiple from the same family."""
    return (root / f"farm_{trial['farm_id']}" / 'models' / trial['feature']
            / trial['family'] / trial['name'])


def _plot_day(frame, prediction, selected, trial, output, timezone):
    """Show the selected model, reviewed target and actual irradiance input for one day."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    part = frame.loc[selected]
    if part.empty:
        return
    times = pd.DatetimeIndex(part.interval_start_utc).tz_convert(timezone)
    fig, (power, ghi) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    prediction_day = np.asarray(prediction)[selected]
    # Expanded cycle views can share physical timestamps; draw each issue separately.
    if times.has_duplicates and 'forecast_cycle_utc' in part:
        observed = part.assign(_time=times).drop_duplicates('_time').sort_values('_time')
        power.plot(observed['_time'], observed.model_target_mw, color='black', label='Reviewed target')
        cycles = pd.DatetimeIndex(part.forecast_cycle_utc)
        for cycle in cycles.unique():
            mask = cycles == cycle
            power.plot(times[mask], prediction_day[mask], label=f'Issue {cycle:%m-%d %H UTC}', alpha=.75)
            ghi.plot(times[mask], part.ghi_input_w_m2.to_numpy()[mask], alpha=.65)
    else:
        power.plot(times, part.model_target_mw, color='black', label='Reviewed target')
        power.plot(times, prediction_day, color='tab:blue', label='Validation-selected model')
        ghi.plot(times, part.ghi_input_w_m2, color='tab:green', label='GHI input used by model')
    synthetic = part.synthetic_target.to_numpy(dtype=bool)
    power.scatter(times[synthetic], part.loc[synthetic, 'model_target_mw'],
                  s=14, color='tab:orange', label='Synthetic/zero target')
    if times.has_duplicates:
        ghi.set_title('Forecast GHI by issue cycle')
    split = part['split'].iloc[0]
    day = part.delivery_day.iloc[0]
    power.set(title=f"Farm {trial['farm_id']}: {trial['family']} / {trial['feature']} · {day} ({split})",
              ylabel='AC power (MW)')
    ghi.set(ylabel='GHI (W/m²)', xlabel=f'Local delivery time ({timezone})')
    for axis in (power, ghi):
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def publish_results(config, root, report, records, winners, selected, refits, identity):
    """Export full rankings, paired metrics, monthly tables and reproducible plots."""
    section = config['pv_reviewed_hpo']
    report_identity = report / 'identity.json'
    if report.exists():
        if not report_identity.exists() or json.loads(report_identity.read_text()).get('identity') != identity:
            raise ValueError('Existing report belongs to another search; choose a fresh report path')
    report.mkdir(parents=True, exist_ok=True)
    _write_json(report_identity, {'identity': identity})
    trial_rows = _trial_rows(records)
    trial_rows.to_csv(report / 'all_trials_validation.csv', index=False)
    winner_rows = []
    selected_rows = []
    metric_rows = []
    monthly_rows = []
    interpolation_rows = []
    baseline_rows = []
    bootstrap_rows = []
    cycle_lead_rows = []
    top_rows = []
    top_metric_rows = []
    top_three = _top_three(records)
    refits_by_trial = {(entry['trial']['farm_id'], entry['trial']['feature'],
                        entry['trial']['family'], entry['trial']['number']): entry for entry in refits}
    for record in refits:
        trial = record['trial']
        key = (trial['farm_id'], trial['feature'], trial['family'])
        if winners[key]['trial']['number'] != trial['number']:
            continue
        summary = {'farm_id': trial['farm_id'], 'feature': trial['feature'], 'family': trial['family'],
                   'name': trial['name'], 'validation_nrmse_percent': record['metrics'][
                       next(number for number, item in enumerate(record['metrics'])
                            if item['split'] == 'validation' and item['target_source'] == 'all')][
                                'day_balanced_nrmse_percent']}
        winner_rows.append(summary)
        for metric in record['metrics']:
            metric_rows.append({**summary, **metric})
    for farm_id, record in selected.items():
        trial = record['trial']
        selected_rows.append({'farm_id': farm_id, 'feature': trial['feature'], 'family': trial['family'],
            'name': trial['name'],
            'validation_nrmse_percent': record['validation']['all']['day_balanced_nrmse_percent'],
            'model_directory': str(_model_directory(root, trial))})
    for farm_id, ranked in top_three.items():
        for rank, record in enumerate(ranked, start=1):
            trial = record['trial']
            key = (farm_id, trial['feature'], trial['family'], trial['number'])
            summary = {'farm_id': farm_id, 'rank': rank, 'feature': trial['feature'],
                'family': trial['family'], 'name': trial['name'],
                'validation_nrmse_percent': record['validation']['all']['day_balanced_nrmse_percent'],
                'model_directory': str(_model_directory(root, trial))}
            top_rows.append(summary)
            top_metric_rows.extend([{**summary, **metric} for metric in refits_by_trial[key]['metrics']])
    for farm_id in section['farm_ids']:
        cache = root / f'farm_{farm_id}' / 'cache'
        frame = pd.read_parquet(cache / 'frame.parquet')
        masks = dict(np.load(cache / 'masks.npz', allow_pickle=False))
        rating = float(config['farms'][str(farm_id)]['ac_capacity_kw']) / 1000
        baseline_rows.extend([{'farm_id': farm_id, **item} for item in _run_baselines(root, frame, masks, rating)])
        best_trial = selected[farm_id]['trial']
        best_prediction = np.load(_model_directory(root, best_trial) / 'prediction_mw.npy')
        reference_daily = _daily_errors(frame, best_prediction, masks['test'], rating)
        for record in [entry for entry in refits if entry['trial']['farm_id'] == farm_id]:
            trial = record['trial']
            prediction = np.load(_model_directory(root, trial) / 'prediction_mw.npy')
            for split in ['train', 'validation', 'test']:
                base = masks[split].copy()
                if split == 'train':
                    base &= frame.evaluation_window.to_numpy(dtype=bool)
                # Validation/test use forecast GHI; the historical fill flag then describes target provenance only.
                is_four_cycle = section.get('data_regime') == 'four_cycle_forecast_ghi'
                provenance_mask = (base if split == 'train' and not is_four_cycle
                                   else base & frame.synthetic_target.to_numpy(dtype=bool))
                for filled in [False, True]:
                    mask = provenance_mask & (frame.ghi_interpolated.to_numpy(dtype=bool) == filled)
                    score = score_predictions(frame, prediction, mask, rating)
                    if score is not None:
                        interpolation_rows.append({'farm_id': farm_id, 'feature': trial['feature'],
                            'family': trial['family'], 'name': trial['name'], 'split': split,
                            'historical_ghi_interpolated': filled,
                            'interpretation': ('training_input' if split == 'train' and not is_four_cycle
                                               else 'synthetic_target_provenance'),
                            **score})
                if is_four_cycle:
                    issue_hour = pd.DatetimeIndex(frame.forecast_cycle_utc).hour.to_numpy()
                    lead = frame.lead_hours.to_numpy(dtype=float)
                    for hour in section['cycle_hours_utc']:
                        for lower in range(0, int(section['issue_horizon_hours']), 6):
                            group = base & (issue_hour == hour) & (lead >= lower) & (lead < lower + 6)
                            for source, source_mask in source_masks(frame, group).items():
                                score = score_predictions(frame, prediction, source_mask, rating)
                                if score is not None:
                                    cycle_lead_rows.append({'farm_id': farm_id, 'feature': trial['feature'],
                                        'family': trial['family'], 'name': trial['name'], 'split': split,
                                        'cycle_hour_utc': hour, 'lead_start_hours': lower,
                                        'lead_end_hours': lower + 6, 'target_source': source, **score})
                for month in sorted(frame.loc[base, 'delivery_day'].str[:7].unique()):
                    month_mask = base & frame.delivery_day.str.startswith(month).to_numpy()
                    for source, mask in source_masks(frame, month_mask).items():
                        score = score_predictions(frame, prediction, mask, rating)
                        if score is not None:
                            monthly_rows.append({'farm_id': farm_id, 'feature': trial['feature'],
                                'family': trial['family'], 'name': trial['name'], 'month': month, 'split': split,
                                'target_source': source, **score})
            daily = _daily_errors(frame, prediction, masks['test'], rating)
            interval = _paired_bootstrap(reference_daily, daily, section['validation'], section['seed'] + farm_id)
            if interval is not None:
                bootstrap_rows.append({'farm_id': farm_id, 'feature': trial['feature'],
                    'family': trial['family'], 'name': trial['name'], 'selected_feature': best_trial['feature'],
                    'selected_family': best_trial['family'], **interval})
        # Plot best, median and worst daily validation/test errors after the winner is frozen.
        for split in ['train', 'validation', 'test']:
            mask = masks[split].copy()
            if split == 'train':
                mask &= frame.evaluation_window.to_numpy(dtype=bool)
            daily = _daily_errors(frame, best_prediction, mask, rating).sort_values()
            if daily.empty:
                continue
            count = int(section['validation']['plot_days_per_split'])
            indexes = np.linspace(0, len(daily) - 1, min(count, len(daily))).round().astype(int)
            for day in daily.index[indexes]:
                selected_day = frame.delivery_day.eq(day).to_numpy() & frame['split'].eq(split).to_numpy()
                output = root / 'plots' / f'farm_{farm_id}' / split / f'{day}.png'
                _plot_day(frame, best_prediction, selected_day, best_trial, output, section['timezone'])
    pd.DataFrame(winner_rows).to_csv(report / 'family_feature_winners.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(report / 'selected_by_farm.csv', index=False)
    pd.DataFrame(top_rows).to_csv(report / 'top_three_by_farm.csv', index=False)
    pd.DataFrame(top_metric_rows).to_csv(report / 'top_three_metrics.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(report / 'winner_metrics.csv', index=False)
    pd.DataFrame(monthly_rows).to_csv(report / 'monthly_metrics.csv', index=False)
    pd.DataFrame(interpolation_rows).to_csv(report / 'ghi_interpolation_metrics.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(report / 'baseline_metrics.csv', index=False)
    pd.DataFrame(bootstrap_rows).to_csv(report / 'paired_test_bootstrap.csv', index=False)
    if section.get('data_regime') == 'four_cycle_forecast_ghi':
        pd.DataFrame(cycle_lead_rows).to_csv(report / 'cycle_lead_metrics.csv', index=False)
    four_cycle = section.get('data_regime') == 'four_cycle_forecast_ghi'
    weather_description = (
        'Training uses historical GHI and weather from all four forecast cycles over the 48 hours from issue; the first six hours are hindsight. '
        'Validation and testing score targets only after the assumed weather release.' if four_cycle else
        f"Training uses historical GHI and {section['night_training_samples_per_farm']} eligible night targets per farm, evenly spread over the year. Validation and testing use the D-1 00 UTC forecast GHI.")
    lines = ['# Reviewed PV HPO', '',
        f'Search identity: `{identity}`. HPO trials and family winners were selected on validation only.',
        'The 2026 report period was previously inspected; these are retrospective comparisons.',
        'The 2026 scoring labels include user-requested pvlib reconstructions, not independent observations.', '',
        weather_description,
        'The month-specific daylight scoring window is used. Hourly radiation means are repeated over',
        'their four quarter-hours. The six-hour release lag is an assumption, not proof of historical availability.', '',
        'No physical PV model or physical-model prediction is a candidate or input feature.',
        'Zero and a training-only day-balanced constant are provided as statistical baselines.',
        'Test bootstrap intervals compare each family winner with the validation-selected model and do not correct',
        'for previous use of the report period or uncertainty in synthetic labels.', '',
        '[Every validation trial](all_trials_validation.csv) · [Family/feature winners](family_feature_winners.csv) ·',
        '[Selected model by farm](selected_by_farm.csv) · [Top three by farm](top_three_by_farm.csv) ·',
        '[Top-three metrics](top_three_metrics.csv) · [Winner metrics](winner_metrics.csv) ·',
        '[Monthly metrics](monthly_metrics.csv) · [Baselines](baseline_metrics.csv) ·',
        '[Historical-GHI interpolation diagnostics](ghi_interpolation_metrics.csv) ·',
        '[Paired test bootstrap](paired_test_bootstrap.csv)', '',
        ('The historical-GHI interpolation flag describes synthetic-target provenance.' if four_cycle else
         'The historical-GHI interpolation flag describes a training input; on validation/test it describes synthetic-target provenance.'), '',
        '| Farm | Feature set | Family | Selected trial | Validation nRMSE (% AC) |',
        '|---|---|---|---|---:|']
    for row in selected_rows:
        lines.append(f"| {row['farm_id']} | {row['feature']} | {row['family']} | {row['name']} | {row['validation_nrmse_percent']:.3f} |")
    lines += ['', '## Top three configurations per farm', '',
              'Rank is based only on validation daylight nRMSE. Scores use all reviewed targets;',
              'the linked CSV separates measured and synthetic targets for every split.', '',
              '| Farm | Rank | Trial | Validation nRMSE (%) | Test nRMSE (%) | Test nMAE (%) | Test bias (%) |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in top_rows:
        ranked = top_three[row['farm_id']][row['rank'] - 1]
        trial = ranked['trial']
        key = (row['farm_id'], trial['feature'], trial['family'], trial['number'])
        test = next(item for item in refits_by_trial[key]['metrics']
                    if item['split'] == 'test' and item['target_source'] == 'all')
        lines.append(f"| {row['farm_id']} | {row['rank']} | {row['name']} | "
                     f"{row['validation_nrmse_percent']:.3f} | {test['day_balanced_nrmse_percent']:.3f} | "
                     f"{test['day_balanced_nmae_percent']:.3f} | {test['day_balanced_bias_percent']:.3f} |")
    lines += ['', 'Selected-model daily plots are grouped by farm, split and local delivery date under',
              f'`{root}/plots/`. Family models, preprocessing and hashes are under each farm’s `models/` directory.']
    if four_cycle:
        lines += ['', '[Metrics by issue cycle and lead time](cycle_lead_metrics.csv)']
    (report / 'README.md').write_text('\n'.join(lines) + '\n')


def run_hpo(config, progress=print):
    """Run the configured search after explicit invocation; never triggered by preflight."""
    section = validate_configuration(config)
    _runtime_details(section, require_free_memory=True)
    minimum = section.get('minimum_available_memory_gib')
    available = _available_memory_gib()
    if minimum is not None and (available is None or available < float(minimum)):
        raise RuntimeError(f"The configured {section['workers']}-worker search requires {minimum} GiB available memory; "
                           f"currently available: {available} GiB")
    root = config_path(config, section['output_root']) / section['run_id']
    report = config_path(config, section['report_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'run.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f'An HPO run may already be active; inspect {lock}') from error
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}) + '\n')
    try:
        with (root / 'run.log').open('a', buffering=1) as log_file:
            def log(message):
                """Keep a durable progress log as well as tmux-visible output."""
                text = f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}"
                print(text, file=log_file, flush=True)
                if progress:
                    progress(text)
            plan, prepared, trials, contract = check_only(config, log)
            identity_path = root / 'identity.json'
            if identity_path.exists():
                previous = json.loads(identity_path.read_text())
                if not section['resume'] or previous['identity'] != plan['identity']:
                    raise ValueError('Existing run has a different identity or resume is disabled')
            else:
                existing = [path.name for path in root.iterdir() if path.name not in {'run.lock', 'run.log'}]
                if existing:
                    raise ValueError('Existing output has no matching identity; choose a new run ID')
                _write_json(identity_path, {'identity': plan['identity'], 'contract': contract})
            _write_json(root / 'preflight.json', plan)
            # Recreate deterministic caches under the run lock before scheduling any candidate.
            for farm_id, item in prepared.items():
                log(f'Preparing checked feature cache for farm {farm_id}')
                _prepare_cache(root, item, config)
            del prepared
            # All numerical libraries and each model family are confined to one thread per worker.
            for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                os.environ[name] = '1'
            jobs = [{'root': str(root), 'trial': trial, 'identity': plan['identity'], 'config': config}
                    for trial in trials]
            log(f"Launching {len(jobs)} HPO fits across {section['workers']} workers; test remains unscored")
            _write_json(root / 'status.json', {'status': 'searching', 'identity': plan['identity'],
                'hpo_fits': len(jobs), 'completed': 0, 'training_started': True})
            records = []
            for record in _bounded_parallel(jobs, _trial_worker, int(section['workers']), log):
                records.append(record)
                if len(records) % 20 == 0 or len(records) == len(jobs):
                    _write_json(root / 'status.json', {'status': 'searching', 'identity': plan['identity'],
                        'hpo_fits': len(jobs), 'completed': len(records), 'training_started': True})
            winners, selected = _family_winners(records)
            top_three = _top_three(records)
            expected = plan['family_winner_refits']
            if len(winners) != expected or len(selected) != len(section['farm_ids']):
                raise ValueError('The completed search does not cover all family/feature/farm groups')
            # Freeze the validation-selected identities before any test scoring occurs.
            frozen = {'identity': plan['identity'], 'selection': 'validation_only',
                'family_winners': [record['trial'] for record in winners.values()],
                'top_three_by_farm': {str(farm): [record['trial'] for record in ranked]
                                      for farm, ranked in top_three.items()},
                'selected_by_farm': {str(farm): record['trial'] for farm, record in selected.items()}}
            _write_json(root / 'frozen_selection.json', frozen)
            # The top three can include multiple trials from a single family/feature pair.
            refit_records = list(winners.values())
            known = {(record['trial']['farm_id'], record['trial']['feature'],
                      record['trial']['family'], record['trial']['number']) for record in refit_records}
            for ranked in top_three.values():
                for record in ranked:
                    trial = record['trial']
                    key = (trial['farm_id'], trial['feature'], trial['family'], trial['number'])
                    if key not in known:
                        refit_records.append(record)
                        known.add(key)
            refit_jobs = [{'root': str(root), 'trial': record['trial'], 'identity': plan['identity'],
                'config': config, 'expected_validation_score': record['validation']['all']['day_balanced_nrmse_percent']}
                for record in refit_records]
            log(f'Refitting {len(refit_jobs)} validation-selected family winners and top-three trials')
            _write_json(root / 'status.json', {'status': 'refitting', 'identity': plan['identity'],
                'hpo_fits': len(jobs), 'completed': len(jobs), 'refits_completed': 0, 'training_started': True})
            refits = []
            for record in _bounded_parallel(refit_jobs, _refit_worker, int(section['workers']), log):
                refits.append(record)
                if len(refits) % 20 == 0 or len(refits) == len(refit_jobs):
                    _write_json(root / 'status.json', {'status': 'refitting', 'identity': plan['identity'],
                        'hpo_fits': len(jobs), 'completed': len(jobs), 'refits_completed': len(refits),
                        'training_started': True})
            log('Publishing validation-selected scores, target-source diagnostics, test comparisons and plots')
            publish_results(config, root, report, records, winners, selected, refits, plan['identity'])
            _write_json(root / 'status.json', {'status': 'complete', 'identity': plan['identity'],
                'hpo_fits': len(jobs), 'completed': len(jobs), 'refits_completed': len(refits),
                'report': str(report), 'training_started': True})
            log(f'Complete: {report}')
            return {'output': str(root), 'report': str(report), 'identity': plan['identity']}
    finally:
        lock.unlink(missing_ok=True)
