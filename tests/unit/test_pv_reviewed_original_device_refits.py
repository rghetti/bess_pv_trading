"""Protect frozen selections, saved refits and original-GPU scheduling."""
from concurrent.futures import Future
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_four_cycle_hpo_data import build_cycle_sequences
from bess_pv_trading.forecasting.training import pv_reviewed_original_device_refits as finish


class ImmediatePool:
    """Return completed futures without starting GPU processes."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, worker, job):
        future = Future()
        try:
            future.set_result(worker(job))
        except Exception as error:
            future.set_exception(error)
        return future


def test_reuse_refit_requires_matching_device_and_checksums(tmp_path):
    """A prior cross-device refit is reused only with verified provenance."""
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': 0, 'name': 'trial'}
    source = {'visible_id': '2', 'uuid': 'original', 'name': 'gpu2'}
    selected = {'trial': trial, 'gpu': source}
    directory = finish.base._model_directory(tmp_path, trial)
    directory.mkdir(parents=True)
    (directory / 'model.pt').write_bytes(b'synthetic model')
    record = {'status': 'complete', 'identity': 'run', 'trial': trial,
              'gpu': {'visible_id': '0'}, 'original_training_gpu': source,
              'validation_replay_difference_percent': 0.,
              'checksums': {'model.pt': finish.base._sha(directory / 'model.pt')}}
    (directory / 'result.json').write_text(json.dumps(record))
    assert finish._saved_refit(tmp_path, selected, 'run', .0001) == record
    (directory / 'model.pt').write_bytes(b'altered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        finish._saved_refit(tmp_path, selected, 'run', .0001)


def test_gpu_oom_retries_on_same_physical_device(monkeypatch):
    """Memory pressure does not move a frozen model to another architecture."""
    monkeypatch.setattr(finish, 'ProcessPoolExecutor', ImmediatePool)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda index: (10 * 1024 ** 3, 12 * 1024 ** 3))
    attempts = []
    messages = []
    devices = [{'visible_id': str(i), 'uuid': f'uuid-{i}', 'workers': 2,
                'minimum_free_gpu_memory_gib': 4., 'total_memory_bytes': 12 * 1024 ** 3}
               for i in range(3)]
    job = {'trial': {'name': 'selected', 'farm_id': 1},
           'gpu': {'visible_id': '1', 'uuid': 'uuid-1'}}

    def worker(item):
        attempts.append(item['gpu']['visible_id'])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError('synthetic memory pressure')
        return {'gpu': item['gpu']}

    monkeypatch.setattr(finish.multi, '_refit_worker', worker)
    records = list(finish._schedule_refits([job], devices, messages.append, poll_seconds=.01))
    assert attempts == ['1', '1']
    assert records[0]['gpu']['visible_id'] == '1'
    assert any('after 5.00 GiB is free' in message for message in messages)


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason='Three CUDA devices are unavailable')
def test_selected_refits_run_on_original_gpu1_and_gpu2(tmp_path, monkeypatch):
    """Real synthetic trials are reproduced by the original-device refit scheduler."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        monkeypatch.setenv(name, '1')
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml')
    config['pv_reviewed_hpo']['neural_training']['max_epochs'] = 2
    config['pv_reviewed_hpo']['neural_training']['patience'] = 2
    devices = finish.multi._hardware(config['pv_reviewed_hpo'])['devices']
    for device in devices:
        device['workers'] = 1
        device['minimum_free_gpu_memory_gib'] = .5
    splits = np.repeat(['train', 'validation', 'test'], 8)
    cycles = pd.to_datetime(np.repeat(['2025-01-01T00:00Z', '2026-01-01T00:00Z', '2026-05-01T00:00Z'], 8))
    frame = pd.DataFrame({'forecast_cycle_utc': cycles, 'split': splits,
        'delivery_day': np.repeat(['2025-01-01', '2026-01-01', '2026-05-01'], 8),
        'model_target_mw': np.linspace(.1, .8, 24), 'synthetic_target': False,
        'night': False, 'evaluation_window': True})
    masks = {name: splits == name for name in ['train', 'validation', 'test']}
    inputs = np.ones((24, 2), dtype=np.float32)
    packed = build_cycle_sequences(frame, inputs, masks, 1., 8)
    cache = tmp_path / 'farm_1' / 'cache'
    cache.mkdir(parents=True)
    frame.to_parquet(cache / 'frame.parquet', index=False)
    np.savez(cache / 'masks.npz', **masks)
    np.save(cache / 'x_core.npy', inputs)
    np.savez(cache / 'packed_core.npz', **packed)
    (cache / 'preprocessor_core.json').write_text(json.dumps({'columns': ['x', 'y']}))
    parameters = {'width': 8, 'depth': 1, 'dropout': .1, 'learning_rate': .001,
                  'weight_decay': 0., 'batch_size': 1, 'loss': 'mse'}
    jobs = []
    for number, device in enumerate(devices[1:]):
        trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': number,
                 'seed': 123 + number, 'parameters': parameters, 'name': f'mlp_core_{number:03d}'}
        gpu = {key: device[key] for key in ('visible_id', 'uuid', 'name')}
        jobs.append({'root': str(tmp_path), 'trial': trial, 'identity': 'synthetic', 'gpu': gpu,
                     'config': finish.multi._single_device_proxy(config)})
    results = list(finish.multi._schedule(jobs, finish.multi._trial_worker, devices, None))
    refit_jobs = [{**job, 'trial': result['trial'], 'gpu': result['gpu'],
                   'expected_validation_score': result['validation']['all']['day_balanced_nrmse_percent']}
                  for job, result in zip(jobs, results)]
    refits = list(finish._schedule_refits(refit_jobs, devices, None, poll_seconds=.1))
    assert {item['trial']['name']: item['gpu'] for item in refits} == {
        item['trial']['name']: item['gpu'] for item in results}
    assert all(item['validation_replay_difference_percent'] == 0 for item in refits)
