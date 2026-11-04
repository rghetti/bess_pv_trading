"""Check multi-GPU scheduling, trial provenance and resume safety."""
from concurrent.futures import Future
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_four_cycle_hpo_data import build_cycle_sequences
from bess_pv_trading.forecasting.training import pv_reviewed_multi_gpu_hpo as multi


class ImmediatePool:
    """Supply completed futures so scheduler decisions are deterministic."""

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


def test_configuration_and_device_mapping(monkeypatch):
    """The study reserves two slots for each of the three physical GPUs."""
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml')
    section = config['pv_reviewed_hpo']
    assert section['workers'] == 6
    assert section['neural_space']['batch_size']['values'] == [16, 32, 48]
    assert [(x['visible_id'], x['workers']) for x in multi._device_specs(section)] == [('0', 2), ('1', 2), ('2', 2)]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,1,0')
    with pytest.raises(ValueError, match='Set CUDA_VISIBLE_DEVICES=0,1,2'):
        multi._hardware(section)


def test_scheduler_pins_retries_and_refits_to_original_gpu(monkeypatch):
    """A memory failure is retried alone on the same device; jobs are not duplicated."""
    monkeypatch.setattr(multi, 'ProcessPoolExecutor', ImmediatePool)
    devices = [{'visible_id': str(number), 'uuid': f'uuid-{number}', 'name': f'gpu-{number}', 'workers': 2}
               for number in range(3)]
    attempts = {}
    log = []
    jobs = [{'trial': {'name': str(number), 'farm_id': 1}} for number in range(6)]

    def worker(job):
        name = job['trial']['name']
        gpu = job['gpu']['visible_id']
        attempts.setdefault(name, []).append(gpu)
        if name == '0' and len(attempts[name]) == 1:
            raise torch.cuda.OutOfMemoryError('synthetic failure')
        return {'trial': job['trial'], 'gpu': job['gpu']}

    records = list(multi._schedule(jobs, worker, devices, log.append))
    assert sorted(item['trial']['name'] for item in records) == [str(number) for number in range(6)]
    assert len(attempts['0']) == 2 and attempts['0'][0] == attempts['0'][1]
    assert any('retrying' in message for message in log)
    pinned_jobs = [{'trial': item['trial'], 'gpu': item['gpu']} for item in records]
    refits = list(multi._schedule(pinned_jobs, lambda job: job, devices, None))
    assert {item['trial']['name']: item['gpu'] for item in records} == {
        item['trial']['name']: item['gpu'] for item in refits}


def test_resume_rejects_changed_device_uuid(tmp_path):
    """Completed scores cannot silently migrate to a different physical device."""
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': 0, 'name': 'trial'}
    gpu = {'visible_id': '0', 'uuid': 'old', 'name': 'device'}
    path = multi.base._trial_path(tmp_path, trial)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'identity': 'run', 'trial': trial, 'gpu': gpu, 'status': 'complete'}))
    with pytest.raises(ValueError, match='GPU provenance'):
        multi._existing_records(tmp_path, [trial], 'run', [{**gpu, 'uuid': 'new'}])


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason='Three CUDA devices are unavailable')
def test_real_device_workers_and_same_gpu_refits(tmp_path, monkeypatch):
    """Fit and refit one tiny synthetic trial on each actual selected device."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml')
    config['pv_reviewed_hpo']['neural_training']['max_epochs'] = 2
    config['pv_reviewed_hpo']['neural_training']['patience'] = 2
    devices = multi._hardware(config['pv_reviewed_hpo'])['devices']
    for item in devices:
        item['workers'] = 1
    splits = np.repeat(['train', 'validation', 'test'], 8)
    cycles = pd.to_datetime(np.repeat(['2025-01-01T00:00Z', '2026-01-01T00:00Z', '2026-05-01T00:00Z'], 8))
    frame = pd.DataFrame({'forecast_cycle_utc': cycles, 'split': splits,
        'delivery_day': np.repeat(['2025-01-01', '2026-01-01', '2026-05-01'], 8),
        'model_target_mw': np.linspace(.1, .8, 24), 'synthetic_target': False,
        'night': False, 'evaluation_window': True})
    masks = {name: splits == name for name in ['train', 'validation', 'test']}
    x = np.ones((24, 2), dtype=np.float32)
    packed = build_cycle_sequences(frame, x, masks, 1., 8)
    cache = tmp_path / 'farm_1' / 'cache'
    cache.mkdir(parents=True)
    frame.to_parquet(cache / 'frame.parquet', index=False)
    np.savez(cache / 'masks.npz', **masks)
    np.save(cache / 'x_core.npy', x)
    np.savez(cache / 'packed_core.npz', **packed)
    (cache / 'preprocessor_core.json').write_text(json.dumps({'columns': ['x', 'y']}))
    parameters = {'width': 8, 'depth': 1, 'dropout': .1, 'learning_rate': .001,
                  'weight_decay': 0., 'batch_size': 1, 'loss': 'mse'}
    jobs = []
    for number in range(3):
        trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': number,
                 'seed': 123 + number, 'parameters': parameters, 'name': f'mlp_core_{number:03d}'}
        jobs.append({'root': str(tmp_path), 'trial': trial, 'identity': 'synthetic',
                     'config': multi._single_device_proxy(config)})
    results = list(multi._schedule(jobs, multi._trial_worker, devices, None))
    assert {item['gpu']['visible_id'] for item in results} == {'0', '1', '2'}
    refit_jobs = [{**job, 'trial': result['trial'], 'gpu': result['gpu'],
                   'expected_validation_score': result['validation']['all']['day_balanced_nrmse_percent']}
                  for job, result in zip(jobs, results)]
    refits = list(multi._schedule(refit_jobs, multi._refit_worker, devices, None))
    assert {item['trial']['name']: item['gpu'] for item in results} == {
        item['trial']['name']: item['gpu'] for item in refits}
    assert all(item['validation_replay_difference_percent'] == 0 for item in refits)
