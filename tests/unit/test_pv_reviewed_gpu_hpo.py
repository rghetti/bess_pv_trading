"""Exercise the isolated GPU search and its device-aware model path."""
import numpy as np
import pandas as pd
import pytest
import torch
import json
from concurrent.futures import Future

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.models.pv_reviewed_gpu_hpo_models import fit_neural, predict_neural
from bess_pv_trading.forecasting.training.pv_reviewed_gpu_hpo import (
    _bounded_parallel, _refit_worker, _runtime_details, _trial_worker, validate_configuration,
)


class ImmediatePool:
    """Provide already-completed futures for deterministic scheduler tests."""

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


def test_parallel_gpu_scheduler_retries_out_of_memory_alone(monkeypatch):
    """An OOM reduces concurrency and retries the failed identity once."""
    monkeypatch.setattr('bess_pv_trading.forecasting.training.pv_reviewed_gpu_hpo.ProcessPoolExecutor', ImmediatePool)
    attempts = {}
    messages = []
    jobs = [{'trial': {'name': str(number), 'farm_id': number}} for number in (1, 2, 3)]

    def worker(job):
        name = job['trial']['name']
        attempts[name] = attempts.get(name, 0) + 1
        if name == '1' and attempts[name] == 1:
            raise torch.cuda.OutOfMemoryError('synthetic GPU exhaustion')
        return name

    records = list(_bounded_parallel(jobs, worker, 2, messages.append))
    assert sorted(records) == ['1', '2', '3']
    assert attempts == {'1': 2, '2': 1, '3': 1}
    assert any('retrying alone' in message for message in messages)


def test_parallel_gpu_scheduler_stops_if_model_cannot_fit_alone(monkeypatch):
    """A permanent memory failure does not cause an endless retry loop."""
    monkeypatch.setattr('bess_pv_trading.forecasting.training.pv_reviewed_gpu_hpo.ProcessPoolExecutor', ImmediatePool)
    job = {'trial': {'name': 'too_large', 'farm_id': 1}}
    calls = []

    def worker(_job):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError('synthetic GPU exhaustion')

    with pytest.raises(RuntimeError, match='ran out of memory running too_large alone'):
        list(_bounded_parallel([job], worker, 4, None))
    assert len(calls) == 2


def test_parallel_gpu_configuration_uses_requested_batch_sizes():
    """The new search has its own identity and four configurable GPU slots."""
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu_parallel.yaml')
    section = validate_configuration(config)
    assert section['workers'] == 4
    assert section['neural_space']['batch_size']['values'] == [16, 32, 48]
    assert section['run_id'] == 'four_cycle_neural_night3000_gpu_parallel_v1'


def test_gpu_configuration_is_isolated_and_does_not_fall_back(monkeypatch):
    """The CUDA run uses its own output and refuses unavailable devices."""
    cpu = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml')['pv_reviewed_hpo']
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu.yaml')
    gpu = validate_configuration(config)
    assert gpu['workers'] == 1
    assert gpu['run_id'] != cpu['run_id']
    assert gpu['report_root'] != cpu['report_root']
    assert gpu['runtime']['device'] == 'cuda:0'
    assert {key: value for key, value in gpu['neural_space'].items() if key != 'batch_size'} == {
        key: value for key, value in cpu['neural_space'].items() if key != 'batch_size'}
    assert gpu['neural_space']['batch_size']['values'] == [2, 4, 8]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        _runtime_details(gpu)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))])
@pytest.mark.parametrize('family', ['mlp', 'tcn', 'bigru', 'bilstm', 'transformer'])
def test_device_aware_fit_and_prediction(device, family, monkeypatch):
    """A small synthetic fit uses its requested device and returns CPU predictions."""
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    frame = pd.DataFrame({'model_target_mw': np.linspace(.1, .8, 16),
                          'delivery_day': ['2025-01-01'] * 8 + ['2026-01-01'] * 8})
    packed = {'x': np.ones((2, 8, 2), dtype=np.float32),
              'y': frame.model_target_mw.to_numpy(dtype=np.float32).reshape(2, 8),
              'indices': np.arange(16).reshape(2, 8),
              'train': np.array([[True] * 8, [False] * 8]),
              'validation': np.array([[False] * 8, [True] * 8])}
    masks = {'validation': np.array([False] * 8 + [True] * 8)}
    parameters = {'width': 8, 'depth': 1, 'dropout': .1, 'kernel_size': 3, 'heads': 2, 'learning_rate': .001,
                  'weight_decay': 0., 'batch_size': 1, 'loss': 'mse'}
    settings = {'device': device, 'max_epochs': 2, 'patience': 2,
                'minimum_improvement': 0., 'gradient_clip_norm': 1.}
    model, info = fit_neural(family, parameters, settings, 123, packed, frame, masks, 1.)
    assert next(model.parameters()).device.type == torch.device(device).type
    assert info['device'] == device
    prediction = predict_neural(model, packed, np.array([0, 1]), 1, len(frame), 1.)
    assert np.isfinite(prediction).all()
    replay, _ = fit_neural(family, parameters, settings, 123, packed, frame, masks, 1.)
    replay_prediction = predict_neural(replay, packed, np.array([0, 1]), 1, len(frame), 1.)
    np.testing.assert_array_equal(prediction, replay_prediction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_spawned_gpu_trial_and_refit_export_cpu_checkpoint(tmp_path, monkeypatch):
    """Run four tiny trials and refits through the spawned GPU worker scheduler."""
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu.yaml')
    config['pv_reviewed_hpo']['neural_training']['max_epochs'] = 2
    config['pv_reviewed_hpo']['neural_training']['patience'] = 2
    names = ['train', 'validation', 'test']
    splits = np.repeat(names, 8)
    frame = pd.DataFrame({'model_target_mw': np.linspace(.1, .8, 24), 'split': splits,
        'delivery_day': np.repeat(['2025-01-01', '2026-01-01', '2026-05-01'], 8),
        'synthetic_target': False, 'night': False, 'evaluation_window': True})
    masks = {name: splits == name for name in names}
    packed = {'x': np.ones((3, 8, 2), dtype=np.float32),
              'y': frame.model_target_mw.to_numpy(dtype=np.float32).reshape(3, 8),
              'indices': np.arange(24).reshape(3, 8), 'split': np.asarray(names),
              **{name: mask.reshape(3, 8) for name, mask in masks.items()}}
    cache = tmp_path / 'farm_1' / 'cache'
    cache.mkdir(parents=True)
    frame.to_parquet(cache / 'frame.parquet', index=False)
    np.savez(cache / 'masks.npz', **masks)
    np.save(cache / 'x_core.npy', np.ones((24, 2), dtype=np.float32))
    np.savez(cache / 'packed_core.npz', **packed)
    (cache / 'preprocessor_core.json').write_text(json.dumps({'columns': ['x', 'y']}))
    parameters = {'width': 8, 'depth': 1, 'dropout': .1, 'learning_rate': .001,
                  'weight_decay': 0., 'batch_size': 1, 'loss': 'mse'}
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': 0,
             'seed': 123, 'parameters': parameters, 'name': 'mlp_core_000'}
    job = {'root': str(tmp_path), 'trial': trial, 'identity': 'synthetic', 'config': config}
    jobs = [{**job, 'trial': {**trial, 'number': number, 'name': f'mlp_core_{number:03d}',
                              'seed': 123 + number}} for number in range(4)]
    results = list(_bounded_parallel(jobs, _trial_worker, 4, None))
    refit_jobs = [{**job, 'trial': result['trial'], 'expected_validation_score':
                   result['validation']['all']['day_balanced_nrmse_percent']} for result in results]
    refits = list(_bounded_parallel(refit_jobs, _refit_worker, 4, None))
    checkpoint = tmp_path / 'farm_1/models/core/mlp/mlp_core_000/model.pt'
    assert all(refit['validation_replay_difference_percent'] == 0 for refit in refits)
    assert all(value.device.type == 'cpu' for value in torch.load(checkpoint, weights_only=True).values())
