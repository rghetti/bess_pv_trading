"""Exercise the isolated GPU search and its device-aware model path."""
import numpy as np
import pandas as pd
import pytest
import torch
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.models.pv_reviewed_gpu_hpo_models import fit_neural, predict_neural
from bess_pv_trading.forecasting.training.pv_reviewed_gpu_hpo import (
    _bounded_parallel, _refit_worker, _runtime_details, _trial_worker, validate_configuration,
)


def test_gpu_configuration_is_isolated_and_does_not_fall_back(monkeypatch):
    """The CUDA run uses its own output and refuses unavailable devices."""
    cpu = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml')['pv_reviewed_hpo']
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu.yaml')
    gpu = validate_configuration(config)
    assert gpu['workers'] == 1
    assert gpu['run_id'] != cpu['run_id']
    assert gpu['report_root'] != cpu['report_root']
    assert gpu['runtime']['device'] == 'cuda:0'
    assert gpu['neural_space'] == cpu['neural_space']
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
    """Run a tiny trial and its refit through the actual spawned worker scheduler."""
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
    result = list(_bounded_parallel([job], _trial_worker, 1, None))[0]
    refit_job = {**job, 'expected_validation_score': result['validation']['all']['day_balanced_nrmse_percent']}
    refit = list(_bounded_parallel([refit_job], _refit_worker, 1, None))[0]
    checkpoint = tmp_path / 'farm_1/models/core/mlp/mlp_core_000/model.pt'
    assert refit['validation_replay_difference_percent'] == 0
    assert all(value.device.type == 'cpu' for value in torch.load(checkpoint, weights_only=True).values())
