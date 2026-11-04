"""Check identity and refit provenance for GPU 0-only recovery."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_four_cycle_hpo_data import build_cycle_sequences
from bess_pv_trading.forecasting.training import pv_reviewed_gpu0_recovery as recovery
from bess_pv_trading.forecasting.training.pv_reviewed_hpo_common import identity_digest


def test_recovery_accepts_only_original_training_contract():
    """Changing data, settings, model code or library versions prevents recovery."""
    config = {'pv_reviewed_hpo': {'run_id': 'existing', 'resume': True}}
    proxy = {'configuration': {'workers': 1}, 'runtime': {'device': 'cuda:0'},
             'datasets': {'1': 'saved_hash'}, 'versions': {'torch': 'saved_version'},
             'code_sha256': {'model.py': 'saved_code'}}
    contract = deepcopy(proxy)
    contract['configuration'] = {'run_id': 'existing'}
    contract['runtime'] = {'devices': ['original_device_map']}
    contract['code_sha256'][recovery.Path(recovery.multi.__file__).name] = recovery.base._sha(
        recovery.Path(recovery.multi.__file__))
    saved = {'identity': identity_digest(contract), 'contract': contract}
    recovery._verify_contract(config, saved, proxy)
    changed = deepcopy(proxy)
    changed['datasets']['1'] = 'different_hash'
    with pytest.raises(ValueError, match='configuration, data, code or versions differ'):
        recovery._verify_contract(config, saved, changed)


def test_recovery_requires_explicit_gpu0_visibility(monkeypatch):
    """Recovery cannot accidentally schedule work on a different physical GPU."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES=0'):
        recovery._gpu0_details({'contract': {'runtime': {'devices': []}}})


def test_recovery_refit_records_original_trial_device(tmp_path, monkeypatch):
    """A GPU 0 refit retains the original validation trial's device provenance."""
    original = {'visible_id': '2', 'uuid': 'source', 'name': 'source_device'}
    current = {'visible_id': '0', 'uuid': 'target', 'name': 'target_device'}
    job = {'root': str(tmp_path), 'trial': {'farm_id': 1, 'feature': 'core', 'family': 'mlp',
                                            'name': 'trial'},
           'gpu': current, 'original_training_gpu': original}
    monkeypatch.setattr(recovery.multi, '_refit_worker', lambda _job: {'gpu': current})
    captured = {}
    monkeypatch.setattr(recovery.base, '_write_json', lambda path, record: captured.update(record))
    result = recovery._refit_worker(job)
    assert result['gpu'] == current
    assert captured['original_training_gpu'] == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_gpu0_trial_refit_and_cross_device_provenance(tmp_path, monkeypatch):
    """A real synthetic GPU 0 refit replays validation and writes both device identities."""
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml')
    config['pv_reviewed_hpo']['neural_training']['max_epochs'] = 2
    config['pv_reviewed_hpo']['neural_training']['patience'] = 2
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
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': 0,
             'seed': 123, 'parameters': parameters, 'name': 'mlp_core_000'}
    properties = torch.cuda.get_device_properties(0)
    gpu0 = {'visible_id': '0', 'uuid': str(properties.uuid), 'name': properties.name}
    job = {'root': str(tmp_path), 'trial': trial, 'identity': 'synthetic', 'gpu': gpu0,
           'config': recovery.multi._single_device_proxy(config)}
    result = recovery.multi._trial_worker(job)
    source = {'visible_id': '1', 'uuid': 'historical_gpu', 'name': 'historical_device'}
    refit_job = {**job, 'original_training_gpu': source, 'expected_validation_score':
                 result['validation']['all']['day_balanced_nrmse_percent']}
    refit = recovery._refit_worker(refit_job)
    assert refit['validation_replay_difference_percent'] == 0
    assert refit['gpu'] == gpu0 and refit['original_training_gpu'] == source
    mismatch_trial = {**trial, 'number': 1, 'name': 'mlp_core_001'}
    mismatch = {**refit_job, 'trial': mismatch_trial,
                'expected_validation_score': refit_job['expected_validation_score'] + 1.0}
    with pytest.raises(ValueError, match='Refit validation replay differs'):
        recovery._refit_worker(mismatch)
