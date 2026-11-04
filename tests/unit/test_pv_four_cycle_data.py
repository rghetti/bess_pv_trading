"""Check split boundaries in overlapping four-cycle weather sequences."""
import numpy as np
import pandas as pd
import json
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_four_cycle_hpo_data import build_cycle_sequences
from bess_pv_trading.forecasting.inference.pv_reviewed_hpo import predict_reviewed_hpo
from bess_pv_trading.forecasting.models.pv_reviewed_hpo_models import predict_neural
from bess_pv_trading.forecasting.training.pv_reviewed_hpo import _plot_day, _refit_worker, _trial_worker, publish_results, run_hpo
from bess_pv_trading.forecasting.training.pv_reviewed_hpo_common import trial_specifications


def test_each_issue_keeps_full_weather_context_but_separate_target_masks():
    cycles = pd.to_datetime(['2025-12-31T18:00Z'] * 96 + ['2026-01-01T00:00Z'] * 96, utc=True)
    frame = pd.DataFrame({'forecast_cycle_utc': cycles, 'model_target_mw': np.arange(192) / 1000,
                          'split': ['train'] * 24 + ['validation'] * 168})
    features = np.arange(384, dtype=np.float32).reshape(192, 2)
    train = np.zeros(192, dtype=bool)
    validation = np.zeros(192, dtype=bool)
    train[:24] = True
    validation[24:96] = True
    validation[120:] = True
    masks = {'train': train, 'validation': validation, 'test': np.zeros(192, dtype=bool)}
    packed = build_cycle_sequences(frame, features, masks, 1., 96, forecast_features=features + 1000)
    assert packed['split'].tolist() == ['train', 'validation', 'validation']
    np.testing.assert_array_equal(packed['indices'][0], packed['indices'][1])
    assert packed['train'][0].sum() == 24 and packed['validation'][0].sum() == 0
    assert packed['train'][1].sum() == 0 and packed['validation'][1].sum() == 72
    assert packed['validation'][2].sum() == 72
    np.testing.assert_array_equal(packed['x'][0], features[:96])
    np.testing.assert_array_equal(packed['x'][1], features[:96] + 1000)
    assert packed['output'][0].sum() == 24 and packed['output'][1].sum() == 72
    class FirstFeature:
        def eval(self):
            return self
        def __call__(self, inputs, valid):
            return inputs[:, :, 0]
    prediction = predict_neural(FirstFeature(), packed, np.array([0, 1, 2]), 3, 192, 1.)
    np.testing.assert_array_equal(prediction[:24], features[:24, 0])
    np.testing.assert_array_equal(prediction[24:96], features[24:96, 0] + 1000)


def test_four_cycle_search_contract_has_no_missing_flags_or_statistical_models():
    section = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml')['pv_reviewed_hpo']
    trials = trial_specifications(section)
    assert len(trials) == 2560 and section['workers'] == 42
    assert section['issue_horizon_hours'] == 48 and not section['include_missing_indicators']
    assert section['training_ghi_source'] == 'historical'
    assert not section['statistical_families']
    assert {trial['parameters']['width'] for trial in trials} == {128, 256, 512}


def test_four_cycle_candidate_and_refit_use_causal_validation(tmp_path):
    """A tiny synthetic cycle search produces a replayable selected checkpoint."""
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml')
    config['pv_reviewed_hpo']['neural_training']['max_epochs'] = 2
    config['pv_reviewed_hpo']['neural_training']['patience'] = 2
    cycles = pd.to_datetime(['2025-01-01T00:00Z'] * 96 + ['2026-01-01T06:00Z'] * 96
                            + ['2026-05-01T12:00Z'] * 96, utc=True)
    splits = np.repeat(['train', 'validation', 'test'], 96)
    frame = pd.DataFrame({'forecast_cycle_utc': cycles, 'split': splits,
        'delivery_day': np.repeat(['2025-01-01', '2026-01-01', '2026-05-01'], 96),
        'model_target_mw': np.tile(np.linspace(0., .5, 96), 3),
        'synthetic_target': False, 'night': False, 'evaluation_window': True,
        'interval_start_utc': cycles + pd.to_timedelta(np.tile(np.arange(96), 3) * 15, unit='min'),
        'lead_hours': np.tile(np.arange(96) / 4, 3), 'ghi_interpolated': False,
        'ghi_input_w_m2': 200.})
    x = np.tile(np.linspace(0., 1., 96), 6).reshape(288, 2).astype(np.float32)
    masks = {name: frame['split'].eq(name).to_numpy(copy=True) for name in ('train', 'validation', 'test')}
    masks['validation'][96:120] = False
    masks['test'][192:216] = False
    packed = build_cycle_sequences(frame, x, masks, 1., 96)
    cache = tmp_path / 'farm_1/cache'
    cache.mkdir(parents=True)
    frame.to_parquet(cache / 'frame.parquet', index=False)
    np.savez(cache / 'masks.npz', **masks)
    np.save(cache / 'x_core.npy', x)
    np.savez(cache / 'packed_core.npz', **packed)
    preprocessor = {'columns': ['ghi_w_m2', 'forecast_lead_hours'], 'median': [0., 0.],
        'mean': [0., 0.], 'scale': [1., 1.], 'clip_standard_deviations': 12.,
        'include_missing_indicators': False}
    (cache / 'preprocessor_core.json').write_text(json.dumps(preprocessor))
    parameters = {'width': 8, 'depth': 1, 'dropout': 0., 'kernel_size': 3, 'heads': 2,
        'learning_rate': .001, 'weight_decay': 0., 'batch_size': 2, 'loss': 'mse'}
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'mlp', 'number': 0,
             'seed': 123, 'parameters': parameters, 'name': 'mlp_core_000'}
    job = {'root': str(tmp_path), 'trial': trial, 'identity': 'fixture', 'config': config}
    result = _trial_worker(job)
    assert result['test_targets_scored'] is False
    assert result['validation']['all']['quarters'] == 72
    refit = _refit_worker({**job, 'expected_validation_score':
                           result['validation']['all']['day_balanced_nrmse_percent']})
    assert refit['input_contract']['ghi_training'] == 'historical measured/interpolated'
    assert next(row for row in refit['metrics'] if row['split'] == 'test'
                and row['target_source'] == 'all')['quarters'] == 72
    times = pd.date_range('2026-08-01T21:00Z', periods=16, freq='15min')
    view = pd.DataFrame({'interval_start_utc': times,
        'delivery_day': times.tz_convert('Europe/Rome').strftime('%Y-%m-%d'),
        'forecast_cycle_utc': pd.Timestamp('2026-08-01T12:00Z'),
        'forecast_available_at_utc': pd.Timestamp('2026-08-01T18:00Z'),
        'decision_cutoff_utc': pd.Timestamp('2026-08-01T19:00Z'),
        'forecast_ghi': .5, 'forecast_bhi': .3})
    bundle = tmp_path / 'farm_1/models/core/mlp/mlp_core_000'
    output = predict_reviewed_hpo(view, bundle, farm_id=1,
                                  night_policy=config['policy']['forecast_pv_night'])
    np.testing.assert_array_equal(output, np.zeros(16))
    config['pv_reviewed_hpo']['farm_ids'] = [1]
    config['pv_reviewed_hpo']['validation']['bootstrap_repetitions'] = 2
    config['pv_reviewed_hpo']['validation']['plot_days_per_split'] = 1
    report = tmp_path / 'report'
    publish_results(config, tmp_path, report, [result], {(1, 'core', 'mlp'): result},
                    {1: result}, [refit], 'fixture')
    scores = pd.read_csv(report / 'cycle_lead_metrics.csv')
    assert set(scores.loc[scores['split'].eq('validation'), 'lead_start_hours']) == {6, 12, 18}


def test_cycle_plot_accepts_overlapping_forecasts(tmp_path):
    times = pd.date_range('2026-05-01T10:00Z', periods=4, freq='15min')
    frame = pd.DataFrame({'interval_start_utc': times.append(times),
        'forecast_cycle_utc': [pd.Timestamp('2026-05-01T00:00Z')] * 4
                              + [pd.Timestamp('2026-05-01T06:00Z')] * 4,
        'delivery_day': '2026-05-01', 'split': 'test', 'model_target_mw': .3,
        'synthetic_target': False, 'ghi_input_w_m2': 200.})
    path = tmp_path / 'plot.png'
    _plot_day(frame, np.full(8, .2), np.ones(8, dtype=bool),
              {'farm_id': 1, 'family': 'mlp', 'feature': 'core'}, path, 'Europe/Rome')
    assert path.is_file()


def test_large_search_refuses_to_start_when_available_memory_is_insufficient(monkeypatch):
    config = load_config('configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml')
    monkeypatch.setattr('bess_pv_trading.forecasting.training.pv_reviewed_hpo._available_memory_gib', lambda: 10.)
    with pytest.raises(RuntimeError, match='requires 64 GiB available memory'):
        run_hpo(config)
