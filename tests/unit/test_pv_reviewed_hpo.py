"""Exercise reviewed HPO data, leakage controls and candidate model contracts."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_reviewed_hpo_data import (
    align_extended_weather, evenly_sample_training_nights, split_weather_inputs, training_mask,
)
from bess_pv_trading.forecasting.models.pv_reviewed_hpo_models import (
    build_neural_network, build_sequences, fit_neural, fit_statistical, predict_neural,
)
from bess_pv_trading.forecasting.inference.pv_reviewed_hpo import inference_features, predict_reviewed_hpo
from bess_pv_trading.forecasting.training.pv_reviewed_hpo_common import (
    score_predictions, trial_specifications,
)
from bess_pv_trading.forecasting.training.pv_reviewed_hpo import (
    _family_winners, _paired_bootstrap, _refit_worker, _top_three, _trial_worker, publish_results,
)


@pytest.fixture
def config():
    """Use the actual 30-worker configuration for all contract checks."""
    return load_config('configs/pv_experiments/reviewed_ghi_hpo.yaml')


def test_trial_grid_covers_all_farms_families_and_both_features(config):
    section = config['pv_reviewed_hpo']
    trials = trial_specifications(section)
    assert len(trials) == 8192
    assert [trial['farm_id'] for trial in trials[:4]] == [1, 345, 901, 910]
    groups = pd.DataFrame(trials).groupby(['farm_id', 'family', 'feature']).size()
    assert len(groups) == 128 and groups.eq(64).all()
    assert not any('pvlib' in trial['family'] for trial in trials)
    core = next(trial for trial in trials if trial['farm_id'] == 1
                and trial['family'] == 'ridge' and trial['feature'] == 'core')
    extended = next(trial for trial in trials if trial['farm_id'] == 1
                    and trial['family'] == 'ridge' and trial['feature'] == 'extended')
    assert core['seed'] == extended['seed'] and core['parameters'] == extended['parameters']
    assert trials == trial_specifications(section)


def test_split_specific_ghi_never_uses_observed_validation_or_test(config):
    section = config['pv_reviewed_hpo']
    frame = pd.DataFrame({'split': ['train', 'train', 'validation', 'test'],
        'night': [False, True, False, False], 'ghi_w_m2': [100., np.nan, 900., 800.],
        'nwp_ghi_w_m2': [10., 0., 200., 300.], 'nwp_bhi_w_m2': [1.] * 4,
        'temperature_c': [20.] * 4, 'wind_u_10m': [1.] * 4, 'wind_v_10m': [2.] * 4})
    weather, zeroed = split_weather_inputs(frame, section)
    assert weather.ghi_w_m2.tolist() == [100., 0., 200., 300.]
    assert zeroed.tolist() == [False, True, False, False]
    changed = frame.copy()
    changed.loc[2:, 'ghi_w_m2'] = 9999.
    pd.testing.assert_frame_equal(weather, split_weather_inputs(changed, section)[0])


def test_night_training_mask_keeps_reviewed_labels_without_inventing_targets(config):
    section = config['pv_reviewed_hpo']
    frame = pd.DataFrame({'split': ['train'] * 5, 'embargoed': [False] * 5,
        'model_target_mw': [0., .1, np.nan, .2, .3], 'night': [True, True, True, True, False],
        'ghi_w_m2': [np.nan, 0., 0., 0., 500.], 'retained_prior_exclusion': [False, False, False, True, False],
        'temperature_c': [20.] * 5, 'wind_u_10m': [1.] * 5, 'wind_v_10m': [1.] * 5})
    assert training_mask(frame, section).tolist() == [True, True, False, False, True]


def test_even_night_sampling_preserves_daylight_and_redistributes_shortfalls():
    """Calendar-day allocation keeps exactly the requested eligible nights."""
    days = pd.date_range('2025-01-01', periods=365, freq='D').strftime('%Y-%m-%d')
    frame = pd.DataFrame({'delivery_day': np.repeat(days, 17), 'split': 'train',
                          'night': np.tile([True] * 16 + [False], len(days))})
    eligible = np.ones(len(frame), dtype=bool)
    eligible[:16] = False
    eligible[17 + 1] = False
    selected = evenly_sample_training_nights(frame, eligible, 3000)
    assert selected[frame.night].sum() == 3000
    assert selected[~frame.night].all()
    assert not selected[~eligible].any()
    assert np.array_equal(selected, evenly_sample_training_nights(frame, eligible, 3000))
    daily = frame.loc[selected & frame.night, 'delivery_day'].value_counts()
    assert len(daily) == 364
    assert daily.min() >= 8 and daily.max() <= 10
    with pytest.raises(ValueError, match='eligible nighttime targets'):
        evenly_sample_training_nights(frame, eligible, 10000)


def test_extended_forecast_alignment_uses_only_recorded_cycle_and_temporal_semantics(config):
    times = pd.date_range('2026-08-02T10:00Z', periods=4, freq='15min')
    leads = np.arange(49)
    raw = leads[None, :].astype(float)
    selected = xr.Dataset({'temperature_2m': (('cycle_time', 'lead_time'), raw),
                           'precipitation': (('cycle_time', 'lead_time'), raw)},
        coords={'cycle_time': [np.datetime64('2026-08-01T00:00')],
                'lead_time': leads.astype('timedelta64[h]')})
    frame = pd.DataFrame({'interval_start_utc': times,
                          'forecast_cycle_utc': pd.Timestamp('2026-08-01T00:00Z')})
    values, _ = align_extended_weather(selected, frame, config, ['temperature_c', 'precipitation_mm'])
    np.testing.assert_allclose(values.temperature_c, [34.125, 34.375, 34.625, 34.875])
    np.testing.assert_allclose(values.precipitation_mm, [35 / 4] * 4)


def test_day_balanced_validation_does_not_weight_longer_days_more():
    frame = pd.DataFrame({'delivery_day': ['2026-01-01', '2026-01-02', '2026-01-02'],
                          'model_target_mw': [0., 0., 0.]})
    score = score_predictions(frame, np.array([1., 3., 3.]), np.ones(3, dtype=bool), 1.)
    assert score['day_balanced_nrmse_percent'] == pytest.approx(np.sqrt(5) * 100)
    assert score['quarter_weighted_nrmse_percent'] == pytest.approx(np.sqrt(19 / 3) * 100)


@pytest.mark.parametrize('family,parameters', [
    ('ridge', {'alpha': 1.0}),
    ('elastic_net', {'alpha': .001, 'l1_ratio': .5}),
    ('huber', {'alpha': .01, 'epsilon': 1.35}),
    ('spline_ridge', {'alpha': 1., 'n_knots': 4, 'degree': 2}),
    ('polynomial_ridge', {'alpha': 1., 'degree': 2}),
    ('rbf_ridge', {'alpha': 1., 'n_components': 16, 'gamma': .1}),
    ('extra_trees', {'n_estimators': 3, 'max_features': 1., 'min_samples_leaf': 2, 'max_depth': 4}),
    ('random_forest', {'n_estimators': 3, 'max_features': 1., 'min_samples_leaf': 2, 'max_depth': 4}),
    ('hist_gradient_boosting', {'max_iter': 3, 'max_leaf_nodes': 7, 'min_samples_leaf': 2,
                                'learning_rate': .1, 'l2_regularization': 1., 'loss': 'squared_error'}),
    ('lightgbm', {'n_estimators': 3, 'num_leaves': 7, 'min_child_samples': 2,
                  'learning_rate': .1, 'reg_lambda': 1., 'objective': 'regression'}),
    ('catboost', {'iterations': 3, 'depth': 4, 'learning_rate': .1,
                  'l2_leaf_reg': 1., 'loss_function': 'RMSE'}),
])
def test_statistical_family_fits_small_deterministic_fixture(family, parameters):
    x = np.arange(120, dtype=float).reshape(40, 3) / 120
    y = .1 + .6 * x[:, 0]
    mask = np.arange(len(x)) < 32
    days = np.repeat(['a', 'b', 'c', 'd'], 8)
    model = fit_statistical(family, parameters, 123, x, y, mask, days)
    assert np.isfinite(model.predict(x[~mask])).all()


@pytest.mark.parametrize('family', ['mlp', 'tcn', 'bigru', 'bilstm', 'transformer'])
def test_neural_family_accepts_valid_night_and_padding(family):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    parameters = {'width': 8, 'depth': 2, 'dropout': 0., 'kernel_size': 3, 'heads': 2}
    model = build_neural_network(family, 4, parameters)
    inputs = torch.ones((2, 4, 4))
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    result = model(inputs, valid)
    assert result.shape == (2, 4)
    assert torch.isfinite(result).all()
    assert (result[valid] > 0).all()
    assert (result[~valid] == 0).all()


def test_sequence_training_keeps_night_and_ignores_test_targets():
    times = pd.DatetimeIndex([*pd.date_range('2025-01-01', periods=6, freq='15min', tz='UTC'),
        *pd.date_range('2026-01-01', periods=6, freq='15min', tz='UTC'),
        *pd.date_range('2026-05-01', periods=6, freq='15min', tz='UTC')])
    frame = pd.DataFrame({'interval_start_utc': times,
        'delivery_day': ['2025-01-01'] * 6 + ['2026-01-01'] * 6 + ['2026-05-01'] * 6,
        'split': ['train'] * 6 + ['validation'] * 6 + ['test'] * 6,
        'forecast_cycle_utc': [pd.Timestamp('2024-12-31T00:00Z')] * 6
            + [pd.Timestamp('2025-12-31T00:00Z')] * 6 + [pd.Timestamp('2026-04-30T00:00Z')] * 6,
        'model_target_mw': [0., 0., .1, .2, .3, .4] * 3,
        'retained_prior_exclusion': False})
    features = np.arange(72, dtype=np.float32).reshape(18, 4) / 100
    masks = {'train': frame['split'].eq('train').to_numpy(),
             'validation': frame['split'].eq('validation').to_numpy(),
             'test': frame['split'].eq('test').to_numpy()}
    packed = build_sequences(frame, features, masks, 1.)
    assert packed['train'].sum() == 6
    parameters = {'width': 8, 'depth': 1, 'dropout': 0., 'kernel_size': 3, 'heads': 2,
        'learning_rate': .001, 'weight_decay': 0., 'batch_size': 2, 'loss': 'mse'}
    options = {'max_epochs': 2, 'patience': 2, 'minimum_improvement': 0.,
               'huber_delta': .05, 'gradient_clip_norm': 1.}
    first, _ = fit_neural('mlp', parameters, options, 42, packed, frame, masks, 1.)
    changed = frame.copy()
    changed.loc[masks['test'], 'model_target_mw'] = 1000.
    altered = build_sequences(changed, features, masks, 1.)
    second, _ = fit_neural('mlp', parameters, options, 42, altered, changed, masks, 1.)
    groups = np.flatnonzero(packed['validation'].any(axis=1))
    a = predict_neural(first, packed, groups, 2, len(frame), 1.)
    b = predict_neural(second, altered, groups, 2, len(frame), 1.)
    np.testing.assert_allclose(a[masks['validation']], b[masks['validation']])


def test_trial_records_only_validation_and_refit_replays_selected_model(tmp_path, config, monkeypatch):
    cache = tmp_path / 'farm_1' / 'cache'
    cache.mkdir(parents=True)
    frame = pd.DataFrame({'delivery_day': ['2025-01-01'] * 4 + ['2026-01-01'] * 4 + ['2026-05-01'] * 4,
        'split': ['train'] * 4 + ['validation'] * 4 + ['test'] * 4,
        'model_target_mw': [.1, .2, .3, .4] * 3,
        'synthetic_target': [False, False, False, False, False, True, False, True] + [False] * 4,
        'evaluation_window': [True] * 12, 'night': [False] * 12,
        'interval_start_utc': [*pd.date_range('2025-01-01T10:00Z', periods=4, freq='15min'),
            *pd.date_range('2026-01-01T10:00Z', periods=4, freq='15min'),
            *pd.date_range('2026-05-01T10:00Z', periods=4, freq='15min')],
        'ghi_input_w_m2': np.arange(12) * 10., 'ghi_interpolated': [False] * 12})
    frame.to_parquet(cache / 'frame.parquet', index=False)
    masks = {name: frame['split'].eq(name).to_numpy() for name in ['train', 'validation', 'test']}
    np.savez(cache / 'masks.npz', **masks)
    np.save(cache / 'x_core.npy', np.arange(24, dtype=np.float32).reshape(12, 2) / 24)
    preprocessor = {'columns': ['ghi_w_m2', 'nwp_bhi_w_m2'], 'median': [0., 0.],
        'mean': [0., 0.], 'scale': [1., 1.], 'clip_standard_deviations': 12.,
        'include_missing_indicators': False}
    (cache / 'preprocessor_core.json').write_text(json.dumps(preprocessor))
    trial = {'farm_id': 1, 'feature': 'core', 'family': 'ridge', 'number': 0,
             'seed': 123, 'parameters': {'alpha': 1.}, 'name': 'ridge_core_000'}
    job = {'root': str(tmp_path), 'trial': trial, 'identity': 'fixture', 'config': config}
    result = _trial_worker(job)
    assert result['test_targets_scored'] is False
    assert set(result['validation']) == {'all', 'measured', 'synthetic'}
    assert not (tmp_path / 'farm_1' / 'models').exists()
    # A completed matching trial can resume without another model fit.
    with monkeypatch.context() as patch:
        patch.setattr('bess_pv_trading.forecasting.training.pv_reviewed_hpo._fit_candidate',
                      lambda *args: (_ for _ in ()).throw(AssertionError('unexpected fit')))
        assert _trial_worker(job) == result
    refit = _refit_worker({**job, 'expected_validation_score':
                            result['validation']['all']['day_balanced_nrmse_percent']})
    assert refit['validation_replay_difference_percent'] <= 0.0001
    assert refit['schema'] == 'reviewed_ghi_forecast_bundle_v1'
    assert refit['timezone'] == 'Europe/Rome'
    assert any(item['split'] == 'test' for item in refit['metrics'])
    bundle = tmp_path / 'farm_1/models/core/ridge/ridge_core_000'
    assert (bundle / 'model.joblib').exists()
    forecast = pd.DataFrame({'interval_start_utc': pd.date_range('2026-08-02T10:00Z', periods=4, freq='15min'),
        'delivery_day': ['2026-08-02'] * 4,
        'forecast_cycle_utc': [pd.Timestamp('2026-08-01T00:00Z')] * 4,
        'forecast_available_at_utc': [pd.Timestamp('2026-08-01T06:00Z')] * 4,
        'decision_cutoff_utc': [pd.Timestamp('2026-08-01T10:00Z')] * 4,
        'forecast_ghi': [.5] * 4, 'forecast_bhi': [.3] * 4})
    policy = config['policy']['forecast_pv_night']
    assert np.isfinite(predict_reviewed_hpo(forecast, bundle, farm_id=1, night_policy=policy)).all()
    night_forecast = forecast.copy()
    night_forecast['interval_start_utc'] = pd.date_range('2026-08-01T22:00Z', periods=4, freq='15min')
    night_forecast['forecast_cycle_utc'] = pd.Timestamp('2026-08-01T12:00Z')
    night_forecast['forecast_available_at_utc'] = pd.Timestamp('2026-08-01T18:00Z')
    night_forecast['decision_cutoff_utc'] = pd.Timestamp('2026-08-01T19:00Z')
    night_forecast.loc[0, 'forecast_ghi'] = np.nan
    np.testing.assert_array_equal(predict_reviewed_hpo(night_forecast, bundle, farm_id=1,
                                                      night_policy=policy), np.zeros(4))
    with pytest.raises(ValueError, match='different farm'):
        predict_reviewed_hpo(forecast, bundle, farm_id=345, night_policy=policy)
    # Exercise the entire report path using only this small synthetic fixture.
    options = deepcopy(config)
    options['pv_reviewed_hpo']['farm_ids'] = [1]
    options['pv_reviewed_hpo']['validation']['bootstrap_repetitions'] = 2
    options['pv_reviewed_hpo']['validation']['plot_days_per_split'] = 1
    report = tmp_path / 'report'
    records = [result]
    refits = [refit]
    for number in (1, 2):
        candidate = {**trial, 'number': number, 'name': f'ridge_core_{number:03d}',
                     'parameters': {'alpha': float(number + 1)}}
        candidate_job = {**job, 'trial': candidate}
        candidate_record = _trial_worker(candidate_job)
        records.append(candidate_record)
        refits.append(_refit_worker({**candidate_job, 'expected_validation_score':
            candidate_record['validation']['all']['day_balanced_nrmse_percent']}))
    winners, selected = _family_winners(records)
    publish_results(options, tmp_path, report, records, winners, selected, refits, 'fixture')
    assert (report / 'monthly_metrics.csv').exists()
    assert (report / 'paired_test_bootstrap.csv').exists()
    assert (report / 'top_three_metrics.csv').exists()
    assert 'Top three configurations per farm' in (report / 'README.md').read_text()
    metrics = pd.read_csv(report / 'top_three_metrics.csv')
    assert set(metrics['split']) == {'train', 'validation', 'test'}
    assert set(metrics['rank']) == {1, 2, 3}
    assert len(pd.read_csv(report / 'family_feature_winners.csv')) == 1
    assert len(list((tmp_path / 'plots').rglob('*.png'))) == 3
    assert not any('test' in name for name in pd.read_csv(report / 'all_trials_validation.csv').columns)


def test_inference_uses_forecast_ghi_and_checks_vintage_and_availability():
    """Future forecasts cannot silently fall back to observed or unavailable weather."""
    frame = pd.DataFrame({'interval_start_utc': pd.date_range('2026-08-02T10:00Z', periods=4, freq='15min'),
        'delivery_day': ['2026-08-02'] * 4,
        'forecast_cycle_utc': [pd.Timestamp('2026-08-01T00:00Z')] * 4,
        'forecast_available_at_utc': [pd.Timestamp('2026-08-01T06:00Z')] * 4,
        'decision_cutoff_utc': [pd.Timestamp('2026-08-01T10:00Z')] * 4,
        'forecast_ghi': [855.] * 4, 'forecast_bhi': [400.] * 4,
        'ghi_w_m2': [9999.] * 4})
    preprocessor = {'columns': ['ghi_w_m2', 'nwp_bhi_w_m2', 'hour_sin']}
    features = inference_features(frame, preprocessor, 'Europe/Rome')
    assert features.ghi_w_m2.tolist() == [855.] * 4
    assert features.nwp_bhi_w_m2.tolist() == [400.] * 4
    changed = frame.copy()
    changed['forecast_available_at_utc'] = pd.Timestamp('2026-08-01T11:00Z')
    with pytest.raises(ValueError, match='available'):
        inference_features(changed, preprocessor, 'Europe/Rome')
    changed = frame.copy()
    changed['forecast_cycle_utc'] = pd.Timestamp('2026-08-01T06:00Z')
    assert len(inference_features(changed, preprocessor, 'Europe/Rome')) == 4
    changed = frame.copy()
    changed.loc[0, 'forecast_ghi'] = np.nan
    with pytest.raises(ValueError, match='Forecast GHI'):
        inference_features(changed, preprocessor, 'Europe/Rome')


def test_family_and_farm_winners_use_validation_scores_only():
    def record(family, feature, value, test_value):
        return {'trial': {'farm_id': 1, 'family': family, 'feature': feature, 'number': 0},
                'validation': {'all': {'day_balanced_nrmse_percent': value}},
                'test_score_unrelated': test_value}
    records = [record('a', 'core', 3., 100.), record('b', 'extended', 4., 0.)]
    families, farms = _family_winners(records)
    assert len(families) == 2
    assert farms[1]['trial']['family'] == 'a'


def test_top_three_ranks_individual_trials_and_ignores_test_scores():
    records = []
    for number, validation, test in [(0, 3., 100.), (1, 2., 200.), (2, 4., 0.), (3, 1., 300.)]:
        records.append({'trial': {'farm_id': 1, 'family': 'ridge', 'feature': 'core', 'number': number},
                        'validation': {'all': {'day_balanced_nrmse_percent': validation}},
                        'test_score_unrelated': test})
    assert [record['trial']['number'] for record in _top_three(records)[1]] == [3, 1, 0]


def test_paired_bootstrap_is_reproducible_for_fixed_seed():
    days = pd.date_range('2026-05-01', periods=14, freq='D').strftime('%Y-%m-%d')
    baseline = pd.Series(np.ones(14), index=days)
    candidate = pd.Series(np.full(14, 4.), index=days)
    options = {'block_bootstrap_days': 7, 'bootstrap_repetitions': 20}
    first = _paired_bootstrap(baseline, candidate, options, 123)
    assert first == _paired_bootstrap(baseline, candidate, options, 123)
    assert first['candidate_minus_selected_nrmse_pp'] == pytest.approx(100.)
