"""Focused tests for hourly PV targets and Mayer-style marginal models."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.integrate import quad
import xarray as xr
from types import SimpleNamespace
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.models.pv import local_delivery_starts
from bess_pv_trading.forecasting.studies.pv_mayer_data import aggregate_hourly, check_coverage
from bess_pv_trading.forecasting.studies.pv_mayer import (censored_normal_mean,
    censored_normal_quantiles, fit_farm, fit_lqr, prepare_farm_parallel, r_models, scoring_rows)


def quarters(day):
    """Build a full, valid, timezone-aware delivery grid with two member paths."""
    starts = local_delivery_starts(day)
    cycle = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize('UTC')
    return pd.DataFrame({'interval_start_utc': starts, 'delivery_day': day,
        'forecast_cycle_utc': cycle, 'forecast_available_at_utc': cycle + pd.Timedelta(hours=6),
        'decision_cutoff_utc': cycle + pd.Timedelta(hours=10), 'measured_ac_mw': .5,
        'eligible': True, 'episode_excluded': False, 'control_mw': .4, 'member_1': .3, 'member_2': .7})


@pytest.mark.parametrize(('day', 'hours'), [('2026-03-29', 23), ('2026-10-25', 25), ('2026-05-01', 24)])
def test_hourly_aggregation_respects_dst_and_missing_targets(day, hours):
    frame = quarters(day)
    frame.loc[0, 'measured_ac_mw'] = 0
    frame.loc[4, 'measured_ac_mw'] = np.nan
    frame.loc[5, 'eligible'] = False
    frame.loc[5, 'episode_excluded'] = True
    out = aggregate_hourly(frame, ['member_1', 'member_2'], 1.)
    assert len(out) == hours
    assert out.loc[0, 'measured_ac_mw'] == pytest.approx(.375)
    assert out.loc[1, 'valid_quarters'] == 2
    assert out.loc[1, 'excluded_quarters'] == 1
    assert np.isnan(out.loc[1, 'measured_ac_mw'])
    assert out.loc[0, 'member_1'] == pytest.approx(.3)
    assert out.loc[0, 'target_available_at_utc'] == out.loc[0, 'hour_start_utc'] + pd.Timedelta(hours=1)


def test_bounds_are_flagged_without_target_clipping():
    frame = quarters('2026-05-01')
    frame.loc[0, 'measured_ac_mw'] = 1.2
    out = aggregate_hourly(frame, ['member_1', 'member_2'], 1.)
    assert out.loc[0, 'out_of_bounds_quarters'] == 1
    assert out.loc[0, 'measured_ac_mw'] == pytest.approx(.675)


def test_censored_gaussian_boundary_atoms_and_mean():
    mu, sigma = np.array([.5]), np.array([.5])
    lower_mass, upper_start = norm.cdf(-1), norm.cdf(1)
    levels = np.array([lower_mass/2, .5, (1+upper_start)/2])
    q = censored_normal_quantiles(mu, sigma, levels)
    assert q[0].tolist() == pytest.approx([0, .5, 1])
    assert censored_normal_mean(mu, sigma)[0] == pytest.approx(.5)
    with pytest.raises(ValueError): censored_normal_quantiles(mu, [0], levels)


def test_lqr_member_permutation_invariance():
    rng = np.random.default_rng(2)
    members = rng.uniform(.1, .9, (80, 4)); control = members.mean(axis=1)
    y = np.clip(control + rng.normal(0, .03, len(control)), 0, 1)
    _, a = fit_lqr(members, control, y, members[:5], control[:5], [.25, .5, .75])
    _, b = fit_lqr(members[:, ::-1], control, y, members[:5, ::-1], control[:5], [.25, .5, .75])
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_crossing_and_out_of_bounds_are_not_silently_corrected():
    frame = pd.DataFrame({'row_id': [0], 'delivery_day': ['2026-10-01'],
        'hour_start_utc': [pd.Timestamp('2026-10-01T12:00Z')], 'lead_hours': [36],
        'measured_ac_mw': [.5], 'y': [.5], 'member_1': [.4], 'member_2': [.6]})
    r = pd.DataFrame({'row_id': [0, 0], 'method': ['cn_emos', 'cn_emos_b'],
                      'location': [.5, .5], 'scale': [.2, .2], 'crps': [.1, .1]})
    scores = scoring_rows(frame, ['member_1', 'member_2'], np.array([.25, .5, .75]), r, np.array([[.6, .4, 1.2]]))
    q = scores.loc[scores.method.eq('linear_quantile')].iloc[0]
    assert q.lqr_crossing and q.lqr_out_of_bounds and not q.common_score
    assert [q.q_01, q.q_02, q.q_03] == pytest.approx([.6, .4, 1.2])
    fixed = scoring_rows(frame, ['member_1', 'member_2'], np.array([.25, .5, .75]), r,
                         np.array([[.6, .4, .8]]), rearrange_quantiles=True)
    row = fixed.loc[fixed.method.eq('linear_quantile')].iloc[0]
    assert row.lqr_crossing and row.lqr_rearranged and row.common_score
    assert [row.q_01, row.q_02, row.q_03] == pytest.approx([.4, .6, .8])
    assert [row.raw_q_01, row.raw_q_02, row.raw_q_03] == pytest.approx([.6, .4, .8])
    outside = scoring_rows(frame, ['member_1', 'member_2'], np.array([.25, .5, .75]), r,
                           np.array([[.6, .4, 1.2]]), rearrange_quantiles=True)
    assert not outside.common_score.any()


def test_current_config_detects_missing_ensemble_periods():
    config = load_config(Path('configs/pv_experiments/mayer_postprocessing_2026.yaml'))
    with pytest.raises(ValueError, match='Incomplete declared ensemble coverage'):
        check_coverage(config)


def test_forecast_cutoff_and_member_alignment_are_checked(monkeypatch):
    """Reject a missing member or a forecast published after the DA submission."""
    config = load_config(Path('configs/pv_experiments/mayer_postprocessing_2026.yaml'))
    s = config['pv_mayer_postprocessing']
    s.update(start_day='2026-05-01', train_end_day='2026-05-01', end_day='2026-05-02',
             dataset_periods=[{'ensemble_zarr': 'synthetic_ensemble', 'start_day': '2026-05-01', 'end_day': '2026-05-02'}])
    config['forecast_mapping'] = {}
    times = local_delivery_starts('2026-05-01').append(local_delivery_starts('2026-05-02')).tz_localize(None)
    cycles = pd.DatetimeIndex(['2026-04-30', '2026-05-01'])
    def open_fake(path, **kwargs):
        if str(path).endswith('synthetic_ensemble'):
            return xr.Dataset(coords={'cycle_time': cycles, 'member': np.arange(1, 51)})
        return xr.Dataset(coords={'cycle_time': cycles, 'time': times})
    monkeypatch.setattr('bess_pv_trading.forecasting.studies.pv_mayer_data.xr.open_zarr', open_fake)
    assert len(check_coverage(config)) == 2
    config['source']['availability_lag_hours'] = 20
    with pytest.raises(ValueError, match='unavailable at DA cutoff'):
        check_coverage(config)
    config['source']['availability_lag_hours'] = 6
    s['member_count'] = 51
    with pytest.raises(ValueError, match='51 distinct members'):
        check_coverage(config)


def test_each_weather_member_is_routed_to_saved_bundle(monkeypatch, tmp_path):
    """Every member gets its own complete day and missing weather is not filled."""
    from bess_pv_trading.forecasting.studies import pv_mayer_data as module
    config = load_config(Path('configs/pv_experiments/mayer_postprocessing_2026.yaml'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'preprocessor': {'columns': ['forecast_ghi']}}))
    config['pv']['forecast_replacement']['models_by_farm']['1']['artifact_path'] = str(manifest)
    metadata = SimpleNamespace(ac_capacity_mw=1., elevation_m=50.)
    monkeypatch.setattr(module, 'load_metadata', lambda *args: {1: metadata})
    monkeypatch.setattr(module, 'farm_settings', lambda *args: {'location': {'latitude': 45., 'longitude': 9.}})
    monkeypatch.setattr(module.xr, 'open_zarr', lambda *args, **kwargs: xr.Dataset())
    def target(dataset, farm_id, starts, config):
        return pd.DataFrame({'interval_start_utc': starts, 'measured_ac_mw': .5, 'eligible': True, 'episode_excluded': False})
    monkeypatch.setattr(module, '_target_frame', target)
    def aligned(dataset, farm_id, cycle, starts, config, ensemble):
        ids = tuple(str(i) for i in range(1, 51)) if ensemble else ('deterministic',)
        parts = []
        for member in ids:
            frame = pd.DataFrame({'member': member, 'forecast_ghi': 100.}, index=starts)
            if ensemble and member == '50': frame.iloc[0, frame.columns.get_loc('forecast_ghi')] = np.nan
            parts.append(frame)
        return SimpleNamespace(members=ids, values=pd.concat(parts))
    monkeypatch.setattr(module, 'align_cycle', aligned)
    calls = []
    def predictor(frame, model, farm_id):
        assert frame.forecast_cycle_utc.nunique() == 1
        assert len(frame) == (95 if frame.member.iloc[0] == '50' else 96)
        calls.append(frame.member.iloc[0])
        return np.full(len(frame), .4)
    monkeypatch.setattr(module, 'predict_bundle', predictor)
    hourly, quarter_output, members, rating, _, _ = module.prepare_farm(config, 1, {pd.Timestamp('2026-05-01'): 'fake'})
    assert len(members) == 50 and len(calls) == 51
    assert '50' in calls and 'deterministic' in calls
    assert not hourly.forecast_valid.iloc[0] and hourly.forecast_valid.iloc[1:].all() and hourly.valid_quarters.eq(4).all()
    assert len(quarter_output) == 96 and quarter_output.member_1.eq(.4).all()
    assert rating == 1.


@pytest.mark.parametrize(('day', 'count'), [('2026-05-01', 96), ('2026-10-25', 100)])
def test_saved_timing_bundle_accepts_synthetic_member_view(day, count):
    """Exercise the real frozen bundle contract with synthetic weather, including DST."""
    import torch
    from bess_pv_trading.forecasting.inference.pv_forecast_bundle import predict_bundle
    torch.set_num_threads(1)
    model = Path('models/pv/farm_901/pilot_2025_train/direct_tcn')
    if not model.exists(): pytest.skip('Saved local neural pilot unavailable')
    starts = local_delivery_starts(day)
    cycle = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize('UTC')
    frame = pd.DataFrame({'interval_start_utc': starts, 'delivery_day': day, 'auction': 'DA',
        'forecast_cycle_utc': cycle, 'lead_hours': (starts - cycle) / pd.Timedelta(hours=1),
        'forecast_ghi': 200., 'forecast_bhi': 100., 'temperature_c': 18.,
        'wind_u_10m': 2., 'wind_v_10m': 1.})
    result = predict_bundle(frame, model, farm_id=901)
    assert len(result) == count and np.isfinite(result).all()


def test_parallel_preparation_combines_days_in_order_without_mixing():
    """Collect asynchronous outputs deterministically and preserve member contracts."""
    class Pool:
        def submit(self, function, config, farm, day, archive):
            frame = pd.DataFrame({'day': [day]})
            result = (frame, frame.copy(), ['member_1'], 1., {'latitude': 45}, str(day))
            return SimpleNamespace(result=lambda: result)
    a = pd.Timestamp('2026-01-01'); b = pd.Timestamp('2026-08-02')
    hourly, quarters, members, _, _, _ = prepare_farm_parallel({}, 1, {b: 'b', a: 'a'}, Pool())
    assert hourly.day.tolist() == [a, b] and quarters.day.tolist() == [a, b]
    assert members == ['member_1']


def test_r_crps_and_boosted_models_are_fit_on_synthetic_cases(tmp_path):
    rscript = Path('/home/rghetti/miniforge3/envs/r-censored/bin/Rscript')
    if not rscript.exists(): pytest.skip('Separate R environment unavailable')
    rng = np.random.default_rng(45); n = 500
    control = rng.uniform(.2, .7, n); spread = rng.uniform(.08, .18, n)
    frame = pd.DataFrame({'row_id': np.arange(n), 'lead': np.repeat([32, 33], n//2),
        'control': control, 'ens_mean': control + rng.normal(0, .03, n),
        'ens_sd': spread, 'variance': spread**2, 'log_variance': np.log(spread**2),
        'y': np.clip(control + rng.normal(0, .14, n), 0, 1)})
    evaluation = frame.iloc[:20].copy()
    evaluation.loc[0, ['variance', 'ens_sd']] = 0.
    evaluation.loc[0, 'log_variance'] = np.nan
    pred, status = r_models(frame, evaluation, tmp_path / 'r', rscript)
    assert set(status.method) == {'cn_emos', 'cn_emos_b'}
    assert set(status.loc[status.method.eq('cn_emos'), 'lead']) == {32, 33}
    assert set(status.status) == {'complete'}, status.to_string()
    assert set(pred.method) == {'cn_emos', 'cn_emos_b'}
    assert 0 not in set(pred.loc[pred.method.eq('cn_emos'), 'row_id'])
    assert 0 in set(pred.loc[pred.method.eq('cn_emos_b'), 'row_id'])
    assert (pred.scale > 0).all() and (pred.crps >= 0).all()
    sample = pred.iloc[0]
    observed = frame.loc[frame.row_id.eq(sample.row_id), 'y'].iloc[0]
    cdf = lambda x: norm.cdf((x - sample.location) / sample.scale)
    integrated = quad(lambda x: cdf(x)**2, 0, observed)[0] + quad(lambda x: (1-cdf(x))**2, observed, 1)[0]
    assert sample.crps == pytest.approx(integrated, abs=1e-7)


def test_complete_synthetic_pipeline_persists_predictions_and_scores(tmp_path):
    """Exercise fit, R subprocess, quantile model and reporting with synthetic cases."""
    rscript = Path('/home/rghetti/miniforge3/envs/r-censored/bin/Rscript')
    if not rscript.exists(): pytest.skip('Separate R environment unavailable')
    config = load_config(Path('configs/pv_experiments/mayer_postprocessing_2026.yaml'))
    config['pv_mayer_postprocessing'].update(train_end_day='2026-08-19', quantile_count=5)
    rng = np.random.default_rng(777); n = 600
    control = rng.uniform(.3, .65, n); spread = rng.uniform(.04, .12, n)
    matrix = control[:, None] + spread[:, None] * np.array([-1, .2, 1.3])
    y = np.clip(control + rng.normal(0, .07, n), 0, 1)
    day = np.where(np.arange(n) < 500, '2026-08-10', '2026-08-20')
    hour = pd.to_datetime(day, utc=True) + pd.to_timedelta(np.arange(n) % 24, unit='h')
    frame = pd.DataFrame({'delivery_day': day, 'hour_start_utc': hour, 'lead_hours': 32.,
        'target_available_at_utc': hour + pd.Timedelta(hours=1), 'valid_quarters': 4,
        'excluded_quarters': 0, 'out_of_bounds_quarters': 0, 'forecast_valid': True,
        'daylight': True, 'measured_ac_mw': y, 'control_mw': control,
        'member_1': matrix[:, 0], 'member_2': matrix[:, 1], 'member_3': matrix[:, 2]})
    frame.loc[495:499, 'target_available_at_utc'] = pd.Timestamp('2026-08-19T23:00Z')
    out = tmp_path / 'farm_1'
    summary = fit_farm(config, 1, frame, ['member_1', 'member_2', 'member_3'], 1., out)
    assert summary['training_hours'] == 495 and summary['evaluation_hours'] == 100
    assert summary['matched_scored_hours'] > 0
    scores = pd.read_csv(out / 'scores.csv')
    assert set(scores.method) == {'cn_emos', 'cn_emos_b', 'linear_quantile', 'raw_ensemble'}
    assert (out / 'r_models' / 'cn_emos_lead_32.rds').exists()
    assert (out / 'quantile_reliability.png').exists()
