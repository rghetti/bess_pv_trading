"""Portable pilot inference preserves the forecast adapter's timing semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.simulation.continuous import ContinuousReplay


def test_reviewed_model_replay_accepts_later_available_cycle_and_zeroes_night():
    """The selected model sees the actual updating cycle and clips local nighttime."""
    from bess_pv_trading.config import load_config
    from bess_pv_trading.data import night_window_mask

    starts = pd.date_range('2026-11-05', periods=96, freq='15min', tz='Europe/Rome').tz_convert('UTC')
    vintage = pd.Timestamp('2026-11-04T12:00Z')
    cutoff = pd.Timestamp('2026-11-04T19:00Z')
    frame = pd.DataFrame({'pv_ac_mw': .1, 'solar_zenith_deg': 40.,
        'forecast_ghi': 200., 'forecast_bhi': 100., 'temperature_c': 20.,
        'wind_u_10m': 1., 'wind_v_10m': 2., 'dew_point_c': 12.,
        'cloud_total_pct': 50., 'cloud_low_pct': 20., 'cloud_mid_pct': 20.,
        'cloud_high_pct': 10., 'precipitation_mm': 0., 'snow_water_equivalent_m': 0.,
        'visibility_km': 10., 'wind_u_100m': 2., 'wind_v_100m': 3.}, index=starts)
    bundle = SimpleNamespace(values=frame, farm_id=901, members=['one'],
        available_at=vintage + pd.Timedelta(hours=6), forecast_vintage=vintage)
    replay = ContinuousReplay.__new__(ContinuousReplay)
    replay.perfect = False
    replay.plant = SimpleNamespace(farm_id=901)
    replay.config = load_config('configs/standard.yaml', overrides={'farm_id': 901})
    replay.config['pv']['forecast_replacement']['enabled'] = True
    replay.prepared = {'forecast_provider': lambda cutoff, requested: bundle,
                       'calendar': SimpleNamespace(timezone='Europe/Rome')}
    values, provenance = replay._forecast(cutoff, starts, 'DA')
    night = night_window_mask(starts, replay.config['policy']['forecast_pv_night'])
    assert np.isfinite(values).all() and np.all(values[night] == 0.)
    assert np.any(values[~night] > 0.)
    assert provenance['pv_forecast_model_kind'] == 'reviewed_hpo'
    assert provenance['pv_forecast_vintage_utc'] == vintage.isoformat()
    # The shared day-ahead and intraday forecast loader uses the same model.
    from bess_pv_trading.data.contracts import ForecastBundle
    from bess_pv_trading.simulation.gme_oracle import _apply_reviewed_forecast
    weather = ForecastBundle(901, vintage, bundle.available_at, frame.copy(), {},
                             provenance={'decision_cutoff_utc': cutoff.isoformat()})
    converted = _apply_reviewed_forecast(replay.config, replay.prepared['calendar'], weather, cutoff, starts)
    np.testing.assert_allclose(converted.values.pv_ac_mw, values)
    assert converted.provenance['reviewed_hpo_model'] == 'reviewed_ghi_bilstm_901'
    assert converted.provenance['reviewed_hpo_training_weather_cycle'] == 'preceding_day_00_utc'
    assert converted.provenance['reviewed_hpo_artifact_sha256']
    # The model cannot be used before its documented publication without an override.
    from dataclasses import replace
    earlier = pd.Timestamp('2026-11-04T02:00Z')
    prior = replace(weather, forecast_vintage=pd.Timestamp('2026-11-03T12:00Z'),
                    available_at=pd.Timestamp('2026-11-03T18:00Z'))
    unavailable = _apply_reviewed_forecast(replay.config, replay.prepared['calendar'], prior, earlier, starts)
    assert unavailable.provenance['reviewed_hpo_status'] == 'fallback_artifact_not_yet_available'


def test_scenario_members_are_interpolated_and_packed_independently():
    """A missing value in one trajectory must never borrow another member's weather."""
    from bess_pv_trading.forecasting.data.pv_neural_data import pack_sequences
    from bess_pv_trading.forecasting.training.pv_forecast_search import interpolate_forecast_gaps

    times = pd.date_range('2026-01-01T10:00Z', periods=3, freq='15min')
    frame = pd.DataFrame({
        'interval_start_utc': list(times) * 2,
        'delivery_day': ['2026-01-01'] * 6,
        'auction': ['DA'] * 6,
        'forecast_cycle_utc': [pd.Timestamp('2025-12-31T00:00Z')] * 6,
        '_scenario_member': ['1'] * 3 + ['2'] * 3,
        'forecast_ghi': [0., np.nan, 2., 100., 101., 102.],
        'measured_ac_mw': [np.nan] * 6,
        'eligible': [False] * 6,
        'daylight': [True] * 6,
        'split': ['prediction'] * 6,
    })
    interpolated, report = interpolate_forecast_gaps(frame)
    assert report['view_count'] == 2
    np.testing.assert_allclose(interpolated.forecast_ghi, [0., 1., 2., 100., 101., 102.])
    packed = pack_sequences(interpolated, interpolated[['forecast_ghi']].to_numpy(), 1.0)
    assert packed['indices'].shape == (2, 3)
    np.testing.assert_array_equal(packed['indices'], [[0, 1, 2], [3, 4, 5]])


def test_reviewed_ensemble_predicts_each_member_independently(monkeypatch):
    """The shared deterministic model must not mix ensemble weather trajectories."""
    import xarray as xr
    from bess_pv_trading.config import load_config
    from bess_pv_trading.simulation.continuous_inputs import _prepare_pv_ensembles

    config = load_config('configs/standard.yaml', overrides={'farm_id': 1})
    config['pv']['forecast_replacement']['enabled'] = True
    config['continuous_optimization']['pv_ensemble'].update(zarr='unused', member_count=2)
    day = '2026-11-06'
    starts = pd.date_range('2026-11-06T09:00Z', periods=4, freq='15min')
    cycle = pd.Timestamp('2026-11-05T00:00Z')
    weather = SimpleNamespace(available_at=cycle + pd.Timedelta(hours=6),
        values=pd.DataFrame({'member': ['a'] * 4 + ['b'] * 4,
            'forecast_ghi': [100.] * 4 + [200.] * 4}, index=starts.append(starts)))
    archive = xr.Dataset(coords={'member': ['a', 'b']})
    monkeypatch.setattr('xarray.open_zarr', lambda *args, **kwargs: archive)
    monkeypatch.setattr('bess_pv_trading.forecasting.data.pv_probabilistic_data.align_cycle',
                        lambda *args, **kwargs: weather)
    monkeypatch.setattr('bess_pv_trading.data.load_metadata', lambda *args, **kwargs: {1: object()})
    seen = []
    def predict(frame, *args, **kwargs):
        """Return a distinct value for each independent member."""
        seen.append((frame['member'].iloc[0], frame.forecast_cycle_utc.iloc[0]))
        return frame.forecast_ghi.to_numpy() / 1000
    monkeypatch.setattr('bess_pv_trading.forecasting.inference.pv_reviewed_hpo.predict_reviewed_hpo', predict)
    days = {day: {'grid': SimpleNamespace(starts=starts),
                  'schedule': [SimpleNamespace(submission_at=cycle + pd.Timedelta(hours=10))]}}
    _prepare_pv_ensembles(config, days)
    assert seen == [('a', cycle), ('b', cycle)]
    np.testing.assert_allclose(days[day]['ensemble_pv'], [[.1] * 4, [.2] * 4])


@pytest.mark.parametrize("context", ["DA", "realtime_mpc"])
def test_replay_uses_seasonal_night_policy_for_neural_forecasts(tmp_path, monkeypatch, context):
    from bess_pv_trading.config import load_config

    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    starts = pd.DatetimeIndex(pd.to_datetime(['2026-01-10 06:30', '2026-01-10 06:45',
                                             '2026-01-10 18:15', '2026-01-10 18:30']))
    starts = starts.tz_localize('Europe/Rome').tz_convert('UTC')
    vintage = pd.Timestamp('2026-01-09T00:00Z')
    frame = pd.DataFrame({'pv_ac_mw': [0.1] * 4, 'solar_zenith_deg': [40.] * 4}, index=starts)
    bundle = SimpleNamespace(values=frame, farm_id=345, members=['one'],
        available_at=vintage, forecast_vintage=vintage)
    replay = ContinuousReplay.__new__(ContinuousReplay)
    replay.perfect = False
    replay.plant = SimpleNamespace(farm_id=345)
    replay.config = load_config('configs/standard.yaml', overrides={'farm_id': 345})
    replay.config['pv']['forecast_replacement'].update(enabled=True, allow_retrospective_artifacts=True)
    replay.config['pv']['forecast_replacement']['models_by_farm']['345']['artifact_path'] = str(manifest)
    replay.config['pv']['forecast_replacement']['models_by_farm']['345']['kind'] = 'weather_bundle'
    replay.prepared = {'forecast_provider': lambda cutoff, requested: bundle,
                       'calendar': SimpleNamespace(timezone='Europe/Rome')}
    captured = {}

    def predict(model_frame, folder, *, farm_id, night_policy):
        captured['policy'] = night_policy
        assert model_frame.auction.eq("DA").all()
        return np.ones(len(model_frame))

    monkeypatch.setattr('bess_pv_trading.forecasting.inference.pv_forecast_bundle.predict_bundle', predict)
    values, provenance = replay._forecast(pd.Timestamp('2026-01-09T08:00Z'), starts, context)
    np.testing.assert_array_equal(values, [0., 1., 1., 0.])
    assert captured['policy'] == replay.config['policy']['realized_pv_night']
    assert provenance['pv_forecast_night_policy'] == 'configured_monthly'


def test_farm_345_replay_uses_unfitted_pvlib_and_seasonal_windows():
    from bess_pv_trading.config import load_config

    paths = ['configs/full_experiments/continuous_2026_01_pv_deterministic_four_farms.yaml',
             'configs/full_experiments/continuous_2026_01_pv_scenarios_four_farms.yaml']
    for path in paths:
        config = load_config(path, overrides={'farm_id': 345})
        pv = config['pv']
        assert pv['forecast_replacement']['models_by_farm']['345'] is None
        assert pv['output']['method'] == 'explicit_efficiency'
        assert pv['output']['calibrated_coefficient'] is None
        assert pv['temperature']['u0'] == 25.0 and pv['temperature']['u1'] == 6.84
        assert config['policy']['forecast_pv_night']['monthly_windows']['01'] == {
            'start': '18:30', 'end': '06:45'}
        assert pv['location']['latitude'] == 44.4056
        assert pv['location']['longitude'] == 8.9463


@pytest.mark.parametrize('provided_lead', [False, True])
def test_weather_bundle_uses_aligned_or_midpoint_lead(tmp_path, monkeypatch, provided_lead):
    """Fallback lead uses quarter midpoints; existing adapter leads remain intact."""
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    starts = pd.date_range('2026-09-02T10:00Z', periods=2, freq='15min')
    vintage = pd.Timestamp('2026-09-02T00:00Z')
    frame = pd.DataFrame({'pv_ac_mw': [0.2, 0.3], 'solar_zenith_deg': [40., 39.]}, index=starts)
    if provided_lead:
        frame['lead_hours'] = [10.125, 10.375]
    bundle = SimpleNamespace(values=frame, farm_id=901, members=['one'],
                             available_at=vintage + pd.Timedelta(hours=6), forecast_vintage=vintage)
    replay = ContinuousReplay.__new__(ContinuousReplay)
    replay.perfect = False
    replay.plant = SimpleNamespace(farm_id=901)
    replay.config = {'source': {'historical_interval_minutes': 15}, 'pv': {'forecast_replacement': {
        'enabled': True, 'models_by_farm': {'901': {'kind': 'weather_bundle', 'name': 'pilot',
        'artifact_path': str(manifest), 'available_at_utc': '2026-09-01T00:00Z'}}}}}
    replay.prepared = {'forecast_provider': lambda cutoff, requested: bundle,
                       'calendar': SimpleNamespace(timezone='Europe/Rome')}
    captured = {}

    def predict(model_frame, folder, *, farm_id):
        captured.update(frame=model_frame, folder=folder, farm_id=farm_id)
        return np.array([0.4, 0.5])

    monkeypatch.setattr('bess_pv_trading.forecasting.inference.pv_forecast_bundle.predict_bundle', predict)
    values, provenance = replay._forecast(pd.Timestamp('2026-09-02T09:00Z'), starts, 'IDA3')
    np.testing.assert_allclose(values, [0.4, 0.5])
    np.testing.assert_allclose(captured['frame'].lead_hours, [10.125, 10.375])
    assert captured['farm_id'] == 901
    assert captured['folder'] == tmp_path
    assert provenance['pv_forecast_model_kind'] == 'weather_bundle'
