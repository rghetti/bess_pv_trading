"""Portable pilot inference preserves the forecast adapter's timing semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.simulation.continuous import ContinuousReplay


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


def test_replay_uses_seasonal_night_policy_for_neural_forecasts(tmp_path, monkeypatch):
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
    replay.prepared = {'forecast_provider': lambda cutoff, requested: bundle,
                       'calendar': SimpleNamespace(timezone='Europe/Rome')}
    captured = {}

    def predict(model_frame, folder, *, farm_id, night_policy):
        captured['policy'] = night_policy
        return np.ones(len(model_frame))

    monkeypatch.setattr('bess_pv_trading.forecasting.inference.pv_forecast_bundle.predict_bundle', predict)
    values, provenance = replay._forecast(pd.Timestamp('2026-01-09T08:00Z'), starts, 'DA')
    np.testing.assert_array_equal(values, [0., 1., 1., 0.])
    assert captured['policy'] == replay.config['policy']['realized_pv_night']
    assert provenance['pv_forecast_night_policy'] == 'configured_monthly'


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
