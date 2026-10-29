"""Check review-plot prediction weighting and delivery-cycle alignment."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.forecasting.plotting import pv_review_predictions as module


def test_historical_mean_requires_all_four_views(monkeypatch):
    """Do not silently replace a missing weather view with unequal weights."""
    times = pd.to_datetime(['2025-12-01T10:00Z'] * 4 + ['2025-12-01T10:15Z'] * 4)
    frame = pd.DataFrame({'interval_start_utc': times, 'ghi_w_m2': 100.,
                          'diffuse_fraction': .9, 'temperature_c': 5., 'wind_speed_m_s': 1.})
    solar = lambda t, *args, **kwargs: pd.DataFrame({'apparent_zenith': np.full(len(t), 50.),
                                                     'azimuth': np.full(len(t), 180.)})
    monkeypatch.setattr(module.pvlib.solarposition, 'get_solarposition', solar)
    monkeypatch.setattr(module.pvlib.irradiance, 'dni', lambda *args, **kwargs: np.zeros(8))
    monkeypatch.setattr(module, 'parameter_defaults', lambda settings: {})
    class FakeModel:
        def __init__(self, *args):
            pass
        def evaluate(self, params):
            return np.array([1., 3., 2., 4., 1., 2., 3., np.nan])
    monkeypatch.setattr(module, 'PhysicalPVModel', FakeModel)
    settings = {'location': {'latitude': 45., 'longitude': 9.}, 'low_sun_zenith_deg': 85.,
                'dni_limit_action': 'cap_and_flag', 'dni_limit_w_m2': 1400.}
    result = module.historical_predictions(frame, SimpleNamespace(elevation_m=25.), settings, {})
    assert result.historical_ghi_prediction_mw.iloc[0] == 2.5
    assert result.historical_weather_views.iloc[1] == 3
    assert np.isnan(result.historical_ghi_prediction_mw.iloc[1])


def test_forecast_uses_preceding_day_and_dst_delivery_grid(monkeypatch):
    """Spring DST has 92 quarters, all tied to the intended preceding 00 UTC cycle."""
    seen = []
    def align(store, farm_id, cycle, starts, config, ensemble):
        seen.append(cycle)
        return SimpleNamespace(available_at=cycle + pd.Timedelta(hours=6), starts=starts)
    def convert(bundle, farm, settings, night):
        return SimpleNamespace(values=pd.DataFrame({'pv_ac_mw': np.ones(len(bundle.starts))}))
    monkeypatch.setattr(module, 'align_cycle', align)
    monkeypatch.setattr(module, 'forecast_pv', convert)
    result = module.forecast_predictions(None, SimpleNamespace(farm_id=1), {}, {'policy': {'night': {}}}, ['2026-03-29'])
    assert seen == [pd.Timestamp('2026-03-28T00:00Z')]
    assert len(result) == 92
    assert result.index.dtype == pd.DatetimeIndex([], tz='UTC').as_unit('ns').dtype
