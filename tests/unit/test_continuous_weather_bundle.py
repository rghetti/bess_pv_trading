"""Portable pilot inference preserves the forecast adapter's timing semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.simulation.continuous import ContinuousReplay


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
