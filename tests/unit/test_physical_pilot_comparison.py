"""Protect matching and weighting in the physical/pilot comparison."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    'physical_pilot_comparison', Path(__file__).resolve().parents[2] / 'scripts/compare_physical_pilot.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_score_balances_day_auction_views():
    """A long view must not outweigh a shorter view in the aggregate."""
    frame = pd.DataFrame({'delivery_day': ['2026-05-01'] * 4, 'auction': ['DA', 'DA', 'DA', 'IDA3'],
                          'measured_ac_mw': [0.] * 4, 'neural_mw': [1., 1., 1., 3.],
                          'fitted_pvlib_mw': [2., 2., 2., 4.]})
    result = comparison.score(frame, 1.)
    assert result[0]['rmse_mw'] == pytest.approx(np.sqrt(5))
    assert result[1]['rmse_mw'] == pytest.approx(np.sqrt(10))


def test_predict_uses_actual_cycle_and_rejects_late_vintage(monkeypatch):
    """Keep each observation attached to its own available forecast cycle."""
    times = pd.to_datetime(['2026-05-01 00:00Z', '2026-05-01 06:00Z'])
    frame = pd.DataFrame({'forecast_cycle_utc': times, 'forecast_available_at_utc': times + pd.Timedelta(hours=6),
                          'decision_cutoff_utc': times + pd.Timedelta(hours=7), 'interval_start_utc': times})
    calls = []
    def fake_predict(bundle, farm, settings, policy):
        calls.append(bundle.forecast_vintage)
        return SimpleNamespace(values=pd.DataFrame({'pv_ac_mw': [float(bundle.forecast_vintage.hour)]}))
    monkeypatch.setattr(comparison, 'forecast_pv', fake_predict)
    assert comparison.predict_physical(frame, SimpleNamespace(farm_id=1), {}, {}).tolist() == [0., 6.]
    assert calls == list(times)
    frame.loc[1, 'decision_cutoff_utc'] = times[1]
    with pytest.raises(ValueError, match='late availability'):
        comparison.predict_physical(frame, SimpleNamespace(farm_id=1), {}, {})
