"""Seasonal measurement clipping stays separate from weather forecast policy."""
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.gme_oracle import _clip_realized_pv_night


def test_monthly_night_windows_clip_measurements_and_record_provenance():
    config = load_config("configs/standard.yaml")
    curated = load_config("configs/pv_experiments/realized_irradiance_four_cycles.yaml")
    assert config["policy"]["realized_pv_night"] == curated["pv_realized_irradiance_search"]["night_policy"]
    starts = pd.DatetimeIndex(pd.to_datetime([
        "2026-01-10 06:30", "2026-01-10 06:45", "2026-01-10 18:15",
        "2026-01-10 18:30", "2026-02-10 19:00", "2026-02-10 19:15",
    ])).tz_localize("Europe/Rome").tz_convert("UTC")
    grid = SimpleNamespace(starts=starts, interval_ids=tuple(str(i) for i in range(len(starts))),
                           delta_hours=np.full(len(starts), 0.25))
    actual = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    clipped, provenance = _clip_realized_pv_night(actual, grid, config["policy"]["realized_pv_night"])
    np.testing.assert_allclose(clipped, [0., 0.2, 0.3, 0., 0.5, 0.])
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert provenance["zeroed_intervals"] == 3
    assert provenance["zeroed_interval_ids"] == ["0", "3", "5"]
    np.testing.assert_allclose(provenance["zeroed_energy_mwh"], 1.1 * 0.25)


def test_disabled_policy_leaves_pv_unchanged():
    grid = SimpleNamespace(starts=pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
                           interval_ids=("a", "b"), delta_hours=np.full(2, 0.25))
    values, provenance = _clip_realized_pv_night([0.2, 0.3], grid, {"enabled": False})
    np.testing.assert_allclose(values, [0.2, 0.3])
    assert provenance["zeroed_intervals"] == 0


def test_replay_distinguishes_missing_night_from_daytime_fallback(monkeypatch):
    from bess_pv_trading.simulation import gme_oracle

    @dataclass
    class Metadata:
        ac_capacity_mw: float = 1.0

    starts = pd.date_range("2026-01-10 05:30", periods=3, freq="15min", tz="Europe/Rome").tz_convert("UTC")
    # The first two starts are inside the January night window; the last is daytime.
    starts = pd.DatetimeIndex([starts[0], starts[1], pd.Timestamp("2026-01-10 06:45", tz="Europe/Rome").tz_convert("UTC")])
    grid = SimpleNamespace(starts=starts, interval_ids=("a", "b", "c"), delta_hours=np.full(3, 0.25))
    config = load_config("configs/standard.yaml", overrides={"farm_id": 1,
        "forecast": {"maximum_missing_realized_intervals": 1}})
    bundle = SimpleNamespace(values=pd.DataFrame({"pv_ac_mw": [0.4, 0.4, 0.7]}),
        forecast_vintage=starts[0], available_at=starts[0], members=("deterministic",),
        member_origin="fixture", provenance={"forecast_missing_filled_night_zero_intervals": 0})
    observed = SimpleNamespace(frame=pd.DataFrame({"variable": ["power"] * 3,
        "time": starts, "raw_value": [np.nan, 100., np.nan]}), provenance={"power_basis": "AC"})
    monkeypatch.setattr(gme_oracle, "open_store", lambda config: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(gme_oracle, "load_metadata", lambda *args: {1: Metadata()})
    monkeypatch.setattr(gme_oracle, "_forecast_pv_at_cutoff", lambda *args: bundle)
    monkeypatch.setattr(gme_oracle, "load_observations", lambda *args, **kwargs: observed)

    _, actual, _, provenance = gme_oracle._load_pv(config, None, grid, decision_cutoff=starts[0])
    np.testing.assert_allclose(actual, [0., 0., 0.7])
    assert provenance["realized_power_raw_missing_intervals"] == 2
    assert provenance["realized_power_missing_fallback_intervals"] == 1
    assert provenance["realized_pv_night_clipping"]["missing_zeroed_interval_ids"] == ["a"]
    assert provenance["realized_pv_night_clipping"]["zeroed_interval_ids"] == ["b"]
