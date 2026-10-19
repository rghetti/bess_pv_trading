"""Use only published weather cycles and retain their auction/interval provenance."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.config import load_config
from bess_pv_trading.data import ForecastBundle
from bess_pv_trading.simulation import gme_oracle, synthetic
from battery_case_config import case_config


def forecast_case(monkeypatch):
    """Supply distinct weather for each configured cycle; mock only PV conversion."""
    case = case_config("rolling_pv_forecast")
    cycles = pd.to_datetime(case["cycles"], utc=True).tz_localize(None)
    shape = (1, len(cycles), case["lead_count"])
    dataset = xr.Dataset({
        "shortwave": (("farm_id", "cycle_time", "lead_time"),
            np.broadcast_to(np.asarray(case["irradiance_by_cycle"])[None, :, None], shape).copy()),
        "temperature": (("farm_id", "cycle_time", "lead_time"),
            np.broadcast_to(np.asarray(case["temperature_by_cycle"])[None, :, None], shape).copy()),
    }, coords={"farm_id": [case["farm_id"]], "cycle_time": cycles,
               "lead_time": np.arange(case["lead_count"])})
    config = {"forecast": {"farm_id": case["farm_id"],
        "missing_forecast_policy": "zero_only_when_policy_marks_night"},
        "source": {"availability_lag_hours": case["availability_lag_hours"]},
        "forecast_mapping": {"forecast_ghi": "shortwave", "temperature_c": "temperature"},
        "forecast_semantics": {"default": "hourly_mean"},
        "policy": {"night": {"enabled": False}}, "pv": {},
        "inputs": {"metadata_parquet": "unused.parquet"}}
    seen = []
    def convert(weather, metadata, settings, night):
        """Capture selected weather, then provide a simple deterministic PV signal."""
        seen.append(weather.values.copy())
        weather.values["pv_ac_mw"] = weather.values.forecast_ghi / case["irradiance_to_mw_divisor"]
        weather.units["pv_ac_mw"] = "MW_AC"
        return weather
    monkeypatch.setattr(gme_oracle, "open_store", lambda config: dataset.copy(deep=True))
    monkeypatch.setattr(gme_oracle, "load_metadata", lambda path, farms: {case["farm_id"]: object()})
    monkeypatch.setattr(gme_oracle, "forecast_pv", convert)
    def forbidden_observations(*args, **kwargs):
        raise AssertionError("Forecast updates must not read realized PV")
    monkeypatch.setattr(gme_oracle, "load_observations", forbidden_observations)
    starts = pd.date_range(case["remaining_start"], periods=case["interval_count"],
                          freq=pd.Timedelta(minutes=case["interval_minutes"]))
    return case, config, dataset, starts, seen


@pytest.mark.parametrize("decision", case_config("rolling_pv_forecast")["decisions"],
                         ids=lambda decision: decision["auction"])
def test_latest_published_cycle_drives_remaining_pv(monkeypatch, decision):
    case, config, _, starts, seen = forecast_case(monkeypatch)
    result = gme_oracle._load_pv_forecast(config, SimpleNamespace(timezone="Europe/Rome"),
                                        pd.Timestamp(decision["cutoff"]), starts)
    index = decision["expected_cycle_index"]
    assert result.forecast_vintage == pd.Timestamp(case["cycles"][index])
    assert result.available_at <= pd.Timestamp(decision["cutoff"])
    assert result.values.index.equals(starts)
    np.testing.assert_allclose(seen[-1].temperature_c, case["temperature_by_cycle"][index])
    np.testing.assert_allclose(result.values.pv_ac_mw,
        case["irradiance_by_cycle"][index] / case["irradiance_to_mw_divisor"])


def test_latest_cycle_daytime_gap_fails_without_using_an_older_cycle(monkeypatch):
    case, config, dataset, starts, _ = forecast_case(monkeypatch)
    decision = next(row for row in case["decisions"] if row["auction"] == "IDA3")
    dataset.shortwave.values[:, decision["expected_cycle_index"], :] = np.nan
    with pytest.raises(ValueError, match="unresolved intervals"):
        gme_oracle._load_pv_forecast(config, SimpleNamespace(timezone="Europe/Rome"),
                                    pd.Timestamp(decision["cutoff"]), starts)


def test_pv_plot_groups_shared_cycle_once_and_keeps_all_distinct_cycles():
    case = case_config("rolling_pv_forecast")
    rows = []
    starts = pd.date_range(case["remaining_start"], periods=case["interval_count"],
                          freq=pd.Timedelta(minutes=case["interval_minutes"]))
    for decision in case["decisions"]:
        if decision["auction"] == "publication_boundary":
            continue
        index = decision["expected_cycle_index"]
        rows.extend({"auction": decision["auction"], "interval_start_utc": str(start),
            "pv_forecast_vintage_utc": case["cycles"][index],
            "pv_ac_forecast_mw": case["irradiance_by_cycle"][index] / case["irradiance_to_mw_divisor"]}
            for start in starts)
    curves = gme_oracle._pv_forecast_curves(pd.DataFrame(rows), "Europe/Rome")
    expected_cycles = {row["expected_cycle_index"] for row in case["decisions"] if row["auction"] != "publication_boundary"}
    assert len(curves) == len(expected_cycles)
    assert "DA / IDA1" in curves[0][0]
    for _, frame in curves:
        assert not frame.interval_start_utc.duplicated().any()
        assert pd.DatetimeIndex(pd.to_datetime(frame.interval_start_utc, utc=True)).equals(starts)


def test_replay_rejects_an_unpublished_update_before_building_a_model(tmp_path, monkeypatch):
    case = case_config("rolling_pv_forecast")
    config = load_config(Path(__file__).resolve().parents[2] / "configs/full_experiments/synthetic_small.yaml")
    fixture = synthetic.synthetic_inputs(config)
    def unpublished(cutoff, starts):
        return ForecastBundle(fixture["plant"].farm_id, cutoff,
            cutoff + pd.Timedelta(minutes=case["future_publication_delay_minutes"]),
            pd.DataFrame({"pv_ac_mw": fixture["pv"]}, index=starts), {"pv_ac_mw": "MW_AC"})
    fixture["pv_forecast_provider"] = unpublished
    def forbidden_model(*args, **kwargs):
        raise AssertionError("Unpublished PV must not enter an optimizer")
    monkeypatch.setattr(synthetic, "build_model", forbidden_model)
    with pytest.raises(ValueError, match="unavailable"):
        synthetic._replay_strategy(config, fixture, "stochastic_integrated", tmp_path / "replay")
