"""PV forecast inputs preserve publication timing, energy and genuine gaps."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.data import ForecastBundle, align_forecast, select_forecast_vintage


def forecast_store():
    """Provide two forecast cycles with no accumulated lead-zero radiation."""
    cycles = pd.to_datetime(["2025-01-01T00:00", "2025-01-01T06:00"]).to_numpy()
    dataset = xr.Dataset(
        {"shortwave": (("farm_id", "cycle_time", "lead_time"), np.full((1, 2, 13), 400.0))},
        coords={"farm_id": [1], "cycle_time": cycles, "lead_time": np.arange(13)})
    dataset.shortwave.loc[{"lead_time": 0}] = np.nan
    return dataset


def test_latency_and_explicit_availability_override_and_no_ensemble():
    dataset = forecast_store()
    mapping = {"forecast_ghi": "shortwave"}
    with pytest.raises(ValueError, match="No forecast"):
        select_forecast_vintage(dataset, 1, "2025-01-01T03:59Z", mapping)
    first = select_forecast_vintage(dataset, 1, "2025-01-01T09:59Z", mapping)
    assert first.forecast_vintage == pd.Timestamp("2025-01-01T00:00Z")
    second = select_forecast_vintage(dataset, 1, "2025-01-01T10:00Z", mapping)
    assert second.forecast_vintage == pd.Timestamp("2025-01-01T06:00Z")
    assert second.members == ("deterministic",) and np.isnan(second.values.forecast_ghi.iloc[0])
    dataset["publication"] = ("cycle_time", pd.to_datetime(["2025-01-01T02:00", "2025-01-01T07:00"]).to_numpy())
    explicit = select_forecast_vintage(dataset, 1, "2025-01-01T08:00Z", mapping, explicit_availability_variable="publication")
    assert explicit.forecast_vintage == pd.Timestamp("2025-01-01T06:00Z")
    assert explicit.available_at == pd.Timestamp("2025-01-01T07:00Z")


def test_end_labeled_radiation_precip_conservation_and_instantaneous_same_cycle():
    index = pd.date_range("2025-01-01T10:00Z", periods=3, freq="h")
    frame = pd.DataFrame({"forecast_ghi": [np.nan, 200, 400], "precipitation_mm": [np.nan, 8, 12], "temperature_c": [-10, 0, 10]}, index=index)
    bundle = ForecastBundle(1, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-01T04:00Z"), frame, {"forecast_ghi": "W/m2", "precipitation_mm": "mm", "temperature_c": "C"})
    starts = pd.date_range("2025-01-01T11:00Z", periods=4, freq="15min")
    aligned = align_forecast(bundle, starts, {"forecast_ghi": "hourly_mean", "precipitation_mm": "hourly_accumulation", "temperature_c": "instantaneous"})
    assert aligned.values.forecast_ghi.tolist() == [400] * 4
    assert aligned.values.forecast_ghi.sum() * 0.25 == 400
    assert aligned.values.precipitation_mm.sum() == 12
    assert aligned.values.temperature_c.tolist() == pytest.approx([1.25, 3.75, 6.25, 8.75])
    assert (aligned.values.forecast_ghi_source_start == pd.Timestamp("2025-01-01T11:00Z")).all()
    assert (aligned.values.forecast_ghi_source_end == pd.Timestamp("2025-01-01T12:00Z")).all()
    missing = align_forecast(bundle, pd.date_range("2025-01-01T09:00Z", periods=4, freq="15min"), {"forecast_ghi": "hourly_mean", "precipitation_mm": "hourly_accumulation"}, ["forecast_ghi"])
    assert missing.values.forecast_ghi.isna().all()


def test_structural_instantaneous_nan_is_not_bridged():
    index = pd.date_range("2025-01-01T10:00Z", periods=3, freq="h")
    bundle = ForecastBundle(1, index[0], index[0], pd.DataFrame({"temperature_c": [0, np.nan, 10]}, index=index), {"temperature_c": "C"})
    result = align_forecast(bundle, pd.date_range(index[0], periods=8, freq="15min"), {"temperature_c": "instantaneous"})
    assert result.values.temperature_c.isna().all()
