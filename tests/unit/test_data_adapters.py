"""Named-dimension alignment and single-vintage interval semantics."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.data import ForecastBundle, align_forecast, forecast_valid_times, load_metadata, load_observations, select_forecast_vintage
from bess_pv_trading.forecasting.pv import local_delivery_starts


def fixture_store():
    farms = [345, 1]
    time = pd.date_range("2025-01-01", periods=4, freq="15min").to_numpy()
    cycles = pd.to_datetime(["2025-01-01T00:00", "2025-01-01T06:00"]).to_numpy()
    lead = np.arange(13)
    dataset = xr.Dataset({"power": (("time", "farm_id"), np.arange(8).reshape(4, 2)), "ghi": (("farm_id", "time"), np.arange(8).reshape(2, 4)), "shortwave": (("farm_id", "cycle_time", "lead_time"), np.ones((2, 2, 13)) * 400)}, coords={"farm_id": farms, "time": time, "cycle_time": cycles, "lead_time": lead})
    dataset.shortwave.loc[{"lead_time": 0}] = np.nan
    return dataset


def test_native_transpose_and_metadata_label_join(tmp_path):
    dataset = fixture_store()
    bundle = load_observations(dataset, [1, 345])
    assert bundle.frame.loc[(bundle.frame.farm_id == 1) & (bundle.frame.variable == "power")].raw_value.tolist() == [1, 3, 5, 7]
    assert bundle.frame.loc[(bundle.frame.farm_id == 345) & (bundle.frame.variable == "ghi")].raw_value.tolist() == [0, 1, 2, 3]
    path = tmp_path / "metadata.parquet"
    pd.DataFrame({"farm_id": [1, 345], "zone": "NORD", "total_power": [953.28, 1524.6], "tracking_type_id": 0, "tilt": 15, "azimuth": 0, "elevation_m": 25}).to_parquet(path)
    metadata = load_metadata(path, [345, 1])
    assert metadata[345].dc_capacity_mw == pytest.approx(1.5246)
    assert metadata[1].dc_capacity_mw == pytest.approx(0.95328)


def test_valid_time_numeric_and_timedelta_equal():
    cycles = ["2025-01-01T00:00Z", "2025-01-01T06:00Z"]
    numeric = forecast_valid_times(cycles, np.array([0, 1, 48]))
    delta = forecast_valid_times(cycles, np.array([0, 1, 48]).astype("timedelta64[h]"))
    assert np.array_equal(numeric, delta)
    assert numeric[1, 1] == np.datetime64("2025-01-01T07:00")


def test_latency_and_explicit_availability_override_and_no_ensemble():
    dataset = fixture_store()
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


def test_local_delivery_day_dst_counts():
    assert len(local_delivery_starts("2025-03-30")) == 92
    assert len(local_delivery_starts("2025-10-26")) == 100
    assert len(local_delivery_starts("2025-10-01")) == 96


def test_instantaneous_alignment_independent_of_datetime_storage_resolution():
    source_index = pd.date_range("2025-01-01T10:00Z", periods=3, freq="h").as_unit("ns")
    bundle = ForecastBundle(1, source_index[0], source_index[0], pd.DataFrame({"temperature_c": [0, 10, 20]}, index=source_index), {"temperature_c": "C"})
    targets = pd.date_range("2025-01-01T11:00Z", periods=4, freq="15min").as_unit("us")
    result = align_forecast(bundle, targets, {"temperature_c": "instantaneous"})
    assert result.values.temperature_c.tolist() == pytest.approx([11.25, 13.75, 16.25, 18.75])


def test_review_decisions_preserved_without_source_file_changes(tmp_path):
    from bess_pv_trading.data import apply_review_exclusions, prepare_observations
    dataset = fixture_store()
    original = dataset.copy(deep=True)
    bundle = load_observations(dataset, [1])
    review = tmp_path / "review.csv"
    text = "farm_id,start_utc,end_utc,decision,notes\n1,2025-01-01T00:15Z,2025-01-01T00:30Z,exclude,manual decision\n"
    review.write_text(text)
    treated = prepare_observations(apply_review_exclusions(bundle, review), {"night": {"enabled": False}})
    assert treated.frame.review_excluded.sum() == 4
    assert (treated.frame.loc[treated.frame.review_excluded, "review_decision"] == "exclude").all()
    assert review.read_text() == text
    xr.testing.assert_identical(dataset, original)


def test_source_config_rejects_invalid_signed_allowlists_without_reads():
    from bess_pv_trading.config import load_config
    from bess_pv_trading.data import validate_source_config
    config = load_config("configs/farms_nord.yaml")
    config["inputs"]["zarr"] = "/missing/never-opened.zarr"
    assert validate_source_config(config)["source_configured"]
    config["policy"]["nonnegative_forecasts"].append("temperature_c")
    with pytest.raises(ValueError, match="signed"):
        validate_source_config(config)
