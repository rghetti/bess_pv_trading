"""Archive metadata fixtures cover new months, DST, gaps and source ambiguity."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.data.ensemble_inventory import inventory_ensemble_sources, resolve_ensemble_sources


def archive(path, cycles, *, members=(0, 1, 2), farms=(1, 345), leads=None, corrupt_valid_time=False):
    """Create tiny weather payloads whose names deliberately carry no date semantics."""
    cycles = pd.DatetimeIndex(cycles)
    leads = pd.timedelta_range("0h", periods=49, freq="h") if leads is None else pd.TimedeltaIndex(leads)
    valid = cycles.to_numpy(dtype="datetime64[ns]")[:, None] + leads.to_numpy()[None, :]
    if corrupt_valid_time:
        valid[0, 0] += np.timedelta64(1, "h")
    shape = (len(farms), len(cycles), len(leads), len(members))
    dataset = xr.Dataset({
        "shortwave_radiation": (("farm_id", "cycle_time", "lead_time", "member"), np.ones(shape)),
        "valid_time": (("cycle_time", "lead_time"), valid),
    }, coords={"farm_id": list(farms), "cycle_time": cycles, "lead_time": leads, "member": list(members)})
    dataset.to_zarr(path, consolidated=True, zarr_format=2)
    return path


def measured_grid(days):
    boundaries = []
    for day in days:
        start = pd.Timestamp(day).tz_localize("Europe/Rome")
        stop = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Rome")
        boundaries.extend(pd.date_range(start, stop, inclusive="left", freq="15min"))
    return pd.DatetimeIndex(boundaries).tz_convert("UTC")


def cutoffs(days):
    return {day: (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize("UTC") + pd.Timedelta(hours=11)
            for day in days}


def test_multiple_archives_resolve_by_stored_cycles_for_any_month_and_dynamic_members(tmp_path):
    early = archive(tmp_path / "source_alpha.zarr", ["2026-01-01", "2026-02-01"], members=(0, 1, 2))
    late = archive(tmp_path / "source_beta.zarr", ["2026-11-01", "2026-12-30"], members=(10, 11, 12, 13, 14))
    days = ["2026-01-02", "2026-02-02", "2026-11-02", "2026-12-31"]
    result = resolve_ensemble_sources([early, late], days, farm_ids=[1], first_cutoffs=cutoffs(days),
        measured_starts=measured_grid(days), required_variables=["shortwave_radiation"])
    assert list(result) == days
    assert result[days[0]]["source"] == str(early)
    assert result[days[-1]]["source"] == str(late)
    assert result[days[0]]["member_count"] == 3
    assert result[days[-1]]["member_count"] == 5
    assert result[days[-1]]["member_ids"] == ["10", "11", "12", "13", "14"]
    assert all(row["availability_cutoff_scope"] == "configured_first_decision" for row in result.values())


def test_dst_days_use_92_and_100_real_quarter_hour_products(tmp_path):
    source = archive(tmp_path / "dst.zarr", ["2026-03-28", "2026-10-24"])
    days = ["2026-03-29", "2026-10-25"]
    result = resolve_ensemble_sources(source, days, farm_ids=[345], first_cutoffs=cutoffs(days), measured_starts=measured_grid(days))
    assert result[days[0]]["interval_count"] == 92
    assert result[days[1]]["interval_count"] == 100
    assert result[days[0]]["lead_coverage"] == {"unbracketed_midpoints": 0, "missing_hourly_endpoints": 0}


def test_missing_cycles_and_future_paths_are_reported_without_invented_months(tmp_path):
    source = archive(tmp_path / "available.zarr", ["2026-08-01"])
    future = tmp_path / "all_of_2026_not_arrived.zarr"
    days = ["2026-08-01", "2026-08-02", "2026-09-01"]
    inventory = inventory_ensemble_sources([source, future], delivery_days=days)
    assert inventory["eligible_delivery_days"] == ["2026-08-02"]
    assert inventory["missing_sources"] == [str(future)]
    assert inventory["days"][0]["forecast_vintage_utc"] == "2026-07-31T00:00:00+00:00"
    assert inventory["days"][0]["ineligibility_reasons"] == ["missing_preceding_day_cycle_or_requested_farm"]
    with pytest.raises(ValueError, match="2026-08-01.*2026-09-01"):
        resolve_ensemble_sources([source, future], days)


def test_default_day_discovery_uses_midnight_cycles_not_archive_filename(tmp_path):
    source = archive(tmp_path / "arbitrary_name.zarr", ["2026-01-31T00:00", "2026-01-31T06:00", "2026-02-01T00:00"])
    inventory = inventory_ensemble_sources(source)
    assert [row["delivery_day"] for row in inventory["days"]] == ["2026-02-01", "2026-02-02"]
    assert inventory["measurements"]["status"] == "not_checked"
    assert all(row["availability_cutoff_scope"] == "physical_delivery_start_only_not_auction_gate" for row in inventory["days"])


def test_overlapping_cycles_fail_until_source_precedence_is_explicit(tmp_path):
    first = archive(tmp_path / "original.zarr", ["2026-05-01"])
    second = archive(tmp_path / "replacement.zarr", ["2026-05-01"], members=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="Ambiguous ensemble cycle"):
        inventory_ensemble_sources([first, second], delivery_days=["2026-05-02"])
    early = resolve_ensemble_sources([first, second], ["2026-05-02"], overlap_policy="first")
    late = resolve_ensemble_sources([first, second], ["2026-05-02"], overlap_policy="last")
    assert early["2026-05-02"]["source"] == str(first)
    assert late["2026-05-02"]["source"] == str(second)
    assert late["2026-05-02"]["source_candidates"] == [str(first), str(second)]
    assert late["2026-05-02"]["member_count"] == 4


def test_disjoint_farm_archives_are_resolved_for_requested_farm(tmp_path):
    first = archive(tmp_path / "farm1.zarr", ["2026-05-01"], farms=(1,))
    second = archive(tmp_path / "farm345.zarr", ["2026-05-01"], farms=(345,))
    selected = resolve_ensemble_sources([first, second], ["2026-05-02"], farm_ids=[345])
    assert selected["2026-05-02"]["source"] == str(second)
    missing = inventory_ensemble_sources(first, delivery_days=["2026-05-02"], farm_ids=[901])
    assert not missing["eligible_delivery_days"]


def test_late_availability_and_incomplete_measurement_dates_are_explicit(tmp_path):
    source = archive(tmp_path / "archive.zarr", ["2026-01-01"])
    day = "2026-01-02"
    starts = measured_grid([day])
    inventory = inventory_ensemble_sources(source, delivery_days=[day], availability_lag_hours=12,
        first_cutoffs=cutoffs([day]), measured_starts=starts.delete(10))
    record = inventory["days"][0]
    assert record["ineligibility_reasons"] == ["forecast_available_after_first_decision", "incomplete_measurement_timestamp_coverage"]
    assert record["missing_measurement_intervals"] == 1
    assert not record["eligible"]


def test_lead_coverage_distinguishes_interpolation_from_required_hour_endpoints(tmp_path):
    leads = pd.timedelta_range("0h", periods=49, freq="h").delete(25)
    source = archive(tmp_path / "lead_gap.zarr", ["2026-01-01"], leads=leads)
    report = inventory_ensemble_sources(source, delivery_days=["2026-01-02"])
    record = report["days"][0]
    assert record["lead_coverage"]["unbracketed_midpoints"] == 0
    assert record["lead_coverage"]["missing_hourly_endpoints"] == 4
    assert not record["eligible"]
    instantaneous = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], alignment_modes=["instantaneous"])
    assert instantaneous["eligible_delivery_days"] == ["2026-01-02"]


def test_source_variables_are_checked_by_dimensions_without_loading_weather(tmp_path, monkeypatch):
    source = archive(tmp_path / "read_only_coordinates.zarr", ["2026-01-01"])
    original = xr.DataArray.values.fget

    def coordinate_values_only(array):
        if array.name == "shortwave_radiation":
            pytest.fail("Inventory loaded full weather data")
        return original(array)

    monkeypatch.setattr(xr.DataArray, "values", property(coordinate_values_only))
    inventory = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], required_variables=["shortwave_radiation"])
    assert inventory["eligible_delivery_days"] == ["2026-01-02"]
    missing = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], required_variables=["nonexistent_variable"])
    assert missing["days"][0]["missing_or_misaligned_variables"] == ["nonexistent_variable"]
    assert "weather payload excluded" in inventory["sources"][0]["checksum_scope"]
    assert len(inventory["sources"][0]["metadata_sha256"]) == 64
    assert len(inventory["sources"][0]["coordinate_sha256"]) == 64
    json.dumps(inventory, allow_nan=False)


def test_measurement_archive_overlap_checks_coordinates_not_power_values(tmp_path, monkeypatch):
    source = archive(tmp_path / "weather.zarr", ["2026-01-01"])
    starts = measured_grid(["2026-01-02"])
    measurements = tmp_path / "measurements.zarr"
    xr.Dataset({"power": (("time", "farm_id"), np.full((len(starts), 1), np.nan))},
        coords={"time": starts.tz_localize(None), "farm_id": [1]}).to_zarr(measurements, consolidated=True, zarr_format=2)
    original = xr.DataArray.values.fget

    def no_measured_values(array):
        if array.name == "power":
            pytest.fail("Inventory must not read historical power data")
        return original(array)

    monkeypatch.setattr(xr.DataArray, "values", property(no_measured_values))
    report = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], measurement_source=measurements, farm_ids=[1])
    assert report["eligible_delivery_days"] == ["2026-01-02"]
    assert "finite power and quality not checked" in report["measurements"]["coverage_scope"]
    missing_farm = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], measurement_source=measurements, farm_ids=[345])
    assert "requested_farm_missing_from_measurements" in missing_farm["days"][0]["ineligibility_reasons"]


def test_invalid_stored_clock_and_duplicate_source_configuration_fail(tmp_path):
    invalid = archive(tmp_path / "invalid_clock.zarr", ["2026-01-01"], corrupt_valid_time=True)
    with pytest.raises(ValueError, match="cycle plus lead"):
        inventory_ensemble_sources(invalid)
    with pytest.raises(ValueError, match="distinct"):
        inventory_ensemble_sources([invalid, invalid])


@pytest.mark.parametrize("settings", [
    {"availability_lag_hours": -1}, {"interval_minutes": 60}, {"overlap_policy": "guess"},
    {"alignment_modes": ["unknown"]}, {"farm_ids": []},
])
def test_invalid_inventory_settings_fail_before_archive_access(tmp_path, settings):
    with pytest.raises(ValueError):
        inventory_ensemble_sources(tmp_path / "not_arrived.zarr", **settings)


def test_missing_measurement_archive_and_naive_requested_cutoffs_are_not_accepted(tmp_path):
    source = archive(tmp_path / "archive.zarr", ["2026-01-01"])
    missing = inventory_ensemble_sources(source, delivery_days=["2026-01-02"], measurement_source=tmp_path / "missing.zarr")
    assert missing["days"][0]["ineligibility_reasons"] == ["measurement_source_missing"]
    with pytest.raises(ValueError, match="timezone aware"):
        inventory_ensemble_sources(source, delivery_days=["2026-01-02"], first_cutoffs={"2026-01-02": "2026-01-01T11:00:00"})
