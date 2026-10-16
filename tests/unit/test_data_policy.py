"""Executable fixtures for the narrow requested policy and its evidence clock."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.data import ObservationBundle, clip_negative_forecasts, night_window_mask, plateau_runs, prepare_observations


def observations(values, variable="power", start="2025-01-01T10:00Z", farm_id=1):
    times = pd.date_range(start, periods=len(values), freq="15min")
    return ObservationBundle(pd.DataFrame({"farm_id": farm_id, "variable": variable, "time": times, "interval_end": times + pd.Timedelta(minutes=15), "available_at": times + pd.Timedelta(minutes=15), "raw_value": values, "review_excluded": False}))


def policy(**updates):
    result = {"version": "test-v1", "plateau": {"minimum_run": 4, "atol_kw": 0.01, "rtol": 0.0001}, "night": {"enabled": False}, "interpolation": {"enabled": True, "variables": ["power", "ghi"], "maximum_gap_hours": None}}
    result.update(updates)
    return result


def test_three_samples_retained_four_zero_entire_run_and_raw_preserved():
    source = observations([10, 10, 10, 20, 20, 20, 20, 30])
    original = source.frame.copy(deep=True)
    result = prepare_observations(source, policy())
    assert result.frame.treated_value.tolist() == [10, 10, 10, 0, 0, 0, 0, 30]
    pd.testing.assert_frame_equal(source.frame, original)
    assert result.frame.plateau_run_length.tolist() == [0, 0, 0, 4, 4, 4, 4, 0]


def test_plateau_anchor_not_adjacent_ramp_and_boundaries():
    p = {"minimum_run": 4, "atol_kw": 0.1, "rtol": 0}
    t = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    assert not plateau_runs([10, 10.08, 10.16, 10.24, 10.32, 10.4], t, p)
    assert not plateau_runs([10, 10, 0, 10, 10, 10], t, p)
    assert not plateau_runs([10, 10, np.nan, 10, 10, 10], t, p)
    gap = t.delete(2)
    assert not plateau_runs([10] * 5, gap, p)
    assert len(plateau_runs([10, 10.01, 10.02, 10.03, 0, 0], t, p)) == 1


def test_plateau_is_farm_specific_and_precedes_interpolation():
    first = observations([10, 10, 10, 10, np.nan, 12]).frame
    second = observations([20, 21, 22, 23, 24, 25], farm_id=345).frame
    result = prepare_observations(ObservationBundle(pd.concat([first, second], ignore_index=True)), policy())
    f1 = result.frame.loc[result.frame.farm_id == 1]
    assert f1.treated_value.tolist() == [0, 0, 0, 0, 6, 12]
    assert result.frame.loc[result.frame.farm_id == 345].treated_value.tolist() == [20, 21, 22, 23, 24, 25]


def test_night_local_boundaries_dst_and_no_ghi_mask():
    local = pd.DatetimeIndex(["2025-03-29T21:15", "2025-03-29T21:30", "2025-03-30T04:45", "2025-03-30T05:00"]).tz_localize("Europe/Rome")
    window = {"enabled": True, "timezone": "Europe/Rome", "start": "21:30", "end": "05:00"}
    assert night_window_mask(local.tz_convert("UTC"), window).tolist() == [False, True, True, False]
    fall = pd.DatetimeIndex(["2025-10-26T00:30Z", "2025-10-26T01:30Z", "2025-10-26T04:00Z"])
    assert night_window_mask(fall, window).tolist() == [True, True, False]
    source = observations([1, np.nan, 3], start="2025-03-29T22:00Z")
    result = prepare_observations(source, policy(night=window))
    assert result.frame.treated_value.tolist() == [0, 0, 0]
    ghi = prepare_observations(observations([1, 2, 3], "ghi", start="2025-03-29T22:00Z"), policy(night=window))
    assert ghi.frame.treated_value.tolist() == [1, 2, 3]


def test_negative_allowlist_preserves_nan_temperatures_wind_prices():
    source = pd.DataFrame({"forecast_ghi": [-1, np.nan, 1], "forecast_bhi": [-2, 1, np.nan], "temperature_c": [-5, 1, 2], "wind_u_10m": [-3, 2, 1], "price": [-50, 0, 1]})
    result, flags = clip_negative_forecasts(source, ["forecast_ghi", "forecast_bhi"])
    assert result.forecast_ghi.iloc[0] == 0 and np.isnan(result.forecast_ghi.iloc[1])
    assert result.temperature_c.iloc[0] == -5 and result.wind_u_10m.iloc[0] == -3 and result.price.iloc[0] == -50
    assert source.forecast_ghi.iloc[0] == -1 and flags.forecast_ghi.sum() == 1


def test_interior_linear_interpolation_no_extrapolation_or_replacement():
    result = prepare_observations(observations([np.nan, 2, np.nan, np.nan, 8, 0, np.nan], "ghi"), policy())
    assert np.isnan(result.frame.treated_value.iloc[0]) and np.isnan(result.frame.treated_value.iloc[-1])
    assert result.frame.treated_value.iloc[1:6].tolist() == [2, 4, 6, 8, 0]
    bounded = prepare_observations(observations([1, np.nan, np.nan, 4], "ghi"), policy(interpolation={"variables": ["ghi"], "maximum_gap_hours": 0.5}))
    assert bounded.frame.treated_value.isna().sum() == 2


def test_asof_plateau_and_interpolation_wait_for_evidence():
    result = prepare_observations(observations([10, 10, 10, 10, np.nan, 30]), policy())
    before = result.as_of("2025-01-01T10:45Z")
    assert before.value.tolist() == [10, 10, 10]
    at_fourth = result.as_of("2025-01-01T11:00Z")
    assert at_fourth.value.tolist() == [0, 0, 0, 0]
    before_right = result.as_of("2025-01-01T11:15Z")
    assert np.isnan(before_right.value.iloc[-1])
    after_right = result.as_of("2025-01-01T11:30Z")
    assert after_right.value.tolist() == [0, 0, 0, 0, 15, 30]
    assert result.frame.evidence_at.iloc[4] == pd.Timestamp("2025-01-01T11:30Z")


def test_interpolation_waits_for_retroactive_endpoint_plateau_mask():
    result = prepare_observations(observations([4, np.nan, 10, 10, 10, 10]), policy())
    assert result.frame.treated_value.iloc[1] == 2
    assert result.frame.evidence_at.iloc[1] == pd.Timestamp("2025-01-01T11:30Z")
    assert np.isnan(result.as_of("2025-01-01T11:00Z").value.iloc[1])


def test_idempotent_flags_review_and_energy_reconcile():
    source = observations([10, 10, 10, 10, np.nan, 20])
    source.frame.loc[5, "review_excluded"] = True
    once = prepare_observations(source, policy())
    twice = prepare_observations(once, policy())
    pd.testing.assert_frame_equal(once.frame, twice.frame)
    energy = once.summary["daily_ac_energy"][0]
    assert energy["energy_change_mwh"] == pytest.approx(energy["paired_adjustment_mwh"] + energy["imputed_energy_mwh"])
    assert once.frame.review_excluded.iloc[5]
    assert np.isnan(once.as_of("2025-01-02").value.iloc[-1])


def test_native_observation_not_available_at_start():
    result = prepare_observations(observations([10]), policy())
    assert result.as_of("2025-01-01T10:00Z").empty
    assert len(result.as_of("2025-01-01T10:15Z")) == 1


def test_infinite_historical_values_not_repaired_as_missing():
    result = prepare_observations(observations([1, np.inf, 3], "ghi"), policy())
    assert np.isinf(result.frame.treated_value.iloc[1])
    assert not result.frame.measurement_interpolated.any()
    assert result.summary["remaining_nonfinite"] == 1


def test_interpolation_with_review_excluded_endpoint_remains_ineligible():
    source = observations([2, np.nan, 8])
    source.frame.loc[0, "review_excluded"] = True
    result = prepare_observations(source, policy())
    assert result.frame.treated_value.iloc[1] == 5
    assert not result.frame.review_excluded.iloc[1]
    assert result.frame.derived_review_excluded.iloc[1]
    assert np.isnan(result.as_of("2025-01-02").value.iloc[1])
