"""PV electrical bases, geometry assumptions, limits and causal calibration."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from bess_pv_trading.config import load_config

from bess_pv_trading.data import FarmMetadata, ForecastBundle, ObservationBundle, prepare_observations
from bess_pv_trading.forecasting.pv import fit_dc_to_ac, forecast_pv, source_azimuth_to_pvlib


def settings():
    """Exercise the nominal physical equations independently of promoted farm fits."""
    config=load_config("configs/farms_nord.yaml")
    return deepcopy(config.get("_pv_shared_settings",config["pv"]))


def farm(tracking=0, capacity=1000, pv_type=777):
    return FarmMetadata(1, "NORD", capacity, tracking, 20, 0, 25, pv_type)


def weather(ghi=800, bhi=500, time="2025-06-01T10:00Z", count=4):
    starts = pd.date_range(time, periods=count, freq="15min")
    values = pd.DataFrame({"forecast_ghi": ghi, "forecast_bhi": bhi, "temperature_c": 25.0, "wind_u_10m": -3.0, "wind_v_10m": 4.0, "fraction_diffuse": np.nan}, index=starts)
    return ForecastBundle(1, pd.Timestamp("2025-05-31T18:00Z"), pd.Timestamp("2025-05-31T22:00Z"), values, {name: "fixture" for name in values}, provenance={"decision_cutoff_utc": "2025-06-01T00:00Z"})


def test_source_azimuth_both_signs():
    assert source_azimuth_to_pvlib(0, "east") == 180
    assert source_azimuth_to_pvlib(-90, "east") == 270
    assert source_azimuth_to_pvlib(90, "west") == 270
    with pytest.raises(ValueError):
        source_azimuth_to_pvlib(0, "unconfirmed")


def test_pv_dc_scaling_ac_conversion_wind_and_raw_preservation():
    source = weather()
    original = source.values.copy(deep=True)
    first = forecast_pv(source, farm(), settings(), {"enabled": False})
    second = forecast_pv(source, farm(capacity=2000), settings(), {"enabled": False})
    pd.testing.assert_frame_equal(source.values, original)
    assert first.values.wind_speed_m_s.tolist() == [5.0] * 4
    assert (first.values.cell_temperature_c > 25).all()
    assert second.values.modeled_dc_mw.tolist() == pytest.approx(first.values.modeled_dc_mw * 2)
    assert first.values.pv_ac_mw.tolist() == pytest.approx(first.values.modeled_dc_mw * settings()["output"]["pv_output_efficiency"])
    assert not first.provenance["bess_converter_efficiency_applied"]
    assert first.provenance["unknown_pv_type_id"] == "retained_without_inferred_electrical_parameters"
    transformer_loss = settings()
    transformer_loss["output"]["transformer_efficiency"] = 0.9
    transformer_loss["output"]["transformer_assumption_status"] = "test_configured_loss"
    transformed = forecast_pv(source, farm(), transformer_loss, {"enabled": False})
    assert transformed.values.pv_ac_mw.tolist() == pytest.approx(
        transformed.values.modeled_dc_mw * transformer_loss["output"]["pv_output_efficiency"] * transformer_loss["output"]["transformer_efficiency"])


def test_dc_reference_stays_fixed_when_fitted_output_coefficient_changes():
    explicit = settings()
    default = forecast_pv(weather(), farm(), explicit, {"enabled": False})
    calibrated = deepcopy(explicit)
    calibrated["output"]["method"] = "calibrated_scale"
    calibrated["output"]["calibrated_coefficient"] = explicit["output"]["pv_output_efficiency"] / 2
    result = forecast_pv(weather(), farm(), calibrated, {"enabled": False})
    assert result.values.modeled_dc_mw.tolist() == pytest.approx(default.values.modeled_dc_mw)
    assert result.values.pv_ac_mw.tolist() == pytest.approx(default.values.pv_ac_mw / 2)
    assert result.provenance["dc_reference_efficiency"] == explicit["dc"]["reference_conversion_efficiency"]


def test_through_origin_mse_fit():
    from bess_pv_trading.forecasting.pv_calibration import fit_scale
    reference = settings()["output"]["pv_output_efficiency"]
    dc = np.array([1.0, 2.0, 3.0])
    measured = reference * dc
    assert fit_scale(dc, measured) == pytest.approx(reference)


def test_horizontal_direct_is_converted_to_dni_and_low_sun_bounded():
    result = forecast_pv(weather(), farm(), settings(), {"enabled": False})
    cos_zenith = np.cos(np.radians(result.values.solar_zenith_deg))
    assert result.values.dni_w_m2.tolist() == pytest.approx(500 / cos_zenith)
    assert not np.allclose(result.values.dni_w_m2, 500)
    dawn = forecast_pv(weather(ghi=100, bhi=80, time="2025-06-01T02:00Z"), farm(), settings(), {"enabled": False})
    assert (dawn.values.dni_w_m2 <= settings()["dni_limit_w_m2"]).all()
    assert dawn.values.low_sun_direct_limited.all()
    assert dawn.values.dni_w_m2.tolist() == [0.0] * 4


def test_inconsistency_invalid_nan_preserved_and_explicit_erbs_alternative():
    mismatch = forecast_pv(weather(ghi=100, bhi=200), farm(), settings(), {"enabled": False})
    assert mismatch.values.pv_ac_mw.isna().all() and mismatch.values.bhi_exceeds_ghi.all()
    missing = forecast_pv(weather(ghi=np.nan), farm(), settings(), {"enabled": False})
    assert missing.values.pv_ac_mw.isna().all()
    config = settings()
    config["ghi_only_decomposition"] = "erbs"
    decomposed = forecast_pv(weather(bhi=np.nan), farm(), config, {"enabled": False})
    assert decomposed.values.ghi_decomposition_used.all() and decomposed.values.pv_ac_mw.notna().all()


def test_night_fixtures_and_production_only_clock_mask():
    fixture = weather(ghi=0, bhi=0, time="2025-06-01T00:00Z")
    result = forecast_pv(fixture, farm(tracking=1), settings(), {"enabled": False})
    assert result.values.pv_ac_mw.tolist() == [0] * 4
    clock = {"enabled": True, "timezone": "Europe/Rome", "start": "11:30", "end": "13:00"}
    masked = forecast_pv(weather(), farm(), settings(), clock)
    assert masked.values.pv_ac_mw.tolist() == [0] * 4
    assert masked.values.forecast_ghi.tolist() == [800] * 4


def test_tracker_axis_is_distinct_and_bifacial_omission_explicit():
    config = settings()
    config["tracker"]["axis_tilt"] = 0
    result = forecast_pv(weather(), farm(tracking=3), config, {"enabled": False})
    assert result.provenance["rear_side_gain"] == "omitted_front_side_only"
    assert (result.values.surface_tilt_deg != 20).any()
    config["bifacial_front_side_only_enabled"] = False
    with pytest.raises(ValueError, match="front-side-only"):
        forecast_pv(weather(), farm(tracking=3), config, {"enabled": False})


def test_fraction_diffuse_and_missing_geometry_require_explicit_status():
    config = settings()
    config["fraction_diffuse_enabled"] = True
    with pytest.raises(ValueError, match="fraction_diffuse"):
        forecast_pv(weather(), farm(), config, {"enabled": False})
    config = settings()
    config["tracker"]["assumptions_acknowledged"] = False
    with pytest.raises(ValueError, match="tracker axis"):
        forecast_pv(weather(), farm(tracking=1), config, {"enabled": False})


def test_measured_power_ac_and_fit_cutoff_are_not_bess_conversion():
    times = pd.date_range("2025-06-01T10:00Z", periods=5, freq="15min")
    dc = pd.Series([1, 2, 3, 4, 5], index=times)
    frame = pd.DataFrame({"farm_id": 1, "variable": "power", "time": times, "interval_end": times + pd.Timedelta(minutes=15), "available_at": times + pd.Timedelta(minutes=15), "raw_value": [900, 1800, 2700, 3600, 500000], "review_excluded": False})
    bundle = prepare_observations(ObservationBundle(frame), {"plateau": {"enabled": False}, "night": {"enabled": False}})
    fit = fit_dc_to_ac(dc, bundle, 1, "2025-06-01T11:00Z", minimum_samples=4)
    assert fit.ac_per_dc == pytest.approx(0.9) and fit.training_samples == 4
    assert bundle.frame.treated_value.iloc[0] == 900
    config = settings()
    config["output"]["method"] = "calibrated_scale"
    config["output"]["transformer_assumption_status"] = "unit_test_no_loss"
    with pytest.raises(ValueError, match="fit available"):
        forecast_pv(weather(), farm(), config, {"enabled": False}, fit)


def test_explicitly_supplied_members_preserved_through_same_pv_interface():
    from bess_pv_trading.data import align_forecast
    times = pd.date_range("2025-06-01T09:00Z", periods=3, freq="h")
    frames = []
    for member, ghi, bhi in [("external_1", 800, 500), ("external_2", 400, 200)]:
        frames.append(pd.DataFrame({"member": member, "forecast_ghi": ghi, "forecast_bhi": bhi, "temperature_c": 25, "wind_u_10m": 3, "wind_v_10m": 4}, index=times))
    source = ForecastBundle(1, pd.Timestamp("2025-05-31T18:00Z"), pd.Timestamp("2025-05-31T22:00Z"), pd.concat(frames), {key: "fixture" for key in frames[0] if key != "member"}, ("external_1", "external_2"), "explicitly_supplied_fixture_members")
    aligned = align_forecast(source, pd.date_range("2025-06-01T10:00Z", periods=4, freq="15min"), {"forecast_ghi": "hourly_mean", "forecast_bhi": "hourly_mean"})
    result = forecast_pv(aligned, farm(), settings(), {"enabled": False})
    assert result.members == source.members and len(result.values) == 8
    assert set(result.values.member) == set(source.members)
    means = result.values.groupby("member").pv_ac_mw.mean()
    assert means["external_1"] > means["external_2"]


def test_exact_grid_ac_point_scoring_reports_missing_and_policy_targets():
    from bess_pv_trading.forecasting.pv import evaluate_pv_forecast
    times = pd.date_range("2025-06-01T10:00Z", periods=3, freq="15min")
    frame = pd.DataFrame({"farm_id": 1, "variable": "power", "time": times, "interval_end": times + pd.Timedelta(minutes=15), "available_at": times + pd.Timedelta(minutes=15), "raw_value": [1000, 2000, 3000], "review_excluded": False})
    observations = prepare_observations(ObservationBundle(frame), {"plateau": {"enabled": False}, "night": {"enabled": False}})
    predictions = pd.DataFrame({"pv_ac_mw": [1.2, 1.8, np.nan], "member": "deterministic", "lead_hours": [24, 24.25, 24.5]}, index=times.as_unit("us"))
    bundle = ForecastBundle(1, times[0] - pd.Timedelta(days=1), times[0] - pd.Timedelta(hours=20), predictions, {"pv_ac_mw": "MW_AC"})
    scored, summary = evaluate_pv_forecast(bundle, observations)
    result = summary["members"][0]["policy_treated_AC_target"]
    assert result["finite_eligible_pairs"] == 2 and result["coverage_fraction"] == pytest.approx(2 / 3)
    assert result["mae_mw"] == pytest.approx(0.2) and result["rmse_mw"] == pytest.approx(0.2)
    assert result["bias_mw"] == pytest.approx(0)
    assert scored.treated_actual_ac_mw.tolist() == [1, 2, 3]
    assert result["paired_measured_energy_mwh"] == pytest.approx(0.75)
