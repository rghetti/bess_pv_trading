"""Transparent pvlib DC diagnostics and separately declared farm AC forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import pvlib

from ...data import (
    FarmMetadata, ForecastBundle, ObservationBundle, align_forecast, apply_review_exclusions, config_path,
    load_metadata, load_observations, night_window_mask, open_store, prepare_observations, select_forecast_vintage,
    utc_timestamp,
)


# Source angles use south-zero; the sign is an explicit dataset assumption.
def source_azimuth_to_pvlib(angle_deg: Any, positive_direction: str) -> Any:
    """Convert both supported south-zero conventions to north-zero clockwise."""
    # Enforce the input contract: Source azimuth positive direction must be explicitly east or west.
    if positive_direction not in {"east", "west"}:
        raise ValueError("Source azimuth positive direction must be explicitly east or west")
    sign = -1 if positive_direction == "east" else 1
    return (180 + sign * np.asarray(angle_deg)) % 360


@dataclass(frozen=True)
class PVCalibration:
    """A fitted DC-to-AC scale with a causal fit cutoff and training diagnostics."""
    ac_per_dc: float
    fit_cutoff: pd.Timestamp
    training_samples: int
    # Count observations adjusted by the declared preprocessing policy.
    adjusted_samples: int
    rmse_mw: float
    target_basis: str = "policy_treated_measured_AC"


# A through-origin mapping preserves zero DC output and uses completed AC samples.
def fit_dc_to_ac(modeled_dc_mw: pd.Series, observations: ObservationBundle, farm_id: int, cutoff: Any, minimum_samples: int = 20, include_interpolated: bool = False) -> PVCalibration:
    """Fit an optional AC calibration using only observations available as of cutoff.

    Model callers must also certify that DC input forecasts were issued by their
    historical prediction times. This function controls target availability.
    """
    view = observations.as_of(cutoff)
    view = view.loc[(view.farm_id == farm_id) & (view.variable == "power")].copy()
    # When `not include_interpolated and 'measurement_interpolated' in view`, set `view` using `view.loc[~view.measurement_interpolated]`.
    if not include_interpolated and "measurement_interpolated" in view:
        view = view.loc[~view.measurement_interpolated]
    series = pd.Series(modeled_dc_mw.to_numpy(), index=pd.to_datetime(modeled_dc_mw.index, utc=True), name="dc_mw")
    joined = view.set_index("time").join(series, how="inner")
    joined = joined.loc[np.isfinite(joined.dc_mw) & np.isfinite(joined.value) & (joined.dc_mw > 0)]
    # Enforce the input contract: Insufficient causal AC samples for PV-output calibration.
    if len(joined) < minimum_samples:
        raise ValueError("Insufficient causal AC samples for PV-output calibration")
    dc, ac = joined.dc_mw.to_numpy(), joined.value.to_numpy() / 1000
    slope = float(np.dot(dc, ac) / np.dot(dc, dc))
    # Enforce the input contract: PV-output calibration slope must be positive and finite.
    if slope <= 0 or not np.isfinite(slope):
        raise ValueError("PV-output calibration slope must be positive and finite")
    adjusted = joined.get("plateau_zero", False) | joined.get("nighttime_zero", False) | joined.get("measurement_interpolated", False)
    return PVCalibration(slope, utc_timestamp(cutoff), len(joined), int(np.sum(adjusted)), float(np.sqrt(np.mean((slope * dc - ac) ** 2))))


# Irradiance diagnostics retain original horizontal means after redistribution.
def forecast_pv(bundle: ForecastBundle, farm: FarmMetadata, settings: dict[str, Any], night_policy: dict[str, Any], calibration: PVCalibration | None = None) -> ForecastBundle:
    """Run pvlib at interval midpoint, returning modeled DC and predicted AC MW."""
    from ..inference.pv_artifacts import resolve_parameters
    # Enforce the input contract: Select either a PV parameter registry or an explicit calibration object.
    if calibration is not None and settings.get("parameter_registry"):
        raise ValueError("Select either a PV parameter registry or an explicit calibration object")
    settings, artifact_provenance = resolve_parameters(settings, farm, night_policy,
        bundle.provenance.get("decision_cutoff_utc", bundle.available_at))
    # Enforce the input contract: fraction_diffuse is not enabled: independent definition/validation is required.
    if settings.get("fraction_diffuse_enabled", False):
        raise ValueError("fraction_diffuse is not enabled: independent definition/validation is required")
    # Enforce the input contract: PV engine requires explicitly declared forecast shortwave GHI.
    if settings.get("shortwave_interpretation") != "ghi":
        raise ValueError("PV engine requires explicitly declared forecast shortwave GHI")
    # Enforce the input contract: PV engine requires explicit horizontal-direct working interpretation.
    if settings.get("direct_radiation_interpretation") != "horizontal_direct":
        raise ValueError("PV engine requires explicit horizontal-direct working interpretation")
    location = settings.get("location", {})
    # Enforce the input contract: Proxy coordinates require explicit acknowledgement.
    if location.get("location_status") == "proxy" and not location.get("acknowledged", False):
        raise ValueError("Proxy coordinates require explicit acknowledgement")
    # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW for each `coordinate` in `('latitude', 'longitude')`.
    for coordinate in ("latitude", "longitude"):
        # Enforce the input contract: PV location requires `coordinate`.
        if coordinate not in location:
            raise ValueError(f"PV location requires {coordinate}")
    frame = bundle.values.copy(deep=True)
    required = ["forecast_ghi", "temperature_c", "wind_u_10m", "wind_v_10m"]
    # When `settings.get('temperature', {}).get('wind_height_m', 10) == 100`, set `required` using `['forecast_ghi', 'temperature_c', 'wind_u_100m', 'wind_v_100m']`.
    if settings.get("temperature", {}).get("wind_height_m", 10) == 100:
        required = ["forecast_ghi", "temperature_c", "wind_u_100m", "wind_v_100m"]
    # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW for each `name` in `required`.
    for name in required:
        # Enforce the input contract: Aligned forecast lacks PV input `name`.
        if name not in frame:
            raise ValueError(f"Aligned forecast lacks PV input {name}")
    times = pd.DatetimeIndex(frame.index) + pd.Timedelta(minutes=7.5)
    solar = pvlib.solarposition.get_solarposition(times, latitude=float(location["latitude"]), longitude=float(location["longitude"]), altitude=farm.elevation_m)
    zenith, azimuth = solar.apparent_zenith.to_numpy(), solar.azimuth.to_numpy()
    ghi = frame.forecast_ghi.to_numpy(dtype=float)
    ghi_valid = np.isfinite(ghi) & (ghi >= 0)
    bhi = frame["forecast_bhi"].to_numpy(dtype=float) if "forecast_bhi" in frame else np.full(len(frame), np.nan)
    direct_valid = np.isfinite(bhi) & (bhi >= 0) & (bhi <= ghi)
    contradiction = np.isfinite(bhi) & np.isfinite(ghi) & (bhi > ghi)
    action = settings.get("direct_exceeds_ghi", "invalidate")
    # Enforce the input contract: direct_exceeds_ghi must be invalidate or clip_and_flag.
    if action not in {"invalidate", "clip_and_flag"}:
        raise ValueError("direct_exceeds_ghi must be invalidate or clip_and_flag")
    # When `action == 'clip_and_flag'`, set `bhi` using `np.where(contradiction, ghi, bhi)`.
    if action == "clip_and_flag":
        bhi = np.where(contradiction, ghi, bhi)
        direct_valid = np.isfinite(bhi) & (bhi >= 0) & (bhi <= ghi)
    dhi = ghi - bhi
    # pvlib.dni derives DNI from horizontal GHI/DHI and suppresses the horizon.
    zenith_limit = float(settings.get("low_sun_zenith_deg", 85))
    # Enforce the input contract: Low-sun zenith limit must lie strictly between 0 and 90 degrees.
    if not 0 < zenith_limit < 90:
        raise ValueError("Low-sun zenith limit must lie strictly between 0 and 90 degrees")
    dni = np.asarray(pvlib.irradiance.dni(pd.Series(ghi), pd.Series(dhi), pd.Series(zenith), zenith_threshold_for_zero_dni=zenith_limit), dtype=float)
    low_sun = (zenith >= zenith_limit) & (bhi > 0) & direct_valid
    dni = np.where((zenith >= zenith_limit) & direct_valid, 0.0, dni)
    dni = np.where(direct_valid & ghi_valid, dni, np.nan)
    decomposition_used = np.zeros(len(frame), dtype=bool)
    decomposition = settings.get("ghi_only_decomposition")
    # Enforce the input contract: Only explicitly selected erbs GHI decomposition is supported.
    if decomposition not in {None, "erbs"}:
        raise ValueError("Only explicitly selected erbs GHI decomposition is supported")
    # When `decomposition == 'erbs'`, set `alternative` using `pvlib.irradiance.erbs(ghi, zenith, times)`.
    if decomposition == "erbs":
        # An explicitly selected model can replace unusable direct input, not GHI.
        alternative = pvlib.irradiance.erbs(ghi, zenith, times)
        decomposition_used = ghi_valid & ~direct_valid & ~contradiction
        dni[decomposition_used] = np.asarray(alternative["dni"])[decomposition_used]
        dhi[decomposition_used] = np.asarray(alternative["dhi"])[decomposition_used]
        dni[decomposition_used & (zenith >= zenith_limit)] = 0
    valid = ghi_valid & (direct_valid | decomposition_used)
    # When `action == 'invalidate'`, set `valid` using `~contradiction`.
    if action == "invalidate":
        valid &= ~contradiction
    dni_limit = float(settings.get("dni_limit_w_m2", 1400))
    # Enforce the input contract: DNI limit must be positive.
    if dni_limit <= 0:
        raise ValueError("DNI limit must be positive")
    dni_limited = np.isfinite(dni) & (dni > dni_limit)
    limit_action = settings.get("dni_limit_action", "cap_and_flag")
    # When `limit_action == 'cap_and_flag'`, set `dni` using `np.minimum(dni, dni_limit)`.
    if limit_action == "cap_and_flag":
        dni = np.minimum(dni, dni_limit)
    # When `limit_action == 'invalidate'`, set `valid` using `~dni_limited`.
    elif limit_action == "invalidate":
        valid &= ~dni_limited
    else:
        raise ValueError("dni_limit_action must be cap_and_flag or invalidate")
    from ..models.pv_physical import PhysicalPVModel
    temperature = settings["temperature"]
    height = int(temperature.get("wind_height_m", 10))
    # Enforce the input contract: Wind source height must be 10 or 100 m.
    if height not in {10, 100}:
        raise ValueError("Wind source height must be 10 or 100 m")
    wind = np.hypot(frame[f"wind_u_{height}m"].to_numpy(), frame[f"wind_v_{height}m"].to_numpy())
    output, dc_settings = settings["output"], settings["dc"]
    method = output["method"]
    # When `method == 'explicit_efficiency'`, set `output_efficiency` using `float(output['pv_output_efficiency'])`.
    if method == "explicit_efficiency":
        output_efficiency = float(output["pv_output_efficiency"])
        # Enforce the input contract: PV-output efficiency requires 0<eta<=1 and assumption provenance.
        if not 0 < output_efficiency <= 1 or not output.get("assumption_status"):
            raise ValueError("PV-output efficiency requires 0<eta<=1 and assumption provenance")
    # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW when `method == 'calibrated_scale'`.
    elif method == "calibrated_scale":
        # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW when `calibration is not None`.
        if calibration is not None:
            # Enforce the input contract: Calibrated PV output requires a fit available by decision cutoff.
            if calibration.fit_cutoff > utc_timestamp(bundle.provenance.get("decision_cutoff_utc", bundle.available_at)):
                raise ValueError("Calibrated PV output requires a fit available by decision cutoff")
            output_efficiency = calibration.ac_per_dc
        # When `output.get('calibrated_coefficient') is not None`, set `output_efficiency` using `float(output['calibrated_coefficient'])`.
        elif output.get("calibrated_coefficient") is not None:
            output_efficiency = float(output["calibrated_coefficient"])
        else:
            raise ValueError("Calibrated PV output requires a fitted coefficient")
    else:
        raise ValueError("PV output requires explicit_efficiency or calibrated_scale")
    # Enforce the input contract: Calibrated PV output coefficient must be positive and finite.
    if not np.isfinite(output_efficiency) or output_efficiency <= 0:
        raise ValueError("Calibrated PV output coefficient must be positive and finite")
    # These target-independent inputs are also saved for efficient fitting.
    for name, values in {"solar_zenith_deg": zenith, "solar_azimuth_deg": azimuth,
        "ghi_w_m2": ghi, "bhi_w_m2": bhi, "dhi_w_m2": dhi, "dni_w_m2": dni,
        "wind_speed_m_s": wind, "pv_input_valid": valid,
        "bhi_exceeds_ghi": contradiction, "low_sun_direct_limited": low_sun,
        "dni_limited": dni_limited, "ghi_decomposition_used": decomposition_used}.items():
        frame[name] = values
    from ..models.pv_physical import parameter_defaults
    model = PhysicalPVModel(frame, farm, settings, night_policy)
    parameters = parameter_defaults(settings)
    parameters["coefficient"] = output_efficiency
    # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW for each `(name, values)` in `model.evaluate(parameters, diagnostics=True).items()`.
    for name, values in model.evaluate(parameters, diagnostics=True).items():
        frame[name] = values
    reference = float(dc_settings["reference_conversion_efficiency"])
    transformer = float(output["transformer_efficiency"])
    scale = output_efficiency * transformer
    conversion_status = {"transformer_efficiency": transformer,
        "overall_dc_to_poi_efficiency": scale, "transformer_assumption": output["transformer_assumption_status"]}
    # Run pvlib at interval midpoint, returning modeled DC and predicted AC MW when `method == 'explicit_efficiency'`.
    if method == "explicit_efficiency":
        conversion_status.update(dc_to_ac_efficiency=output_efficiency, dc_to_ac_assumption=output["assumption_status"])
    else:
        conversion_status.update(calibrated_dc_to_ac_scale=output_efficiency,
            calibration_source="causal_fitted_object" if calibration else "configured_static_metadata",
            dc_reference_conversion_efficiency=reference,
            assumption_status=output.get("assumption_status"))
    bifacial = bool(settings.get("bifacial", {}).get("enabled", False))
    provenance = {**bundle.provenance, "pvlib_version": pvlib.__version__,
        "pv_target": "farm_AC_at_POI_MW", "farm_ac_capacity_mw": farm.ac_capacity_mw,
        "dc_reference_capacity_kw": farm.ac_capacity_kw / reference, "dc_reference_efficiency": reference,
        "modeled_dc_unit": "MW", "pv_output_model": conversion_status, "pv_output_scale": scale,
        "ac_limit_mw": model.ac_limit, "geometry_location": location,
        "orientation": "fixed_source_surface_orientation" if farm.tracking_type_id == 0 else "configured_tracker_axis_distinct_from_source_module_orientation",
        "rear_side_gain": "infinite_sheds" if bifacial else ("omitted_front_side_only" if farm.tracking_type_id == 3 else "not_bifacial"),
        "transposition_model": settings.get("transposition_model", "isotropic"),
        "unknown_pv_type_id": "retained_without_inferred_electrical_parameters",
        "bhi_to_dni": "pvlib.irradiance.dni;uniform_hourly_means_at_quarter_hour_midpoint;low_sun_and_cap_flagged",
        "temperature_model": "pvlib.temperature.faiman", "wind_height_proxy_m": height,
        "settings": settings, "night_policy": night_policy, "bess_converter_efficiency_applied": False,
        "parameter_artifact": artifact_provenance}
    return ForecastBundle(bundle.farm_id, bundle.forecast_vintage, bundle.available_at, frame, {**bundle.units, "pv_ac_mw": "MW_AC", "modeled_dc_mw": "MW_DC"}, bundle.members, bundle.member_origin, provenance)


# DST-aware delivery days are built in local time and represented internally UTC.
def local_delivery_starts(day: str, timezone: str = "Europe/Rome") -> pd.DatetimeIndex:
    """Return the real 92/96/100 quarter-hour products of a local calendar day."""
    start = pd.Timestamp(day).tz_localize(timezone)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


# CLI integration saves the exact selected vintage, inputs, diagnostics and gaps.
def forecast_pv_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run one farm/day forecast with clearly disclosed assumptions and validity."""
    run = config.get("forecast_run", {})
    starts = local_delivery_starts(str(run["delivery_day"]), run.get("delivery_timezone", "Europe/Rome"))
    farm_id = int(config.get("farm_id", 1))
    metadata = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
    source = config.get("source", {})
    dataset = open_store(config)
    try:
        arguments = {"dataset": dataset, "farm_id": farm_id, "cutoff": run["decision_cutoff"], "mapping": config["forecast_mapping"], "latency_hours": float(source.get("availability_lag_hours", 4)), "explicit_availability_variable": source.get("explicit_availability_variable"), "numeric_lead_units": source.get("numeric_lead_units", "hours")}
        weather = select_forecast_vintage(**arguments)
        weather = align_forecast(weather, starts, config["forecast_semantics"], config.get("policy", {}).get("nonnegative_forecasts", []))
        pv = forecast_pv(weather, metadata, config["pv"], config.get("policy", {}).get("night", {}))
        fallback = run.get("earlier_complete_vintage_fallback", False)
        attempted_vintages = [pv.forecast_vintage.isoformat()]
        # When `fallback and pv.values.pv_ac_mw.isna().any()`, set `selected_latest` using `pv`.
        if fallback and pv.values.pv_ac_mw.isna().any():
            selected_latest = pv
            cursor = pv.forecast_vintage
            maximum_lead = pd.Timedelta(hours=float(dataset.lead_time.values[-1] / np.timedelta64(1, "h"))) if np.issubdtype(dataset.lead_time.dtype, np.timedelta64) else pd.Timedelta(hours=float(dataset.lead_time.values[-1]))
            # Continue until `True` is false.
            while True:
                previous = dataset.sel(cycle_time=slice(None, (cursor - pd.Timedelta(nanoseconds=1)).tz_localize(None)))
                # Stop processing when `previous.sizes.get('cycle_time', 0) == 0`.
                if previous.sizes.get("cycle_time", 0) == 0:
                    break
                try:
                    weather_source = select_forecast_vintage(**{**arguments, "dataset": previous})
                except ValueError:
                    break
                cursor = weather_source.forecast_vintage
                # All still older vintages share the lead horizon and cannot
                # cover the requested end once this candidate falls short.
                if cursor + maximum_lead < starts[-1] + pd.Timedelta(minutes=15):
                    break
                weather = align_forecast(weather_source, starts, config["forecast_semantics"], config.get("policy", {}).get("nonnegative_forecasts", []))
                candidate = forecast_pv(weather, metadata, config["pv"], config.get("policy", {}).get("night", {}))
                attempted_vintages.append(candidate.forecast_vintage.isoformat())
                # When `candidate.values.pv_ac_mw.notna().all()`, set `pv` using `candidate`.
                if candidate.values.pv_ac_mw.notna().all():
                    pv = candidate
                    break
            # When `pv.values.pv_ac_mw.isna().any()`, set `pv` using `selected_latest`.
            if pv.values.pv_ac_mw.isna().any():
                pv = selected_latest
        pv.provenance["earlier_complete_vintage_fallback_requested"] = bool(fallback)
        pv.provenance["attempted_vintages"] = attempted_vintages
        # Full selected-farm native history supplies boundary/endpoint evidence;
        # this is ex-post scoring only and is never fed back into forecast inputs.
        observations = load_observations(dataset, [farm_id], variables=("power",), telemetry_latency_minutes=float(source.get("telemetry_latency_minutes", 0)))
        review = config.get("review", {})
        review_path = config_path(config, review["exclusions_csv"]) if review.get("exclusions_csv") else None
        observations = apply_review_exclusions(observations, review_path, review.get("excluded_decisions"))
        observations = prepare_observations(observations, config.get("policy", {}))
        evaluation_frame, evaluation = evaluate_pv_forecast(pv, observations)
        pv.values = evaluation_frame
    finally:
        dataset.close()
    output = config_path(config, run.get("output_dir", "outputs/pv/latest"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "pv_forecast.csv"
    # The long export states electrical bases, decision cutoff and member origin.
    pv.values["farm_id"] = farm_id
    pv.values["forecast_cycle_time"] = pv.forecast_vintage
    pv.values["forecast_available_at"] = pv.available_at
    pv.values["source_lead_time"] = (pd.DatetimeIndex(pv.values.index).floor("h") + pd.Timedelta(hours=1) - pv.forecast_vintage)
    pv.values["interval_start_utc"] = pv.values.index
    pv.values["interval_end_utc"] = pv.values.index + pd.Timedelta(minutes=15)
    pv.values["member_id"] = pv.values.member
    pv.values["member_origin"] = pv.member_origin
    pv.values["member_probability"] = 1.0
    pv.values["pv_dc_mw"] = pv.values.modeled_dc_mw
    pv.values["input_eligibility"] = pv.values.pv_input_valid
    pv.values["review_status"] = np.where(pv.values.derived_review_excluded, "ex_post_observation_review_excluded", "no_configured_observation_review_exclusion")
    flag_columns = [name for name in pv.values if name.endswith("_negative_forecast_zero")] + ["bhi_exceeds_ghi", "low_sun_direct_limited", "dni_limited", "ghi_decomposition_used", "nighttime_zero"]
    pv.values["treatment_flags"] = pv.values[flag_columns].apply(lambda row: ",".join(name for name, value in row.items() if value), axis=1)
    pv.values["policy_version"] = config.get("policy", {}).get("version", "requested-v1")
    pv.values["assumptions_id"] = config["pv"].get("assumptions_id", "configured_PV_assumptions_unfitted")
    pv.values.to_csv(path)
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str) + "\n")
    frame = pv.values
    valid = frame.pv_ac_mw.notna()
    forecast_status = "complete_exploratory_forecast" if valid.all() else "ineligible_missing_required_model_inputs"
    conversion = pv.provenance["pv_output_model"]
    calibration_status = ("issue_time_parameter_artifact" if pv.provenance["parameter_artifact"]["status"] == "fitted"
                          else conversion.get("calibration_source", "not_fitted_explicit_PV_output_assumption"))
    summary = {"farm_id": farm_id, "delivery_day": str(run["delivery_day"]), "intervals": len(frame), "forecast_vintage_utc": pv.forecast_vintage.isoformat(), "available_at_utc": pv.available_at.isoformat(), "member_count": len(pv.members), "member_origin": pv.member_origin, "forecast_status": forecast_status, "missing_pv_intervals": int((~valid).sum()), "forecast_energy_finite_mwh": float(frame.pv_ac_mw.sum() * 0.25), "night_mask_adjustment_mwh": float((frame.pv_ac_mw - frame.pv_ac_before_night_mask_mw).sum() * 0.25), "flags": {name: int(frame[name].sum()) for name in ("bhi_exceeds_ghi", "low_sun_direct_limited", "dni_limited", "ghi_decomposition_used", "nighttime_zero")}, "output_path": str(path), "attempted_vintages": attempted_vintages, "provenance": pv.provenance, "calibration_status": calibration_status, "ex_post_point_evaluation": evaluation}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _plot_pv_forecast(frame, output)
    summary["plot_path"] = str(output / "pv_forecast.png")
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    # A saved exploratory forecast may contain gaps; the optimizer must reject it.
    if config.get("review", {}).get("mode", "exploratory") == "strict" and not valid.all():
        raise ValueError(f"Strict PV forecast has missing model inputs; see {output / 'manifest.json'}")
    return summary


# DC diagnostics and farm AC forecasts use visibly distinct electrical labels.
def _plot_pv_forecast(frame: pd.DataFrame, output: Path) -> None:
    """Save forecast output/radiation and display required-input gaps explicitly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ...reporting.plots import plot_interval_series
    def plot_values(axis, values, **kwargs):
        """Close the last forecast interval when its explicit end is available."""
        # Return the selected result for this condition.
        if "interval_end_utc" in frame:
            return plot_interval_series(axis, frame.index,
                pd.to_datetime(frame.interval_end_utc, utc=True), values, **kwargs)
        return axis.plot(frame.index, values, **kwargs)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plot_values(axes[0], frame.modeled_dc_mw, label="modeled DC diagnostic MW")
    plot_values(axes[0], frame.pv_ac_mw, label="predicted farm AC MW")
    # Save forecast output/radiation and display required-input gaps explicitly when `'raw_actual_ac_mw' in frame`.
    if "raw_actual_ac_mw" in frame:
        plot_values(axes[0], frame.raw_actual_ac_mw, label="raw measured AC MW", linewidth=1.2)
        plot_values(axes[0], frame.treated_actual_ac_mw, linestyle="--", label="policy-treated AC MW", linewidth=1.0)
    missing = frame.pv_ac_mw.isna()
    axes[0].scatter(frame.index[missing], np.zeros(missing.sum()), color="red", marker="x", label="missing required PV input")
    axes[0].set_ylabel("MW")
    plot_values(axes[1], frame.forecast_ghi, label="hourly forecast GHI, uniform redistribution")
    plot_values(axes[1], frame.forecast_bhi, label="forecast BHI working interpretation")
    axes[1].set_ylabel("W/m²")
    axes[1].set_xlabel("UTC delivery time (including available interval ends)")
    # Save forecast output/radiation and display required-input gaps explicitly when `'interval_end_utc' in frame and len(frame)`.
    if "interval_end_utc" in frame and len(frame):
        axes[0].set_xlim(frame.index[0], pd.to_datetime(frame.interval_end_utc, utc=True).iloc[-1])
    # Save forecast output/radiation and display required-input gaps explicitly for each `axis` in `axes`.
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "pv_forecast.png", dpi=130)
    plt.close(fig)


# Scoring uses exact native interval starts and is explicitly ex-post accounting.
def evaluate_pv_forecast(bundle: ForecastBundle, observations: ObservationBundle, lead_bin_hours: float = 12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare AC forecasts to raw/treated AC, separately by member and lead bin."""
    # Enforce the input contract: Lead-bin width must be positive.
    if lead_bin_hours <= 0:
        raise ValueError("Lead-bin width must be positive")
    power = observations.frame.loc[(observations.frame.farm_id == bundle.farm_id) & (observations.frame.variable == "power")].set_index("time")
    # Enforce the input contract: Measured native power interval keys must be unique.
    if power.index.has_duplicates:
        raise ValueError("Measured native power interval keys must be unique")
    frame = bundle.values.copy()
    actual = power.reindex(frame.index)
    frame["raw_actual_ac_mw"] = actual.raw_value.to_numpy(dtype=float) / 1000
    frame["treated_actual_ac_mw"] = actual.treated_value.to_numpy(dtype=float) / 1000
    frame["review_excluded"] = actual.review_excluded.fillna(True).to_numpy(dtype=bool)
    frame["derived_review_excluded"] = actual.get("derived_review_excluded", actual.review_excluded).fillna(True).to_numpy(dtype=bool)
    # Compare AC forecasts to raw/treated AC, separately by member and lead bin for each `column` in `('measurement_interpolated', 'plateau_zero', 'nighttime_zero')`.
    for column in ("measurement_interpolated", "plateau_zero", "nighttime_zero"):
        frame[f"observation_{column}"] = actual[column].fillna(False).to_numpy(dtype=bool)
    frame["observation_evidence_at"] = actual.evidence_at.to_numpy()
    frame["lead_bin_start_hours"] = np.floor(frame.lead_hours / lead_bin_hours) * lead_bin_hours

    def metrics(group: pd.DataFrame, target: str, excluded: str) -> dict[str, Any]:
        forecast = group.pv_ac_mw.to_numpy()
        measured = group[target].to_numpy()
        valid = np.isfinite(forecast) & np.isfinite(measured) & ~group[excluded].to_numpy(dtype=bool)
        error = forecast[valid] - measured[valid]
        return {"intervals": len(group), "finite_eligible_pairs": int(valid.sum()), "coverage_fraction": float(valid.mean()) if len(group) else 0.0, "forecast_missing_intervals": int((~np.isfinite(forecast)).sum()), "target_missing_intervals": int((~np.isfinite(measured)).sum()), "review_excluded_intervals": int(group[excluded].sum()), "mae_mw": float(np.mean(np.abs(error))) if len(error) else None, "rmse_mw": float(np.sqrt(np.mean(error ** 2))) if len(error) else None, "bias_mw": float(np.mean(error)) if len(error) else None, "paired_forecast_energy_mwh": float(np.sum(forecast[valid]) * 0.25), "paired_measured_energy_mwh": float(np.sum(measured[valid]) * 0.25)}

    members: list[dict[str, Any]] = []
    # Compare AC forecasts to raw/treated AC, separately by member and lead bin for each `(member, group)` in `frame.groupby('member', sort=False)`.
    for member, group in frame.groupby("member", sort=False):
        members.append({"member_id": str(member), "raw_AC_target": metrics(group, "raw_actual_ac_mw", "review_excluded"), "policy_treated_AC_target": metrics(group, "treated_actual_ac_mw", "derived_review_excluded"), "adjusted_observation_intervals": int((group.observation_measurement_interpolated | group.observation_plateau_zero | group.observation_nighttime_zero).sum()), "lead_bins": [{"lead_bin_start_hours": float(lead), "raw_AC_target": metrics(lead_group, "raw_actual_ac_mw", "review_excluded"), "policy_treated_AC_target": metrics(lead_group, "treated_actual_ac_mw", "derived_review_excluded")} for lead, lead_group in group.groupby("lead_bin_start_hours", sort=True)]})
    summary = {"evaluation_type": "ex_post_exact_grid_point_forecast_by_vintage_not_ensemble_calibration", "farm_id": bundle.farm_id, "forecast_vintage_utc": bundle.forecast_vintage.isoformat(), "source_member_origin": bundle.member_origin, "target_basis": "measured_AC_MW", "period_start_utc": frame.index.min().isoformat(), "period_end_utc": frame.index.max().isoformat(), "lead_bin_hours": lead_bin_hours, "members": members, "limitations": "Policy-treated observations may include curtailment/outages and are not certified unconstrained availability; one deterministic vintage is not an ensemble sample."}
    return frame, summary
