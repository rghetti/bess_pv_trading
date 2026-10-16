"""Read-only, named-dimension adapters for the verified farm Zarr and Parquet."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import xarray as xr

from .contracts import FarmMetadata, ForecastBundle, ObservationBundle, utc_timestamp
from .policy import clip_negative_forecasts, night_window_mask, observation_summary, prepare_observations


# Resolve paths against the repository, never an inferred arbitrary data file.
def config_path(config: dict[str, Any], path: str | Path) -> Path:
    """Resolve a configured relative path against the explicit project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(config.get("_project_root", ".")) / candidate


def open_store(config: dict[str, Any]) -> xr.Dataset:
    """Open consolidated Zarr metadata without loading forecast arrays."""
    path = config_path(config, config["inputs"]["zarr"])
    return xr.open_zarr(path, chunks=None, consolidated=True)


# Metadata joins always use farm labels rather than Parquet/Zarr row order.
def load_metadata(path: str | Path, farm_ids: list[int] | None = None) -> dict[int, FarmMetadata]:
    """Read the explicitly selected metadata file and validate farm labels."""
    table = pd.read_parquet(path)
    required = {"farm_id", "zone", "total_power", "tracking_type_id", "tilt", "azimuth", "elevation_m"}
    if not required <= set(table):
        raise ValueError(f"Metadata missing fields: {sorted(required - set(table))}")
    if table.farm_id.duplicated().any() or table.farm_id.isna().any():
        raise ValueError("Metadata farm IDs must be unique and present")
    metadata: dict[int, FarmMetadata] = {}
    for row in table.itertuples(index=False):
        farm_id = int(row.farm_id)
        if farm_ids is not None and farm_id not in farm_ids:
            continue
        if not np.isfinite(row.total_power) or row.total_power <= 0:
            raise ValueError(f"Invalid DC nameplate capacity for farm {farm_id}")
        metadata[farm_id] = FarmMetadata(farm_id, str(row.zone), float(row.total_power), int(row.tracking_type_id), float(row.tilt), float(row.azimuth), float(row.elevation_m), float(row.pv_type_id) if hasattr(row, "pv_type_id") else None)
    if farm_ids is not None and set(farm_ids) != set(metadata):
        raise ValueError("Requested farm IDs absent from metadata")
    return metadata


# The store currently has timedeltas; numeric-hour fixtures are also supported.
def lead_timedeltas(coordinate: Any, numeric_units: str = "hours") -> pd.TimedeltaIndex:
    """Convert forecast lead encodings explicitly, rejecting unknown units."""
    values = np.asarray(coordinate)
    if np.issubdtype(values.dtype, np.timedelta64):
        return pd.TimedeltaIndex(values)
    if np.issubdtype(values.dtype, np.number):
        units = {"hours": "h", "hour": "h", "h": "h"}
        if numeric_units not in units:
            raise ValueError("Numeric forecast lead units must explicitly denote hours")
        return pd.to_timedelta(values, unit=units[numeric_units])
    raise ValueError("Forecast leads must be timedeltas or numeric hours")


def forecast_valid_times(cycles: Any, leads: Any, numeric_units: str = "hours") -> np.ndarray:
    """Return the cycle-by-lead UTC valid-time matrix."""
    cycles_utc = pd.DatetimeIndex(pd.to_datetime(cycles, utc=True)).tz_localize(None).to_numpy()
    return cycles_utc[:, None] + lead_timedeltas(leads, numeric_units).to_numpy()[None, :]


# Inspection reads only coordinates, metadata and a few already-declared attributes.
def inspect_data(config: dict[str, Any]) -> dict[str, Any]:
    """Describe the actual store contract without scanning or hashing its values."""
    dataset = open_store(config)
    try:
        metadata = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]))
        mapping = config.get("forecast_mapping", {})
        absent = [raw for raw in mapping.values() if raw not in dataset]
        if absent:
            raise ValueError(f"Configured raw forecast fields absent: {absent}")
        farm_ids = list(map(int, dataset.farm_id.values))
        if set(farm_ids) != set(metadata):
            raise ValueError("Zarr and Parquet farm IDs disagree")
        expected_dims = {"power": ("time", "farm_id"), "ghi": ("farm_id", "time"), "fraction_diffuse": ("farm_id", "time")}
        for name, dimensions in expected_dims.items():
            if name not in dataset or set(dataset[name].dims) != set(dimensions):
                raise ValueError(f"Unexpected historical dimension contract: {name}")
        for canonical, raw in mapping.items():
            if set(dataset[raw].dims) != {"farm_id", "cycle_time", "lead_time"}:
                raise ValueError(f"Unexpected forecast dimension contract: {canonical}/{raw}")
        leads = lead_timedeltas(dataset.lead_time.values, config.get("source", {}).get("numeric_lead_units", "hours"))
        times, cycles = pd.to_datetime(dataset.time.values, utc=True), pd.to_datetime(dataset.cycle_time.values, utc=True)
        # Validate supplied coordinates without reading any forecast data arrays.
        valid_matches = True
        if "valid_time" in dataset:
            valid_matches = bool(np.array_equal(dataset.valid_time.values, forecast_valid_times(cycles, leads)))
        if not valid_matches:
            raise ValueError("Stored valid_time does not equal cycle_time + lead_time")
        metadata_file = config_path(config, config["inputs"]["metadata_parquet"])
        marker = config_path(config, config["inputs"]["zarr"]) / ".zmetadata"
        stat = marker.stat()
        return {"zarr": str(config["inputs"]["zarr"]), "metadata_parquet": str(config["inputs"]["metadata_parquet"]), "dimensions": dict(dataset.sizes), "historical_start_utc": times[0].isoformat(), "historical_end_utc": times[-1].isoformat(), "cycle_start_utc": cycles[0].isoformat(), "cycle_end_utc": cycles[-1].isoformat(), "lead_encoding": str(dataset.lead_time.dtype), "lead_hours": list(map(float, leads / pd.Timedelta(hours=1))), "valid_time_verified": valid_matches, "farm_ids": farm_ids, "farms": [asdict(metadata[farm]) for farm in farm_ids], "variables": {name: {"dimensions": list(variable.dims), "dtype": str(variable.dtype), "attributes": dict(variable.attrs)} for name, variable in dataset.data_vars.items()}, "attributes": dict(dataset.attrs), "ensemble_axis_present": any("member" in name or "ensemble" in name for name in dataset.dims), "provenance": {"metadata_size_bytes": metadata_file.stat().st_size, "zarr_metadata_size_bytes": stat.st_size, "zarr_metadata_mtime_ns": stat.st_mtime_ns, "inspection": "coordinates_and_consolidated_metadata_only", "measured_power_basis": "AC_user_confirmed", "capacity_basis": "DC_user_confirmed"}}
    finally:
        dataset.close()


# Historical availability is interval completion plus declared telemetry latency.
def load_observations(dataset: xr.Dataset, farm_ids: list[int], variables: tuple[str, ...] = ("power", "ghi"), start: Any | None = None, end: Any | None = None, telemetry_latency_minutes: float = 0) -> ObservationBundle:
    """Read a chosen farm/date slice, transposing strictly by dimension names."""
    if telemetry_latency_minutes < 0:
        raise ValueError("Telemetry latency must be nonnegative")
    selected = dataset.sel(farm_id=farm_ids)
    if start is not None or end is not None:
        selected = selected.sel(time=slice(utc_timestamp(start).tz_localize(None) if start is not None else None, utc_timestamp(end).tz_localize(None) if end is not None else None))
    times = pd.to_datetime(selected.time.values, utc=True)
    rows: list[pd.DataFrame] = []
    for variable in variables:
        if variable not in selected:
            raise ValueError(f"Historical variable absent: {variable}")
        values = selected[variable].transpose("farm_id", "time").values
        for position, farm in enumerate(farm_ids):
            rows.append(pd.DataFrame({"farm_id": farm, "variable": variable, "time": times, "interval_end": times + pd.Timedelta(minutes=15), "available_at": times + pd.Timedelta(minutes=15 + telemetry_latency_minutes), "raw_value": values[position], "review_excluded": False}))
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ObservationBundle(frame, {"historical_interval_label": "start", "power_basis": "AC", "source_units": {name: dataset[name].attrs.get("units") for name in variables}, "telemetry_latency_minutes": telemetry_latency_minutes})


# Apply configured exclusion decisions without changing the original review file.
def apply_review_exclusions(bundle: ObservationBundle, review_path: str | Path | None, excluded_decisions: list[str] | None = None) -> ObservationBundle:
    """Carry forward explicitly reviewed exclusion intervals and review decisions."""
    if review_path is None:
        return bundle
    review = pd.read_csv(review_path)
    required = {"farm_id", "start_utc", "end_utc", "decision"}
    if not required <= set(review):
        raise ValueError("Review CSV requires farm_id,start_utc,end_utc,decision")
    frame = bundle.frame.copy()
    frame["review_decision"] = ""
    excluded = set(excluded_decisions or ["exclude", "excluded", "reject"])
    for row in review.itertuples(index=False):
        if pd.isna(row.decision):
            continue
        farm_mask = np.ones(len(frame), dtype=bool) if str(row.farm_id) == "portfolio" else frame.farm_id == int(row.farm_id)
        mask = farm_mask & (frame.time >= utc_timestamp(row.start_utc)) & (frame.time <= utc_timestamp(row.end_utc))
        frame.loc[mask, "review_decision"] = str(row.decision)
        if str(row.decision).lower() in excluded:
            frame.loc[mask, "review_excluded"] = True
    return ObservationBundle(frame, {**bundle.provenance, "review_file": str(review_path), "review_interval_end": "inclusive_native_start"})


# Native observations are treated before trimming, preserving endpoint evidence.
def prepare_analysis_data(config: dict[str, Any]) -> dict[str, Any]:
    """Write a small requested analysis slice with raw/treated values and manifest."""
    analysis = config.get("analysis", {})
    if analysis.get("policy_context", "complete_selected_farm_history") != "complete_selected_farm_history":
        raise ValueError("Only complete_selected_farm_history policy context is supported")
    farms = [int(config.get("farm_id", 1))]
    dataset = open_store(config)
    try:
        bundle = load_observations(dataset, farms, telemetry_latency_minutes=float(config.get("source", {}).get("telemetry_latency_minutes", 0)))
    finally:
        dataset.close()
    review = config.get("review", {})
    review_path = config_path(config, review["exclusions_csv"]) if review.get("exclusions_csv") else None
    bundle = apply_review_exclusions(bundle, review_path, review.get("excluded_decisions"))
    treated = prepare_observations(bundle, config.get("policy", {}))
    frame = treated.frame
    if analysis.get("start"):
        frame = frame.loc[frame.time >= utc_timestamp(analysis["start"])]
    if analysis.get("end"):
        frame = frame.loc[frame.time <= utc_timestamp(analysis["end"])]
    if frame.empty:
        raise ValueError("Requested analysis slice contains no observations")
    summary = observation_summary(frame)
    output = config_path(config, analysis.get("output_dir", "outputs/analysis/latest"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "observations_raw_and_treated.csv"
    frame.to_csv(path, index=False)
    manifest = {**summary, "farm_ids": farms, "start_utc": frame.time.min().isoformat(), "end_utc": frame.time.max().isoformat(), "output_path": str(path), "provenance": treated.provenance, "data_quality_status": "policy_treated_exploratory_not_fault_free_ground_truth", "review_exclusions_configured": review_path is not None}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _plot_observations(frame, output)
    manifest["plot_path"] = str(output / "raw_and_treated.png")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if review.get("mode", "exploratory") == "strict" and (summary["remaining_nonfinite"] or summary["flags"]["derived_review_excluded"]):
        raise ValueError(f"Strict analysis slice has missing or review-excluded observations; see {output / 'manifest.json'}")
    return manifest


# All forecast values from an eligible cycle share its publication time.
def select_forecast_vintage(dataset: xr.Dataset, farm_id: int, cutoff: Any, mapping: dict[str, str], latency_hours: float = 4, explicit_availability_variable: str | None = None, numeric_lead_units: str = "hours") -> ForecastBundle:
    """Select the latest available vintage; never average overlapping cycles."""
    if latency_hours < 0:
        raise ValueError("Forecast availability latency must be nonnegative")
    cycles = pd.DatetimeIndex(pd.to_datetime(dataset.cycle_time.values, utc=True))
    availability = cycles + pd.Timedelta(hours=latency_hours)
    status = "assumed_cycle_plus_latency"
    if explicit_availability_variable:
        variable = dataset[explicit_availability_variable]
        if set(variable.dims) == {"farm_id", "cycle_time"}:
            variable = variable.sel(farm_id=farm_id)
        if variable.dims != ("cycle_time",):
            raise ValueError("Explicit availability metadata requires cycle_time, optionally farm_id")
        availability = pd.DatetimeIndex(pd.to_datetime(variable.values, utc=True))
        status = "explicit_metadata_override"
    eligible = np.flatnonzero((availability <= utc_timestamp(cutoff)) & (cycles <= utc_timestamp(cutoff)))
    if not len(eligible):
        raise ValueError("No forecast vintage is available by this decision cutoff")
    position = eligible[np.argmax(cycles.asi8[eligible])]
    cycle = cycles[position]
    selected = dataset.sel(farm_id=farm_id).isel(cycle_time=int(position))
    leads = lead_timedeltas(selected.lead_time.values, numeric_lead_units)
    values = pd.DataFrame(index=cycle + leads)
    values.index.name = "valid_time"
    units: dict[str, str] = {}
    for canonical, raw in mapping.items():
        if raw not in selected:
            raise ValueError(f"Configured forecast mapping is absent: {canonical} -> {raw}")
        if selected[raw].dims != ("lead_time",):
            raise ValueError(f"Forecast {raw} does not have named lead_time dimension after farm/cycle selection")
        values[canonical] = selected[raw].values
        units[canonical] = selected[raw].attrs.get("units", "unknown")
    values["lead_hours"] = leads / pd.Timedelta(hours=1)
    provenance = {"availability_policy": status, "assumed_latency_hours": latency_hours, "source_lead_dtype": str(dataset.lead_time.dtype), "source_lead_encoding_units": dataset.lead_time.encoding.get("units", numeric_lead_units), "source_units": dict(units), "source_lead_attributes": dict(dataset.lead_time.attrs), "source_lead_values": list(map(str, selected.lead_time.values)), "canonical_to_raw": mapping, "decision_cutoff_utc": utc_timestamp(cutoff).isoformat(), "vintage_selection": "latest_available_single_cycle", "ensemble_axis": "explicit_length_one", "future_forecast_revision_assumption": "freeze_until_new_cycle_available"}
    return ForecastBundle(int(farm_id), cycle, availability[position], values, units, provenance=provenance)


# Mean/accumulation redistribution respects source interval endpoints.
def align_forecast(bundle: ForecastBundle, starts: Any, semantics: dict[str, str], nonnegative_allowlist: list[str] | None = None) -> ForecastBundle:
    """Map one eligible hourly vintage to 15-minute interval model inputs."""
    if len(bundle.members) > 1:
        if "member" not in bundle.values or set(bundle.values.member.unique()) != set(bundle.members):
            raise ValueError("Supplied multiple members require an explicit matching member column")
        aligned_members = []
        for member in bundle.members:
            selected = ForecastBundle(bundle.farm_id, bundle.forecast_vintage, bundle.available_at, bundle.values.loc[bundle.values.member == member].copy(), dict(bundle.units), (member,), bundle.member_origin, dict(bundle.provenance))
            aligned_members.append(align_forecast(selected, starts, semantics, nonnegative_allowlist))
        return ForecastBundle(bundle.farm_id, bundle.forecast_vintage, bundle.available_at, pd.concat([member.values for member in aligned_members]), aligned_members[0].units, bundle.members, bundle.member_origin, aligned_members[0].provenance)
    starts = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    if np.any(starts.minute % 15) or np.any(starts.second) or np.any(starts.microsecond):
        raise ValueError("Model intervals must start on quarter-hour boundaries")
    if starts.has_duplicates or not starts.is_monotonic_increasing:
        raise ValueError("Model interval starts must be unique and increasing")
    source, flags = clip_negative_forecasts(bundle.values, nonnegative_allowlist or [])
    frame = pd.DataFrame(index=starts)
    frame.index.name = "time"
    centers = starts + pd.Timedelta(minutes=7.5)
    source_ns = source.index.as_unit("ns").asi8
    for canonical in bundle.units:
        kind = semantics.get(canonical, semantics.get("default", "instantaneous"))
        if kind in {"hourly_mean", "hourly_accumulation"}:
            # For an exact hour start use the following hour's END label.
            ends = starts.floor("h") + pd.Timedelta(hours=1)
            data = source[canonical].reindex(ends).to_numpy(dtype=float)
            changed = flags[canonical].reindex(ends, fill_value=False).to_numpy(dtype=bool)
            if kind == "hourly_accumulation":
                data = data / 4.0
            frame[canonical] = data
            frame[f"{canonical}_negative_forecast_zero"] = changed
            frame[f"{canonical}_source_start"] = ends - pd.Timedelta(hours=1)
            frame[f"{canonical}_source_end"] = ends
        elif kind == "instantaneous":
            # Adjacent forecast values may interpolate only inside this vintage;
            # NaN endpoints are structural gaps and are never skipped across.
            target_ns = centers.as_unit("ns").asi8
            right = np.searchsorted(source_ns, target_ns, side="left")
            left = np.clip(right - 1, 0, max(len(source_ns) - 1, 0))
            right_clipped = np.clip(right, 0, max(len(source_ns) - 1, 0))
            raw = source[canonical].to_numpy(dtype=float)
            data = np.full(len(starts), np.nan)
            if len(raw):
                exact = (right < len(raw)) & (source_ns[right_clipped] == target_ns)
                data[exact] = raw[right_clipped[exact]]
                inside = (right > 0) & (right < len(raw)) & ~exact
                fraction = (target_ns[inside] - source_ns[left[inside]]) / (source_ns[right_clipped[inside]] - source_ns[left[inside]])
                data[inside] = raw[left[inside]] + fraction * (raw[right_clipped[inside]] - raw[left[inside]])
            frame[canonical] = data
            frame[f"{canonical}_negative_forecast_zero"] = flags[canonical].iloc[left].to_numpy() | flags[canonical].iloc[right_clipped].to_numpy() if len(raw) else False
        else:
            raise ValueError(f"Unknown forecast temporal semantics: {canonical}={kind}")
    frame["forecast_vintage"] = bundle.forecast_vintage
    frame["available_at"] = bundle.available_at
    frame["member"] = bundle.members[0]
    frame["lead_hours"] = (centers - bundle.forecast_vintage) / pd.Timedelta(hours=1)
    if nonnegative_allowlist:
        baseline = align_forecast(bundle, starts, semantics, None)
        for canonical in bundle.units:
            frame[f"{canonical}_raw_aligned"] = baseline.values[canonical].to_numpy()
    else:
        for canonical in bundle.units:
            frame[f"{canonical}_raw_aligned"] = frame[canonical].to_numpy()
    units = dict(bundle.units)
    if "precipitation_mm" in units:
        units["precipitation_mm"] = "mm_per_model_interval"
    return ForecastBundle(bundle.farm_id, bundle.forecast_vintage, bundle.available_at, frame, units, bundle.members, bundle.member_origin, {**bundle.provenance, "alignment": "uniform_conservative_hourly_interval_means_and_accumulations;instantaneous_at_interval_midpoint", "model_interval_minutes": 15, "forecast_gaps_filled": False, "raw_aligned_fields": "source_values_before_negative_floor_after_identical_temporal_alignment"})


# Configuration checks are pure and do not open datasets or fetch dependencies.
def validate_source_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the declared source/policy/PV contract without reading data."""
    if not config.get("inputs"):
        return {"source_configured": False}
    for key in ("zarr", "metadata_parquet"):
        if not isinstance(config["inputs"].get(key), str) or not config["inputs"][key]:
            raise ValueError(f"inputs.{key} must explicitly name a source path")
    source = config.get("source", {})
    if source.get("timezone", "UTC") != "UTC":
        raise ValueError("The verified source timezone is UTC")
    if source.get("measured_power_basis", "AC") != "AC":
        raise ValueError("Measured farm power is already AC")
    if source.get("historical_label", "start") != "start" or source.get("forecast_radiation_label", "end") != "end":
        raise ValueError("Historical and forecast radiation labels must be start/end")
    for key in ("availability_lag_hours", "telemetry_latency_minutes"):
        value = float(source.get(key, 4 if key == "availability_lag_hours" else 0))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"source.{key} must be finite and nonnegative")
    mapping = config.get("forecast_mapping", {})
    if not mapping or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("forecast_mapping must explicitly map canonical names to raw fields")
    if "forecast_ghi" not in mapping:
        raise ValueError("forecast_ghi must have its own forecast-variable mapping")
    if mapping["forecast_ghi"] == "ghi":
        raise ValueError("Historical satellite ghi cannot substitute for forecast shortwave")
    policy = config.get("policy", {})
    plateau = policy.get("plateau", {})
    minimum = plateau.get("minimum_run", 4)
    if not isinstance(minimum, int) or minimum < 4:
        raise ValueError("policy plateau minimum_run must be an integer >=4")
    for key in ("atol_kw", "rtol"):
        value = float(plateau.get(key, 0.01 if key == "atol_kw" else 0.0001))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Plateau {key} must be finite and nonnegative")
    maximum = policy.get("interpolation", {}).get("maximum_gap_hours")
    if maximum is not None and (not np.isfinite(float(maximum)) or float(maximum) < 0):
        raise ValueError("Interpolation maximum_gap_hours must be nonnegative or null")
    signed = {"temperature_c", "dew_point_c", "wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m", "price", "price_eur_mwh"}
    allowlist = policy.get("nonnegative_forecasts", [])
    if not isinstance(allowlist, list) or signed.intersection(allowlist):
        raise ValueError("Nonnegative-forecast allowlist cannot contain signed weather/prices")
    night_window_mask(pd.DatetimeIndex(["2025-01-01T00:00Z"]), policy.get("night", {}))
    if config.get("review", {}).get("mode", "exploratory") not in {"strict", "exploratory"}:
        raise ValueError("review.mode must be strict or exploratory")
    pv = config.get("pv", {})
    if pv:
        if pv.get("fraction_diffuse_enabled", False):
            raise ValueError("fraction_diffuse remains disabled until independently validated")
        if pv.get("source_azimuth_positive") not in {"east", "west"}:
            raise ValueError("PV source azimuth direction must explicitly be east or west")
        if pv.get("direct_radiation_interpretation") != "horizontal_direct" or pv.get("shortwave_interpretation") != "ghi":
            raise ValueError("PV config requires GHI and horizontal-direct interpretations")
        location = pv.get("location", {})
        for key, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
            value = float(location.get(key, np.nan))
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"PV {key} lies outside its geographic range")
        if location.get("location_status") == "proxy" and not location.get("acknowledged", False):
            raise ValueError("Proxy PV coordinates must be explicitly acknowledged")
        for key, default, lower, upper in (("albedo", 0.2, 0, 1), ("low_sun_zenith_deg", 85, 0, 90)):
            value = float(pv.get(key, default))
            if not np.isfinite(value) or not lower <= value <= upper or (key == "low_sun_zenith_deg" and value in {0, 90}):
                raise ValueError(f"Invalid PV {key}")
        if not np.isfinite(float(pv.get("dni_limit_w_m2", 1400))) or float(pv.get("dni_limit_w_m2", 1400)) <= 0:
            raise ValueError("DNI limiting threshold must be finite and positive")
        tracker = pv.get("tracker", {})
        if tracker:
            if not tracker.get("assumptions_acknowledged", False):
                raise ValueError("Tracker geometry assumptions must be acknowledged")
            gcr = float(tracker.get("gcr", np.nan))
            if not np.isfinite(gcr) or not 0 < gcr <= 1:
                raise ValueError("Tracker gcr must lie in (0,1]")
        temperature = pv.get("temperature", {})
        if temperature.get("wind_height_m", 10) not in {10, 100}:
            raise ValueError("PV wind source height must be 10 or 100 m")
        if not temperature.get("wind_as_module_proxy_acknowledged", False):
            raise ValueError("PV wind-to-module proxy must be acknowledged")
        gamma = float(pv.get("dc", {}).get("gamma_pdc_per_c", np.nan))
        if not np.isfinite(gamma):
            raise ValueError("PVWatts gamma coefficient must be finite and configured")
        output = pv.get("output", {})
        if output.get("ac_limit_mw") is not None and (not np.isfinite(float(output["ac_limit_mw"])) or float(output["ac_limit_mw"]) <= 0):
            raise ValueError("PV AC limit must be finite and positive or null")
        if output.get("method") not in {"explicit_efficiency", "calibrated_scale"}:
            raise ValueError("PV-output conversion method must be explicit")
        if output.get("method") == "explicit_efficiency":
            value = float(output.get("pv_output_efficiency", np.nan))
            if not np.isfinite(value) or not 0 < value <= 1 or not output.get("assumption_status"):
                raise ValueError("PV-output efficiency requires a fraction and assumption status")
    return {"source_configured": True, "policy_version": policy.get("version", "requested-v1"), "source_timezone": "UTC", "measured_power_basis": "AC", "review_mode": config.get("review", {}).get("mode", "exploratory")}


# Figures show policy adjustments explicitly, without treating them as diagnoses.
def _plot_observations(frame: pd.DataFrame, output: Path) -> None:
    """Save non-interactive native power/GHI comparisons for the output slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variables = list(frame.variable.unique())
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 3 * len(variables)), squeeze=False, sharex=True)
    for axis, variable in zip(axes[:, 0], variables):
        group = frame.loc[frame.variable == variable]
        axis.plot(group.time, group.raw_value, label="raw measured AC kW" if variable == "power" else "raw satellite GHI W/m²", linewidth=1.4)
        axis.plot(group.time, group.treated_value, "--", label="policy-treated", linewidth=1.1)
        filled = group.measurement_interpolated
        axis.scatter(group.loc[filled, "time"], group.loc[filled, "treated_value"], marker="x", color="red", label="interpolated", s=24)
        axis.set_ylabel(variable)
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("UTC native interval start")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "raw_and_treated.png", dpi=130)
    plt.close(fig)
