"""Frozen auction-time weather views and unchanged AC targets for PV calibration."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import merge_mapping
from ..data import (align_forecast, config_path, load_metadata, night_window_mask,
                    open_store, select_forecast_vintage)
from ..data.policy import plateau_runs
from ..markets import AuctionCalendar
from .pv import forecast_pv
from .pv_artifacts import digest
from .pv_outages import apply_sustained_zero_exclusion


def farm_settings(config: dict, farm) -> dict:
    """Resolve farm overrides after replacing the selected-farm automatic AC cap."""
    declared = config["farms"][str(farm.farm_id)]
    if not np.isclose(float(declared["ac_capacity_kw"]), farm.ac_capacity_kw):
        raise ValueError("Configured farm AC rating disagrees with measured metadata")
    settings = deepcopy(config.get("_pv_shared_settings", config["pv"]))
    settings["output"]["ac_limit_mw"] = farm.ac_capacity_mw
    settings = merge_mapping(settings, declared.get("pv") or {})
    if settings["output"].get("ac_limit_mw") == "auto":
        settings["output"]["ac_limit_mw"] = farm.ac_capacity_mw
    settings["parameter_registry"] = None
    return settings


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash actual aligned values and keys rather than file modification times."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _source_fingerprint(dataset, config, farm, settings) -> str:
    """Read bounded source slices; later appended history does not alter old hashes."""
    hasher = hashlib.sha256()
    contract = {"metadata": asdict(farm), "pv": settings, "source": config["source"],
        "mapping": config["forecast_mapping"], "semantics": config["forecast_semantics"],
        "policy": config["policy"], "review": config.get("review"),
        "market": config["market"], "start": config["pv_calibration"]["start_day"],
        "end": config["pv_calibration"]["end_day_exclusive"],
        "scale": config["forecast"]["realized_power_scale_to_mw"],
        "outage_threshold": config["pv_calibration"]["outage_ghi_threshold_w_m2"],
        "outage_exclusion": config["pv_data_quality"]["sustained_zero_outage"]}
    hasher.update(digest(contract).encode())
    for name in sorted(set(config["forecast_mapping"].values()) | {"time", "cycle_time", "lead_time", "power", "ghi"}):
        array = dataset[name]
        hasher.update(digest({"name": name, "dims": array.dims, "shape": array.shape,
                              "dtype": str(array.dtype), "attrs": array.attrs}).encode())
        hasher.update(np.ascontiguousarray(array.values).tobytes())
    explicit = config["source"].get("explicit_availability_variable")
    if explicit:
        hasher.update(np.ascontiguousarray(dataset[explicit].values).tobytes())
    review = config.get("review", {}).get("exclusions_csv")
    if review:
        hasher.update(config_path(config, review).read_bytes())
    return hasher.hexdigest()


def prepare_corpus(config: dict, farm_id: int, destination: Path, progress=None, *, retain_weather=False):
    """Cache one fixed set of forecast/measurement pairs, preserving all missing rows."""
    section = config["pv_calibration"]
    timezone = section["delivery_timezone"]
    start = pd.Timestamp(section["start_day"]).tz_localize(timezone).tz_convert("UTC")
    end = pd.Timestamp(section["end_day_exclusive"]).tz_localize(timezone).tz_convert("UTC")
    farm = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
    settings = farm_settings(config, farm)
    manifest_path = destination / "json" / "input_provenance.json"
    corpus_path = destination / "parquet" / "corpus.parquet"
    with open_store(config) as store:
        dataset = store.sel(farm_id=[farm_id],
            time=slice(start.tz_localize(None), end.tz_localize(None)),
            cycle_time=slice((start-pd.Timedelta(days=3)).tz_localize(None), end.tz_localize(None))).load()
    fingerprint = _source_fingerprint(dataset, config, farm, settings)
    if manifest_path.exists() and corpus_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_fingerprint"] != fingerprint:
            raise ValueError(f"PV study source fingerprint changed for farm {farm_id}; choose a fresh run ID")
        cached = pd.read_parquet(corpus_path)
        if frame_fingerprint(cached) != manifest["aligned_fingerprint"]:
            raise ValueError("PV calibration corpus checksum mismatch")
        if retain_weather and not set(config["forecast_mapping"]).issubset(cached.columns):
            raise ValueError("Cached corpus lacks requested weather features; choose a fresh run ID")
        dataset.close()
        return cached, farm, settings, manifest
    # The calendar supplies DST-aware quarter-hours and auction-specific suffixes.
    calendar_settings = deepcopy(config["market"])
    calendar_settings.update(valid_from=section["start_day"],
        valid_to=(pd.Timestamp(section["end_day_exclusive"])-pd.Timedelta(days=1)).date().isoformat())
    calendar = AuctionCalendar.from_config(calendar_settings)
    mapping = config["forecast_mapping"]
    rows = []
    days = pd.date_range(section["start_day"], section["end_day_exclusive"], freq="D", inclusive="left")
    for number, day in enumerate(days):
        grid = calendar.delivery_grid(day.date())
        for event in calendar.schedule(day.date()):
            starts = grid.starts[list(event.eligible_intervals)]
            if not len(starts):
                continue
            weather = select_forecast_vintage(dataset, farm_id, event.submission_at, mapping,
                latency_hours=float(config["source"]["availability_lag_hours"]),
                explicit_availability_variable=config["source"].get("explicit_availability_variable"),
                numeric_lead_units=config["source"].get("numeric_lead_units", "hours"))
            weather = align_forecast(weather, starts, config["forecast_semantics"], config["policy"]["nonnegative_forecasts"])
            pv = forecast_pv(weather, farm, settings, config["policy"]["night"])
            frame = pv.values.copy()
            frame["delivery_day"] = day.date().isoformat()
            frame["auction"] = event.auction
            frame["decision_cutoff_utc"] = event.submission_at
            frame["forecast_cycle_utc"] = weather.forecast_vintage
            frame["forecast_available_at_utc"] = weather.available_at
            frame["interval_start_utc"] = starts
            frame["baseline_mw"] = frame.pv_ac_mw
            night = night_window_mask(starts, config["policy"]["night"])
            frame.loc[night & ~np.isfinite(frame.baseline_mw), "baseline_mw"] = 0.0
            frame["daylight"] = (frame.solar_zenith_deg < 90) & ~night
            rows.append(frame)
        if progress and (number % 60 == 0 or number+1 == len(days)):
            progress(f"farm {farm_id}: aligned {number+1}/{len(days)} delivery days")
    result = pd.concat(rows, ignore_index=True)
    # Measurements are quarter-hour AC means; no loss, interpolation or forecast fill.
    times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values, utc=True))
    power = pd.Series(dataset.power.sel(farm_id=farm_id).values, index=times)
    ghi = pd.Series(dataset.ghi.sel(farm_id=farm_id).values, index=times)
    index = pd.DatetimeIndex(result.interval_start_utc)
    result["measured_ac_mw"] = power.reindex(index).to_numpy() * float(config["forecast"]["realized_power_scale_to_mw"])
    result["satellite_ghi_w_m2"] = ghi.reindex(index).to_numpy()
    result["target_available_at_utc"] = index + pd.Timedelta(minutes=15 + float(config["source"]["telemetry_latency_minutes"]))
    plateau = pd.Series(False, index=times)
    for run in plateau_runs(power.values, times, config["policy"]["plateau"]):
        plateau.iloc[run] = True
    result["plateau_flag"] = plateau.reindex(index, fill_value=False).to_numpy()
    result["suspected_outage"] = ((result.measured_ac_mw == 0)
        & (result.satellite_ghi_w_m2 > float(section["outage_ghi_threshold_w_m2"])))
    result["review_excluded"] = False
    result["eligible"] = (np.isfinite(result.measured_ac_mw) & (result.measured_ac_mw >= 0)
        & np.isfinite(result.baseline_mw))
    result=apply_sustained_zero_exclusion(result,config,farm_id=farm_id)
    # Retain only columns needed by the common forward model and diagnostics.
    keep = ["interval_start_utc", "delivery_day", "auction", "decision_cutoff_utc",
        "forecast_cycle_utc", "forecast_available_at_utc", "target_available_at_utc", "lead_hours",
        "solar_zenith_deg", "solar_azimuth_deg", "ghi_w_m2", "bhi_w_m2", "dhi_w_m2", "dni_w_m2",
        "temperature_c", "wind_speed_m_s", "pv_input_valid", "bhi_exceeds_ghi", "dni_limited",
        "baseline_mw", "measured_ac_mw", "satellite_ghi_w_m2", "daylight", "plateau_flag",
        "suspected_outage", "outage_excluded", "review_excluded", "eligible"]
    if retain_weather:
        keep += [name for name in mapping if name not in keep]
    result = result[keep]
    manifest = {"source_fingerprint": fingerprint, "aligned_fingerprint": frame_fingerprint(result),
        "farm_metadata": asdict(farm), "rows": len(result), "eligible_rows": int(result.eligible.sum()),
        "missing_or_invalid_target_rows": int((~np.isfinite(result.measured_ac_mw) | (result.measured_ac_mw < 0)).sum()),
        "outage_excluded_views": int(result.outage_excluded.sum()),
        "outage_excluded_physical_quarters": int(result.loc[result.outage_excluded,'interval_start_utc'].nunique()),
        "forecast_availability_lag_hours": config["source"]["availability_lag_hours"],
        "target": "raw_measured_AC_MW_no_losses_no_imputation", "forecast_selection": "auction_submission_cutoff",
        "satellite_ghi_role": "diagnostics_only", "auction_views": "scheduled_information_times_not_market_participation",
        "settings": settings}
    result.to_parquet(corpus_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    dataset.close()
    return result, farm, settings, manifest
