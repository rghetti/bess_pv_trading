"""Metadata-only discovery of delivery-day weather ensemble coverage.

Archives are selected by their stored preceding-day 00 UTC cycles, never by
filename or month. Checksums here cover metadata and coordinate values only;
finite weather values, measured power and PV conversion still require preflight.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _utc_coordinate(values, name):
    """Stored Zarr clock coordinates follow the project's explicit UTC convention."""
    result = pd.DatetimeIndex(values)
    if result.hasnans or result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError(f"{name} must be unique increasing nonmissing timestamps")
    return result.tz_localize("UTC") if result.tz is None else result.tz_convert("UTC")


def _metadata_checksum(path):
    """Hash consolidated metadata without reading any weather or power chunks."""
    for filename in (".zmetadata", "zarr.json"):
        candidate = path / filename
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    raise ValueError(f"No consolidated Zarr metadata found at {path}")


def _coordinate_checksum(values):
    """Hash semantic coordinates in a portable representation, separately from payload."""
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inspect_archive(path):
    """Load only coordinate arrays; weather variables are inspected through shapes."""
    import xarray as xr

    metadata_sha256 = _metadata_checksum(path)
    with xr.open_zarr(path, consolidated=True, chunks=None, decode_timedelta=True) as dataset:
        required = {"farm_id", "cycle_time", "lead_time", "member"}
        if not required.issubset(dataset.coords):
            raise ValueError(f"Ensemble archive {path} lacks required coordinates: {sorted(required - set(dataset.coords))}")
        for name in required:
            if dataset[name].dims != (name,):
                raise ValueError(f"Ensemble archive {path} requires a one-dimensional {name} coordinate")
        cycles = _utc_coordinate(dataset.cycle_time.values, "Ensemble cycle_time")
        leads = np.asarray(dataset.lead_time.values)
        if not np.issubdtype(leads.dtype, np.timedelta64):
            raise ValueError("Ensemble lead_time must decode to a duration coordinate")
        lead_times = pd.TimedeltaIndex(leads).as_unit("ns")
        if (not len(cycles) or not len(lead_times) or lead_times.hasnans or lead_times.has_duplicates
                or not lead_times.is_monotonic_increasing or np.any(lead_times.asi8 < 0)):
            raise ValueError("Ensemble cycles and nonnegative leads must be nonempty, unique and increasing")
        farms = tuple(str(value) for value in dataset.farm_id.values)
        members = tuple(str(value) for value in dataset.member.values)
        if not farms or len(set(farms)) != len(farms) or not members or len(set(members)) != len(members):
            raise ValueError("Ensemble farm and member coordinates must be nonempty and unique")
        valid_time_checked = "valid_time" in dataset
        if valid_time_checked:
            if dataset.valid_time.dims != ("cycle_time", "lead_time"):
                raise ValueError("Ensemble valid_time must align with (cycle_time, lead_time)")
            expected = cycles.tz_localize(None).to_numpy(dtype="datetime64[ns]")[:, None] + lead_times.to_numpy()[None, :]
            if not np.array_equal(dataset.valid_time.values.astype("datetime64[ns]"), expected):
                raise ValueError("Ensemble valid_time differs from cycle plus lead")
        # Variable dimensions are metadata, not a read of the weather values.
        variables = {name: list(array.dims) for name, array in dataset.data_vars.items() if name != "valid_time"}
    coordinates = {"cycle_times_utc": [stamp.isoformat() for stamp in cycles],
                   "lead_nanoseconds": lead_times.asi8.tolist(), "farm_ids": list(farms), "member_ids": list(members)}
    record = {"path": str(path), "status": "present", "metadata_sha256": metadata_sha256,
              "coordinate_sha256": _coordinate_checksum(coordinates),
              "checksum_scope": "consolidated metadata and coordinate values; weather payload excluded",
              "cycle_count": len(cycles), "cycle_times_utc": coordinates["cycle_times_utc"],
              "lead_hours": (lead_times.total_seconds() / 3600).tolist(),
              "farm_ids": list(farms), "member_ids": list(members), "member_count": len(members),
              "variables": variables, "valid_time_checked": valid_time_checked,
              "stored_clock_convention": "naive Zarr clock coordinates are UTC"}
    return record, cycles, lead_times


def _measurement_coordinates(path, coordinate, variable):
    """Inspect timestamp overlap; do not mistake it for finite per-farm production."""
    import xarray as xr

    with xr.open_zarr(path, consolidated=True, chunks=None, decode_timedelta=True) as dataset:
        if coordinate not in dataset.coords or dataset[coordinate].dims != (coordinate,):
            raise ValueError(f"Measurement archive lacks one-dimensional {coordinate}")
        if variable not in dataset or coordinate not in dataset[variable].dims:
            raise ValueError(f"Measurement archive lacks time-aligned {variable}")
        starts = _utc_coordinate(dataset[coordinate].values, "Measurement timestamps")
        farms = [str(value) for value in dataset.farm_id.values] if "farm_id" in dataset.coords else []
    metadata = {"path": str(path), "status": "present", "metadata_sha256": _metadata_checksum(path),
                "coordinate_sha256": _coordinate_checksum({"starts_utc": starts.astype(str).tolist(), "farm_ids": farms}),
                "timestamp_count": len(starts), "farm_ids": farms,
                "coverage_scope": "timestamps and farm coordinates; finite power and quality not checked"}
    return starts, metadata


def _delivery_grid(day, timezone, interval_minutes):
    """Use local midnight boundaries so spring and autumn DST lengths stay exact."""
    start = pd.Timestamp(day).tz_localize(timezone)
    stop = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(start, stop, freq=f"{interval_minutes}min", inclusive="left").tz_convert("UTC")


def _lead_coverage(cycle, leads, starts, alignment_modes):
    """Mirror midpoint interpolation and hourly endpoint selection without weather reads."""
    source = (cycle + leads).as_unit("ns")
    centers = starts + (starts[1] - starts[0]) / 2
    missing = {}
    if "instantaneous" in alignment_modes:
        # Linear interpolation can bridge stored leads; extrapolation is forbidden.
        outside = (centers < source[0]) | (centers > source[-1])
        missing["unbracketed_midpoints"] = int(outside.sum())
    if set(alignment_modes) & {"hourly_mean", "hourly_accumulation"}:
        endpoints = starts.floor("h") + pd.Timedelta(hours=1)
        missing["missing_hourly_endpoints"] = int((source.get_indexer(endpoints) < 0).sum())
    return missing


def inventory_ensemble_sources(paths, *, delivery_days=None, farm_ids=None, timezone="Europe/Rome",
                               availability_lag_hours=6, first_cutoffs=None, measured_starts=None,
                               measurement_source=None, measurement_time_coordinate="time", measurement_variable="power",
                               interval_minutes=15, required_variables=(),
                               alignment_modes=("instantaneous", "hourly_mean"), overlap_policy="error"):
    """Report actual coordinate coverage and the reason every requested date fails.

    Explicit ``first_cutoffs`` should come from the configured auction calendar.
    Without them only availability before physical delivery is checked, and that
    weaker scope is recorded. Measurement coordinates may be supplied directly or
    loaded from ``measurement_source``; neither path checks measured values.

    Overlapping archives fail by default because metadata cannot prove equal
    weather payloads. An explicit first/last policy records all candidate sources.
    """
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    sources = [Path(path).resolve() for path in raw_paths]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Ensemble inventory needs nonempty distinct source paths")
    if (isinstance(availability_lag_hours, bool) or not np.isfinite(availability_lag_hours)
            or availability_lag_hours < 0):
        raise ValueError("Ensemble availability lag must be finite and nonnegative")
    if (isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int)
            or interval_minutes != 15):
        raise ValueError("The current ensemble alignment contract uses 15-minute delivery intervals")
    modes = tuple(alignment_modes)
    if not modes or set(modes) - {"instantaneous", "hourly_mean", "hourly_accumulation"}:
        raise ValueError("Unknown or empty ensemble alignment modes")
    if overlap_policy not in {"error", "first", "last"}:
        raise ValueError("Ensemble overlap_policy must be error, first or last")
    if measured_starts is not None and measurement_source is not None:
        raise ValueError("Supply measured_starts or measurement_source, not both")
    requested_farms = None if farm_ids is None else tuple(str(value) for value in farm_ids)
    if requested_farms is not None and (not requested_farms or len(set(requested_farms)) != len(requested_farms)):
        raise ValueError("Requested ensemble farms must be nonempty and unique")
    archive_records, archives, absent = [], [], []
    for path in sources:
        if not path.exists():
            absent.append(str(path))
            archive_records.append({"path": str(path), "status": "missing"})
            continue
        record, cycles, leads = _inspect_archive(path)
        archive_records.append(record)
        archives.append((record, set(cycles), leads))
    if delivery_days is None:
        candidates = {stamp.date().isoformat() for _, cycles, _ in archives for cycle in cycles
                      if cycle.hour == 0 and cycle.minute == 0 and cycle.second == 0
                      for stamp in [cycle + pd.Timedelta(days=1)]}
        days = sorted(candidates)
    else:
        supplied = list(delivery_days)
        days = sorted(str(day) for day in supplied)
        if len(set(days)) != len(days):
            raise ValueError("Requested delivery days must be unique")
        parsed = pd.to_datetime(days, format="%Y-%m-%d", errors="raise")
        if list(parsed.strftime("%Y-%m-%d")) != days:
            raise ValueError("Requested delivery days must be YYYY-MM-DD dates")
    measurement = {"status": "not_checked", "coverage_scope": "measurement overlap was not supplied"}
    measurements = None
    if measurement_source is not None:
        path = Path(measurement_source).resolve()
        if path.exists():
            measurements, measurement = _measurement_coordinates(path, measurement_time_coordinate, measurement_variable)
        else:
            measurement = {"path": str(path), "status": "missing"}
    elif measured_starts is not None:
        stamps = pd.DatetimeIndex(measured_starts)
        if stamps.tz is None:
            raise ValueError("Caller-supplied measured timestamps must be timezone aware")
        measurements = _utc_coordinate(stamps, "Measured timestamps")
        measurement = {"status": "supplied_coordinates", "timestamp_count": len(measurements),
            "coordinate_sha256": _coordinate_checksum(measurements.astype(str).tolist()),
            "coverage_scope": "timestamps only; finite power and quality not checked"}
    records = []
    for day in days:
        starts = _delivery_grid(day, timezone, interval_minutes)
        cycle = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize("UTC")
        available = cycle + pd.Timedelta(hours=availability_lag_hours)
        if first_cutoffs is not None:
            if day not in first_cutoffs:
                raise ValueError(f"No first decision cutoff was supplied for {day}")
            cutoff = pd.Timestamp(first_cutoffs[day])
            if pd.isna(cutoff) or cutoff.tzinfo is None:
                raise ValueError("First decision cutoffs must be timezone aware")
            cutoff = cutoff.tz_convert("UTC")
            cutoff_scope = "configured_first_decision"
        else:
            cutoff, cutoff_scope = starts[0], "physical_delivery_start_only_not_auction_gate"
        candidates = [archive for archive in archives if cycle in archive[1]
                      and (requested_farms is None or set(requested_farms).issubset(archive[0]["farm_ids"]))]
        if len(candidates) > 1 and overlap_policy == "error":
            raise ValueError(f"Ambiguous ensemble cycle {cycle.isoformat()} for {day}: "
                             f"{[candidate[0]['path'] for candidate in candidates]}; select an explicit overlap policy")
        chosen = None if not candidates else candidates[-1] if overlap_policy == "last" else candidates[0]
        reasons = []
        if chosen is None:
            reasons.append("missing_preceding_day_cycle_or_requested_farm")
        if available > cutoff:
            reasons.append("forecast_available_after_first_decision")
        missing_measurements = None
        if measurements is not None:
            missing_measurements = int((measurements.get_indexer(starts) < 0).sum())
            if missing_measurements:
                reasons.append("incomplete_measurement_timestamp_coverage")
        elif measurement.get("status") == "missing":
            reasons.append("measurement_source_missing")
        if (requested_farms is not None and measurement.get("farm_ids") is not None
                and not set(requested_farms).issubset(measurement["farm_ids"])):
            reasons.append("requested_farm_missing_from_measurements")
        missing_leads, missing_variables = {}, []
        if chosen is not None:
            source, _, leads = chosen
            missing_leads = _lead_coverage(cycle, leads, starts, modes)
            if any(missing_leads.values()):
                reasons.append("incomplete_forecast_lead_coverage")
            expected_dimensions = {"farm_id", "cycle_time", "lead_time", "member"}
            missing_variables = [name for name in required_variables
                if name not in source["variables"] or set(source["variables"][name]) != expected_dimensions]
            if missing_variables:
                reasons.append("missing_or_misaligned_weather_variables")
        records.append({"delivery_day": day, "forecast_vintage_utc": cycle.isoformat(),
            "available_at_utc": available.isoformat(), "availability": "assumed cycle plus configured latency",
            "first_decision_cutoff_utc": cutoff.isoformat(), "availability_cutoff_scope": cutoff_scope,
            "interval_count": len(starts), "delivery_start_utc": starts[0].isoformat(),
            "delivery_end_utc": (starts[-1] + pd.Timedelta(minutes=interval_minutes)).isoformat(),
            "eligible": not reasons, "ineligibility_reasons": reasons,
            "source": None if chosen is None else chosen[0]["path"],
            "source_candidates": [candidate[0]["path"] for candidate in candidates],
            "member_count": None if chosen is None else chosen[0]["member_count"],
            "member_ids": [] if chosen is None else chosen[0]["member_ids"],
            "missing_measurement_intervals": missing_measurements,
            "lead_coverage": missing_leads, "missing_or_misaligned_variables": missing_variables})
    return {"schema": "ensemble-delivery-inventory-v1", "sources": archive_records, "days": records,
            "eligible_delivery_days": [row["delivery_day"] for row in records if row["eligible"]],
            "missing_sources": absent, "measurements": measurement,
            "settings": {"timezone": timezone, "availability_lag_hours": float(availability_lag_hours),
                "interval_minutes": interval_minutes, "farm_ids": None if requested_farms is None else list(requested_farms),
                "alignment_modes": list(modes), "required_variables": list(required_variables),
                "overlap_policy": overlap_policy},
            "eligibility_scope": "coordinate coverage and declared availability only; numerical values require preflight"}


def resolve_ensemble_sources(paths, delivery_days, **settings):
    """Resolve every requested day to one archive or fail with all coverage reasons."""
    inventory = inventory_ensemble_sources(paths, delivery_days=delivery_days, **settings)
    rejected = [row for row in inventory["days"] if not row["eligible"]]
    if rejected:
        explanation = "; ".join(f"{row['delivery_day']}: {', '.join(row['ineligibility_reasons'])}" for row in rejected)
        raise ValueError(f"Ensemble delivery coverage is incomplete: {explanation}")
    sources = {row["path"]: row for row in inventory["sources"] if row["status"] == "present"}
    return {row["delivery_day"]: {**row, "source_metadata": sources[row["source"]]} for row in inventory["days"]}
