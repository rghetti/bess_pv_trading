"""Projected DuckDB feature reads using the existing feature study's semantics.

Like the features checkout's ``price_benchmark.data`` adapter, this reads one
consistent current catalog, preserves snapshot identities, keeps native interval
values constant, and distinguishes publication evidence from study assumptions.
It does not import that checkout or read weather payloads into every worker.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.duckdb_store import DuckDBCache
from .energy_prices import _aware, _aware_series


def _publication_clocks(frame, spec, timezone):
    """Apply one explicitly configured evidence, inferred-clock or assumed policy."""
    column = spec.get("availability_column")
    clock = spec.get("publication_local_time")
    lag = spec.get("publication_lag_hours")
    if sum(value is not None for value in (column, clock, lag)) != 1:
        raise ValueError("Each feature requires exactly one explicit publication policy")
    if column is not None:
        if column in {"retrieved_at", "cached_at", "downloaded_at"}:
            raise ValueError("Retrieval time is not historical feature publication")
        result = pd.to_datetime(frame[column], utc=True)
        kind = spec.get("availability_kind", "recorded_publication")
    elif clock is not None:
        offset = spec.get("publication_day_offset", -1)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("Feature publication_day_offset must be an integer")
        dates = pd.DatetimeIndex(frame.product_start).tz_convert(timezone).date
        by_day = {day: pd.Timestamp(f"{day + timedelta(days=offset)} {clock}", tz=timezone)
                  for day in np.unique(dates)}
        result = (pd.DatetimeIndex([by_day[day] for day in dates]).tz_convert("UTC")
                  if len(dates) else pd.DatetimeIndex([], tz="UTC"))
        kind = "assumed_delivery_day_publication_clock"
    else:
        if isinstance(lag, bool) or not np.isfinite(lag) or lag < 0:
            raise ValueError("Feature publication lag must be finite and nonnegative")
        result = frame.product_end + pd.Timedelta(hours=lag)
        kind = "assumed_lag_after_product_end"
    return result, kind


def _interval_features(frame, spec, interval_minutes):
    """Repeat native interval means without interpolating across missing products."""
    duration = (frame.product_end - frame.product_start).dt.total_seconds().to_numpy() / 60
    counts = duration / interval_minutes
    if not np.isfinite(counts).all() or np.any(counts < 1) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError(f"Feature {spec['name']} has unresolved or incompatible native intervals")
    counts = counts.astype(int)
    parents = np.repeat(np.arange(len(frame)), counts)
    offsets = np.arange(len(parents)) - np.repeat(np.cumsum(counts) - counts, counts)
    starts = pd.DatetimeIndex(frame.product_start.iloc[parents]) + pd.to_timedelta(offsets * interval_minutes, unit="min")
    return pd.DataFrame({"product_start": starts, "available_at": frame.available_at.iloc[parents].to_numpy(),
                         "value": frame[spec.get("value_column", "value")].iloc[parents].to_numpy()})


def _instantaneous_weather(frame, spec, interval_minutes):
    """Interpolate within one forecast cycle and only across complete hourly brackets."""
    output = []
    column = spec.get("value_column", "value")
    max_gap = float(spec.get("maximum_interpolation_gap_hours", 1.0))
    for cycle, group in frame.groupby("cycle_time", sort=True):
        group = group.sort_values("valid_time").drop_duplicates(["valid_time", column, "available_at"])
        if group.valid_time.duplicated().any():
            raise ValueError("Conflicting weather values within one cycle require explicit source selection")
        source_times = pd.DatetimeIndex(group.valid_time)
        times = source_times.as_unit("ns")
        values = group[column].to_numpy(float)
        if len(times) < 2:
            continue
        # Missing or nonadjacent observations break interpolation; expand only
        # valid brackets and construct all their quarter-hours in one batch.
        lengths = np.diff(times.asi8)
        valid = ((lengths > 0) & (lengths <= pd.Timedelta(hours=max_gap).value)
                 & np.isfinite(values[:-1]) & np.isfinite(values[1:]))
        left = np.flatnonzero(valid)
        if not len(left):
            continue
        step = pd.Timedelta(minutes=interval_minutes).value
        counts = (lengths[left] + step - 1) // step
        parents = np.repeat(left, counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        start_ns = times.asi8[parents] + offsets * step
        fractions = (start_ns - times.asi8[parents]) / lengths[parents]
        source_availability = pd.DatetimeIndex(group.available_at)
        availability = source_availability.as_unit("ns").asi8
        available_ns = np.maximum(availability[parents], availability[parents + 1])
        output.append(pd.DataFrame({"product_start": pd.to_datetime(start_ns, utc=True).as_unit(source_times.unit),
            "available_at": pd.to_datetime(available_ns, utc=True).as_unit(source_availability.unit),
            "value": values[parents] + fractions * (values[parents + 1] - values[parents])}))
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=["product_start", "available_at", "value"])


def load_catalog_features(database, selections, *, start, end, timezone="Europe/Rome",
                          interval_minutes=15, database_threads=1):
    """Load selected fundamentals/weather with small column projections and lineage.

    Normalized snapshot hashes are retained; projected SQL rows receive their own
    content hash. This avoids claiming that unselected weather payloads were
    redecoded and independently verified during feature preparation.
    """
    if Path(database).suffix != ".duckdb":
        raise ValueError("Feature preparation requires the DuckDB catalog")
    start, end = _aware(start, "feature start"), _aware(end, "feature end")
    if start >= end:
        raise ValueError("Feature start must precede exclusive end")
    if len({item["name"] for item in selections}) != len(selections):
        raise ValueError("Feature selection names must be unique")
    store = DuckDBCache(database, threads=database_threads)
    outputs, evidence = [], []
    with store.read_session():
        connection = store._session_connection
        if connection is None:
            raise FileNotFoundError(database)
        columns = {row[0] for row in connection.execute("DESCRIBE current_api_records").fetchall()}
        for spec in selections:
            value_column = spec.get("value_column", "value")
            component = spec.get("json_component")
            if component is not None and (not isinstance(component, dict)
                    or set(component) != {"column", "key"}
                    or not all(isinstance(component[name], str) and component[name] for name in ("column", "key"))):
                raise ValueError("json_component requires nonempty column and key strings")
            requested = {"_snapshot_id", "_row_id", "product_start", "product_end", "unit"}
            requested.add(value_column if component is None else component["column"])
            weather = spec.get("kind", "interval") == "weather"
            gas = spec.get("kind", "interval") == "daily_gas"
            if weather:
                requested.update({"cycle_time", "valid_time", "temporal_semantics", "source_kind"})
            if gas:
                requested.add("gas_delivery_day")
            if spec.get("availability_column"):
                requested.add(spec["availability_column"])
            if not requested.issubset(columns):
                raise ValueError(f"Feature {spec['name']} lacks catalog columns {sorted(requested - columns)}")
            projection = ",".join('"' + name.replace('"', '""') + '"' for name in sorted(requested))
            shift = pd.Timedelta(hours=spec.get("target_shift_hours", 0))
            reference = "valid_time" if weather else "TRY_CAST(gas_delivery_day AS TIMESTAMPTZ)" if gas else "product_start"
            conditions = ["provider=?", "dataset=?", "zone=?", f"{reference}>=?", f"{reference}<?"]
            parameters = [spec["provider"], spec["dataset"], spec["zone"],
                          start - shift - pd.Timedelta(hours=1 if weather else 0),
                          end - shift + pd.Timedelta(hours=1 if weather else 0)]
            # Restrict weather variables/locations before rows leave DuckDB.
            for name, value in spec.get("selectors", {}).items():
                if name not in {"location_id", "variable", "model", "source_kind", "gas_product"} or name not in columns:
                    raise ValueError(f"Unsupported feature selector {name}")
                conditions.append(f'"{name}"=?')
                parameters.append(value)
            frame = connection.execute(f"SELECT {projection} FROM current_api_records WHERE " + " AND ".join(conditions), parameters).fetchdf()
            frame = frame.sort_values(["_snapshot_id", "_row_id"], kind="stable").reset_index(drop=True)
            snapshot_ids = sorted(frame._snapshot_id.unique()) if len(frame) else []
            sources = []
            if snapshot_ids:
                placeholders = ",".join("?" for _ in snapshot_ids)
                sources = connection.execute(f"SELECT snapshot_id,raw_sha256,normalized_sha256 FROM api_snapshots WHERE snapshot_id IN ({placeholders}) ORDER BY snapshot_id", snapshot_ids).fetchdf().to_dict("records")
            row_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
            source_count = len(frame)
            if frame.empty:
                evidence.append({"name": spec["name"], "source_rows": 0, "prepared_rows": 0,
                                 "selected_row_sha256": row_hash, "snapshot_sources": sources, "selection": spec})
                continue
            if "expected_unit" in spec and not frame.unit.eq(spec["expected_unit"]).all():
                raise ValueError(f"Feature {spec['name']} does not use the declared unit")
            if component is not None:
                # Extract the declared generation component while retaining the
                # raw JSON column in the source-row fingerprint above.
                def extract_component(value):
                    if value is None or pd.isna(value):
                        return np.nan
                    parsed = json.loads(value)
                    if not isinstance(parsed, dict):
                        raise ValueError("Generation component source must be a JSON object")
                    return parsed.get(component["key"], np.nan)

                frame["_selected_component"] = pd.to_numeric(
                    frame[component["column"]].map(extract_component), errors="coerce")
                spec = {**spec, "value_column": "_selected_component"}
            if gas:
                days = [pd.Timestamp(value).date() for value in frame.gas_delivery_day]
                frame["product_start"] = pd.DatetimeIndex([pd.Timestamp(day, tz=timezone) for day in days]).tz_convert("UTC")
                frame["product_end"] = pd.DatetimeIndex([pd.Timestamp(day + timedelta(days=1), tz=timezone) for day in days]).tz_convert("UTC")
            # Unknown release clocks and optional hindcast exclusions remain documented gaps.
            hindcasts = (frame.source_kind.fillna("").str.contains("hindcast", case=False)
                         if weather else pd.Series(False, index=frame.index))
            excluded_hindcasts = int(hindcasts.sum()) if not spec.get("allow_hindcasts", False) else 0
            frame = frame.loc[~hindcasts] if excluded_hindcasts else frame
            clocks, availability = _publication_clocks(frame, spec, timezone)
            frame = frame.copy()
            frame["available_at"] = clocks
            unknown = int(frame.available_at.isna().sum())
            frame = frame.loc[frame.available_at.notna()].copy()
            for name in ("product_start", "product_end", "cycle_time", "valid_time"):
                if name in frame:
                    frame[name] = pd.to_datetime(frame[name], utc=True)
            if weather and set(frame.temporal_semantics.dropna()) == {"instantaneous"}:
                prepared = _instantaneous_weather(frame, spec, interval_minutes)
            else:
                prepared = _interval_features(frame, spec, interval_minutes)
            prepared.product_start += shift
            missing = int((~np.isfinite(pd.to_numeric(prepared.value, errors="coerce"))).sum())
            prepared = prepared.loc[np.isfinite(pd.to_numeric(prepared.value, errors="coerce"))].copy()
            prepared = prepared.loc[(prepared.product_start >= start) & (prepared.product_start < end)].drop_duplicates()
            if prepared.duplicated(["product_start", "available_at"]).any():
                raise ValueError(f"Feature {spec['name']} contains conflicting source values at one vintage")
            prepared["feature"] = spec["name"]
            outputs.append(prepared)
            evidence.append({"name": spec["name"], "source_rows": source_count, "prepared_rows": len(prepared),
                             "unknown_publication_rows": unknown, "excluded_hindcast_rows": excluded_hindcasts,
                             "missing_value_rows": missing, "selected_row_sha256": row_hash,
                             "availability": availability, "snapshot_sources": sources, "selection": spec})
    frame = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=["product_start", "available_at", "value", "feature"])
    return frame, {"database": str(Path(database).resolve()), "features": evidence,
                   "snapshot_verification": "recorded_immutable_snapshot_hashes_and_hash_of_projected_rows",
                   "feature_semantics": "source_intervals_constant_instantaneous_weather_interpolated_within_cycle"}
