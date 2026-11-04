"""Shared configuration, provenance and DuckDB access for price feature preparation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .duckdb_store import DuckDBCache


def settings(config):
    """Validate the explicitly configured feature window and availability policy."""
    result = dict(config["price_features"])
    if not result["start"] or not result["end_exclusive"]:
        raise ValueError("Configure explicit price_features.start and end_exclusive dates")
    if result["availability_policy"] not in {"historical_as_of", "assume_day_ahead_known"}:
        raise ValueError("Unknown price_features.availability_policy")
    if not isinstance(result["gas_delivery_day_assumed_before_auctions"], bool):
        raise ValueError("price_features.gas_delivery_day_assumed_before_auctions must be boolean")
    if not isinstance(result["include_weather_features"], bool):
        raise ValueError("price_features.include_weather_features must be boolean")
    first = pd.Timestamp(result["start"], tz=result["timezone"])
    stop = pd.Timestamp(result["end_exclusive"], tz=result["timezone"])
    if stop <= first:
        raise ValueError("Feature end_exclusive must follow start")
    if not set(result["auctions"]).issubset({"DA", "IDA1"}) or not result["auctions"]:
        raise ValueError("Feature preparation currently supports DA and IDA1")
    if not {7, 14}.issubset(result["price_lag_days"]) or any(
            not isinstance(day, int) or isinstance(day, bool) or day < 1 for day in result["price_lag_days"]):
        raise ValueError("Price lags must be positive local-day offsets including 7 and 14")
    allowed = {"load_forecast", "total_load_forecast", "solar_forecast", "wind_onshore_forecast",
               "wind_forecast", "generation_forecast", "preliminary_load_forecast"}
    for item in result["fundamentals"]:
        if item["dataset"] not in allowed:
            raise ValueError("Only declared forecast datasets may use relaxed feature availability")
        if item.get("interval_minutes") and item.get("timestamp_label") not in {"start", "end"}:
            raise ValueError("An interval assumption also requires an explicit timestamp_label")
    result["first"], result["stop"] = first.tz_convert("UTC"), stop.tz_convert("UTC")
    return result


def cache(config):
    """Use the shared immutable catalog without selecting the legacy file cache."""
    return DuckDBCache(config["price_features"]["database_path"],
                       threads=config["api"]["database_threads"])


def json_value(value):
    """Serialize timestamps and numerical scalars in evidence manifests."""
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    """Keep manifests readable and reject nonstandard NaN JSON values."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def new_output(config, kind):
    """Protect completed feature packages and group artifacts by run."""
    selected = config["price_features"]
    run = selected["run_id"] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(selected["output_root"]) / f"{kind}-{run}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def read_current(config):
    """Pin a consistent SQL snapshot and retain row and source checksum identities."""
    selected = settings(config)
    with cache(config)._connect() as connection:
        if connection is None:
            raise FileNotFoundError("Feature source DuckDB is absent; import or acquire data first")
        connection.execute("BEGIN TRANSACTION")
        names = {row[0] for row in connection.execute("DESCRIBE current_api_records").fetchall()}
        wanted = ("_snapshot_id", "_row_id", "provider", "dataset", "zone", "value", "unit",
            "product_start", "product_end", "reference_time", "historical_available_at",
            "scheduled_available_at", "availability_provenance", "source_document_time", "retrieved_at",
            "historical_feature_eligible", "value_mw", "variable", "cycle_time", "valid_time",
            "lead_hours", "location_id", "temporal_semantics", "model", "source_kind", "gas_delivery_day",
            "gas_product", "record_status", "source_block_repeated", "source_block_start", "source_block_end")
        projection = ",".join('"' + name + '"' for name in wanted if name in names)
        # Keep lag warm-up and weather interpolation endpoints without reading unrelated years.
        earliest = selected["first"] - pd.Timedelta(days=max(selected["price_lag_days"]) + 2)
        reference = "coalesce(product_start,reference_time)" if "reference_time" in names else "product_start"
        weather_filter = "" if selected["include_weather_features"] else " AND dataset != 'weather_forecast'"
        frame = connection.execute(f"SELECT {projection} FROM current_api_records WHERE {reference} >= ? AND {reference} <= ?{weather_filter}",
                                   [earliest, selected["stop"]]).fetchdf()
        sources = connection.execute("""SELECT snapshot_id, provider, dataset, zone,
            raw_sha256, normalized_sha256, manifest_json FROM api_snapshots
            JOIN api_current_requests USING(snapshot_id) ORDER BY snapshot_id""").fetchdf()
        connection.execute("COMMIT")
    return frame, sources


def identity(provider, dataset, zone, first, stop, **options):
    """Keep import/parser changes in request identity, never overwrite raw revisions."""
    return dict(provider=provider, dataset=dataset, zone=zone,
                start=pd.Timestamp(first).isoformat(), end=pd.Timestamp(stop).isoformat(),
                options={"parser": "public-features-v1", **options})


def persist(store, key, raw, frame, metadata):
    """Make repeat imports idempotent while verifying authoritative snapshot bytes."""
    # Reuse one connection for each local snapshot operation, never over an HTTP request.
    with store.batch_write():
        existing = store.load(key)
        if existing is not None:
            old, manifest = existing
            if manifest["raw_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("Immutable feature identity collides with different source bytes")
            revision = metadata.get("normalizer_revision")
            if revision is None or manifest.get("metadata", {}).get("normalizer_revision") == revision:
                return manifest
            # Parser corrections publish a new immutable revision of the same original file.
        manifest = store.save(key, raw, frame, metadata)
        store.verify_native_snapshot(key, snapshot_id=manifest["snapshot_id"])
        return manifest
