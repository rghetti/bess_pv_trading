"""Build repeatable price-search inputs using only an explicit DuckDB catalog."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import duckdb
import pandas as pd


def _quote(name: str) -> str:
    """Quote configured SQL identifiers while binding all selection values separately."""
    return '"' + name.replace('"', '""') + '"'


def _digest(path: Path) -> str:
    """Fingerprint an artifact without loading its complete contents into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def build_price_search_inputs(config: Mapping[str, Any], *, pinned_manifest: str | None = None) -> dict:
    """Export native source intervals and pin immutable snapshots before parallel model fits."""
    settings = config["price_search_inputs"]
    output = Path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(Path(pinned_manifest).read_text()) if pinned_manifest else None
    # Machine storage paths and SQL thread counts do not change the scientific input selection.
    selection_contract = {key: value for key, value in settings.items()
                          if key not in {"database_path", "output_directory", "database_threads"}}
    configuration_hash = hashlib.sha256(json.dumps(selection_contract, sort_keys=True, default=str).encode()).hexdigest()
    if pinned is not None and pinned["selection_sha256"] != configuration_hash:
        raise ValueError("Pinned input manifest uses a different source selection configuration")
    first = pd.Timestamp(settings["start"], tz=settings["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(settings["end_exclusive"], tz=settings["timezone"]).tz_convert("UTC")
    tables = {}
    # One read transaction fixes all source selections; workers later read prepared arrays, not DuckDB.
    with duckdb.connect(str(settings["database_path"]), read_only=True,
                        config={"threads": int(settings["database_threads"])}) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute("BEGIN TRANSACTION")
        available = {row[0] for row in connection.execute("DESCRIBE api_records").fetchall()}
        for selection in settings["sources"]:
            name = selection["name"]
            if name in tables or selection["provider"] in settings.get("excluded_providers", []):
                raise ValueError(f"Duplicate or excluded source selection: {name}")
            columns = list(dict.fromkeys(["_snapshot_id", "_row_id", *selection["columns"]]))
            if not set(columns).issubset(available):
                raise ValueError(f"Source {name} lacks columns {sorted(set(columns) - available)}")
            placeholders = ",".join("?" for _ in selection["datasets"])
            conditions = ["provider = ?", "zone = ?", f"dataset IN ({placeholders})"]
            parameters = [selection["provider"], selection["zone"], *selection["datasets"]]
            clock = selection.get("date_column", "product_start")
            if clock not in available:
                raise ValueError(f"Source {name} lacks its date column")
            # Daily gas indices use local delivery dates; timestamped records retain native intervals.
            if clock == "gas_delivery_day":
                conditions.extend([f"TRY_CAST({_quote(clock)} AS DATE) >= ?", f"TRY_CAST({_quote(clock)} AS DATE) < ?"])
                parameters.extend([settings["start"], settings["end_exclusive"]])
            else:
                conditions.extend([f"{_quote(clock)} >= ?", f"{_quote(clock)} < ?"])
                parameters.extend([first, stop])
            for selector, value in selection.get("selectors", {}).items():
                if selector not in available:
                    raise ValueError(f"Unknown selector column {selector}")
                conditions.append(f"{_quote(selector)} = ?")
                parameters.append(value)
            relation = "current_api_records"
            if pinned is not None:
                # Historical immutable records remain addressable even after newer revisions arrive.
                snapshot_ids = [source["snapshot_id"] for source in pinned["tables"][name]["snapshot_sources"]]
                if not snapshot_ids:
                    raise ValueError(f"Pinned source {name} has no snapshots")
                relation = "api_records"
                conditions.append("_snapshot_id IN (" + ",".join("?" for _ in snapshot_ids) + ")")
                parameters.extend(snapshot_ids)
            projection = ",".join(_quote(column) for column in columns)
            query = f"SELECT {projection} FROM {relation} WHERE " + " AND ".join(conditions)
            frame = connection.execute(query + f" ORDER BY {_quote(clock)}, _snapshot_id, _row_id", parameters).df()
            if frame.empty and selection.get("required", True):
                raise ValueError(f"Required source {name} has no rows in the configured period")
            # Fingerprint projected SQL values as well as immutable snapshot metadata.
            row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy(dtype="uint64")
            selected_row_hash = hashlib.sha256(row_hashes.astype("<u8").tobytes()).hexdigest()
            if pinned is not None and selected_row_hash != pinned["tables"][name]["selected_row_sha256"]:
                raise ValueError(f"Pinned selected SQL values changed for {name}")
            snapshot_ids = sorted(frame["_snapshot_id"].unique())
            snapshot_sources = []
            if snapshot_ids:
                query = """SELECT snapshot_id, provider, dataset, zone, raw_sha256, normalized_sha256, row_count
                           FROM api_snapshots WHERE snapshot_id IN (""" + ",".join("?" for _ in snapshot_ids) + ") ORDER BY snapshot_id"
                snapshot_sources = connection.execute(query, snapshot_ids).df().to_dict("records")
            if pinned is not None and snapshot_sources != pinned["tables"][name]["snapshot_sources"]:
                raise ValueError(f"Pinned source snapshot checksums changed for {name}")
            path = output / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            details = {"path": path.name, "rows": len(frame), "sha256": _digest(path),
                       "selected_row_sha256": selected_row_hash,
                       "selection": selection, "snapshot_sources": snapshot_sources,
                       "missing_by_column": {column: int(frame[column].isna().sum()) for column in columns}}
            if clock in frame:
                details["source_start"] = str(frame[clock].min())
                details["source_last"] = str(frame[clock].max())
            if "product_start" in frame and "product_end" in frame:
                duration = (pd.to_datetime(frame["product_end"], utc=True) -
                            pd.to_datetime(frame["product_start"], utc=True)).dt.total_seconds() / 60
                details["native_interval_minutes"] = {str(key): int(value) for key, value in duration.value_counts().items()}
            tables[name] = details
        connection.execute("COMMIT")
    manifest = {
        "status": "complete", "database_only_data_dependency": True, "farm_files_required": False,
        "database_path": str(settings["database_path"]), "selection_sha256": configuration_hash,
        "selection": settings, "pinned_from": pinned_manifest,
        "implementation_sha256": _digest(Path(__file__)), "tables": tables,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(),
                     **{package: version(package) for package in ("numpy", "pandas", "pyarrow", "duckdb")}},
        "native_price_resolution_policy": "retain native auction intervals; fit separate market/resolution groups",
        "same_day_realized_feature_policy": "explicitly retrospective; publication timing bypassed by user request",
        "forecast_availability_policy": "previous-day forecasts assumed usable at all decision gates by user request",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
