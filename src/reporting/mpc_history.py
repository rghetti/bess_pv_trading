"""Immutable daily Parquet partitions for large MPC horizon diagnostics.

Only completed delivery days are spilled. Controller state, outstanding actions,
and the current shadow plan remain in memory and in the ordinary checkpoint.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mpc-horizon-history-v1"
MANIFEST_PATH = Path("json/mpc_horizon_history.json")


def validate_mpc_history_settings(config):
    """Keep legacy in-memory output unless the study explicitly opts into partitions."""
    settings = config.get("realtime_mpc_outputs", {})
    if not isinstance(settings, dict) or set(settings) - {"horizon_history"}:
        raise ValueError("Unknown realtime_mpc_outputs settings")
    mode = settings.get("horizon_history", "memory")
    if mode not in {"memory", "daily_parquet"}:
        raise ValueError("realtime_mpc_outputs.horizon_history must be memory or daily_parquet")
    return mode


def _json_default(value):
    """Keep numerical and timestamp provenance serializable without lossy repr strings."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported MPC history value {type(value).__name__}")


def _checksum(path):
    """Hash streamed file bytes without loading an entire daily partition again."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_path(destination, relative):
    """Reject archive references escaping their replay directory, including symlinks."""
    destination = Path(destination).resolve()
    candidate = (destination / relative).resolve()
    if not candidate.is_relative_to(destination):
        raise ValueError("MPC history partition escapes its replay directory")
    return candidate


def _encode_rows(rows):
    """Store scalar columns natively and encode nested provenance with a declared schema."""
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="string") for name in (
            "delivery_day", "solve_id", "interval_id")}), []
    frame = pd.DataFrame(rows)
    json_columns = []
    memo = {}

    def encode(value):
        if value is None or isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            identity = id(value)
            if identity not in memo:
                memo[identity] = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                            default=_json_default, allow_nan=False)
            return memo[identity]
        return json.dumps(value, default=_json_default, allow_nan=False)

    # Shared provenance objects recur across every interval of one solved horizon.
    for name in frame.columns:
        if frame[name].map(lambda value: isinstance(value, (dict, list, tuple, np.ndarray))).any():
            frame[name] = frame[name].map(encode)
            json_columns.append(name)
    return frame, json_columns


def _write_partition(destination, day, kind, rows):
    """Publish by content hash; abandoned checkpoint attempts cannot overwrite data."""
    directory = Path(destination) / "diagnostics/mpc_horizon" / day
    directory.mkdir(parents=True, exist_ok=True)
    frame, json_columns = _encode_rows(rows)
    with tempfile.NamedTemporaryFile(prefix=f".{kind}-", suffix=".parquet", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        checksum = _checksum(temporary)
        path = directory / f"{kind}-{checksum}.parquet"
        if path.exists():
            if _checksum(path) != checksum:
                raise ValueError("Existing immutable MPC history partition has changed")
            temporary.unlink()
        else:
            temporary.replace(path)
        return {"path": str(path.relative_to(destination)), "sha256": checksum,
                "records": len(rows), "json_columns": json_columns}
    finally:
        # Remove only a temporary file created by this operation after an I/O failure.
        if temporary.exists():
            temporary.unlink()


def write_history_manifest(destination, archives):
    """Publish a small partition index; it never materializes archived horizon rows."""
    manifest = {"schema_version": SCHEMA_VERSION, "storage": "daily_parquet",
                "partitions": archives,
                "plan_records": sum(item["plans"]["records"] for item in archives),
                "scenario_records": sum(item["scenarios"]["records"] for item in archives),
                "reader": "bess_pv_trading.reporting.mpc_history.iter_mpc_history",
                "legacy_horizon_csv_json_written": False}
    path = Path(destination) / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return manifest


def archive_completed_history(destination, state, archives, completed_days):
    """Archive only named completed days, clearing buffers after all files are durable."""
    completed = {date.fromisoformat(str(day)).isoformat() for day in completed_days}
    index = list(archives)
    existing = {item["delivery_day"] for item in index}
    if len(existing) != len(index):
        raise ValueError("Duplicate delivery days in MPC history archive index")
    for rows in (state.plans, state.scenarios):
        if any("delivery_day" not in row for row in rows):
            raise ValueError("MPC horizon rows require delivery_day before archiving")
        if any(row["delivery_day"] in existing for row in rows):
            raise ValueError("MPC history buffer duplicates an already archived delivery day")
    pending = completed - existing
    for day in sorted(pending):
        plans = [row for row in state.plans if row["delivery_day"] == day]
        scenarios = [row for row in state.scenarios if row["delivery_day"] == day]
        index.append({"delivery_day": day,
                      "plans": _write_partition(destination, day, "plans", plans),
                      "scenarios": _write_partition(destination, day, "scenarios", scenarios)})
    index.sort(key=lambda item: item["delivery_day"])
    write_history_manifest(destination, index)
    # The live first action, latest plan, solves and action log are deliberately retained.
    state.plans = [row for row in state.plans if row["delivery_day"] not in completed]
    state.scenarios = [row for row in state.scenarios if row["delivery_day"] not in completed]
    return index


def verify_history_archives(destination, archives):
    """Validate every checkpoint reference before accepting a resumed controller."""
    import pyarrow.parquet as parquet

    if not isinstance(archives, list):
        raise ValueError("MPC history archive index must be a list")
    days = set()
    for item in archives:
        day = date.fromisoformat(item["delivery_day"]).isoformat()
        if day in days:
            raise ValueError("Duplicate delivery days in MPC history archive index")
        days.add(day)
        for kind in ("plans", "scenarios"):
            entry = item[kind]
            path = _partition_path(destination, entry["path"])
            if not path.is_file() or _checksum(path) != entry["sha256"]:
                raise ValueError(f"MPC history checksum mismatch or missing partition: {entry['path']}")
            metadata = parquet.ParquetFile(path)
            if metadata.metadata.num_rows != entry["records"]:
                raise ValueError("MPC history partition row count differs from its checkpoint")
            if set(entry.get("json_columns", [])) - set(metadata.schema_arrow.names):
                raise ValueError("MPC history JSON column schema mismatch")
    return archives


def iter_mpc_history(destination, *, kind="plans", archives=None, delivery_days=None,
                     batch_size=65536, verify_checksums=True):
    """Yield bounded DataFrame batches from selected days, decoding nested provenance."""
    import pyarrow.parquet as parquet

    if kind not in {"plans", "scenarios"}:
        raise ValueError("MPC history kind must be plans or scenarios")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("MPC history batch_size must be a positive integer")
    if archives is None:
        manifest = json.loads((Path(destination) / MANIFEST_PATH).read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unknown MPC horizon archive schema")
        archives = manifest["partitions"]
    selected = None if delivery_days is None else set(delivery_days)
    for item in archives:
        if selected is not None and item["delivery_day"] not in selected:
            continue
        entry = item[kind]
        path = _partition_path(destination, entry["path"])
        if verify_checksums and _checksum(path) != entry["sha256"]:
            raise ValueError("MPC history checksum mismatch while streaming")
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            for name in entry.get("json_columns", []):
                frame[name] = frame[name].map(lambda value: None if value is None else json.loads(value))
            yield frame
