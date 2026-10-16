"""Verify legacy API snapshots and import an exact, rebuildable DuckDB archive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

import pandas as pd

from .api import DataAPIError, DownloadCache, _atomic_bytes, _check_nonsecret, _json_bytes


@dataclass(frozen=True)
class _Snapshot:
    """A source whose manifest and exact payload bytes passed preflight checks."""

    directory: Path
    identity: dict
    manifest: dict
    current: bool
    first_product: str | None
    last_product: str | None


@dataclass(frozen=True)
class _Receipt:
    """An absence or unparsed-response receipt with its original retrieval time."""

    directory: Path
    identity: dict
    receipt: dict
    kind: str


def _read_json(path: Path) -> dict:
    """Reject unreadable metadata without printing provider response contents."""
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError, UnicodeError):
        raise DataAPIError(f"Invalid API migration metadata at {path}") from None


def _timestamp(value: Any, path: Path) -> pd.Timestamp:
    """Require an explicit timezone so importing cannot reinterpret source clocks."""
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError
        return timestamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        raise DataAPIError(f"Invalid timezone-aware source timestamp at {path}") from None


def _identity(cache: DownloadCache, directory: Path, metadata: dict) -> dict:
    """Match the complete nonsecret identity to its canonical hashed directory."""
    try:
        identity = metadata["request"]
        if not isinstance(identity, dict):
            raise ValueError
        _check_nonsecret(identity)
        expected = cache.location(identity)
        actual = directory.parent.parent if directory.parent.name == "revisions" else directory
        if expected != actual or identity.get("schema_version") != "api-cache-v1":
            raise ValueError
        if identity.get("provider") not in {"entsoe", "terna"}:
            raise ValueError
        return identity
    except (KeyError, TypeError, ValueError):
        raise DataAPIError(f"API migration identity/path mismatch at {directory}") from None


def _payloads(directory: Path, identity: Mapping[str, Any], manifest: dict) -> tuple[bytes, bytes, pd.DataFrame]:
    """Verify original raw and Parquet hashes before trusting normalized records."""
    expected_raw = "response.xml" if identity["provider"] == "entsoe" else "response.json"
    if manifest.get("schema_version") != "api-cache-v1" or manifest.get("raw_file") != expected_raw:
        raise DataAPIError(f"Invalid API migration manifest at {directory}")
    try:
        payload = (directory / expected_raw).read_bytes()
        normalized = (directory / "records.parquet").read_bytes()
        for data, field in ((payload, "raw_sha256"), (normalized, "normalized_sha256")):
            if hashlib.sha256(data).hexdigest() != manifest[field]:
                raise DataAPIError(f"API migration {field} checksum mismatch at {directory}")
        frame = pd.read_parquet(BytesIO(normalized))
        if not isinstance(manifest["records"], int) or manifest["records"] != len(frame):
            raise DataAPIError(f"API migration record-count mismatch at {directory}")
        return payload, normalized, frame
    except (OSError, KeyError, ValueError, TypeError) as error:
        if isinstance(error, DataAPIError):
            raise
        raise DataAPIError(f"Incomplete/corrupt API migration snapshot at {directory}") from None


def _product_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """Describe native source coverage without inferring historical availability."""
    first = last = None
    for name in ("product_start", "reference_time"):
        if name in frame and frame[name].notna().any():
            first = str(frame[name].dropna().min())
            last = str(frame[name].dropna().max())
            break
    if "product_end" in frame and frame["product_end"].notna().any():
        last = str(frame["product_end"].dropna().max())
    return first, last


def _unparsed_payload(directory: Path, receipt: dict) -> bytes:
    """Verify the retained unparsed provider data independently of its receipt."""
    try:
        payload = (directory / "unparsed_response.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["raw_sha256"]:
            raise ValueError
        return payload
    except (OSError, KeyError, ValueError):
        raise DataAPIError(f"Unparsed API migration payload checksum mismatch at {directory}") from None


def _scan(cache: DownloadCache) -> tuple[list[_Snapshot], list[_Receipt], list[Path], list[Path], list[Path]]:
    """Preflight canonical current/revision directories and count unfinished writes."""
    snapshots, receipts, partial, files = [], [], [], []
    ignored = sorted(path for path in cache.directory.iterdir()
                     if path.is_dir() and path.name not in {"entsoe", "terna"})
    for provider in ("entsoe", "terna"):
        root = cache.directory / provider
        if not root.exists():
            continue
        if not root.is_dir():
            raise DataAPIError(f"Malformed API migration provider directory at {root}")
        for current in sorted(path for path in root.glob("*/*/*/*") if path.is_dir()):
            relative = current.relative_to(cache.directory).parts
            if (len(relative) != 5 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in relative[:3])
                    or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", relative[3])
                    or not re.fullmatch(r"[a-f0-9]{20}", relative[4])):
                raise DataAPIError(f"Malformed API migration cache path at {current}")
            revisions = current / "revisions"
            versions = [(current, True)]
            if revisions.exists():
                if not revisions.is_dir():
                    raise DataAPIError(f"Malformed API migration revisions directory at {revisions}")
                versions += [(path, False) for path in sorted(revisions.iterdir()) if path.is_dir()]
            for directory, is_current in versions:
                if not is_current and not re.fullmatch(r"\d{8}T\d{12}Z", directory.name):
                    raise DataAPIError(f"Malformed API migration revision path at {directory}")
                files.extend(path for path in sorted(directory.iterdir()) if path.is_file())
                published = False
                manifest_path = directory / "manifest.json"
                if manifest_path.is_file():
                    manifest = _read_json(manifest_path)
                    identity = _identity(cache, directory, manifest)
                    _timestamp(manifest.get("cached_at"), manifest_path)
                    _, _, frame = _payloads(directory, identity, manifest)
                    first, last = _product_bounds(frame)
                    snapshots.append(_Snapshot(directory, identity, manifest, is_current, first, last))
                    published = True
                for filename, kind in (("no_data.json", "no-data"), ("schema_error.json", "unparsed-data")):
                    path = directory / filename
                    if not path.is_file():
                        continue
                    receipt = _read_json(path)
                    identity = _identity(cache, directory, receipt)
                    if receipt.get("status") != kind or (kind == "no-data" and receipt.get("records") != 0):
                        raise DataAPIError(f"Invalid API migration receipt at {path}")
                    _timestamp(receipt.get("retrieved_at"), path)
                    if kind == "unparsed-data":
                        _unparsed_payload(directory, receipt)
                    receipts.append(_Receipt(directory, identity, receipt, kind))
                    published = True
                if not published:
                    partial.append(directory)
        scanned = set(files)
        for path in root.rglob("*"):
            if path.name in {"manifest.json", "no_data.json", "schema_error.json"} and path.is_file() and path not in scanned:
                raise DataAPIError(f"Malformed API migration metadata path at {path}")
    return snapshots, receipts, partial, files, ignored


def _coverage(snapshots: list[_Snapshot], receipts: list[_Receipt]) -> list[dict]:
    """Aggregate current datasets by requested local month; keep revisions separate."""
    grouped = {}
    for source in [*snapshots, *receipts]:
        identity = source.identity
        key = tuple(identity[name] for name in ("provider", "dataset", "zone"))
        group = grouped.setdefault(key, {"provider": key[0], "dataset": key[1], "zone": key[2],
                                        "current_snapshots": 0, "archived_snapshots": 0,
                                        "records": 0, "local_days": set(), "months": {}})
        day = identity["local_day"]
        month = group["months"].setdefault(day[:7], {"month": day[:7], "current_snapshots": 0,
                                                   "archived_snapshots": 0, "records": 0,
                                                   "no_data_receipts": 0, "unparsed_receipts": 0})
        if isinstance(source, _Snapshot):
            field = "current_snapshots" if source.current else "archived_snapshots"
            group[field] += 1
            month[field] += 1
            if source.current:
                group["records"] += source.manifest["records"]
                month["records"] += source.manifest["records"]
                group["local_days"].add(day)
        else:
            month["no_data_receipts" if source.kind == "no-data" else "unparsed_receipts"] += 1
    for group in grouped.values():
        days = sorted(group.pop("local_days"))
        group.update({"first_local_day": days[0] if days else None,
                      "last_local_day": days[-1] if days else None, "distinct_local_days": len(days)})
        group["months"] = [group["months"][key] for key in sorted(group["months"])]
    return [grouped[key] for key in sorted(grouped)]


def _plot_coverage(coverage: list[dict], destination: Path) -> str | None:
    """Save a compact monthly count plot beside the migration verification report."""
    if not coverage:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    months = sorted({month["month"] for group in coverage for month in group["months"]})
    values = np.zeros((len(coverage), len(months)), dtype=int)
    for row, group in enumerate(coverage):
        for month in group["months"]:
            values[row, months.index(month["month"])] = month["current_snapshots"]
    figure, axis = plt.subplots(figsize=(max(7, len(months) * .55), max(3, len(coverage) * .4)), layout="constrained")
    heatmap = axis.imshow(values, aspect="auto", cmap="Blues", vmin=0)
    axis.set_xticks(range(len(months)), months, rotation=55, ha="right")
    axis.set_yticks(range(len(coverage)), [f"{item['provider']}/{item['dataset']}/{item['zone']}" for item in coverage])
    axis.set_title("Verified current API snapshots by requested local month")
    for row in range(len(coverage)):
        for column in range(len(months)):
            if values[row, column]:
                axis.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=7)
    figure.colorbar(heatmap, ax=axis, label="Daily request snapshots (absence receipts excluded)")
    path = destination / "monthly_coverage.png"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return str(path)


def _import_verified(cache, snapshots: list[_Snapshot], receipts: list[_Receipt], progress) -> tuple[int, int, int]:
    """Import exact revisions and verify parity while retaining newer live DB work."""
    imported, preserved, preserved_receipts = 0, 0, 0
    fields = ("cached_at", "raw_sha256", "normalized_sha256", "records")
    with cache.batch_write():
        prior_current = {str(item.directory): cache.manifest(item.identity)
                         for item in snapshots if item.current}
        for index, item in enumerate(snapshots, 1):
            prior = prior_current.get(str(item.directory))
            newer = (prior is not None and _timestamp(prior["cached_at"], item.directory)
                     > _timestamp(item.manifest["cached_at"], item.directory))
            identical_current = prior is not None and all(prior.get(field) == item.manifest.get(field) for field in fields)
            publish = item.current and not newer and not identical_current
            if item.current and newer:
                preserved += 1
            payload, normalized, expected = _payloads(item.directory, item.identity, item.manifest)
            imported += int(cache.save_snapshot(item.identity, payload, normalized, item.manifest, make_current=publish))
            loaded = cache.load_snapshot(item.identity, manifest=item.manifest)
            if loaded is None:
                raise DataAPIError(f"Missing imported API snapshot for {item.directory}")
            actual, manifest = loaded
            try:
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
            except AssertionError:
                raise DataAPIError(f"API migration DataFrame parity mismatch for {item.directory}") from None
            stored_raw = cache.load_raw(item.identity, snapshot_id=manifest["snapshot_id"])
            if stored_raw != payload or manifest["raw_sha256"] != item.manifest["raw_sha256"]:
                raise DataAPIError(f"API migration raw-payload parity mismatch for {item.directory}")
            if progress and (index % 100 == 0 or index == len(snapshots)):
                progress(f"Verified DuckDB parity for {index}/{len(snapshots)} API snapshots")
        for index, item in enumerate(snapshots, 1):
            # Recheck SQL values after every provider schema has been unioned.
            cache.verify_native_snapshot(item.identity, manifest=item.manifest)
            if item.current:
                prior = prior_current[str(item.directory)]
                expected = (prior if prior is not None and _timestamp(prior["cached_at"], item.directory)
                            > _timestamp(item.manifest["cached_at"], item.directory) else item.manifest)
                current = cache.manifest(item.identity)
                if current is None or any(current.get(field) != expected.get(field) for field in fields):
                    raise DataAPIError(f"API migration current-snapshot mismatch for {item.directory}")
            if progress and (index % 100 == 0 or index == len(snapshots)):
                progress(f"Verified native SQL parity for {index}/{len(snapshots)} API snapshots")
        for item in sorted(receipts, key=lambda source: _timestamp(source.receipt["retrieved_at"], source.directory)):
            if item.kind == "no-data":
                prior = cache.no_data_receipt(item.identity)
            else:
                loaded = cache.load_unparsed(item.identity)
                prior = loaded[1] if loaded is not None else None
            if prior is not None and _timestamp(prior["retrieved_at"], item.directory) > _timestamp(item.receipt["retrieved_at"], item.directory):
                preserved_receipts += 1
                continue
            if item.kind == "no-data":
                cache.save_no_data_receipt(item.identity, item.receipt)
                stored = cache.no_data_receipt(item.identity)
                if stored != item.receipt:
                    raise DataAPIError(f"API migration no-data receipt parity mismatch for {item.directory}")
            else:
                payload = _unparsed_payload(item.directory, item.receipt)
                cache.save_unparsed_receipt(item.identity, payload, item.receipt)
                loaded = cache.load_unparsed(item.identity)
                if loaded is None or loaded[0] != payload or loaded[1] != item.receipt:
                    raise DataAPIError(f"API migration unparsed-payload parity mismatch for {item.directory}")
    return imported, preserved, preserved_receipts


def migrate_api_cache(config: Mapping[str, Any], progress: Callable[[str], None] | None = None) -> dict:
    """Import verified legacy snapshots without changing source files or timestamps.

    All source manifests and receipts pass preflight before DuckDB is opened.
    Exact Parquet bytes remain the fidelity copy, while DuckDB supplies a shared
    catalog and queryable records. Archived revisions never replace the original
    current pointer, and expired absence receipts keep their original TTL.
    """
    from .duckdb_store import DuckDBCache

    settings = config.get("api", {}) if isinstance(config, Mapping) else None
    if not isinstance(settings, Mapping):
        raise DataAPIError("API migration requires an api configuration mapping")
    try:
        source = Path(settings.get("cache_directory", "data/api")).resolve()
        database = Path(settings.get("database_path", "data/api/market.duckdb")).resolve()
        output_root = Path(settings.get("migration_output_directory", "outputs/api/migrations"))
    except (TypeError, ValueError, OSError):
        raise DataAPIError("API migration cache, database and output paths must be valid filesystem paths") from None
    if database.suffix != ".duckdb":
        raise DataAPIError("api.database_path must end in .duckdb")
    cache = DuckDBCache(database, threads=settings.get("database_threads", 4))
    if not source.is_dir():
        raise DataAPIError(f"API migration source directory does not exist: {source}")
    if progress:
        progress(f"Verifying legacy API snapshots in {source}")
    snapshots, receipts, partial, files, ignored = _scan(DownloadCache(source))
    # Import archived versions first, then explicitly publish each source current.
    snapshots.sort(key=lambda item: (item.current, _timestamp(item.manifest["cached_at"], item.directory), str(item.directory)))
    imported, preserved, preserved_receipts = _import_verified(cache, snapshots, receipts, progress)
    coverage = _coverage(snapshots, receipts)
    run = datetime.now(timezone.utc).strftime("migration-%Y%m%dT%H%M%S%fZ")
    output = output_root / run
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    current_count = sum(item.current for item in snapshots)
    report = {
        "schema_version": "api-duckdb-migration-v1", "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source), "database": str(database), "report": str(report_path),
        "counts": {"snapshots": len(snapshots), "current_snapshots": current_count,
                   "archived_snapshots": len(snapshots) - current_count,
                   "imported_snapshots": imported, "existing_snapshots": len(snapshots) - imported,
                   "preserved_newer_current_snapshots": preserved, "preserved_newer_receipts": preserved_receipts,
                   "raw_hashes_verified": len(snapshots) + sum(item.kind == "unparsed-data" for item in receipts),
                   "normalized_hashes_verified": len(snapshots), "parity_verified_snapshots": len(snapshots),
                   "native_parity_verified_snapshots": len(snapshots),
                   "records": sum(item.manifest["records"] for item in snapshots),
                   "current_records": sum(item.manifest["records"] for item in snapshots if item.current),
                   "no_data_receipts": sum(item.kind == "no-data" for item in receipts),
                   "unparsed_receipts": sum(item.kind == "unparsed-data" for item in receipts),
                   "partial_directories": len(partial), "ignored_directories": len(ignored),
                   "source_files": len(files)},
        "datasets": coverage,
        "snapshots": [{"directory": str(item.directory), "current": item.current,
                       "provider": item.identity["provider"], "dataset": item.identity["dataset"],
                       "zone": item.identity["zone"], "local_day": item.identity["local_day"],
                       "cached_at": item.manifest["cached_at"], "records": item.manifest["records"],
                       "raw_sha256": item.manifest["raw_sha256"],
                       "normalized_sha256": item.manifest["normalized_sha256"],
                       "first_product": item.first_product, "last_product": item.last_product}
                      for item in snapshots],
        "partial_directories": [str(path.relative_to(source)) for path in partial],
        "ignored_directories": [str(path.relative_to(source)) for path in ignored],
        "source_files": [{"path": str(path.relative_to(source)), "bytes": path.stat().st_size} for path in files],
        "interpretation": "Exact cached snapshots and source availability metadata preserved; retrieval time does not establish historical publication. Source files retained unchanged. Nested legacy roots require a separate migration.",
    }
    report["coverage_plot"] = _plot_coverage(coverage, output)
    _atomic_bytes(report_path, _json_bytes(report))
    if progress:
        progress(f"API migration complete: {imported} new snapshots, {len(partial)} partial directories skipped; report {report_path}")
    return report
