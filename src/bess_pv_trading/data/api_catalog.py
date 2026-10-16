"""Read complete API snapshots and inspect progressive monthly coverage offline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .api import (
    DataAPIError,
    DownloadCache,
    cache_from_location,
    cache_from_settings,
    plan_downloads,
)


@dataclass(frozen=True)
class ApiCacheEntry:
    """A published daily snapshot, with a filesystem or virtual source locator."""

    manifest_path: Path
    identity: dict
    manifest: dict

    @property
    def directory(self) -> Path:
        """Locate filesystem payloads or identify a database request virtually."""
        return self.manifest_path.parent


def discover_api_caches(cache_directory: str | Path, *, provider: str | None = None,
                        dataset: str | None = None, zone: str | None = None) -> list[ApiCacheEntry]:
    """Find published current snapshots without opening credentials or networking.

    A manifest is the final atomic write of a daily cache. Directories containing
    only a payload, and manifests below ``revisions/``, are deliberately excluded.
    Discovery validates identity/path consistency; the reader verifies payloads.
    A ``.duckdb`` path reads the current database snapshots instead of files.
    """
    return _discover_api_caches(cache_from_location(cache_directory), provider=provider,
                                dataset=dataset, zone=zone)


def _discover_api_caches(cache, *, provider=None, dataset=None, zone=None) -> list[ApiCacheEntry]:
    """Reuse an existing database read session when discovering its snapshots."""
    if not isinstance(cache, DownloadCache):
        return [ApiCacheEntry(Path(item["source"]), item["identity"], item["manifest"])
                for item in cache.entries(provider=provider, dataset=dataset, zone=zone)]
    entries = []
    for path in sorted(cache.directory.glob("*/*/*/*/*/manifest.json")):
        # Filter path components before reading unrelated provider manifests.
        components = path.relative_to(cache.directory).parts
        if any(wanted is not None and actual != wanted
               for wanted, actual in zip((provider, dataset, zone), components[:3])):
            continue
        try:
            manifest = json.loads(path.read_text())
            identity = manifest["request"]
            expected = cache.location(identity)
        except (KeyError, TypeError, ValueError, OSError):
            raise DataAPIError(f"Invalid API cache manifest at {path}") from None
        if path.parent != expected or manifest.get("schema_version") != "api-cache-v1":
            raise DataAPIError(f"API cache identity/path mismatch at {path}")
        entries.append(ApiCacheEntry(path, identity, manifest))
    return entries


def _aware_bound(value: Any, label: str) -> pd.Timestamp | None:
    """Require explicit clocks when limiting reusable records by delivery time."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataAPIError(f"API catalog {label} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def load_api_cache_records(cache_directory: str | Path, *, provider: str | None = None,
                           dataset: str | None = None, zone: str | None = None,
                           start: Any = None, end: Any = None,
                           verify_checksums: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Load complete immutable daily snapshots and their exact source provenance.

    Records retain native parser fields, repeated-block markers and unverified
    historical availability. The cache location can be a legacy directory or a
    DuckDB file. No publication clocks, intervals or observations are manufactured.
    Optional bounds select complete products inside [start, end); absence is
    returned as an empty frame, never as zero-valued records.
    """
    first, stop = _aware_bound(start, "start"), _aware_bound(end, "end")
    if first is not None and stop is not None and first >= stop:
        raise DataAPIError("API catalog start must precede exclusive end")
    cache = cache_from_location(cache_directory)
    session = nullcontext() if isinstance(cache, DownloadCache) else cache.read_session()
    with session:
        return _load_api_cache_records(cache, provider=provider, dataset=dataset, zone=zone,
                                        first=first, stop=stop, verify_checksums=verify_checksums)


def _load_api_cache_records(cache, *, provider, dataset, zone, first, stop,
                            verify_checksums: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Read pinned revisions through one database connection and transaction."""
    database_backend = not isinstance(cache, DownloadCache)
    frames, provenance = [], []
    for entry in _discover_api_caches(cache, provider=provider, dataset=dataset, zone=zone):
        identity = entry.identity
        if first is not None and pd.Timestamp(identity["end"]) <= first:
            continue
        if stop is not None and pd.Timestamp(identity["start"]) >= stop:
            continue
        if verify_checksums or database_backend:
            loaded = (cache.load_snapshot(identity, snapshot_id=entry.manifest["snapshot_id"],
                                           verify_checksums=verify_checksums)
                      if database_backend else cache.load(identity))
            if loaded is None:  # A concurrent writer has not published this cache yet.
                continue
            frame, manifest = loaded
        else:
            manifest = entry.manifest
            try:
                frame = pd.read_parquet(entry.directory / "records.parquet")
            except (ValueError, OSError):
                raise DataAPIError(f"Incomplete/corrupt API cache at {entry.directory}") from None
        if first is not None or stop is not None:
            if not {"product_start", "product_end"}.issubset(frame.columns):
                raise DataAPIError("Delivery bounds require explicit product_start/product_end columns")
            keep = pd.Series(True, index=frame.index)
            if first is not None:
                keep &= frame.product_start >= first
            if stop is not None:
                keep &= frame.product_end <= stop
            frame = frame.loc[keep].copy()
        frames.append(frame)
        source = {"manifest": str(entry.manifest_path), "directory": str(entry.directory),
                  "request": identity, "cached_at": manifest.get("cached_at"),
                  "raw_sha256": manifest.get("raw_sha256"),
                  "normalized_sha256": manifest.get("normalized_sha256"),
                  "records": manifest.get("records"), "selected_records": len(frame),
                  "checksums_verified": verify_checksums}
        if database_backend:
            # Snapshot identifiers make an experiment traceable after a refresh.
            source["storage"] = manifest["storage"]
            if "source_directory" in manifest:
                source["source_directory"] = manifest["source_directory"]
        provenance.append(source)
    if not frames:
        return pd.DataFrame(), provenance
    combined = pd.concat(frames, ignore_index=True)
    columns = [name for name in ("product_start", "product_end", "provider", "dataset", "zone")
               if name in combined]
    if columns:
        combined = combined.sort_values(columns, kind="stable", ignore_index=True)
    return combined, provenance


def summarize_api_cache(config: Mapping[str, Any], *, verify_checksums: bool = False) -> dict:
    """Report exact configured daily request coverage, grouped by local month.

    Cached no-data responses are explicit gaps and do not make a month ready for
    experiments. A ready month means all configured daily payloads are present;
    it does not establish historical forecast publication or validate their
    scientific suitability. Metadata-only status avoids decoding every Parquet;
    model ingestion verifies both checksums. Database status reads metadata
    without accessing legacy source directories and requires no active writer.
    """
    settings = config.get("api", {})
    cache = cache_from_settings(settings)
    session = nullcontext() if isinstance(cache, DownloadCache) else cache.read_session()
    with session:
        return _summarize_api_cache(config, cache, verify_checksums=verify_checksums)


def _summarize_api_cache(config, cache, *, verify_checksums: bool) -> dict:
    """Batch database metadata so absent historical requests require no queries."""
    settings = config.get("api", {})
    database_backend = not isinstance(cache, DownloadCache)
    if database_backend:
        database_manifests = {cache.location(entry["identity"]): entry["manifest"]
                              for entry in cache.entries()}
        no_data_locations = {cache.location(entry["identity"]) for entry in cache.no_data_entries()}
    no_data_hours = float(settings.get("no_data_cache_hours", 24))
    months: dict[str, dict] = {}
    total = {name: 0 for name in ("expected", "cached", "no_data", "missing", "records")}
    for plan in plan_downloads(config):
        identity = plan["identity"]
        month = identity["local_day"][:7]
        group = months.setdefault(month, {"month": month, "expected": 0, "cached": 0,
                                         "no_data": 0, "missing": 0, "records": 0,
                                         "datasets": {}})
        key = "/".join(identity[name] for name in ("provider", "dataset", "zone"))
        dataset_group = group["datasets"].setdefault(key, {"provider": identity["provider"],
            "dataset": identity["dataset"], "zone": identity["zone"], "expected": 0,
            "cached": 0, "no_data": 0, "missing": 0, "records": 0})
        directory = cache.location(identity)
        manifest_path = directory / "manifest.json"
        records, status = 0, "missing"
        if database_backend:
            manifest = database_manifests.get(directory)
            if manifest is not None:
                if verify_checksums:
                    loaded = cache.load_snapshot(identity, snapshot_id=manifest["snapshot_id"])
                    if loaded is not None:
                        frame, manifest = loaded
                        records, status = len(frame), "cached"
                else:
                    records, status = int(manifest["records"]), "cached"
            elif directory in no_data_locations and cache.load_no_data(identity, no_data_hours) is not None:
                status = "no_data"
        elif manifest_path.is_file():
            if verify_checksums:
                loaded = cache.load(identity)
                if loaded is not None:
                    frame, manifest = loaded
                    status, records = "cached", len(frame)
            else:
                try:
                    manifest = json.loads(manifest_path.read_text())
                    expected_raw = "response.xml" if identity["provider"] == "entsoe" else "response.json"
                    valid = (manifest.get("request") == identity and manifest.get("raw_file") == expected_raw
                             and (directory / expected_raw).is_file()
                             and (directory / "records.parquet").is_file())
                    if not valid:
                        raise DataAPIError(f"Incomplete/corrupt API cache at {directory}")
                    records, status = int(manifest["records"]), "cached"
                except (OSError, KeyError, TypeError, ValueError):
                    raise DataAPIError(f"Incomplete/corrupt API cache at {directory}") from None
        elif cache.load_no_data(identity, no_data_hours) is not None:
            status = "no_data"
        for target in (total, group, dataset_group):
            target["expected"] += 1
            target[status] += 1
            target["records"] += records
    for group in months.values():
        group["ready"] = group["cached"] == group["expected"]
        group["all_requests_resolved"] = group["missing"] == 0
        group["datasets"] = list(group["datasets"].values())
        for dataset_group in group["datasets"]:
            dataset_group["ready"] = dataset_group["cached"] == dataset_group["expected"]
    return {"status": "complete" if total["cached"] == total["expected"] else "in-progress",
            "cache_directory": str(cache.directory), "checksums_verified": verify_checksums,
            "storage_backend": "duckdb" if database_backend else "files",
            "totals": total, "ready_months": [month for month, group in sorted(months.items()) if group["ready"]],
            "months": [group for _, group in sorted(months.items())],
            "interpretation": "ready means all configured daily snapshots are cached; no-data receipts remain gaps, and publication availability requires separate validation"}
