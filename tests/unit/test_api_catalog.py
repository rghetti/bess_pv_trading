"""Offline catalog checks cover partial downloads, provenance and monthly gaps."""

from copy import deepcopy

import pandas as pd
import pytest

from bess_pv_trading.data.api import DataAPIError, DownloadCache, plan_downloads
from bess_pv_trading.data.api_catalog import (
    discover_api_caches,
    load_api_cache_records,
    summarize_api_cache,
)


def configuration(tmp_path):
    """Plan two datasets spanning a month boundary with no authentication."""
    return {"api": {"start": "2025-01-31", "end": "2025-02-02", "timezone": "Europe/Rome",
                    "cache_directory": str(tmp_path / "cache"), "no_data_cache_hours": 24,
                    "requests": [{"provider": "entsoe", "dataset": name,
                                  "options": {"zone": "NORD"}}
                                 for name in ("day_ahead_prices", "load_forecast")]}}


def save_snapshot(config, identity, *, value=-12.5):
    """Publish a small valid cache whose historical publication stays unknown."""
    start = pd.Timestamp(identity["start"])
    frame = pd.DataFrame({"product_start": [start], "product_end": [start + pd.Timedelta(hours=1)],
                          "value": [value], "historical_feature_eligible": [False],
                          "historical_available_at": [pd.NaT]})
    cache = DownloadCache(config["api"]["cache_directory"])
    cache.save(identity, b"synthetic offline source", frame, {})
    return cache


def test_reader_ignores_partial_and_archived_snapshots_and_preserves_provenance(tmp_path):
    config = configuration(tmp_path)
    identities = [item["identity"] for item in plan_downloads(config)
                  if item["identity"]["dataset"] == "day_ahead_prices"]
    cache = save_snapshot(config, identities[0], value=1)
    save_snapshot(config, identities[0], value=-12.5)  # Archive exists but is not a new observation.
    partial = cache.location(identities[1])
    partial.mkdir(parents=True)
    (partial / "response.xml").write_bytes(b"inflight response without final manifest")
    entries = discover_api_caches(cache.directory, dataset="day_ahead_prices")
    assert len(entries) == 1
    records, sources = load_api_cache_records(cache.directory, provider="entsoe",
                                             dataset="day_ahead_prices", zone="NORD")
    assert len(records) == 1
    assert records.value.tolist() == [-12.5]
    assert records.historical_feature_eligible.tolist() == [False]
    assert records.historical_available_at.isna().all()
    assert len(sources) == 1 and sources[0]["checksums_verified"]
    assert len(sources[0]["normalized_sha256"]) == 64


def test_reader_rejects_corrupt_payload_instead_of_using_csv(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    cache = save_snapshot(config, identity)
    (cache.location(identity) / "records.parquet").write_bytes(b"corruption")
    with pytest.raises(DataAPIError, match="checksum mismatch"):
        load_api_cache_records(cache.directory)


def test_catalog_month_readiness_keeps_explicit_no_data_separate(tmp_path):
    config = configuration(tmp_path)
    plans = plan_downloads(config)
    cache = DownloadCache(config["api"]["cache_directory"])
    for plan in plans:
        identity = plan["identity"]
        if identity["local_day"] == "2025-01-31":
            save_snapshot(config, identity)
        elif identity["dataset"] == "load_forecast":
            cache.save_no_data(identity)
    status = summarize_api_cache(config)
    assert status["totals"] == {"expected": 4, "cached": 2, "no_data": 1, "missing": 1, "records": 2}
    assert status["ready_months"] == ["2025-01"]
    assert [month["ready"] for month in status["months"]] == [True, False]
    assert status["months"][1]["datasets"][1]["no_data"] == 1
    assert summarize_api_cache(config, verify_checksums=True)["totals"] == status["totals"]


def test_catalog_coverage_uses_exact_request_identity(tmp_path):
    config = configuration(tmp_path)
    save_snapshot(config, plan_downloads(config)[0]["identity"])
    changed = deepcopy(config)
    changed["api"]["requests"][0]["vintage_revision"] = "different experiment vintage"
    assert summarize_api_cache(changed)["totals"]["cached"] == 0


def test_reader_delivery_bounds_require_aware_complete_products(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    cache = save_snapshot(config, identity)
    frame, _ = load_api_cache_records(cache.directory, start=identity["start"], end=identity["end"])
    assert len(frame) == 1
    frame, _ = load_api_cache_records(cache.directory, start=pd.Timestamp(identity["start"]) + pd.Timedelta(minutes=15))
    assert frame.empty
    with pytest.raises(DataAPIError, match="timezone-aware"):
        load_api_cache_records(cache.directory, start="2025-01-31")
