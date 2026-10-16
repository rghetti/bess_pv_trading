"""Database catalogs preserve offline records, provenance and coverage semantics."""

from copy import deepcopy
from pathlib import Path

import pandas as pd

from bess_pv_trading.data.api import cache_from_settings, plan_downloads
from bess_pv_trading.data.api_catalog import (
    discover_api_caches,
    load_api_cache_records,
    summarize_api_cache,
)


def configuration(tmp_path):
    """Use exact daily requests spanning two months in one offline database."""
    return {"api": {"start": "2025-01-31", "end": "2025-02-02", "timezone": "Europe/Rome",
                    "storage_backend": "duckdb", "database_path": str(tmp_path / "market.duckdb"),
                    "cache_directory": str(tmp_path / "legacy"), "no_data_cache_hours": 24,
                    "requests": [{"provider": "entsoe", "dataset": name,
                                  "options": {"zone": "NORD"}}
                                 for name in ("day_ahead_prices", "load_forecast")]}}


def save_snapshot(cache, identity, value):
    """Keep negative observations and unknown historical availability intact."""
    start = pd.Timestamp(identity["start"])
    frame = pd.DataFrame({"product_start": [start], "product_end": [start + pd.Timedelta(hours=1)],
                          "value": [value], "historical_feature_eligible": [False],
                          "historical_available_at": [pd.NaT]})
    return cache.save(identity, b"synthetic offline source", frame, {})


def test_database_catalog_reads_only_current_snapshot_with_exact_provenance(tmp_path):
    config = configuration(tmp_path)
    cache = cache_from_settings(config["api"])
    identities = [plan["identity"] for plan in plan_downloads(config)
                  if plan["identity"]["dataset"] == "day_ahead_prices"]
    save_snapshot(cache, identities[0], 100)
    first = save_snapshot(cache, identities[0], -12.5)
    save_snapshot(cache, identities[1], 23)
    forecast = next(plan["identity"] for plan in plan_downloads(config)
                    if plan["identity"]["dataset"] == "load_forecast")
    save_snapshot(cache, forecast, 350)

    entries = discover_api_caches(cache.directory, provider="entsoe", dataset="day_ahead_prices", zone="NORD")
    assert len(entries) == 2
    assert not Path(config["api"]["cache_directory"]).exists()
    records, sources = load_api_cache_records(cache.directory, provider="entsoe", dataset="day_ahead_prices", zone="NORD")
    assert records.value.tolist() == [-12.5, 23]
    assert records.historical_feature_eligible.tolist() == [False, False]
    assert records.historical_available_at.isna().all()
    assert all(source["checksums_verified"] for source in sources)
    source = next(source for source in sources if source["request"] == identities[0])
    assert source["raw_sha256"] == first["raw_sha256"]
    assert source["normalized_sha256"] == first["normalized_sha256"]
    assert source["storage"]["backend"] == "duckdb"
    assert source["storage"]["database"] == str(cache.directory)
    assert source["storage"]["snapshot_id"] == first["storage"]["snapshot_id"]


def test_database_delivery_bounds_and_unverified_read_use_database_payload(tmp_path):
    config = configuration(tmp_path)
    cache = cache_from_settings(config["api"])
    identity = plan_downloads(config)[0]["identity"]
    save_snapshot(cache, identity, -12.5)
    records, sources = load_api_cache_records(cache.directory, start=identity["start"], end=identity["end"],
                                              verify_checksums=False)
    assert records.value.tolist() == [-12.5]
    assert sources[0]["checksums_verified"] is False
    assert not (cache.location(identity) / "records.parquet").is_file()
    records, sources = load_api_cache_records(cache.directory,
                                              start=pd.Timestamp(identity["start"]) + pd.Timedelta(minutes=15))
    assert records.empty
    assert sources[0]["selected_records"] == 0


def test_database_reader_pins_discovered_revision_when_current_changes(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    cache = cache_from_settings(config["api"])
    identity = plan_downloads(config)[0]["identity"]
    first = save_snapshot(cache, identity, 100)
    discovered = cache.entries()
    save_snapshot(cache, identity, -12.5)

    # Model a refresh published after discovery but before the payload read.
    monkeypatch.setattr(type(cache), "entries", lambda *args, **kwargs: discovered)
    records, sources = load_api_cache_records(cache.directory)
    assert records.value.tolist() == [100]
    assert sources[0]["storage"]["snapshot_id"] == first["storage"]["snapshot_id"]


def test_database_coverage_preserves_no_data_gaps_and_request_identity(tmp_path):
    config = configuration(tmp_path)
    cache = cache_from_settings(config["api"])
    for plan in plan_downloads(config):
        identity = plan["identity"]
        if identity["local_day"] == "2025-01-31":
            save_snapshot(cache, identity, -12.5)
        elif identity["dataset"] == "load_forecast":
            cache.save_no_data(identity)
    status = summarize_api_cache(config)
    assert status["storage_backend"] == "duckdb"
    assert status["totals"] == {"expected": 4, "cached": 2, "no_data": 1, "missing": 1, "records": 2}
    assert status["ready_months"] == ["2025-01"]
    assert summarize_api_cache(config, verify_checksums=True)["totals"] == status["totals"]

    changed = deepcopy(config)
    changed["api"]["requests"][0]["vintage_revision"] = "different experiment vintage"
    assert summarize_api_cache(changed)["totals"]["cached"] == 1


def test_missing_database_discovery_and_status_do_not_create_storage(tmp_path):
    config = configuration(tmp_path)
    database = config["api"]["database_path"]
    assert discover_api_caches(database) == []
    records, sources = load_api_cache_records(database)
    assert records.empty and sources == []
    assert summarize_api_cache(config)["totals"]["missing"] == 4
    assert not tmp_path.joinpath("market.duckdb").exists()
    assert not tmp_path.joinpath("legacy").exists()


def test_database_metadata_status_does_not_decode_payloads(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    cache = cache_from_settings(config["api"])
    save_snapshot(cache, plan_downloads(config)[0]["identity"], -12.5)

    def unexpected_payload_read(*args, **kwargs):
        raise AssertionError("metadata-only status must not decode records")

    monkeypatch.setattr(type(cache), "load_snapshot", unexpected_payload_read)
    assert summarize_api_cache(config)["totals"] == {
        "expected": 4, "cached": 1, "no_data": 0, "missing": 3, "records": 1}
