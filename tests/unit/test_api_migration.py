"""Legacy imports retain exact values, native dtypes, revisions and receipt clocks."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.data.api import DataAPIError, DownloadCache, plan_downloads
from bess_pv_trading.data.api_migration import migrate_api_cache


pytest.importorskip("duckdb")

from bess_pv_trading.data.duckdb_store import DuckDBCache


def configuration(tmp_path):
    """Keep migration fixtures independent of credentials and provider networking."""
    return {"api": {"start": "2025-01-31", "end": "2025-02-03", "timezone": "Europe/Rome",
                    "cache_directory": str(tmp_path / "cache"),
                    "database_path": str(tmp_path / "market.duckdb"),
                    "migration_output_directory": str(tmp_path / "migration"),
                    "requests": [{"provider": "entsoe", "dataset": "day_ahead_prices",
                                  "options": {"zone": "NORD"}}]}}


def typed_frame(identity, value):
    """Exercise missing source availability alongside nullable native columns."""
    start = pd.Timestamp(identity["start"])
    return pd.DataFrame({
        "product_start": pd.Series([start, start + pd.Timedelta(hours=1)]),
        "product_end": pd.Series([start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=2)]),
        "value": pd.Series([value, pd.NA], dtype="Float64"),
        "native_interval_minutes": pd.Series([60, pd.NA], dtype="Int64"),
        "historical_feature_eligible": pd.Series([False, pd.NA], dtype="boolean"),
        "historical_available_at": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns, UTC]"),
        "source_note": pd.Series(["original", pd.NA], dtype="string"),
    })


def source_hashes(root):
    """Check retained legacy files, including manifests, remain byte-for-byte intact."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_migration_retains_revisions_nulls_dtypes_and_is_idempotent(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    old = legacy.save(identity, b"old source", typed_frame(identity, -12.5), {"retrieved_at": "2026-10-01T12:00:00Z"})
    current_frame = typed_frame(identity, 47.25)
    current = legacy.save(identity, b"current source", current_frame, {"retrieved_at": "2026-10-02T12:00:00Z"})
    before = source_hashes(legacy.directory)

    report = migrate_api_cache(config)
    assert report["counts"]["current_snapshots"] == 1
    assert report["counts"]["archived_snapshots"] == 1
    assert report["counts"]["imported_snapshots"] == report["counts"]["parity_verified_snapshots"] == 2
    assert report["counts"]["native_parity_verified_snapshots"] == 2
    assert report["counts"]["current_records"] == 2
    assert Path(report["report"]).is_file() and Path(report["coverage_plot"]).is_file()
    assert json.loads(Path(report["report"]).read_text()) == report

    database = DuckDBCache(config["api"]["database_path"])
    frame, manifest = database.load(identity)
    pd.testing.assert_frame_equal(frame, current_frame, check_exact=True)
    assert manifest["metadata"] == current["metadata"]
    assert manifest["normalized_sha256"] == current["normalized_sha256"]
    history, history_manifest = database.load_snapshot(identity, manifest=old)
    pd.testing.assert_frame_equal(history, typed_frame(identity, -12.5), check_exact=True)
    assert database.load_raw(identity, snapshot_id=history_manifest["snapshot_id"]) == b"old source"
    assert len(database.snapshots(identity)) == 2

    second = migrate_api_cache(config)
    assert second["counts"]["imported_snapshots"] == 0
    assert second["counts"]["existing_snapshots"] == 2
    assert len(database.snapshots(identity)) == 2
    assert source_hashes(legacy.directory) == before


@pytest.mark.parametrize("filename", ["response.xml", "records.parquet"])
def test_corrupt_source_rejected_before_database_created(tmp_path, filename):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save(identity, b"valid source", typed_frame(identity, 1), {})
    (legacy.location(identity) / filename).write_bytes(b"corrupt")

    with pytest.raises(DataAPIError, match="checksum mismatch"):
        migrate_api_cache(config)
    assert not Path(config["api"]["database_path"]).exists()


def test_corrupt_archived_revision_is_rejected_before_any_import(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save(identity, b"old", typed_frame(identity, 1), {})
    legacy.save(identity, b"new", typed_frame(identity, 2), {})
    archived = next((legacy.location(identity) / "revisions").glob("*/response.xml"))
    archived.write_bytes(b"damaged history")

    with pytest.raises(DataAPIError, match="checksum mismatch"):
        migrate_api_cache(config)
    assert not Path(config["api"]["database_path"]).exists()


def test_original_current_snapshot_wins_over_newer_archive_timestamp(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save(identity, b"archive", typed_frame(identity, 1), {})
    legacy.save(identity, b"current", typed_frame(identity, 2), {})
    path = legacy.location(identity) / "manifest.json"
    current = json.loads(path.read_text())
    current["cached_at"] = "2001-01-01T00:00:00+00:00"
    path.write_text(json.dumps(current))

    migrate_api_cache(config)
    database = DuckDBCache(config["api"]["database_path"])
    assert database.manifest(identity)["cached_at"] == current["cached_at"]
    assert database.load(identity)[0].value.iloc[0] == 2
    migrate_api_cache(config)
    assert database.load(identity)[0].value.iloc[0] == 2


def test_rerun_preserves_later_database_download_and_newer_receipts(tmp_path):
    config = configuration(tmp_path)
    first, second, third = [item["identity"] for item in plan_downloads(config)]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save(first, b"legacy source", typed_frame(first, 1), {})
    legacy.save_no_data(second)
    legacy.save_unparsed(third, b'{"legacy_schema": []}', "2025-01-01T00:00:00Z")
    migrate_api_cache(config)
    database = DuckDBCache(config["api"]["database_path"])
    refreshed = database.save(first, b"new live source", typed_frame(first, 9), {})
    absence = database.save_no_data(second)
    database.save_unparsed(third, b'{"new_schema": []}', "2026-01-01T00:00:00Z")

    report = migrate_api_cache(config)
    assert database.manifest(first)["cached_at"] == refreshed["cached_at"]
    assert database.load(first)[0].value.iloc[0] == 9
    assert len(database.snapshots(first)) == 2
    assert database.no_data_receipt(second)["retrieved_at"] == absence["retrieved_at"]
    assert database.load_unparsed(third)[0] == b'{"new_schema": []}'
    assert report["counts"]["preserved_newer_current_snapshots"] == 1
    assert report["counts"]["preserved_newer_receipts"] == 2


def test_receipts_keep_original_ttl_and_unparsed_payload_for_offline_reparse(tmp_path):
    config = configuration(tmp_path)
    first, second, third = [item["identity"] for item in plan_downloads(config)]
    legacy = DownloadCache(config["api"]["cache_directory"])
    receipt = legacy.save_no_data(first)
    receipt["retrieved_at"] = "2000-01-01T00:00:00+00:00"
    (legacy.location(first) / "no_data.json").write_text(json.dumps(receipt))
    legacy.save_unparsed(second, b'{"unrecognized": [1]}', "2000-02-01T00:00:00+00:00")
    partial = legacy.location(third)
    partial.mkdir(parents=True)
    (partial / "response.xml").write_bytes(b"interrupted response")
    ignored = legacy.directory / "cache" / "nested-smoke"
    ignored.mkdir(parents=True)
    (ignored / "manifest.json").write_text("invalid unrelated metadata")
    before = source_hashes(legacy.directory)

    report = migrate_api_cache(config)
    database = DuckDBCache(config["api"]["database_path"])
    assert database.load_no_data(first, max_age_hours=24) is None
    original = database.load_no_data(first, max_age_hours=1_000_000)
    assert original["retrieved_at"] == receipt["retrieved_at"]
    payload, retained = database.load_unparsed(second)
    assert payload == b'{"unrecognized": [1]}'
    assert retained["retrieved_at"] == "2000-02-01T00:00:00+00:00"
    assert report["counts"]["no_data_receipts"] == 1
    assert report["counts"]["unparsed_receipts"] == 1
    assert report["counts"]["partial_directories"] == 1
    assert report["partial_directories"] == [str(partial.relative_to(legacy.directory))]
    assert report["ignored_directories"] == ["cache"]
    assert source_hashes(legacy.directory) == before


def test_unparsed_checksum_and_receipt_identity_are_verified(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save_unparsed(identity, b'{"unrecognized": []}', "2025-01-01T00:00:00Z")
    (legacy.location(identity) / "unparsed_response.json").write_bytes(b"corrupt")
    with pytest.raises(DataAPIError, match="checksum mismatch"):
        migrate_api_cache(config)


def test_misplaced_manifest_and_malformed_cache_paths_fail_clearly(tmp_path):
    config = configuration(tmp_path)
    identity = plan_downloads(config)[0]["identity"]
    legacy = DownloadCache(config["api"]["cache_directory"])
    legacy.save(identity, b"source", typed_frame(identity, 1), {})
    original = legacy.location(identity)
    misplaced = original.with_name("0" * 20)
    original.rename(misplaced)
    with pytest.raises(DataAPIError, match="identity/path mismatch"):
        migrate_api_cache(config)
    misplaced.rename(misplaced.with_name("bad-key"))
    with pytest.raises(DataAPIError, match="Malformed.*cache path"):
        migrate_api_cache(config)


@pytest.mark.parametrize("bad_config, message", [
    ({"api": None}, "configuration mapping"),
    ({"api": {"database_path": None}}, "valid filesystem paths"),
    ({"api": {"database_path": "source.parquet"}}, "must end in .duckdb"),
    ({"api": {"database_threads": 0}}, "positive integer"),
])
def test_malformed_migration_configuration_fails_before_mutation(tmp_path, bad_config, message):
    config = configuration(tmp_path)
    if isinstance(bad_config["api"], dict):
        config["api"].update(bad_config["api"])
    else:
        config.update(bad_config)
    with pytest.raises(DataAPIError, match=message):
        migrate_api_cache(config)
    assert not (tmp_path / "market.duckdb").exists()
