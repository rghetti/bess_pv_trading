"""Offline storage checks for exact bytes, revision isolation and SQL schemas."""

from copy import deepcopy
import builtins
import hashlib
from io import BytesIO
import json
import subprocess
import sys

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from bess_pv_trading.data.api import DataAPIError, DownloadCache
from bess_pv_trading.data.duckdb_store import DuckDBCache


def identity(*, dataset="day_ahead_prices", vintage="latest_at_retrieval"):
    """Specify a nonsecret request with an exact vintage and schema identity."""
    return {"provider": "entsoe", "dataset": dataset, "zone": "NORD",
            "start": "2025-01-01T00:00:00+00:00", "end": "2025-01-02T00:00:00+00:00",
            "local_day": "2025-01-01", "timezone": "UTC", "options": {"zone": "NORD"},
            "request": {"url": "https://web-api.tp.entsoe.eu/api", "params": {}, "headers": {}},
            "schema_version": "api-cache-v1", "vintage_revision": vintage}


def records(value=-12.5):
    """Include nullable values and unknown historical availability explicitly."""
    return pd.DataFrame({"product_start": pd.to_datetime(["2025-01-01T00:00:00Z"]),
                         "product_end": pd.to_datetime(["2025-01-01T01:00:00Z"]),
                         "value": [value], "quality": pd.Series([None], dtype="string"),
                         "native_count": pd.Series([1], dtype="Int64"),
                         "historical_feature_eligible": [False],
                         "historical_available_at": [pd.NaT]})


def test_exact_source_bytes_and_dataframe_provenance_survive_import(tmp_path):
    """Import an existing cache without reserializing its authoritative Parquet."""
    request = identity()
    source = DownloadCache(tmp_path / "legacy")
    original = source.save(request, b"original raw source", records(), {"unit": "EUR/MWh", "native_interval": "PT60M"})
    normalized = (source.location(request) / "records.parquet").read_bytes()
    store = DuckDBCache(tmp_path / "api.duckdb")
    assert store.save_snapshot(request, b"original raw source", normalized, original)
    assert not store.save_snapshot(request, b"original raw source", normalized, original)
    frame, manifest = store.load(request)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(BytesIO(normalized)))
    assert store.load_raw(request) == b"original raw source"
    assert manifest["cached_at"] == original["cached_at"]
    assert manifest["metadata"] == original["metadata"]
    assert manifest["source_directory"] == original["directory"]
    assert manifest["normalized_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert manifest["storage_backend"] == "duckdb"
    assert not store.location(request).exists()
    with duckdb.connect(str(store.directory), read_only=True) as connection:
        raw, parquet, saved_manifest = connection.execute("SELECT raw_payload, normalized_parquet, manifest_json FROM api_snapshots").fetchone()
    assert raw == b"original raw source" and parquet == normalized
    assert json.loads(saved_manifest) == original


def test_revisions_are_immutable_and_current_pointer_respects_import_order(tmp_path):
    """A refresh changes current data without duplicating old observations."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    request = identity()
    first = store.save(request, b"first", records(1), {})
    second = store.save(request, b"second", records(2), {})
    other = identity(vintage="separate snapshot")
    store.save(other, b"separate", records(3), {})
    assert store.load(request)[0].value.tolist() == [2]
    assert store.load_snapshot(request, snapshot_id=first["snapshot_id"])[0].value.tolist() == [1]
    assert store.load_raw(request, snapshot_id=first["snapshot_id"]) == b"first"
    assert len(store.snapshots(request)) == 2
    assert len(store.entries(provider="entsoe", dataset="day_ahead_prices", zone="NORD")) == 2
    assert store.entries(provider="terna") == []
    with duckdb.connect(str(store.directory), read_only=True) as connection:
        assert connection.execute("SELECT value FROM current_api_records ORDER BY value").fetchall() == [(2,), (3,)]
        assert connection.execute("SELECT count(*) FROM api_records").fetchone() == (3,)
        original = json.loads(connection.execute("SELECT manifest_json FROM api_snapshots WHERE snapshot_id = ?", [first["snapshot_id"]]).fetchone()[0])
        normalized = connection.execute("SELECT normalized_parquet FROM api_snapshots WHERE snapshot_id = ?", [first["snapshot_id"]]).fetchone()[0]
    assert not store.save_snapshot(request, b"first", normalized, original, make_current=False)
    assert store.manifest(request)["snapshot_id"] == second["snapshot_id"]
    assert not store.save_snapshot(request, b"first", normalized, original, make_current=True)
    assert store.load(request)[0].value.tolist() == [1]


def test_native_schema_promotes_initial_null_columns_and_keeps_utc(tmp_path):
    """A null-only first provider schema must not force later values to integers."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    first = pd.DataFrame({"value": [1], "optional": [None], "published_at": [pd.NaT]})
    second = pd.DataFrame({"value": [2.5], "optional": ["native source text"],
                           "published_at": pd.to_datetime(["2025-01-01T02:00:00+02:00"], utc=True),
                           "new_field": [True]})
    with store.batch_write():
        store.save(identity(), b"first", first, {})
        store.save(identity(dataset="load_forecast"), b"second", second, {})
        assert len(store.entries()) == 2
    with duckdb.connect(str(store.directory), read_only=True) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        rows = connection.execute("SELECT value, optional, published_at, new_field FROM current_api_records ORDER BY value").fetchall()
        types = dict((row[0], row[1]) for row in connection.execute("DESCRIBE api_records").fetchall())
    assert rows[0] == (1.0, None, None, None)
    assert rows[1][:2] == (2.5, "native source text")
    assert rows[1][2] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert types["optional"] == "VARCHAR"
    assert types["published_at"] == "TIMESTAMP WITH TIME ZONE"
    assert types["value"] == "DOUBLE"
    for entry in store.entries():
        assert store.verify_native_snapshot(entry["identity"], snapshot_id=entry["manifest"]["snapshot_id"])


def test_native_verification_rejects_tampering_even_when_blob_checksums_pass(tmp_path):
    """SQL analytics must match the authoritative Parquet independently of BLOBs."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    request = identity()
    saved = store.save(request, b"raw", records(1), {})
    assert store.verify_native_snapshot(request, snapshot_id=saved["snapshot_id"])
    with duckdb.connect(str(store.directory)) as connection:
        connection.execute("UPDATE api_records SET value = 999")
    assert store.load(request)[0].value.tolist() == [1]
    with pytest.raises(DataAPIError, match="Native DuckDB record parity mismatch"):
        store.verify_native_snapshot(request)


def test_failed_refresh_does_not_publish_partial_snapshot_and_batch_recovers(tmp_path, monkeypatch):
    """Rollback includes BLOBs, schema edits, rows and the current pointer."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    request = identity()
    old = store.save(request, b"old", records(1), {})
    insert = store._insert_records

    def broken_insert(connection, table, snapshot_id):
        """Fail after inserting rows to exercise the complete transaction boundary."""
        insert(connection, table, snapshot_id)
        raise RuntimeError("simulated interrupted publication")

    with store.batch_write():
        monkeypatch.setattr(store, "_insert_records", broken_insert)
        with pytest.raises(RuntimeError, match="interrupted publication"):
            store.save(request, b"failed", records(2), {})
        monkeypatch.setattr(store, "_insert_records", insert)
        assert store.manifest(request)["snapshot_id"] == old["snapshot_id"]
        assert len(store.snapshots(request)) == 1
        store.save(request, b"good", records(3), {})
    assert store.load(request)[0].value.tolist() == [3]
    assert len(store.snapshots(request)) == 2


@pytest.mark.parametrize("column, expected", [("raw_payload", "API payload"), ("normalized_parquet", "normalized records")])
def test_corruption_is_rejected_without_csv_fallback(tmp_path, column, expected):
    """Checksum verification applies to both independent authoritative byte sets."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    store.save(identity(), b"raw", records(), {})
    with duckdb.connect(str(store.directory)) as connection:
        connection.execute(f"UPDATE api_snapshots SET {column} = ?", [b"corruption"])
    with pytest.raises(DataAPIError, match=expected + " checksum mismatch"):
        store.load(identity())


def test_missing_database_reads_are_inert_and_read_session_reuses_one_connection(tmp_path):
    """Status and experimentation must never create an empty cache accidentally."""
    store = DuckDBCache(tmp_path / "absent" / "api.duckdb")
    with store.read_session():
        assert store.load(identity()) is None
        assert store.load_raw(identity()) is None
        assert store.manifest(identity()) is None
        assert store.entries() == store.snapshots(identity()) == store.no_data_entries() == []
        assert store.load_no_data(identity(), 24) is None
        assert store.load_unparsed(identity()) is None
    assert not store.directory.parent.exists()
    store.save(identity(), b"raw", records(), {})
    with store.read_session():
        connection = store._session_connection
        assert store.load(identity()) is not None
        assert store.manifest(identity()) is not None
        assert connection is store._session_connection
        with pytest.raises(DataAPIError, match="read-only"):
            store.save(identity(), b"new", records(2), {})
    assert store._session_connection is None


def test_no_data_ttl_and_retained_unparsed_response_keep_original_clocks(tmp_path):
    """Absence remains a gap; parser retries retain the original retrieval time."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    request = identity()
    expired = {"request": request, "status": "no-data", "records": 0,
               "retrieved_at": "2020-01-01T00:00:00Z"}
    store.save_no_data_receipt(request, expired)
    assert store.load_no_data(request, 24) is None
    assert store.no_data_receipt(request) == expired
    assert store.no_data_entries()[0]["receipt"] == expired
    receipt = store.save_no_data(request)
    assert store.load_no_data(request, 24)["retrieved_at"] == receipt["retrieved_at"]
    store.save(request, b"normalized raw", records(), {})
    assert store.load_no_data(request, 24) is None
    store.save_no_data(request)
    assert store.load(request) is not None
    payload = b'{"unexpected_array":[]}'
    store.save_unparsed(request, payload, "2025-01-01T03:00:00Z")
    raw, retained = store.load_unparsed(request)
    assert raw == payload and retained["retrieved_at"] == "2025-01-01T03:00:00Z"
    with duckdb.connect(str(store.directory)) as connection:
        connection.execute("UPDATE api_unparsed_responses SET payload = ?", [b"changed"])
    with pytest.raises(DataAPIError, match="Invalid retained API response"):
        store.load_unparsed(request)


@pytest.mark.parametrize("payload", [b"synthetic-secret", b'{"echo":"synthetic-secre\\u0074"}'])
def test_sensitive_unparsed_response_is_rejected_before_database_creation(tmp_path, payload):
    """Decoded escapes cannot hide echoed authentication material in retained JSON."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    with pytest.raises(DataAPIError, match="authentication material"):
        store.save_unparsed(identity(), payload, "2025-01-01T03:00:00Z", ["synthetic-secret"])
    assert not store.directory.exists()


def test_invalid_import_checksum_and_credential_identity_do_not_touch_store(tmp_path):
    """Validate source manifests and the complete identity before opening a writer."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    request = identity()
    source = DownloadCache(tmp_path / "legacy")
    manifest = source.save(request, b"raw", records(), {})
    normalized = (source.location(request) / "records.parquet").read_bytes()
    with pytest.raises(DataAPIError, match="checksum mismatch"):
        store.save_snapshot(request, b"wrong", normalized, manifest)
    assert not store.directory.exists()
    secret_request = deepcopy(request)
    secret_request["options"]["securityToken"] = "synthetic-secret"
    with pytest.raises(DataAPIError, match="Credentials"):
        store.save(secret_request, b"raw", records(), {})
    assert not store.directory.exists()


def test_cross_process_lock_has_actionable_error_without_provider_payload(tmp_path):
    """Independent experiments cannot read a database held by another writer."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    store.save(identity(), b"private provider payload", records(), {})
    script = """import json, sys
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.api import DataAPIError
try:
    DuckDBCache(sys.argv[1]).load(json.loads(sys.argv[2]))
except DataAPIError as error:
    print(str(error))
    sys.exit(2)
"""
    with duckdb.connect(str(store.directory)):
        result = subprocess.run([sys.executable, "-c", script, str(store.directory), json.dumps(identity())],
                                capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "is locked" in result.stdout and "separate database snapshot" in result.stdout
    assert "private provider payload" not in result.stdout + result.stderr


def test_missing_package_has_explicit_conda_guidance(tmp_path, monkeypatch):
    """An optional package failure must explain the configured environment step."""
    store = DuckDBCache(tmp_path / "api.duckdb")
    store.save(identity(), b"raw", records(), {})
    original_import = builtins.__import__

    def without_duckdb(name, *args, **kwargs):
        """Simulate an environment without the selected storage dependency."""
        if name == "duckdb":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_duckdb)
    with pytest.raises(DataAPIError, match="python-duckdb in the bess conda environment"):
        store.load(identity())
