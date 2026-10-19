"""Transactional API snapshots with exact source bytes and native SQL records.

The database is a self-contained cache. Parquet bytes remain authoritative for
DataFrame fidelity; ``api_records`` also exposes every parser field for SQL.
Connections last only for a local operation, never for a provider HTTP request.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = 1


def _error(message: str):
    """Resolve the public exception lazily so the cache factory can import us."""
    from .api import DataAPIError
    return DataAPIError(message)


def _json_bytes(value: Any) -> bytes:
    """Use the existing canonical encoding for stable full request identities."""
    from .api import _json_bytes as encode
    return encode(value)


def _identity_hash(identity: Mapping[str, Any]) -> str:
    """Reject credentials and unsafe identity components before persistence."""
    from .api import _check_nonsecret
    _check_nonsecret(identity)
    for name in ("provider", "dataset", "zone"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(identity.get(name, ""))):
            raise _error("Unsafe provider/dataset cache component")
    return hashlib.sha256(_json_bytes(dict(identity))).hexdigest()


def _stamp(value: Any, label: str) -> pd.Timestamp:
    """Keep source clocks explicit without inventing historical availability."""
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError
        return stamp.tz_convert("UTC")
    except (TypeError, ValueError):
        raise _error(f"API cache {label} must be a timezone-aware timestamp") from None


def _quote(identifier: str) -> str:
    """Quote provider field names as SQL identifiers, never SQL expressions."""
    return '"' + identifier.replace('"', '""') + '"'


class DuckDBCache:
    """Store immutable revisions and publish a current snapshot atomically."""

    def __init__(self, path: str | Path, *, threads: int = 4):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise _error("DuckDB cache threads must be a positive integer")
        self.directory = Path(path)
        self.threads = threads
        self._session_connection = None
        self._session_writable = False

    def location(self, identity: Mapping[str, Any]) -> Path:
        """Return a virtual provenance locator; it is never a disk directory."""
        digest = _identity_hash(identity)
        return Path(str(self.directory) + "::requests") / digest

    @contextmanager
    def _connect(self, *, write: bool = False, transaction: bool = True):
        """Bound database locks to one operation and leave absent reads inert."""
        if not write and not self.directory.is_file():
            if self.directory.exists():
                raise _error(f"DuckDB API cache path is not a file: {self.directory}")
            yield None
            return
        try:
            import duckdb
        except ImportError:
            raise _error("DuckDB API storage requires python-duckdb in the bess conda environment; install it with conda after obtaining permission") from None
        connection = self._session_connection
        owns_connection = connection is None
        if write and connection is not None and not self._session_writable:
            raise _error("Cannot write through a read-only DuckDB cache session")
        try:
            if owns_connection:
                if write:
                    self.directory.parent.mkdir(parents=True, exist_ok=True)
                connection = duckdb.connect(str(self.directory), read_only=not write,
                                            config={"threads": self.threads,
                                                    "autoinstall_known_extensions": False})
                connection.execute("SET TimeZone = 'UTC'")
            if write and transaction:
                connection.execute("BEGIN TRANSACTION")
            if owns_connection:
                if write:
                    self._ensure_schema(connection)
                else:
                    self._check_schema(connection)
            yield connection
            if write and transaction:
                connection.execute("COMMIT")
        except BaseException as error:
            if connection is not None and write and transaction:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            if not isinstance(error, (duckdb.Error, OSError)):
                raise
            if "lock" in str(error).lower():
                raise _error(f"DuckDB API cache is locked at {self.directory}; stop the other downloader/reader process and retry, or use a separate database snapshot for experiments") from None
            raise _error(f"Could not {'write' if write else 'read'} DuckDB API cache at {self.directory}; verify the database and schema ({type(error).__name__})") from None
        finally:
            if connection is not None and owns_connection:
                connection.close()

    @contextmanager
    def _session(self, *, write: bool):
        """Reuse one local connection for a bounded sequential offline operation."""
        if self._session_connection is not None:
            if write and not self._session_writable:
                raise _error("Cannot write through a read-only DuckDB cache session")
            yield self
            return
        with self._connect(write=write, transaction=False) as connection:
            self._session_connection = connection
            self._session_writable = write
            if not write and connection is not None:
                connection.execute("BEGIN TRANSACTION")
            try:
                yield self
                if not write and connection is not None:
                    connection.execute("COMMIT")
            except BaseException:
                if not write and connection is not None:
                    connection.execute("ROLLBACK")
                raise
            finally:
                self._session_connection = None
                self._session_writable = False

    def batch_write(self):
        """Reuse an offline migration connection, committing each snapshot alone."""
        return self._session(write=True)

    def read_session(self):
        """Reuse a read connection for bulk ingestion; never hold it over HTTP."""
        return self._session(write=False)

    @staticmethod
    def _check_schema(connection) -> None:
        """Refuse incompatible stores instead of silently reinterpreting them."""
        try:
            row = connection.execute("SELECT value FROM api_cache_metadata WHERE name = 'schema_version'").fetchone()
        except Exception:
            raise _error("Unrecognized DuckDB API cache schema; use a new database and migrate the verified source cache") from None
        if row != (str(SCHEMA_VERSION),):
            raise _error("Unsupported DuckDB API cache schema version; migrate with a compatible application")

    def _ensure_schema(self, connection) -> None:
        """Create the manifest, immutable payload, receipt and native row tables."""
        connection.execute("CREATE TABLE IF NOT EXISTS api_cache_metadata (name VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
        connection.execute("INSERT INTO api_cache_metadata VALUES ('schema_version', ?) ON CONFLICT DO NOTHING", [str(SCHEMA_VERSION)])
        self._check_schema(connection)
        connection.execute("""CREATE TABLE IF NOT EXISTS api_snapshots (
            snapshot_id VARCHAR PRIMARY KEY, identity_hash VARCHAR NOT NULL,
            identity_json VARCHAR NOT NULL, provider VARCHAR NOT NULL,
            dataset VARCHAR NOT NULL, zone VARCHAR NOT NULL, cached_at TIMESTAMPTZ NOT NULL,
            manifest_json VARCHAR NOT NULL, raw_payload BLOB NOT NULL,
            normalized_parquet BLOB NOT NULL, raw_sha256 VARCHAR NOT NULL,
            normalized_sha256 VARCHAR NOT NULL, row_count BIGINT NOT NULL)""")
        connection.execute("CREATE TABLE IF NOT EXISTS api_current_requests (identity_hash VARCHAR PRIMARY KEY, snapshot_id VARCHAR NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS api_no_data_receipts (identity_hash VARCHAR PRIMARY KEY, identity_json VARCHAR NOT NULL, receipt_json VARCHAR NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS api_unparsed_responses (identity_hash VARCHAR PRIMARY KEY, identity_json VARCHAR NOT NULL, receipt_json VARCHAR NOT NULL, payload BLOB NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS api_records (_snapshot_id VARCHAR NOT NULL, _row_id BIGINT NOT NULL)")
        self._refresh_view(connection)

    @staticmethod
    def _refresh_view(connection) -> None:
        """Expose current revisions once, while leaving historical rows queryable."""
        connection.execute("""CREATE OR REPLACE VIEW current_api_records AS
            SELECT records.* FROM api_records records
            JOIN api_current_requests current ON records._snapshot_id = current.snapshot_id""")

    def _decorate_manifest(self, identity, manifest, snapshot_id: str) -> dict:
        """Separate source paths from the database's current provenance locator."""
        result = dict(manifest)
        if "directory" in result:
            result["source_directory"] = result["directory"]
        result["directory"] = str(self.location(identity))
        result["storage_backend"] = "duckdb"
        result["database_path"] = str(self.directory)
        result["snapshot_id"] = snapshot_id
        result["storage"] = {"backend": "duckdb", "database": str(self.directory),
                             "snapshot_id": snapshot_id, "schema_version": SCHEMA_VERSION}
        return result

    def _entry(self, identity_json, manifest_json, snapshot_id) -> dict:
        """Describe a current or historical snapshot without reading its BLOBs."""
        identity = json.loads(identity_json)
        manifest = self._decorate_manifest(identity, json.loads(manifest_json), snapshot_id)
        return {"identity": identity, "manifest": manifest,
                "source": str(self.location(identity) / "manifest.json")}

    def entries(self, *, provider=None, dataset=None, zone=None) -> list[dict]:
        """List current snapshots with exact request identity and source metadata."""
        filters, parameters = [], []
        for name, value in (("provider", provider), ("dataset", dataset), ("zone", zone)):
            if value is not None:
                filters.append(f"snapshot.{name} = ?")
                parameters.append(value)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._connect() as connection:
            if connection is None:
                return []
            rows = connection.execute("""SELECT snapshot.identity_json, snapshot.manifest_json,
                snapshot.snapshot_id FROM api_snapshots snapshot
                JOIN api_current_requests current USING (snapshot_id)""" + where +
                " ORDER BY snapshot.provider, snapshot.dataset, snapshot.zone, snapshot.identity_hash", parameters).fetchall()
        return [self._entry(*row) for row in rows]

    def snapshots(self, identity: Mapping[str, Any]) -> list[dict]:
        """List all retained revisions, preserving the source's original manifest."""
        digest = _identity_hash(identity)
        with self._connect() as connection:
            if connection is None:
                return []
            rows = connection.execute("""SELECT identity_json, manifest_json, snapshot_id
                FROM api_snapshots WHERE identity_hash = ? ORDER BY cached_at, snapshot_id""", [digest]).fetchall()
        return [self._entry(*row) for row in rows]

    def manifest(self, identity: Mapping[str, Any]) -> dict | None:
        """Read current provenance cheaply for offline coverage reports."""
        digest = _identity_hash(identity)
        with self._connect() as connection:
            if connection is None:
                return None
            row = connection.execute("""SELECT snapshot.manifest_json, snapshot.snapshot_id
                FROM api_snapshots snapshot JOIN api_current_requests current USING (snapshot_id)
                WHERE current.identity_hash = ?""", [digest]).fetchone()
        if row is None:
            return None
        manifest = json.loads(row[0])
        if manifest.get("request") != dict(identity):
            raise _error("DuckDB API cache identity mismatch")
        return self._decorate_manifest(identity, manifest, row[1])

    def _snapshot_row(self, identity, *, manifest=None, snapshot_id=None):
        """Select exact immutable bytes with one short consistent read operation."""
        digest = _identity_hash(identity)
        parameters = [digest]
        if snapshot_id is not None:
            selector = "snapshot.snapshot_id = ?"
            parameters.append(snapshot_id)
        elif manifest is not None:
            selector = "snapshot.cached_at = ? AND snapshot.raw_sha256 = ? AND snapshot.normalized_sha256 = ?"
            parameters.extend([_stamp(manifest["cached_at"], "cached_at").to_pydatetime(),
                               manifest["raw_sha256"], manifest["normalized_sha256"]])
        else:
            selector = "snapshot.snapshot_id = (SELECT snapshot_id FROM api_current_requests WHERE identity_hash = ?)"
            parameters.append(digest)
        with self._connect() as connection:
            if connection is None:
                return None
            return connection.execute("""SELECT snapshot.snapshot_id, snapshot.manifest_json,
                snapshot.raw_payload, snapshot.normalized_parquet FROM api_snapshots snapshot
                WHERE snapshot.identity_hash = ? AND """ + selector, parameters).fetchone()

    def load_snapshot(self, identity: Mapping[str, Any], manifest=None, *, snapshot_id=None,
                      verify_checksums: bool = True) -> tuple[pd.DataFrame, dict] | None:
        """Verify an immutable revision and restore its exact Parquet dtypes."""
        row = self._snapshot_row(identity, manifest=manifest, snapshot_id=snapshot_id)
        if row is None:
            return None
        selected_id, manifest_json, raw, normalized = row
        stored = json.loads(manifest_json)
        if stored.get("request") != dict(identity):
            raise _error("DuckDB API cache identity mismatch")
        try:
            if verify_checksums:
                if hashlib.sha256(raw).hexdigest() != stored["raw_sha256"]:
                    raise _error("Cached API payload checksum mismatch in DuckDB")
                if hashlib.sha256(normalized).hexdigest() != stored["normalized_sha256"]:
                    raise _error("Cached normalized records checksum mismatch in DuckDB")
            frame = pd.read_parquet(BytesIO(normalized))
            if len(frame) != stored["records"]:
                raise _error("Cached normalized record count mismatch in DuckDB")
        except (KeyError, OSError, ValueError) as error:
            from .api import DataAPIError
            if isinstance(error, DataAPIError):
                raise
            raise _error("Incomplete/corrupt API snapshot in DuckDB; refresh this request explicitly") from None
        return frame, self._decorate_manifest(identity, stored, selected_id)

    def load(self, identity: Mapping[str, Any], *, verify_checksums: bool = True):
        """Load the current published snapshot without consulting credentials."""
        return self.load_snapshot(identity, verify_checksums=verify_checksums)

    def load_raw(self, identity: Mapping[str, Any], *, snapshot_id=None) -> bytes | None:
        """Read a verified provider payload for offline parser correctness checks."""
        row = self._snapshot_row(identity, snapshot_id=snapshot_id)
        if row is None:
            return None
        manifest = json.loads(row[1])
        if manifest.get("request") != dict(identity) or hashlib.sha256(row[2]).hexdigest() != manifest.get("raw_sha256"):
            raise _error("Cached API payload checksum mismatch in DuckDB")
        return bytes(row[2])

    def verify_native_snapshot(self, identity: Mapping[str, Any], manifest=None, *,
                               snapshot_id=None) -> bool:
        """Check native SQL rows against Parquet after every schema promotion.

        Row ordinals preserve multiplicity and ordering. DuckDB's common SQL
        types handle intentional schema widening; fields absent from the source
        must remain null for this revision. Authoritative bytes are checked too.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = self._snapshot_row(identity, manifest=manifest, snapshot_id=snapshot_id)
        if row is None:
            raise _error("Missing API snapshot during native DuckDB verification")
        selected_id, manifest_json, raw, normalized = row
        source = json.loads(manifest_json)
        if (source.get("request") != dict(identity)
                or hashlib.sha256(raw).hexdigest() != source.get("raw_sha256")
                or hashlib.sha256(normalized).hexdigest() != source.get("normalized_sha256")):
            raise _error("API snapshot checksum/identity mismatch during native DuckDB verification")
        table = pq.read_table(pa.BufferReader(normalized))
        table = table.append_column("_row_id", pa.array(range(len(table)), type=pa.int64()))
        columns = ", ".join(_quote(name) for name in table.column_names)
        with self._connect() as connection:
            connection.register("_parity_expected", table)
            try:
                actual = f"SELECT {columns} FROM api_records WHERE _snapshot_id = ?"
                expected = f"SELECT {columns} FROM _parity_expected"
                different = connection.execute(
                    f"SELECT EXISTS (({actual} EXCEPT ALL {expected}) UNION ALL ({expected} EXCEPT ALL {actual}))",
                    [selected_id, selected_id]).fetchone()[0]
                native_names = [column[0] for column in connection.execute("DESCRIBE api_records").fetchall()]
                extras = [name for name in native_names if name not in table.column_names and name != "_snapshot_id"]
                if extras:
                    nonnull = " OR ".join(f"{_quote(name)} IS NOT NULL" for name in extras)
                    different |= connection.execute(f"SELECT EXISTS (SELECT 1 FROM api_records WHERE _snapshot_id = ? AND ({nonnull}))", [selected_id]).fetchone()[0]
            finally:
                connection.unregister("_parity_expected")
        if different:
            raise _error(f"Native DuckDB record parity mismatch for snapshot {selected_id}; rebuild the analysis database from verified source caches")
        return True

    def save(self, identity: Mapping[str, Any], payload: bytes, frame: pd.DataFrame,
             metadata: Mapping[str, Any]) -> dict:
        """Publish raw response, typed values and retrieval provenance together."""
        normalized = frame.to_parquet(index=False)
        manual_source = metadata.get("source_kind") == "manual_gme_xlsx"
        raw_file = metadata.get("source_file") if manual_source else None
        if raw_file is None:
            raw_file = "response.xml" if identity["provider"] == "entsoe" else "response.json"
        manifest = {"schema_version": "api-cache-v1", "request": dict(identity),
                    "raw_file": raw_file,
                    "raw_sha256": hashlib.sha256(payload).hexdigest(),
                    "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                    "records": len(frame), "cached_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": dict(metadata),
                    "historical_availability": "source provenance only; download/cache time does not establish historical publication"}
        self.save_snapshot(identity, payload, normalized, manifest)
        return self._decorate_manifest(identity, manifest, self._snapshot_id(identity, manifest))

    @staticmethod
    def _snapshot_id(identity, manifest) -> str:
        """Make migration repeatable without collapsing distinct retrieval clocks."""
        key = [_identity_hash(identity), manifest["cached_at"], manifest["raw_sha256"], manifest["normalized_sha256"]]
        return hashlib.sha256(_json_bytes(key)).hexdigest()

    def save_snapshot(self, identity: Mapping[str, Any], payload: bytes, normalized_bytes: bytes,
                      manifest: Mapping[str, Any], *, make_current: bool = True) -> bool:
        """Import exact source bytes idempotently; one transaction publishes them."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        digest = _identity_hash(identity)
        source = dict(manifest)
        manual_source = source.get("metadata", {}).get("source_kind") == "manual_gme_xlsx"
        expected_raw = source.get("metadata", {}).get("source_file") if manual_source else (
            "response.xml" if identity["provider"] == "entsoe" else "response.json")
        if manual_source and (identity.get("provider") != "gme"
                              or not isinstance(expected_raw, str)
                              or Path(expected_raw).name != expected_raw
                              or Path(expected_raw).suffix.lower() != ".xlsx"):
            raise _error("Manual GME snapshots require a safe XLSX source filename")
        if (source.get("request") != dict(identity) or source.get("schema_version") != "api-cache-v1"
                or source.get("raw_file") != expected_raw):
            raise _error("Invalid source API snapshot manifest/identity")
        if hashlib.sha256(payload).hexdigest() != source.get("raw_sha256"):
            raise _error("Cached API payload checksum mismatch during DuckDB import")
        if hashlib.sha256(normalized_bytes).hexdigest() != source.get("normalized_sha256"):
            raise _error("Cached normalized records checksum mismatch during DuckDB import")
        cached_at = _stamp(source.get("cached_at"), "cached_at")
        try:
            table = pq.read_table(pa.BufferReader(normalized_bytes))
        except (ValueError, OSError, pa.ArrowException):
            raise _error("Invalid normalized Parquet snapshot during DuckDB import") from None
        if len(table) != source.get("records"):
            raise _error("Cached normalized record count mismatch during DuckDB import")
        names = table.column_names
        if (len({name.lower() for name in names}) != len(names)
                or any(name.lower() in {"_snapshot_id", "_row_id"} for name in names)):
            raise _error("Normalized API fields conflict with reserved DuckDB record columns")
        snapshot_id = self._snapshot_id(identity, source)
        with self._connect(write=True) as connection:
            exists = connection.execute("SELECT 1 FROM api_snapshots WHERE snapshot_id = ?", [snapshot_id]).fetchone() is not None
            if not exists:
                connection.execute("INSERT INTO api_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [snapshot_id, digest, _json_bytes(dict(identity)).decode(), identity["provider"],
                     identity["dataset"], identity["zone"], cached_at.to_pydatetime(),
                     _json_bytes(source).decode(), payload, normalized_bytes, source["raw_sha256"],
                     source["normalized_sha256"], len(table)])
                self._insert_records(connection, table, snapshot_id)
            if make_current:
                connection.execute("""INSERT INTO api_current_requests VALUES (?, ?)
                    ON CONFLICT (identity_hash) DO UPDATE SET snapshot_id = excluded.snapshot_id""", [digest, snapshot_id])
                connection.execute("DELETE FROM api_no_data_receipts WHERE identity_hash = ?", [digest])
            self._refresh_view(connection)
        return not exists

    def _insert_records(self, connection, table, snapshot_id: str) -> None:
        """Union native schemas and promote initially null fields when data arrives."""
        import pyarrow as pa
        table = table.append_column("_snapshot_id", pa.array([snapshot_id] * len(table), type=pa.string()))
        table = table.append_column("_row_id", pa.array(range(len(table)), type=pa.int64()))
        connection.register("_incoming_records", table)
        existing = {row[0]: row[1] for row in connection.execute("DESCRIBE api_records").fetchall()}
        incoming = {row[0]: row[1] for row in connection.execute("DESCRIBE SELECT * FROM _incoming_records").fetchall()}
        connection.execute("DROP VIEW IF EXISTS current_api_records")
        for name, incoming_type in incoming.items():
            quoted = _quote(name)
            if name not in existing:
                connection.execute(f"ALTER TABLE api_records ADD COLUMN {quoted} {incoming_type}")
            elif incoming_type != existing[name] and table[name].null_count < len(table):
                has_values = connection.execute(f"SELECT EXISTS (SELECT 1 FROM api_records WHERE {quoted} IS NOT NULL)").fetchone()[0]
                desired = incoming_type
                if has_values:
                    desired = connection.execute(f"DESCRIBE SELECT {quoted} FROM api_records UNION ALL SELECT {quoted} FROM _incoming_records").fetchone()[1]
                if desired != existing[name]:
                    connection.execute(f"ALTER TABLE api_records ALTER COLUMN {quoted} TYPE {desired}")
        connection.execute("INSERT INTO api_records BY NAME SELECT * FROM _incoming_records")
        connection.unregister("_incoming_records")

    def save_no_data(self, identity: Mapping[str, Any]) -> dict:
        """Record temporary provider absence without erasing a valid revision."""
        receipt = {"request": dict(identity), "status": "no-data", "records": 0,
                   "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                   "directory": str(self.location(identity)),
                   "interpretation": "no rows for the requested series after provider parsing/filtering; not zero values or inferred historical coverage"}
        self.save_no_data_receipt(identity, receipt)
        return receipt

    def save_no_data_receipt(self, identity: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        """Import the original no-data clock so migration cannot extend its TTL."""
        digest = _identity_hash(identity)
        if receipt.get("request") != dict(identity) or receipt.get("status") != "no-data":
            raise _error("No-data receipt identity/status mismatch")
        _stamp(receipt.get("retrieved_at"), "retrieved_at")
        with self._connect(write=True) as connection:
            connection.execute("""INSERT INTO api_no_data_receipts VALUES (?, ?, ?)
                ON CONFLICT (identity_hash) DO UPDATE SET identity_json = excluded.identity_json,
                receipt_json = excluded.receipt_json""", [digest, _json_bytes(dict(identity)).decode(), _json_bytes(dict(receipt)).decode()])

    def no_data_entries(self) -> list[dict]:
        """List original absence receipts, including expired ones, for audits."""
        with self._connect() as connection:
            if connection is None:
                return []
            rows = connection.execute("SELECT identity_json, receipt_json FROM api_no_data_receipts ORDER BY identity_hash").fetchall()
        return [{"identity": json.loads(row[0]), "receipt": json.loads(row[1])} for row in rows]

    def no_data_receipt(self, identity: Mapping[str, Any]) -> dict | None:
        """Read an original absence receipt, including an expired migration source."""
        digest = _identity_hash(identity)
        with self._connect() as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT receipt_json FROM api_no_data_receipts WHERE identity_hash = ?", [digest]).fetchone()
        if row is None:
            return None
        receipt = json.loads(row[0])
        if receipt.get("request") != dict(identity):
            raise _error("No-data receipt identity mismatch")
        return receipt

    def load_no_data(self, identity: Mapping[str, Any], max_age_hours: float) -> dict | None:
        """Bound absence lifetime exactly as the existing file cache does."""
        digest = _identity_hash(identity)
        if max_age_hours <= 0:
            return None
        with self._connect() as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT receipt_json FROM api_no_data_receipts WHERE identity_hash = ?", [digest]).fetchone()
        if row is None:
            return None
        receipt = json.loads(row[0])
        if receipt.get("request") != dict(identity):
            raise _error("No-data receipt identity mismatch")
        age = (pd.Timestamp.now(tz="UTC") - _stamp(receipt["retrieved_at"], "retrieved_at")).total_seconds() / 3600
        if not 0 <= age < max_age_hours:
            return None
        receipt["directory"] = str(self.location(identity))
        return receipt

    def save_unparsed(self, identity: Mapping[str, Any], payload: bytes, retrieved_at,
                      sensitive_values=()) -> Path:
        """Retain reparsable responses only after detecting echoed credentials."""
        secrets = [str(value) for value in sensitive_values if value]

        def contains_secret(value):
            """Inspect decoded JSON too, including escaped credential characters."""
            if isinstance(value, str):
                return any(secret in value for secret in secrets)
            if isinstance(value, dict):
                return any(contains_secret(key) or contains_secret(child) for key, child in value.items())
            if isinstance(value, list):
                return any(contains_secret(child) for child in value)
            return False

        try:
            decoded_contains_secret = contains_secret(json.loads(payload))
        except (ValueError, UnicodeError):
            decoded_contains_secret = False
        if decoded_contains_secret or any(value.encode() in payload for value in secrets):
            raise _error("Terna returned authentication material in its data response; the response was not retained")
        receipt = {"request": dict(identity), "status": "unparsed-data", "retrieved_at": str(retrieved_at),
                   "raw_sha256": hashlib.sha256(payload).hexdigest(),
                   "interpretation": "Retained provider data awaiting a schema correction; not a successful normalized cache or a no-data receipt"}
        self.save_unparsed_receipt(identity, payload, receipt)
        return self.location(identity) / "unparsed_response.json"

    def save_unparsed_receipt(self, identity: Mapping[str, Any], payload: bytes,
                              receipt: Mapping[str, Any]) -> None:
        """Migrate retained responses without changing their retrieval clocks."""
        digest = _identity_hash(identity)
        if (receipt.get("request") != dict(identity) or receipt.get("status") != "unparsed-data"
                or hashlib.sha256(payload).hexdigest() != receipt.get("raw_sha256")):
            raise _error("Invalid retained API response during DuckDB import")
        _stamp(receipt.get("retrieved_at"), "retrieved_at")
        with self._connect(write=True) as connection:
            connection.execute("""INSERT INTO api_unparsed_responses VALUES (?, ?, ?, ?)
                ON CONFLICT (identity_hash) DO UPDATE SET identity_json = excluded.identity_json,
                receipt_json = excluded.receipt_json, payload = excluded.payload""",
                [digest, _json_bytes(dict(identity)).decode(), _json_bytes(dict(receipt)).decode(), payload])

    def load_unparsed(self, identity: Mapping[str, Any]) -> tuple[bytes, dict] | None:
        """Verify retained responses before retrying a corrected parser offline."""
        digest = _identity_hash(identity)
        with self._connect() as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT payload, receipt_json FROM api_unparsed_responses WHERE identity_hash = ?", [digest]).fetchone()
        if row is None:
            return None
        payload, receipt_json = row
        receipt = json.loads(receipt_json)
        if receipt.get("request") != dict(identity) or hashlib.sha256(payload).hexdigest() != receipt.get("raw_sha256"):
            raise _error("Invalid retained API response in DuckDB; refresh this request explicitly")
        return bytes(payload), receipt
