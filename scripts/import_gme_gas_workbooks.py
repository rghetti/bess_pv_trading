"""Validate and import GME IG Index workbooks, retaining raw source bytes.

Stage into a separate DuckDB first. Merge checks every overlapping gas day,
stores only novel values, and optionally deletes verified source workbooks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.gme_workbooks import _price, _read_sheet


def parse_gas_workbook(path):
    """Read gas-day values without inventing electricity intervals or publication times."""
    path = Path(path)
    payload = path.read_bytes()
    rows = _read_sheet(payload, "xl/worksheets/sheet1.xml")
    if rows[0] != ["Giorno Gas", "Prezzo (€/MWh)"]:
        raise ValueError(f"Unexpected IG Index columns: {path.name}")
    source_sha = hashlib.sha256(payload).hexdigest()
    records = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise ValueError(f"Unexpected gas workbook row: {path.name}:{row_number}")
        day = datetime.strptime(row[0], "%d/%m/%Y").date()
        value, status = _price(row[1], ",", "", {"", "-", "—"})
        if status != "observed" or not np.isfinite(value):
            raise ValueError(f"Missing gas index value: {path.name}:{row_number}")
        records.append({"provider": "gme", "dataset": "gas_igi", "zone": "PSV", "value": value,
            "unit": "EUR/MWh", "record_status": status, "source_file": path.name,
            "source_row": row_number, "source_value_text": row[1], "gas_delivery_day": day.isoformat(),
            "gas_product": "unspecified", "historical_available_at": pd.NaT,
            "historical_feature_eligible": False, "availability_provenance": "unknown_manual_export",
            "reference_time": pd.Timestamp(day, tz="Europe/Rome").tz_convert("UTC"),
            "interval_provenance": "daily_gas_index; gas_delivery_day_retained_separately",
            "revision": "workbook_sha256:" + source_sha})
    frame = pd.DataFrame(records)
    if frame.empty or frame.gas_delivery_day.duplicated().any():
        raise ValueError(f"Empty or duplicate gas days: {path.name}")
    return payload, frame.sort_values("gas_delivery_day").reset_index(drop=True), source_sha


def stage(paths, database):
    """Archive validated sources in an isolated staging catalog."""
    cache = DuckDBCache(database, threads=2)
    if cache.directory.exists():
        raise FileExistsError("Use a fresh gas staging database")
    results = []
    seen = {}
    for path in paths:
        payload, frame, source_sha = parse_gas_workbook(path)
        for row in frame.itertuples():
            if row.gas_delivery_day in seen and not np.isclose(seen[row.gas_delivery_day], row.value, atol=1e-9):
                raise ValueError(f"Conflicting source gas values on {row.gas_delivery_day}")
            seen[row.gas_delivery_day] = row.value
        identity = {"provider": "gme", "dataset": "gas_igi", "zone": "PSV",
                    "source_file": path.name, "source_sha256": source_sha, "schema": "gas-day-xlsx-v1"}
        manifest = cache.save(identity, payload, frame, {"source_kind": "manual_gme_xlsx",
            "source_file": path.name, "source_path_configured": str(path.resolve()),
            "historical_availability": "unknown; gas day is not a publication clock"})
        cache.verify_native_snapshot(identity, snapshot_id=manifest["snapshot_id"])
        results.append({"file": str(path.resolve()), "sha256": source_sha, "rows": len(frame),
                        "first_day": frame.gas_delivery_day.min(), "last_day": frame.gas_delivery_day.max()})
    return results


def merge(stage_database, target_database, delete_verified):
    """Atomically merge nonconflicting gas days and verify before deleting files."""
    staging = DuckDBCache(stage_database, threads=2)
    sources = []
    with staging.read_session():
        for entry in staging.entries(provider="gme", dataset="gas_igi", zone="PSV"):
            identity = entry["identity"]
            frame, manifest = staging.load_snapshot(identity, snapshot_id=entry["manifest"]["snapshot_id"])
            staging.verify_native_snapshot(identity, snapshot_id=manifest["snapshot_id"])
            sources.append((identity, frame, manifest, staging.load_raw(identity)))
    if not sources:
        raise ValueError("No staged gas workbooks found")
    target = DuckDBCache(target_database, threads=2)
    if not target.directory.is_file():
        raise FileNotFoundError("Target market database does not exist")
    inserted = reused = 0
    # Commit all files together only after every overlap and native snapshot validates.
    with target.atomic_write():
        frames = [target.load_snapshot(e["identity"], snapshot_id=e["manifest"]["snapshot_id"])[0]
                  for e in target.entries(provider="gme", dataset="gas_igi", zone="PSV")]
        existing = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        values = {}
        if not existing.empty:
            for row in existing.itertuples():
                day = str(row.gas_delivery_day)[:10]
                if day in values and not np.isclose(values[day], row.value, atol=1e-9, rtol=0):
                    raise ValueError(f"Conflicting existing gas prices on {day}")
                values[day] = row.value
        for identity, frame, manifest, payload in sources:
            novel = []
            for index, row in frame.iterrows():
                day = row.gas_delivery_day
                if day in values:
                    if not np.isclose(values[day], row.value, atol=1e-9, rtol=0):
                        raise ValueError(f"Conflicting imported gas prices on {day}")
                    reused += 1
                else:
                    novel.append(index)
                    values[day] = row.value
            # An identical previously imported file keeps its original current snapshot.
            existing_manifest = target.manifest(identity)
            if existing_manifest is None:
                saved = target.save(identity, payload, frame.loc[novel].reset_index(drop=True),
                    {**manifest["metadata"], "staged_snapshot_id": manifest["snapshot_id"]})
                target.verify_native_snapshot(identity, snapshot_id=saved["snapshot_id"])
                inserted += len(novel)
            elif novel:
                raise ValueError("A known gas snapshot is missing its previously imported products")
    # Reopen after commit to verify the published data and exact archived source bytes.
    with target.read_session():
        for identity, frame, manifest, payload in sources:
            stored = target.load_raw(identity)
            if stored != payload:
                raise ValueError("Committed gas raw workbook does not match staged bytes")
            target.verify_native_snapshot(identity)
    removed = []
    if delete_verified:
        for _, _, manifest, payload in sources:
            path = Path(manifest["metadata"]["source_path_configured"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"Workbook changed after staging; refusing to delete {path}")
            path.unlink()
            removed.append(str(path))
    return {"status": "merged", "inserted_days": inserted, "identical_days_reused": reused,
            "sources_verified": len(sources), "removed_workbooks": removed}


def main():
    """Stage a source directory or merge a previously staged gas database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "merge"))
    parser.add_argument("--source-directory", default="data/api")
    parser.add_argument("--stage-database", required=True)
    parser.add_argument("--database", help="Existing shared market database for merge")
    parser.add_argument("--delete-verified", action="store_true")
    args = parser.parse_args()
    if args.action == "stage":
        paths = sorted(Path(args.source_directory).glob("IGIndex_Giornaliero_*.xlsx"))
        if not paths:
            raise FileNotFoundError("No matching GME IG Index workbooks")
        result = stage(paths, args.stage_database)
    else:
        if not args.database:
            parser.error("merge requires --database")
        result = merge(args.stage_database, args.database, args.delete_verified)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
