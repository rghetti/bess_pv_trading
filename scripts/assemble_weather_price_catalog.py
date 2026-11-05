"""Copy a market catalog and append verified, nonoverlapping weather snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb

from bess_pv_trading.data.duckdb_store import DuckDBCache


def checksum(path: Path) -> str:
    """Identify the exact catalog file used to assemble an experiment copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(base: Path, weather: Path, destination: Path, report: Path) -> dict:
    """Preserve original snapshots while importing a later weather-only archive."""
    paths = [path.resolve() for path in (base, weather, destination, report)]
    if len(set(paths)) != 4 or destination.exists() or report.exists():
        raise ValueError("Input, new catalog and new report must be distinct; outputs must not exist")
    if not base.is_file() or not weather.is_file():
        raise ValueError("Both source catalogs must exist")
    # A live writer could leave either on-disk database incomplete for copying.
    if any(Path(str(path) + ".wal").exists() for path in (base, weather)):
        raise ValueError("Close catalog writers and checkpoint before assembly")
    with duckdb.connect(str(base), read_only=True) as connection:
        base_last = connection.execute("""SELECT max(cycle_time) FROM current_api_records
            WHERE provider='open_meteo' AND dataset='weather_forecast'""").fetchone()[0]
    with duckdb.connect(str(weather), read_only=True) as connection:
        first, last, count = connection.execute("""SELECT min(cycle_time), max(cycle_time), count(*)
            FROM current_api_records WHERE provider='open_meteo' AND dataset='weather_forecast'""").fetchone()
        rows = connection.execute("""SELECT s.snapshot_id, s.identity_json, s.manifest_json,
            s.raw_payload, s.normalized_parquet FROM api_snapshots s
            JOIN api_current_requests c USING(snapshot_id)
            WHERE s.provider='open_meteo' AND s.dataset='weather_forecast'
            ORDER BY s.snapshot_id""").fetchall()
        total = connection.execute("SELECT count(*) FROM api_current_requests").fetchone()[0]
    if not rows or len(rows) != total or not count or first is None or last is None:
        raise ValueError("Weather source must contain only nonempty current weather snapshots")
    if base_last is not None and first <= base_last:
        raise ValueError("Weather cycles overlap the original catalog")
    evidence = {"base_catalog": str(base.resolve()), "base_sha256": checksum(base),
                "weather_catalog": str(weather.resolve()), "weather_sha256": checksum(weather),
                "base_last_cycle": str(base_last), "weather_first_cycle": str(first),
                "weather_last_cycle": str(last), "weather_rows": count, "snapshot_ids": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing studies even if a prior run was interrupted.
    created = False
    try:
        with base.open("rb") as source, destination.open("xb") as target:
            created = True
            shutil.copyfileobj(source, target, length=1024 * 1024)
        target_cache = DuckDBCache(destination, threads=1)
        with target_cache.batch_write():
            for snapshot_id, identity_text, manifest_text, raw, normalized in rows:
                identity = json.loads(identity_text)
                manifest = json.loads(manifest_text)
                if target_cache.manifest(identity) is not None:
                    raise ValueError("Weather request identity already exists in the copied catalog")
                # save_snapshot checks hashes, row count, request identity and Parquet bytes.
                if target_cache._snapshot_id(identity, manifest) != snapshot_id:
                    raise ValueError("Weather snapshot identity does not match its manifest")
                target_cache.save_snapshot(identity, raw, normalized, manifest)
                evidence["snapshot_ids"].append(snapshot_id)
        evidence["destination_catalog"] = str(destination.resolve())
        evidence["destination_sha256"] = checksum(destination)
        with report.open("x") as stream:
            stream.write(json.dumps(evidence, indent=2) + "\n")
        return evidence
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
            Path(str(destination) + ".wal").unlink(missing_ok=True)
        raise


def main() -> None:
    """Expose source and output paths explicitly for reproducible study assembly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.base, args.weather, args.destination, args.report), indent=2))


if __name__ == "__main__":
    main()
