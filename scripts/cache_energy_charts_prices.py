"""Validate and cache a downloaded Energy Charts response in DuckDB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.energy_charts import parse_energy_charts_prices


def main():
    """Publish one immutable source snapshot, preserving the original response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/energy_charts_nord_fall2026.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--payload", help="Previously downloaded, unmodified API response")
    source.add_argument("--fetch", action="store_true", help="Retrieve the public API response")
    parser.add_argument("--database")
    parser.add_argument("--start", help="First local delivery date, YYYY-MM-DD")
    parser.add_argument("--end-exclusive", help="First local date not requested, YYYY-MM-DD")
    args = parser.parse_args()
    settings = load_config(args.config)["energy_charts_prices"]
    start = args.start or settings["start"]
    end_exclusive = args.end_exclusive or settings["end_exclusive"]
    final_day = (pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)).date().isoformat()
    url = settings["endpoint"] + "?" + urlencode({"bzn": settings["bidding_zone"],
        "start": start, "end": final_day})
    if args.fetch:
        # Finish the network request before opening the embedded database for writes.
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        retrieved_at = datetime.now(timezone.utc)
    else:
        path = Path(args.payload)
        payload = path.read_bytes()
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    frame, details = parse_energy_charts_prices(payload, zone=settings["zone"],
        start=start, end_exclusive=end_exclusive, timezone=settings["timezone"])
    cache = DuckDBCache(args.database or settings["database_path"], threads=settings["database_threads"])
    local_days = frame.product_start.dt.tz_convert(settings["timezone"]).dt.strftime("%Y-%m-%d")
    snapshots = []
    # Stable per-day identities prevent overlapping update requests from duplicating current records.
    for day, rows in frame.groupby(local_days, sort=True):
        identity = {"schema_version": "api-cache-v1", "provider": "energy_charts",
            "dataset": "day_ahead_prices", "zone": settings["zone"], "local_day": day,
            "bidding_zone": settings["bidding_zone"]}
        manifest = cache.save(identity, payload, rows.reset_index(drop=True),
            {"retrieved_at": retrieved_at.isoformat(), "source_kind": "energy_charts_public_api",
             "historical_availability": "unknown", "source_url": url, **details})
        snapshots.append({"day": day, "rows": len(rows), "snapshot_id": manifest.get("snapshot_id")})
    print(json.dumps({"rows": len(frame), "first": str(frame.product_start.min()),
        "last": str(frame.product_start.max()), "snapshots": snapshots,
        "raw_sha256": manifest["raw_sha256"], "database": str(cache.directory)}, indent=2))


if __name__ == "__main__":
    main()
