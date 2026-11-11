"""Validate and cache a downloaded Energy Charts response in DuckDB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.energy_charts import parse_energy_charts_prices


def save_daily_prices(cache, frame, payload, settings, *, retrieved_at, url, details, progress=None):
    """Reuse unchanged days and atomically publish only new or revised prices."""
    local_days = frame.product_start.dt.tz_convert(settings["timezone"]).dt.strftime("%Y-%m-%d")
    planned = []
    snapshots = []
    preserved = []
    grouped = list(frame.groupby(local_days, sort=True))
    for index, (day, day_frame) in enumerate(grouped, start=1):
        if progress and (index == 1 or index % 5 == 0 or index == len(grouped)):
            progress(f"Energy Charts: checking {index}/{len(grouped)} delivery days")
        rows = day_frame.reset_index(drop=True)
        identity = {"schema_version": "api-cache-v1", "provider": "energy_charts",
            "dataset": "day_ahead_prices", "zone": settings["zone"], "local_day": day,
            "bidding_zone": settings["bidding_zone"]}
        previous = cache.load(identity)
        if previous is not None:
            old, manifest = previous
            # An upstream partial response must not delete earlier observed products.
            lost = pd.DatetimeIndex(old.product_start).difference(pd.DatetimeIndex(rows.product_start))
            if len(lost):
                preserved.append({"day": day, "previous_rows": len(old), "received_rows": len(rows),
                    "missing_previous_products": len(lost), "snapshot_id": manifest["snapshot_id"]})
                continue
            if set(rows.columns).issubset(old.columns):
                try:
                    pd.testing.assert_frame_equal(old[rows.columns].reset_index(drop=True), rows,
                                                  check_dtype=False, check_like=False)
                    snapshots.append({"day": day, "rows": len(old), "snapshot_id": manifest["snapshot_id"],
                                      "status": "unchanged"})
                    continue
                except AssertionError:
                    pass
        planned.append((day, rows, identity))
    if planned:
        if progress:
            progress(f"Energy Charts: writing {len(planned)} new or revised delivery days")
        with cache.batch_write():
            for day, rows, identity in planned:
                manifest = cache.save(identity, payload, rows, {"retrieved_at": retrieved_at.isoformat(),
                    "source_kind": "energy_charts_public_api", "historical_availability": "unknown",
                    "source_url": url, **details})
                snapshots.append({"day": day, "rows": len(rows), "snapshot_id": manifest["snapshot_id"],
                                  "status": "updated"})
    return {"status": "partial" if preserved else "complete", "snapshots": sorted(snapshots, key=lambda x: x["day"]),
        "unchanged_days": sum(item["status"] == "unchanged" for item in snapshots),
        "updated_days": len(planned), "preserved_previous_days": preserved}


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
        print(f"Energy Charts: fetching {start} through {final_day}", file=sys.stderr, flush=True)
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
    result = save_daily_prices(cache, frame, payload, settings, retrieved_at=retrieved_at,
        url=url, details=details, progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(json.dumps({"rows": len(frame), "first": str(frame.product_start.min()),
        "last": str(frame.product_start.max()), **result,
        "raw_sha256": hashlib.sha256(payload).hexdigest(), "database": str(cache.directory)}, indent=2))


if __name__ == "__main__":
    main()
