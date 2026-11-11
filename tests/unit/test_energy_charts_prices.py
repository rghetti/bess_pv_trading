"""Validate Energy Charts normalization without internet or external services."""

import json

import pandas as pd
import pytest

from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.energy_charts import parse_energy_charts_prices
from scripts.cache_energy_charts_prices import save_daily_prices


def test_quarter_hour_prices_preserve_fall_dst_and_missing_values():
    """Both repeated local hours remain distinct and null prices remain absent."""
    times = pd.DatetimeIndex(["2026-10-25T00:30Z", "2026-10-25T01:30Z", "2026-10-25T01:45Z"]).as_unit("ns")
    payload = json.dumps({"unit": "EUR / MWh", "unix_seconds": (times.asi8 // 10**9).tolist(),
                          "price": [12.0, -2.0, None], "license_info": "CC BY 4.0"})
    frame, metadata = parse_energy_charts_prices(payload, zone="NORD", start="2026-10-25",
                                                  end_exclusive="2026-10-26")
    assert frame.value.tolist() == [12.0, -2.0]
    assert frame.product_start.nunique() == 2
    assert frame.product_start.dt.tz_convert("Europe/Rome").dt.hour.tolist() == [2, 2]
    assert frame.published_at.isna().all()
    assert metadata["null_prices"] == 1


def test_rejects_duplicate_product_times():
    """Duplicate prices cannot silently create ambiguous market products."""
    payload = json.dumps({"unit": "EUR / MWh", "unix_seconds": [1792890000, 1792890000], "price": [1, 2]})
    with pytest.raises(ValueError, match="unique"):
        parse_energy_charts_prices(payload, zone="NORD", start="2026-10-24", end_exclusive="2026-10-26")


def test_daily_refresh_reuses_unchanged_days_and_preserves_partial_response(tmp_path):
    """A repeat does not create revisions; a price correction is still imported."""
    times = pd.DatetimeIndex(["2026-10-25T00:30Z", "2026-10-25T00:45Z", "2026-10-26T00:00Z"]).as_unit("ns")
    payload = json.dumps({"unit": "EUR / MWh", "unix_seconds": (times.asi8 // 10**9).tolist(),
                          "price": [12.0, 13.0, 20.0]}).encode()
    settings = {"timezone": "Europe/Rome", "zone": "NORD", "bidding_zone": "IT-North"}
    frame, details = parse_energy_charts_prices(payload, zone="NORD", start="2026-10-25", end_exclusive="2026-10-27")
    cache = DuckDBCache(tmp_path / "market.duckdb", threads=1)
    stamp = pd.Timestamp("2026-10-27T12:00Z")
    first = save_daily_prices(cache, frame, payload, settings, retrieved_at=stamp, url="https://example.test", details=details)
    assert first["updated_days"] == 2
    second = save_daily_prices(cache, frame, payload, settings, retrieved_at=stamp, url="https://example.test", details=details)
    assert second["updated_days"] == 0
    assert second["unchanged_days"] == 2
    assert [x["snapshot_id"] for x in first["snapshots"]] == [x["snapshot_id"] for x in second["snapshots"]]
    corrected = frame.copy()
    corrected.loc[0, "value"] = 14.0
    third = save_daily_prices(cache, corrected, payload, settings, retrieved_at=stamp, url="https://example.test", details=details)
    assert third["updated_days"] == 1
    assert third["unchanged_days"] == 1
    partial = corrected.drop(index=0).reset_index(drop=True)
    fourth = save_daily_prices(cache, partial, payload, settings, retrieved_at=stamp, url="https://example.test", details=details)
    assert fourth["status"] == "partial"
    assert fourth["preserved_previous_days"][0]["day"] == "2026-10-25"
    assert fourth["preserved_previous_days"][0]["snapshot_id"] == third["snapshots"][0]["snapshot_id"]
