"""Regression checks for cache reuse and strict native auction products."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import api
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.simulation import gme_oracle
from test_terna_monthly_downloads import MonthlyTransport, configuration


@pytest.mark.parametrize("backend", ["duckdb", "files"])
def test_gme_zone_sibling_reuses_cached_raw_without_credentials(tmp_path, monkeypatch, backend):
    """Lazy raw reads must preserve cross-zone reuse of one GME response."""
    from bess_pv_trading.data import gme

    config = configuration(tmp_path, "2025-01-01", "2025-01-02", backend)
    config["api"]["requests"] = [{"provider": "gme", "dataset": "day_ahead_prices",
        "options": {"zone": zone}} for zone in ("NORD", "SUD")]
    plans = api.plan_downloads(config)
    assert plans[0]["specification"] == plans[1]["specification"]
    cache = api.cache_from_settings(config["api"])
    cache.save(plans[0]["identity"], b'{"fixture":"both zones"}', pd.DataFrame({"value": [10.0]}),
        {"retrieved_at": "2025-01-03T00:00:00+00:00"})

    def forbidden(*args, **kwargs):
        raise AssertionError("A cached multi-zone payload must not require credentials or HTTP")

    monkeypatch.setattr(api, "_secret", forbidden)
    monkeypatch.setattr(gme, "parse_gme_response", lambda *args, **kwargs: pd.DataFrame({"value": [20.0]}))
    transport = type("OfflineTransport", (), {"post_json": forbidden})()
    result = api.download_from_config(config, transport=transport)
    assert result["cached_chunks"] == result["downloaded_chunks"] == 1
    assert result["data_http_operations"] == result["authentication_operations"] == 0
    assert cache.load(plans[1]["identity"])[0].value.tolist() == [20.0]


def test_snapshot_cache_pins_revision_verification_and_evicts(tmp_path, monkeypatch):
    """A refresh and a stronger verification request must both miss the memo."""
    from bess_pv_trading.data.api_catalog import SnapshotReadCache, load_api_cache_records

    monkeypatch.setattr(api, "_secret", lambda *args, **kwargs: "fixture-private")
    config = configuration(tmp_path, "2025-01-01", "2025-01-02")
    config["api"]["requests"] = config["api"]["requests"][:1]
    api.download_from_config(config, transport=MonthlyTransport())
    cache = api.cache_from_settings(config["api"])
    identity = api.plan_downloads(config)[0]["identity"]
    original = DuckDBCache.load_snapshot
    reads = []

    def record_read(self, identity, **kwargs):
        reads.append(kwargs)
        return original(self, identity, **kwargs)

    monkeypatch.setattr(DuckDBCache, "load_snapshot", record_read)
    memo = SnapshotReadCache(max_entries=1)
    options = dict(provider="terna", dataset="total_load", zone="NORD", snapshot_cache=memo)
    first, provenance = load_api_cache_records(cache.directory, **options, verify_checksums=False)
    load_api_cache_records(cache.directory, **options)
    second, repeated = load_api_cache_records(cache.directory, **options)
    assert len(reads) == 2
    pd.testing.assert_frame_equal(first, second)
    assert repeated[0]["checksums_verified"] is True
    # Mutating the public result must never poison subsequent reads.
    second.loc[:, "value"] = -1
    again, _ = load_api_cache_records(cache.directory, **options)
    assert (again.value >= 0).all()
    manifest = cache.manifest(identity)
    changed = first.copy()
    changed.loc[:, "value"] += 7
    cache.save(identity, cache.load_raw(identity), changed, manifest["metadata"])
    latest, sources = load_api_cache_records(cache.directory, **options)
    assert len(reads) == 3
    assert sources[0]["storage"]["snapshot_id"] != provenance[0]["storage"]["snapshot_id"]
    np.testing.assert_array_equal(latest.value, first.value + 7)
    memo.load(cache, identity, snapshot_id=provenance[0]["storage"]["snapshot_id"], verify_checksums=True)
    assert len(reads) == 4


@pytest.mark.parametrize("backend", ["duckdb", "files"])
def test_cached_resume_never_reloads_raw_responses(tmp_path, monkeypatch, backend):
    """A fully cached two-field batch requires no raw-only rereads or network."""
    monkeypatch.setattr(api, "_secret", lambda *args, **kwargs: "fixture-private")
    config = configuration(tmp_path, "2025-01-01", "2025-01-03", backend)
    api.download_from_config(config, transport=MonthlyTransport())
    cache_type = DuckDBCache if backend == "duckdb" else api.DownloadCache

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached normalized results do not need a second raw payload read")

    monkeypatch.setattr(cache_type, "load_raw", forbidden)
    transport = MonthlyTransport()
    result = api.download_from_config(config, transport=transport)
    assert result["cached_chunks"] == 4
    assert transport.auth_calls == 0
    assert not transport.requests


def test_missing_sibling_reads_one_pinned_raw_snapshot(tmp_path, monkeypatch):
    """Recover missing days/fields without decoding every cached sibling frame."""
    monkeypatch.setattr(api, "_secret", lambda *args, **kwargs: "fixture-private")
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    requests = config["api"]["requests"]
    config["api"]["requests"] = requests[1:]
    api.download_from_config(config, transport=MonthlyTransport())
    config["api"]["requests"] = requests
    original = DuckDBCache.load_raw
    reads = []

    def record_read(self, identity, *, snapshot_id=None):
        assert snapshot_id is not None
        reads.append(snapshot_id)
        return original(self, identity, snapshot_id=snapshot_id)

    monkeypatch.setattr(DuckDBCache, "load_raw", record_read)
    transport = MonthlyTransport()
    result = api.download_from_config(config, transport=transport)
    assert result["downloaded_chunks"] == result["cached_chunks"] == 3
    assert len(reads) == 1
    assert not transport.requests


@pytest.mark.parametrize("day", ["2025-03-30", "2025-10-26", "2026-01-01"])
@pytest.mark.parametrize("corruption", [None, "end", "fractional", "missing"])
def test_native_auction_products_validate_duration_and_map_dst(monkeypatch, day, corruption):
    """Correct curves cover DST days; short ends and invalid labels cannot pass."""
    from bess_pv_trading.data.gme import native_price_resolution_minutes

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/full_experiments/gme_oracle_2025-01-01.yaml")
    study = config["gme_oracle_experiment"]
    study["delivery_day"] = day
    study["participating_auctions"] = ["DA", "IDA1", "IDA2", "IDA3"]
    config["market"].update(valid_from="2025-01-01", valid_to="2026-12-31")
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(day).to_domain()
    events = calendar.schedule(day)
    expected_curves = {}

    # Construct the real native product grids, including the shorter IDA3 window.
    for auction, item in study["auction_datasets"].items():
        item["native_interval_minutes"] = native_price_resolution_minutes(item["dataset"], date.fromisoformat(day))

    def read_prices(database, *, dataset, **kwargs):
        auction = next(name for name, item in study["auction_datasets"].items() if item["dataset"] == dataset)
        event = next(event for event in events if event.auction == auction)
        minutes = native_price_resolution_minutes(dataset, date.fromisoformat(day))
        starts = gme_oracle._expected_product_starts(grid, event.eligible_intervals, minutes, calendar.timezone)
        frame = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=minutes),
            "native_interval_minutes": float(minutes), "price_eur_mwh": np.arange(len(starts), dtype=float) - 20})
        # The slow reference checks containment independently of the vectorized lookup.
        expected_curves[auction] = np.array([next((float(row.price_eur_mwh) for row in frame.itertuples()
            if row.product_start <= stamp < row.product_end), 0.0) for stamp in grid.starts])
        if corruption == "end":
            frame.loc[0, "product_end"] -= pd.Timedelta(minutes=1)
        elif corruption == "fractional":
            frame.loc[0, "native_interval_minutes"] += 0.5
        elif corruption == "missing":
            frame.loc[0, "native_interval_minutes"] = np.nan
        return frame, []

    monkeypatch.setattr(gme_oracle, "load_api_cache_records", read_prices)
    if corruption:
        with pytest.raises(ValueError, match="duration|native resolution"):
            gme_oracle._load_auction_prices(config, calendar, grid, events)
    else:
        actual, _, _ = gme_oracle._load_auction_prices(config, calendar, grid, events)
        for auction, expected in expected_curves.items():
            np.testing.assert_array_equal(actual[auction], expected)
