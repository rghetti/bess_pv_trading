"""Official-calendar clock semantics and strict manual provider boundaries."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.data.market_prices import (
    MarketCache, MarketCacheKey, MarketDataError, available_records, import_market_csv,
    import_market_from_config, import_market_xml, validate_market_records,
)
from bess_pv_trading.markets import AuctionCalendar


def record(price=-12.0, revision="1", published="2025-10-01T11:30:00Z"):
    return {"provider": "manual_gme", "dataset": "zonal_prices", "zone": "NORD", "auction": "MGP",
            "product_start": "2025-10-01T22:00:00Z", "product_end": "2025-10-01T22:15:00Z",
            "price_eur_mwh": price, "published_at": published,
            "publication_provenance": "recorded export timestamp", "revision": revision}


def test_calendar_dst_products_gates_and_results_are_separate():
    calendar = AuctionCalendar()
    assert len(calendar.delivery_grid("2026-03-29")) == 92
    assert len(calendar.delivery_grid("2025-10-26")) == 100
    ordinary = calendar.delivery_grid("2025-10-20")
    assert len(ordinary) == 96
    assert np.allclose(ordinary.delta_hours, .25)
    assert len(ordinary.to_domain().interval_ids) == 96
    summer = calendar.schedule("2025-10-20")
    winter = calendar.schedule("2025-10-28")
    assert summer[0].gate_at.hour == 10
    assert winter[0].gate_at.hour == 11
    assert summer[0].submission_at == summer[0].gate_at - pd.Timedelta(minutes=5)
    assert "study_assumption" in summer[0].result_provenance
    assert summer[0].results_at > summer[0].gate_at
    assert len(summer[3].eligible_intervals) == 48
    assert all(ordinary.starts[t].tz_convert("Europe/Rome").hour >= 12 for t in summer[3].eligible_intervals)


def test_historical_validity_missing_results_cancellation():
    calendar = AuctionCalendar()
    with pytest.raises(ValueError, match="outside configured auction-calendar validity"):
        calendar.delivery_grid("2025-09-30")
    events = calendar.schedule("2025-10-02", recorded_results={"DA": "2025-10-01T12:00:00Z", "IDA1": None}, cancelled=("IDA2",))
    assert events[0].result_provenance == "recorded_publication"
    assert not events[1].available and events[1].results_at is None
    assert not events[2].available
    boundary = calendar.schedule("2026-01-13")
    assert boundary[0].rule_version == "GME-DTF03-rev13"
    assert boundary[3].rule_version == "GME-DTF03-rev14"
    with pytest.raises(ValueError, match="explicit timezone"):
        calendar.schedule("2025-10-02", recorded_results={"DA": "2025-10-01T13:30:00"})


def test_import_retains_negative_prices_and_rejects_missing_duplicate_naive(tmp_path):
    path = tmp_path / "gme.csv"
    pd.DataFrame([record()]).to_csv(path, index=False)
    frame = import_market_csv(path)
    assert frame.auction.iloc[0] == "DA" and frame.price_eur_mwh.iloc[0] == -12
    for changes, message in [({"price_eur_mwh": np.nan}, "Missing/nonfinite"),
                             ({"product_start": "2025-10-01T22:00:00"}, "offset-aware")]:
        with pytest.raises(MarketDataError, match=message):
            validate_market_records(pd.DataFrame([{**record(), **changes}]))
    with pytest.raises(MarketDataError, match="Duplicate"):
        validate_market_records(pd.DataFrame([record(), record()]))


def test_revision_selection_cache_keys_and_retrieval_separate(tmp_path):
    records = validate_market_records(pd.DataFrame([record(), record(22, "2", "2025-10-02T12:00:00Z")]))
    assert available_records(records, "2025-10-01T14:00:00Z").price_eur_mwh.tolist() == [-12]
    assert available_records(records, "2025-10-03T14:00:00Z").price_eur_mwh.tolist() == [22]
    key = MarketCacheKey("gme", "zonal", "NORD", "DA", "2025-10-02", "2025-10-03", "2")
    assert key.digest() != replace(key, vintage_revision="3").digest()
    cache = MarketCache(tmp_path / "cache")
    assert cache.load(key) is None
    saved = cache.save(key, records)
    assert len(cache.load(key)) == 2
    metadata = json.loads(saved.with_suffix(".json").read_text())
    assert metadata["retrieved_at"] != records.published_at.iloc[0].isoformat()
    assert "never retrieved_at" in metadata["historical_availability"]


def test_explicit_xml_mapping_and_unsupported_provider(tmp_path):
    path = tmp_path / "manual.xml"
    item = record()
    path.write_text("<root><price>" + "".join(f"<{key}>{value}</{key}>" for key, value in item.items()) + "</price></root>")
    imported = import_market_xml(path, record_tag="price", field_mapping={key: key for key in item})
    assert imported.price_eur_mwh.iloc[0] == -12
    with pytest.raises(MarketDataError, match="download is not implemented"):
        import_market_from_config({"market_data": {"provider": "terna"}})


def test_import_reports_product_gaps_and_cache_hit_without_source(tmp_path):
    from bess_pv_trading.data.market_prices import import_market_from_config
    source = tmp_path / "source.csv"
    pd.DataFrame([record()]).to_csv(source, index=False)
    config = {"market_data": {"input_path": str(source), "output_path": str(tmp_path / "canonical.csv"),
        "cache_directory": str(tmp_path / "cache"), "requested_auctions": ["DA"],
        "cache_key": {"provider": "manual_gme", "dataset": "zonal_prices", "zone": "NORD", "product": "DA",
            "start": "2025-10-02", "end": "2025-10-03", "vintage_revision": "1"}}}
    imported = import_market_from_config(config)
    assert imported["status"] == "imported" and imported["missing_products"] == 95
    source.unlink()
    cached = import_market_from_config(config)
    assert cached["status"] == "cached" and cached["missing_products"] == 95
    assert (tmp_path / "canonical.coverage.json").exists()
