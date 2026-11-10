"""Verify reservoir observations, local publication assumptions, and snapshot caching."""

import pandas as pd
import pytest

from bess_pv_trading.data.swiss_reservoir import download_reservoir_levels, parse_reservoir_levels


def _payload():
    """Build two consecutive local weeks that straddle the spring clock change."""
    return ("Datum,Wallis_speicherinhalt_gwh,Graubuenden_speicherinhalt_gwh,"
            "Tessin_speicherinhalt_gwh,UebrigCH_speicherinhalt_gwh,"
            "TotalCH_speicherinhalt_gwh,TotalCH_max_speicherinhalt_gwh\n"
            "2026-03-23,1,2,3,4,10,40\n"
            "2026-03-30,2,3,4,5,14,40\n").encode()


def test_local_week_dst_availability_and_percent():
    frame = parse_reservoir_levels(_payload(), timezone="Europe/Zurich", publication_delay_days=4)
    total = frame[frame.variable == "total_ch_gwh"].reset_index(drop=True)
    assert total.product_start.iloc[0] == pd.Timestamp("2026-03-22T23:00Z")
    assert total.product_end.iloc[0] == pd.Timestamp("2026-03-29T22:00Z")
    assert total.scheduled_available_at.iloc[1] == pd.Timestamp("2026-04-02T22:00Z")
    assert total.historical_available_at.isna().all()
    assert frame.loc[frame.variable == "total_ch_percent", "value"].tolist() == [25.0, 35.0]


def test_duplicate_reference_rejected():
    with pytest.raises(ValueError, match="unique increasing"):
        parse_reservoir_levels(_payload().replace(b"2026-03-30", b"2026-03-23"),
                               timezone="Europe/Zurich", publication_delay_days=4)


def test_cached_public_csv_snapshot(tmp_path):
    config = {"api": {"database_threads": 1}, "swiss_reservoir": {
        "database_path": str(tmp_path / "market.duckdb"),
        "url": "https://www.bfe-ogd.ch/ogd17/ogd17_fuellungsgrad_speicherseen.csv",
        "timezone": "Europe/Zurich", "publication_delay_days": 4}}

    class Transport:
        def __init__(self):
            self.calls = 0

        def get(self, request):
            self.calls += 1
            return _payload()

    transport = Transport()
    first = download_reservoir_levels(config, transport=transport)
    second = download_reservoir_levels(config, transport=transport)
    assert first["status"] == "complete"
    assert first["rows"] == 12
    assert first["snapshot"]["raw_file"] == "response.csv"
    assert second["status"] == "cached"
    assert transport.calls == 1
