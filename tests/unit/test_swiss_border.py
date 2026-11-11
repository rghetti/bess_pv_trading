"""Swissgrid ingestion preserves source semantics, missingness and DST folds."""

from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.data.swiss_border import discover_swiss_csv, download_swiss_border, parse_swiss_border
from bess_pv_trading.data.country_balances import _rte_coverage, download_french, parse_french_export


def _payload():
    """Construct both occurrences of the European autumn repeated hour."""
    starts = pd.date_range("2026-10-25T00:00Z", periods=8, freq="15min")
    labels = starts.tz_convert("Europe/Zurich").strftime("%d.%m.%Y %H:%M")
    frame = pd.DataFrame({"Date Time": labels, "H:Capacity [MW]": [1, 2, 3, 4, 5, 6, 7, np.nan],
                          "R:Spread [EUR/MWh]": range(8)})
    return frame.to_csv(index=False, sep=";").encode()


def test_swiss_border_preserves_autumn_folds_missing_values_and_units():
    columns = {"ntc_ch_to_it": "H:Capacity [MW]", "spread_it_nord_ch": "R:Spread [EUR/MWh]"}
    frame = parse_swiss_border(_payload(), year=2026, timezone="Europe/Zurich", columns=columns)
    capacity = frame[frame.variable == "ntc_ch_to_it"]
    assert len(capacity) == 8
    assert capacity.product_start.is_unique
    assert capacity.value.isna().sum() == 1
    assert capacity.unit.unique().tolist() == ["MW"]
    assert frame.loc[frame.variable == "spread_it_nord_ch", "unit"].unique().tolist() == ["EUR/MWh"]


def test_rte_parser_retains_nuclear_hydro_and_border_source_fields():
    """The downloaded RTE row remains available for later feature extraction."""
    record = {"date": "2026-01-15", "heure": "12:00", "date_heure": "2026-01-15T11:00:00Z",
              "consommation": 20, "pompage": -1, "nucleaire": 10, "hydraulique": 4,
              "ech_comm_italie": -3}
    settings = {"france": {"date_field": "date", "hour_field": "heure", "timestamp_field": "date_heure",
        "timezone": "Europe/Paris", "generation_fields": ["nucleaire", "hydraulique"],
        "optional_generation_fields": [], "optional_consumption_fields": [], "consumption_field": "consommation",
        "pumping_field": "pompage", "pumping_multiplier": 1.0}}
    frame = parse_french_export(json.dumps([record]).encode(), start=pd.Timestamp("2026-01-15T00:00Z"),
        end=pd.Timestamp("2026-01-16T00:00Z"), config=settings)
    assert json.loads(frame.generation_components_json.iloc[0])["nucleaire"] == 10
    assert json.loads(frame.generation_components_json.iloc[0])["hydraulique"] == 4
    assert json.loads(frame.source_record.iloc[0])["ech_comm_italie"] == -3


def test_empty_rte_month_is_incomplete_and_retried(tmp_path):
    """An empty provider response is evidence of a gap, not a completed month."""
    france = {"dataset": "electricity_balance", "endpoint": "https://odre.opendatasoft.com/api/export",
        "date_field": "date", "hour_field": "heure", "timestamp_field": "date_heure",
        "timezone": "Europe/Paris", "generation_fields": ["nucleaire", "hydraulique"],
        "optional_generation_fields": [], "optional_consumption_fields": [], "consumption_field": "consommation",
        "pumping_field": "pompage", "pumping_multiplier": 1.0}
    config = {"api": {"database_threads": 1}, "country_balances": {"database_path": str(tmp_path / "market.duckdb"),
        "start": "2026-09-01", "end_exclusive": "2026-10-01", "france": france}}

    class EmptyTransport:
        def __init__(self):
            self.calls = 0

        def get(self, request):
            self.calls += 1
            return b"[]"

    transport = EmptyTransport()
    first = download_french(config, transport=transport)
    second = download_french(config, transport=transport)
    assert first["status"] == second["status"] == "partial"
    assert first["remaining_requests"] == second["remaining_requests"] == 1
    assert first["months"][0]["records"] == 0
    assert transport.calls == 2


def test_rte_open_month_coverage_stops_at_last_completed_half_hour():
    """An unfinished month is not reported as missing its future intervals."""
    start = pd.Timestamp("2026-11-01", tz="Europe/Paris").tz_convert("UTC")
    end = pd.Timestamp("2026-12-01", tz="Europe/Paris").tz_convert("UTC")
    now = pd.Timestamp("2026-11-02T11:17Z")
    expected = pd.date_range(start, now.floor("30min"), freq="30min", inclusive="left")
    frame = pd.DataFrame({"product_start": expected, "value": 1.0})
    result = _rte_coverage(frame, start, end, now)
    assert result["window_open"] is True
    assert result["expected_full_window_intervals"] == 1440
    assert result["expected_elapsed_intervals"] == 72
    assert result["status"] == "complete_through_as_of"
    missing = _rte_coverage(frame.iloc[:-1], start, end, now)
    assert missing["missing_elapsed_intervals"] == 1
    assert missing["status"] == "missing_elapsed_intervals"


def test_rte_coverage_counts_autumn_dst_fold():
    """A 25-hour local day has 50 distinct elapsed half-hours."""
    start = pd.Timestamp("2026-10-25", tz="Europe/Paris").tz_convert("UTC")
    end = pd.Timestamp("2026-10-26", tz="Europe/Paris").tz_convert("UTC")
    grid = pd.date_range(start, end, freq="30min", inclusive="left")
    result = _rte_coverage(pd.DataFrame({"product_start": grid, "value": 1.0}), start, end, end)
    assert result["expected_full_window_intervals"] == 50
    assert result["status"] == "complete_window"


def test_discovery_rejects_unofficial_or_ambiguous_annual_files():
    page_url = "https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html"
    official = b'<a href="/dam/jcr:x/Grenzfluesse-2026.csv">file</a>'
    assert discover_swiss_csv(official, page_url, 2026).startswith("https://www.swissgrid.ch/")
    with pytest.raises(ValueError, match="found 0"):
        discover_swiss_csv(b'<a href="https://elsewhere.invalid/Grenzfluesse-2026.csv">x</a>', page_url, 2026)
    with pytest.raises(ValueError, match="found 2"):
        discover_swiss_csv(official + b'<a href="/dam/jcr:y/Grenzfluesse-2026.csv">y</a>', page_url, 2026)


def test_download_caches_and_verifies_raw_csv_snapshot(tmp_path):
    page_url = "https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html"
    columns = {"ntc_ch_to_it": "H:Capacity [MW]", "spread_it_nord_ch": "R:Spread [EUR/MWh]"}
    config = {"api": {"database_threads": 1}, "swiss_border": {"database_path": str(tmp_path / "market.duckdb"),
        "page_url": page_url, "year": 2026, "timezone": "Europe/Zurich", "columns": columns}}

    class Transport:
        def __init__(self):
            self.calls = []

        def get(self, request):
            self.calls.append(request.url)
            return (b'<a href="/dam/jcr:x/Grenzfluesse-2026.csv">file</a>'
                    if request.url == page_url else _payload())

    transport = Transport()
    first = download_swiss_border(config, transport=transport)
    second = download_swiss_border(config, transport=transport)
    assert first["status"] == "complete"
    assert first["rows"] == 16
    assert first["snapshot"]["raw_file"] == "response.csv"
    assert second["status"] == "cached"
    assert len(transport.calls) == 2
    refreshed = download_swiss_border(config, transport=transport, refresh=True)
    assert refreshed["status"] == "cached"
    assert refreshed["snapshot"]["snapshot_id"] == first["snapshot"]["snapshot_id"]
    assert len(transport.calls) == 4


def test_french_refresh_reuses_an_unchanged_nonempty_snapshot(tmp_path):
    """Repeated RTE refreshes do not write identical monthly data again."""
    france = {"dataset": "electricity_balance", "endpoint": "https://odre.opendatasoft.com/api/export",
        "date_field": "date", "hour_field": "heure", "timestamp_field": "date_heure",
        "timezone": "Europe/Paris", "generation_fields": ["nucleaire", "hydraulique"],
        "optional_generation_fields": [], "optional_consumption_fields": [], "consumption_field": "consommation",
        "pumping_field": "pompage", "pumping_multiplier": 1.0}
    config = {"api": {"database_threads": 1}, "country_balances": {"database_path": str(tmp_path / "market.duckdb"),
        "start": "2026-01-01", "end_exclusive": "2026-02-01", "france": france}}
    record = {"date": "2026-01-15", "heure": "12:00", "date_heure": "2026-01-15T11:00:00Z",
              "consommation": 20, "pompage": -1, "nucleaire": 10, "hydraulique": 4}

    class Transport:
        def get(self, request):
            return json.dumps([record]).encode()

    first = download_french(config, transport=Transport(), refresh=True)
    second = download_french(config, transport=Transport(), refresh=True)
    assert first["status"] == second["status"] == "complete"
    assert second["months"][0]["unchanged"] is True
    assert first["months"][0]["snapshot_id"] == second["months"][0]["snapshot_id"]


def test_french_refresh_preserves_existing_intervals_if_response_shrinks(tmp_path):
    """An empty or truncated refresh cannot replace an existing usable snapshot."""
    france = {"dataset": "electricity_balance", "endpoint": "https://odre.opendatasoft.com/api/export",
        "date_field": "date", "hour_field": "heure", "timestamp_field": "date_heure",
        "timezone": "Europe/Paris", "generation_fields": ["nucleaire", "hydraulique"],
        "optional_generation_fields": [], "optional_consumption_fields": [], "consumption_field": "consommation",
        "pumping_field": "pompage", "pumping_multiplier": 1.0}
    config = {"api": {"database_threads": 1}, "country_balances": {"database_path": str(tmp_path / "market.duckdb"),
        "start": "2026-01-01", "end_exclusive": "2026-02-01", "france": france}}
    record = {"date": "2026-01-15", "heure": "12:00", "date_heure": "2026-01-15T11:00:00Z",
              "consommation": 20, "pompage": -1, "nucleaire": 10, "hydraulique": 4}

    class Transport:
        def __init__(self, records):
            self.raw = json.dumps(records).encode()

        def get(self, request):
            return self.raw

    first = download_french(config, transport=Transport([record]), refresh=True)
    empty = download_french(config, transport=Transport([]), refresh=True)
    assert empty["status"] == "partial"
    assert empty["coverage_status"] == "incomplete"
    assert empty["months"][0]["preserved_previous"] is True
    assert empty["months"][0]["snapshot_id"] == first["months"][0]["snapshot_id"]
    repeat = download_french(config, transport=Transport([record]), refresh=True)
    assert repeat["months"][0]["unchanged"] is True
    assert repeat["months"][0]["snapshot_id"] == first["months"][0]["snapshot_id"]
    later = {**record, "heure": "12:30", "date_heure": "2026-01-15T11:30:00Z"}
    longer = download_french(config, transport=Transport([record, later]), refresh=True)
    truncated = download_french(config, transport=Transport([later]), refresh=True)
    assert longer["status"] == "complete"
    assert truncated["status"] == "partial"
    assert truncated["months"][0]["preserved_previous"] is True
    assert truncated["months"][0]["snapshot_id"] == longer["months"][0]["snapshot_id"]
