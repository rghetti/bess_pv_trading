"""Terna documented requests, parser provenance, and adversarial timestamp cases."""
import json

import pandas as pd
import pytest

from bess_pv_trading.data.market_prices import MarketDataError
from bess_pv_trading.data.terna import TOKEN_URL, parse_terna_json, parse_token, request_spec


# Fixtures use the field names and units in Terna's official public documentation.
def response(dataset="total_load", **changes):
    record = {"date": "2025-07-01 12:00:00", "date_tz": "Europe/Rome", "date_offset": "+02:00", "total_load_MW": "5033.597", "forecast_total_load_MW": "5048.25", "bidding_zone": "North"}
    key = "total_load"
    if dataset in {"load_forecast", "wind_forecast"}:
        key = "forecast_load"
        record = {"reference_date": "2025-07-01 12:00:00", "date_tz": "Europe/Rome", "zone": "Centro-Nord", "session_type": "MGP" if dataset == "load_forecast" else "MSD1", "load_MW": "2102"}
        if dataset == "wind_forecast":
            record.update(publication_date="2025-06-30", prevision_date="2025-06-30 09:00:00")
    if dataset == "renewable_generation":
        key = "renewable_generation"
        record = {"date": "2025-07-01 12:00:00", "date_tz": "Europe/Rome", "date_offset": "+02:00", "renewable_generation_GWh": "0.187", "energy_source": "Photovoltaic"}
    record.update(changes)
    return json.dumps({"result": {"message": "OK", "status": "Completed"}, key: [record]}).encode()


def parse(payload=None, dataset="total_load", options=None):
    return parse_terna_json(payload or response(dataset), dataset, options, "2026-01-01T12:00:00Z")


def test_observed_wind_schema_preserves_forecast_publication_and_original_field():
    """The live wind contract has its own array/value names, unlike the docs example."""
    document={"result":{"status":"Completed"},"wind_production_forecast":[{
        "publication_date":"2025-06-30T16:10:00","reference_date":"2025-07-01T00:00:00",
        "date_tz":"Europe/Rome","date_offset":"+02:00","zone":"North",
        "session_type":"MSD1","forecast":"19.99"}]}
    frame=parse(json.dumps(document).encode(),"wind_forecast",{"zone":"NORD","session_type":"MSD1"})
    assert frame.value.tolist()==[19.99] and frame.value_mw.tolist()==[19.99]
    assert frame.source_array.tolist()==["wind_production_forecast"]
    assert frame.source_field.tolist()==["forecast"]
    assert frame.historical_available_at.iloc[0]==pd.Timestamp("2025-06-30T14:10:00Z")
    assert frame.historical_feature_eligible.all()
    assert frame.product_start.isna().all()
    assert "no unit suffix" in frame.unit_provenance.iloc[0]
    document["forecast_load"]=[]
    with pytest.raises(MarketDataError,match="ambiguous"):
        parse(json.dumps(document).encode(),"wind_forecast")


def test_observed_wind_schema_does_not_guess_other_numeric_fields():
    document={"result":{"status":"Completed"},"wind_production_forecast":[{
        "reference_date":"2025-07-01T00:00:00","date_tz":"Europe/Rome","date_offset":"+02:00",
        "zone":"North","session_type":"MSD1","load_MW":"19.99"}]}
    with pytest.raises(MarketDataError,match="numeric forecast"):
        parse(json.dumps(document).encode(),"wind_forecast")


def test_documented_endpoints_dates_repeated_filters_and_no_auth_in_spec():
    spec = request_spec("total_load", "2025-07-01", "2025-07-02", {"bidding_zones": ["North", "Italy"]})
    assert spec.url == "https://api.terna.it/load/v2.0/total-load"
    assert spec.params == {"dateFrom": "01/07/2025", "dateTo": "02/07/2025", "biddingZone": ["North", "Italy"]}
    assert spec.headers == {"Accept": "application/json"}
    assert "Authorization" not in spec.headers
    assert request_spec("total_load_forecast", "2025-07-01", "2025-07-01", {}).url == spec.url
    assert request_spec("load_forecast", "2025-07-01", "2025-07-01", {"session_type": "MGP"}).params["sessionType"] == "MGP"
    renewable = request_spec("renewable_generation", "2025-07-01", "2025-07-01", {"energy_sources": ["Wind", "Photovoltaic"]})
    assert renewable.params["type"] == ["Wind", "Photovoltaic"]


@pytest.mark.parametrize("dataset,options,match", [("total_load", {"bidding_zones": ["NORD"]}, "English"), ("total_load", {"bidding_zones": "North"}, "list"), ("renewable_generation", {"energy_sources": ["Solar"]}, "source names"), ("wind_forecast", {"session_type": "MGP"}, "MSD1"), ("load_forecast", {}, "session_type"), ("total_load", {"page": 2}, "Unknown"), ("IDA1", {}, "No verified"), ("daily_prices", {}, "conflicts")])
def test_unsupported_or_silently_empty_queries_fail_before_network(dataset, options, match):
    with pytest.raises(MarketDataError, match=match):
        request_spec(dataset, "2025-07-01", "2025-07-01", options)


@pytest.mark.parametrize("start,end", [("01/07/2025", "2025-07-02"), ("2025-07-02", "2025-07-01"), ("2025-02-30", "2025-03-01"), (pd.Timestamp("2025-07-01T00:00Z"), "2025-07-01")])
def test_requests_use_explicit_ordered_local_dates(start, end):
    with pytest.raises(MarketDataError):
        request_spec("total_load", start, end)


def test_token_response_and_errors_never_echo_secret_payloads():
    assert TOKEN_URL == "https://api.terna.it/public-api/access-token"
    assert parse_token(b'{"access_token":"test-token","token_type":"bearer","expires_in":300}') == ("test-token", 300)
    for payload in [b"secret-invalid-json", b'{"access_token":"secret-value","token_type":"basic","expires_in":300}', b'{"access_token":"secret\\r\\nvalue","token_type":"bearer","expires_in":300}', b'{"access_token":"secret-value","token_type":"bearer","expires_in":true}']:
        with pytest.raises(MarketDataError) as error:
            parse_token(payload)
        assert "secret" not in str(error.value)


def test_actual_and_forecast_values_preserve_unknown_intervals_and_availability():
    actual = parse()
    forecast = parse(dataset="total_load_forecast")
    assert actual.value.iloc[0] == pytest.approx(5033.597)
    assert forecast.value.iloc[0] == pytest.approx(5048.25)
    assert actual.reference_time.iloc[0] == pd.Timestamp("2025-07-01T10:00Z")
    assert actual.zone.iloc[0] == "NORD"
    assert actual.product_start.isna().all() and actual.product_end.isna().all()
    assert actual.historical_available_at.isna().all()
    assert not actual.historical_feature_eligible.any()
    assert actual.retrieved_at.iloc[0] == pd.Timestamp("2026-01-01T12:00Z")
    assert actual.attrs["product_interval_semantics"] == "unresolved"


def test_declared_end_label_maps_interval_without_changing_publication_provenance():
    frame = parse(options={"interval_minutes": 15, "timestamp_label": "end"})
    assert frame.product_start.iloc[0] == pd.Timestamp("2025-07-01T09:45Z")
    assert frame.product_end.iloc[0] == pd.Timestamp("2025-07-01T10:00Z")
    assert frame.interval_provenance.iloc[0] == "configured_interval_assumption"
    assert frame.historical_available_at.isna().all()


@pytest.mark.parametrize("options", [{"interval_minutes": 15}, {"timestamp_label": "start"}, {"interval_minutes": True, "timestamp_label": "end"}, {"interval_minutes": 0, "timestamp_label": "start"}])
def test_interval_alignment_requires_complete_explicit_assumption(options):
    with pytest.raises(MarketDataError, match="interval alignment"):
        parse(options=options)


def test_wind_date_only_publication_does_not_use_prevision_or_retrieval_as_availability():
    frame = parse(dataset="wind_forecast", options={"session_type": "MSD1"})
    assert frame.source_document_time.iloc[0] == pd.Timestamp("2025-06-30T07:00Z")
    assert frame.source_publication_date.iloc[0] == "2025-06-30"
    assert frame.availability_provenance.iloc[0] == "unknown_date_only_publication"
    assert frame.historical_available_at.isna().all()
    assert not frame.historical_feature_eligible.any()


def test_clock_publication_is_distinct_from_document_time_and_retrieval():
    payload = response("wind_forecast", publication_date="2025-06-30 10:00:00")
    frame = parse(payload, "wind_forecast")
    assert frame.historical_available_at.iloc[0] == pd.Timestamp("2025-06-30T08:00Z")
    assert frame.source_document_time.iloc[0] == pd.Timestamp("2025-06-30T07:00Z")
    assert frame.historical_feature_eligible.all()


def test_publication_on_another_day_does_not_reuse_reference_dst_offset():
    payload = response("wind_forecast", reference_date="2025-03-30 12:00:00", offset="+02:00", publication_date="2025-03-29 12:00:00")
    frame = parse(payload, "wind_forecast")
    assert frame.historical_available_at.iloc[0] == pd.Timestamp("2025-03-29T11:00Z")


def test_fall_repeated_hour_is_distinguished_by_explicit_offset():
    first = parse(response(date="2025-10-26 02:00:00", date_offset="+02:00"))
    second = parse(response(date="2025-10-26 02:00:00", date_offset="+01:00"))
    assert second.reference_time.iloc[0] - first.reference_time.iloc[0] == pd.Timedelta(hours=1)


@pytest.mark.parametrize("changes", [{"date": "2025-03-30 02:00:00", "date_offset": "+01:00"}, {"date": "2025-10-26 02:00:00", "date_offset": None}, {"date_offset": "+01:00"}, {"date_tz": "Invalid/Timezone"}])
def test_ambiguous_nonexistent_and_inconsistent_timestamps_fail(changes):
    with pytest.raises(MarketDataError, match="timestamp"):
        parse(response(**changes))


def test_renewable_generation_remains_energy_and_national_actual_not_forecast():
    frame = parse(dataset="renewable_generation")
    assert frame.value.iloc[0] == pytest.approx(0.187)
    assert frame.value_gwh.iloc[0] == pytest.approx(0.187)
    assert frame.value_mw.isna().all()
    assert frame.unit.iloc[0] == "GWh" and frame.zone.iloc[0] == "IT"
    assert frame.energy_source.iloc[0] == "Photovoltaic"
    assert not frame.historical_feature_eligible.any()


def test_server_errors_missing_arrays_duplicates_and_undocumented_pagination_fail_safely():
    valid = json.loads(response())
    duplicate = json.loads(response())
    duplicate["total_load"] *= 2
    paginated = {**valid, "nextPage": "secret-continuation"}
    for data in [{"result": {"status": "ERROR", "message": "secret-body"}}, {"result": {"status": "Completed"}}, duplicate, paginated]:
        with pytest.raises(MarketDataError) as error:
            parse(json.dumps(data).encode())
        assert "secret" not in str(error.value)
    empty = parse(b'{"result":{"status":"Completed"},"total_load":[]}')
    assert empty.empty and "reference_time" in empty


@pytest.mark.parametrize("value", [None, "", "NaN", "Infinity", "1,234", True])
def test_missing_nonfinite_and_ambiguous_numbers_never_fill(value):
    with pytest.raises(MarketDataError, match="numeric"):
        parse(response(total_load_MW=value))


def test_zone_filter_and_request_session_validation():
    assert parse(options={"zone": "CNOR"}).empty
    with pytest.raises(MarketDataError, match="session_type differs"):
        parse(response("load_forecast"), "load_forecast", {"session_type": "MSD1"})
    with pytest.raises(MarketDataError, match="timezone-aware"):
        parse_terna_json(response(), "total_load", {}, "2026-01-01")


# Exercise authentication, source parsing, and cache reuse without real credentials.
def test_mock_transport_auth_download_then_cache_needs_no_credentials(tmp_path, monkeypatch):
    from bess_pv_trading.data.api import download_from_config

    class MockTransport:
        def __init__(self):
            self.auth_calls = 0
            self.data_calls = 0

        def post_form(self, url, fields):
            self.auth_calls += 1
            assert url == TOKEN_URL
            assert fields == {"client_id": "fake-client-id", "client_secret": "fake-client-secret", "grant_type": "client_credentials"}
            return b'{"access_token":"fake-access-token","token_type":"bearer","expires_in":300}'

        def get(self, specification, *, bearer_token):
            self.data_calls += 1
            assert specification.url == "https://api.terna.it/load/v2.0/total-load"
            assert specification.params["dateFrom"] == "01/07/2025"
            assert specification.params["dateTo"] == "01/07/2025"
            assert bearer_token == "fake-access-token"
            return response()

    monkeypatch.setenv("TEST_TERNA_ID", "fake-client-id")
    monkeypatch.setenv("TEST_TERNA_SECRET", "fake-client-secret")
    config = {"api": {"start": "2025-07-01", "end": "2025-07-02", "timezone": "Europe/Rome", "cache_directory": str(tmp_path / "cache"), "output_directory": str(tmp_path / "output"), "plots": False, "credentials": {"terna_client_id_env": "TEST_TERNA_ID", "terna_client_secret_env": "TEST_TERNA_SECRET"}, "requests": [{"provider": "terna", "dataset": "total_load_forecast", "options": {"zone": "NORD", "bidding_zones": ["North"]}}]}}
    transport = MockTransport()
    first = download_from_config(config, transport=transport)
    assert first["downloaded_chunks"] == 1
    assert first["items"][0]["records"] == 1
    assert first["items"][0]["historically_eligible_records"] == 0
    monkeypatch.delenv("TEST_TERNA_ID")
    monkeypatch.delenv("TEST_TERNA_SECRET")
    second = download_from_config(config, transport=transport)
    assert second["cached_chunks"] == 1
    assert transport.auth_calls == transport.data_calls == 1
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            contents = path.read_text()
            assert "fake-client-secret" not in contents
            assert "fake-access-token" not in contents
