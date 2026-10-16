"""Downloader contract checks with synthetic credentials and offline transports."""
from copy import deepcopy
import json
from pathlib import Path
import traceback

import pandas as pd
import pytest
import requests

from bess_pv_trading.data import api
from bess_pv_trading.data.entsoe import ENTSOE_ENDPOINT, NORD_EIC


FAKE_TOKEN = "synthetic-unit-test-token"
TEST_TOKEN_ENV = "BESS_UNIT_ENTSOE_TOKEN"


class StubResponse:
    """Expose close semantics without a network or provider dependency."""
    def __init__(self, status=200, content=b"ok", headers=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def close(self):
        self.closed = True


class StubSession:
    """Return a prescribed response/exception sequence and capture private calls."""
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.outcomes:
            raise AssertionError("Unexpected network attempt")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def transport(session, **settings):
    """Disable pacing in offline tests; retries are inspected without sleeping."""
    return api.SecureTransport(session=session, minimum_request_interval_seconds=0,
                               retry_delay_seconds=0, **settings)


def configuration(tmp_path, *, start="2025-01-01", end="2025-01-02"):
    """Use isolated data/output roots and a named synthetic credential only."""
    return {"api": {"start": start, "end": end, "timezone": "Europe/Rome",
                    "cache_directory": str(tmp_path / "cache"),
                    "output_directory": str(tmp_path / "outputs"), "plots": False,
                    "credentials": {"entsoe_token_env": TEST_TOKEN_ENV},
                    "requests": [{"provider": "entsoe", "dataset": "day_ahead_prices",
                                  "options": {"zone": "NORD", "zone_eic": NORD_EIC}}]}}


def price_xml(start="2024-12-31T23:00Z", end="2025-01-01T23:00Z"):
    """Represent invented hourly negative prices in provider-syntax XML."""
    return (f'<Publication_MarketDocument xmlns="urn:synthetic:api-test">'
            f'<mRID>synthetic-doc</mRID><revisionNumber>1</revisionNumber><type>A44</type>'
            '<createdDateTime>2025-01-05T00:00Z</createdDateTime><TimeSeries><mRID>1</mRID>'
            f'<in_Domain.mRID>{NORD_EIC}</in_Domain.mRID><out_Domain.mRID>{NORD_EIC}</out_Domain.mRID>'
            '<currency_Unit.name>EUR</currency_Unit.name><price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
            '<curveType>A03</curveType><Period><timeInterval>'
            f'<start>{start}</start><end>{end}</end></timeInterval><resolution>PT60M</resolution>'
            '<Point><position>1</position><price.amount>-12.5</price.amount></Point>'
            '</Period></TimeSeries></Publication_MarketDocument>').encode()


@pytest.mark.parametrize("url", ["http://web-api.tp.entsoe.eu/api",
                                     "https://outside.invalid/api",
                                     "https://web-api.tp.entsoe.eu/api?securityToken=synthetic-secret",
                                     "https://synthetic-secret@web-api.tp.entsoe.eu/api",
                                     "https://web-api.tp.entsoe.eu/api#synthetic-secret"])
def test_request_spec_rejects_credential_urls_and_unofficial_destinations(url):
    with pytest.raises(api.DataAPIError) as caught:
        api.ApiRequest(url)
    assert "synthetic-secret" not in str(caught.value)


@pytest.mark.parametrize("key", ["securityToken", "ACCESS_TOKEN", "client_secret", "Authorization"])
def test_request_spec_rejects_secret_fields_in_public_identity(key):
    with pytest.raises(api.DataAPIError, match="transport-only"):
        api.ApiRequest(ENTSOE_ENDPOINT, params={key: FAKE_TOKEN})


def test_transient_http_errors_retry_bounded_and_close_responses(monkeypatch):
    delays = []
    monkeypatch.setattr(api.time, "sleep", delays.append)
    responses = [StubResponse(429, headers={"Retry-After": "1000000"}),
                 StubResponse(503), StubResponse(200, b"actual-payload")]
    session = StubSession(responses)
    assert transport(session).get(api.ApiRequest(ENTSOE_ENDPOINT)) == b"actual-payload"
    assert len(session.calls) == 3
    assert all(response.closed for response in responses)
    assert delays[0] <= 10


def test_terna_specific_pacing_leaves_other_provider_interval_unchanged(monkeypatch):
    delays=[]
    monkeypatch.setattr(api.time,"monotonic",lambda:100.0)
    monkeypatch.setattr(api.time,"sleep",delays.append)
    session=StubSession([StubResponse(),StubResponse(),StubResponse()])
    http=transport(session,terna_minimum_request_interval_seconds=2.0)
    http.get(api.ApiRequest(ENTSOE_ENDPOINT))
    http.get(api.ApiRequest("https://api.terna.it/load/v2.0/total-load"),bearer_token=FAKE_TOKEN)
    http.get(api.ApiRequest(ENTSOE_ENDPOINT))
    assert delays==[2.0]


def test_configurable_outage_backoff_announces_retries_without_credentials(monkeypatch):
    delays,messages=[],[]
    monkeypatch.setattr(api.time,"sleep",delays.append)
    session=StubSession([StubResponse(503,FAKE_TOKEN.encode()) for _ in range(4)]+[StubResponse(content=b'ok')])
    http=transport(session,attempts=5,max_retry_delay_seconds=5,retry_callback=messages.append)
    http.retry_delay=2
    assert http.get(api.ApiRequest("https://api.terna.it/load/v2.0/total-load"),bearer_token=FAKE_TOKEN)==b'ok'
    assert delays==[2,4,5,5]
    assert len(messages)==4 and "retry 2/5" in messages[0] and "retry 5/5" in messages[-1]
    assert all("HTTP 503" in message and FAKE_TOKEN not in message for message in messages)


def test_unavailable_service_stops_at_budget_and_reports_attempt_count(monkeypatch):
    monkeypatch.setattr(api.time,"sleep",lambda _:None)
    session=StubSession([StubResponse(503),StubResponse(503)])
    with pytest.raises(api.DataAPIError,match="HTTP 503 after 2 attempts"):
        transport(session,attempts=2).get(api.ApiRequest(ENTSOE_ENDPOINT))
    assert len(session.calls)==2


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_http_errors_are_redacted_and_not_retried(status):
    response = StubResponse(status, content=f"securityToken={FAKE_TOKEN}".encode())
    session = StubSession([response])
    with pytest.raises(api.DataAPIError) as caught:
        transport(session).get(api.ApiRequest(ENTSOE_ENDPOINT), query_secrets={"securityToken": FAKE_TOKEN})
    assert str(status) in str(caught.value)
    assert FAKE_TOKEN not in str(caught.value)
    assert len(session.calls) == 1
    assert response.closed


def test_transport_exception_traceback_suppresses_credential_url(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    private_error = requests.Timeout(f"request failed: {ENTSOE_ENDPOINT}?securityToken={FAKE_TOKEN}")
    session = StubSession([private_error, private_error])
    with pytest.raises(api.DataAPIError) as caught:
        transport(session, attempts=2).get(api.ApiRequest(ENTSOE_ENDPOINT), query_secrets={"securityToken": FAKE_TOKEN})
    rendered = "".join(traceback.format_exception(caught.value))
    assert FAKE_TOKEN not in rendered
    assert "Timeout" in str(caught.value)
    assert len(session.calls) == 2


def test_redirect_does_not_forward_credentials_or_follow_location():
    response = StubResponse(302, content=FAKE_TOKEN.encode(), headers={"Location": "https://outside.invalid/collect"})
    session = StubSession([response])
    with pytest.raises(api.DataAPIError) as caught:
        transport(session).get(api.ApiRequest(ENTSOE_ENDPOINT), query_secrets={"securityToken": FAKE_TOKEN})
    assert len(session.calls) == 1
    assert session.calls[0]["allow_redirects"] is False
    assert FAKE_TOKEN not in str(caught.value)
    assert response.closed


def test_credentials_added_only_to_private_transport_call():
    public = api.ApiRequest(ENTSOE_ENDPOINT, params={"documentType": "A44"})
    session = StubSession([StubResponse()])
    transport(session).get(public, query_secrets={"securityToken": FAKE_TOKEN})
    assert public.params == {"documentType": "A44"}
    assert session.calls[0]["params"]["securityToken"] == FAKE_TOKEN


def test_terna_authentication_form_and_bearer_token_stay_outside_specs():
    session = StubSession([StubResponse(content=b"token-body"), StubResponse(content=b"data-body")])
    http = transport(session)
    http.post_form("https://api.terna.it/public-api/access-token", {"client_id": "synthetic-client", "client_secret": FAKE_TOKEN})
    spec = api.ApiRequest("https://api.terna.it/load/v2.0/total-load", params={"dateFrom": "01/01/2025"})
    http.get(spec, bearer_token=FAKE_TOKEN)
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["data"]["client_secret"] == FAKE_TOKEN
    assert session.calls[1]["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert not spec.headers


def test_terna_business_ids_are_transport_only_and_preserve_explicit_support_ids():
    """Physical requests get support IDs without changing logical request/cache keys."""
    from dataclasses import asdict, replace
    from uuid import UUID
    session = StubSession([StubResponse(), StubResponse(), StubResponse(), StubResponse(403)])
    http = transport(session)
    specification = api.ApiRequest("https://api.terna.it/load/v2.0/total-load",
        params={"dateFrom": "01/07/2025"}, headers={"Accept": "application/json"})
    original = asdict(specification)
    http.get(specification, bearer_token=FAKE_TOKEN)
    http.get(specification, bearer_token=FAKE_TOKEN)
    first, second = [call["headers"]["businessID"] for call in session.calls]
    assert UUID(first) != UUID(second)
    explicit = replace(specification, headers={**specification.headers, "businessID": first})
    http.get(explicit, bearer_token=FAKE_TOKEN)
    assert session.calls[-1]["headers"]["businessID"] == first
    with pytest.raises(api.DataAPIError) as caught:
        http.get(specification, bearer_token=FAKE_TOKEN)
    assert session.calls[-1]["headers"]["businessID"] in str(caught.value)
    assert FAKE_TOKEN not in str(caught.value)
    assert asdict(specification) == original


@pytest.mark.parametrize("start,end,hours", [("2025-03-29", "2025-04-01", [24, 23, 24]),
                                            ("2025-10-25", "2025-10-28", [24, 25, 24])])
def test_planner_preserves_local_midnights_and_dst_lengths(tmp_path, start, end, hours):
    plans = api.plan_downloads(configuration(tmp_path, start=start, end=end))
    assert len(plans) == len(hours)
    for plan, duration in zip(plans, hours):
        identity = plan["identity"]
        first, last = pd.Timestamp(identity["start"]), pd.Timestamp(identity["end"])
        assert (last - first).total_seconds() / 3600 == duration
        assert first.tz_convert("Europe/Rome").hour == last.tz_convert("Europe/Rome").hour == 0
        assert plan["specification"].params["periodStart"] == first.strftime("%Y%m%d%H%M")


def test_month_first_plan_completes_all_datasets_before_next_local_month(tmp_path):
    config=configuration(tmp_path,start="2025-01-30",end="2025-02-03")
    config["api"]["requests"].append({"provider":"entsoe","dataset":"load_forecast",
        "options":{"zone":"NORD","zone_eic":NORD_EIC}})
    plans=api.plan_downloads(config)
    assert [(plan["identity"]["local_day"],plan["identity"]["dataset"]) for plan in plans]==[
        ("2025-01-30","day_ahead_prices"),("2025-01-31","day_ahead_prices"),
        ("2025-01-30","load_forecast"),("2025-01-31","load_forecast"),
        ("2025-02-01","day_ahead_prices"),("2025-02-02","day_ahead_prices"),
        ("2025-02-01","load_forecast"),("2025-02-02","load_forecast")]


def test_dataset_first_compatibility_preserves_the_same_cache_identities(tmp_path):
    config=configuration(tmp_path,start="2025-01-31",end="2025-02-02")
    config["api"]["requests"].append({"provider":"entsoe","dataset":"load_forecast",
        "options":{"zone":"NORD","zone_eic":NORD_EIC}})
    monthly=api.plan_downloads(config)
    config["api"]["download_order"]="dataset_first"
    previous=api.plan_downloads(config)
    assert [plan["identity"]["dataset"] for plan in previous]==[
        "day_ahead_prices","day_ahead_prices","load_forecast","load_forecast"]
    cache=api.DownloadCache(tmp_path/"cache")
    assert {cache.location(plan["identity"]) for plan in monthly}=={
        cache.location(plan["identity"]) for plan in previous}


def test_invalid_download_order_is_rejected_before_any_download(tmp_path):
    config=configuration(tmp_path)
    config["api"]["download_order"]="unrecognized"
    with pytest.raises(api.DataAPIError,match="api.download_order"):
        api.download_from_config(config,transport=transport(StubSession([])))


def test_completed_dataset_reuses_cache_after_switch_to_month_first(tmp_path,monkeypatch):
    config=configuration(tmp_path,start="2025-01-31",end="2025-02-02")
    config["api"]["download_order"]="dataset_first"
    monkeypatch.setenv(TEST_TOKEN_ENV,FAKE_TOKEN)
    payloads=[price_xml("2025-01-30T23:00Z","2025-01-31T23:00Z"),
              price_xml("2025-01-31T23:00Z","2025-02-01T23:00Z")]
    api.download_from_config(config,transport=transport(StubSession([
        StubResponse(content=payload) for payload in payloads])))
    # A second requested snapshot makes scheduling across datasets observable
    # while preserving the real provider parser and fixed daily boundaries.
    config["api"]["requests"].append({**deepcopy(config["api"]["requests"][0]),
        "vintage_revision":"second-snapshot"})
    config["api"]["download_order"]="month_first"
    progress=[]
    session=StubSession([StubResponse(content=payload) for payload in payloads])
    result=api.download_from_config(config,transport=transport(session),progress=progress.append)
    assert result["cached_chunks"]==result["downloaded_chunks"]==2
    assert [item["status"] for item in result["items"]]==["cached","downloaded","cached","downloaded"]
    assert [call["params"]["periodStart"] for call in session.calls]==["202501302300","202501312300"]
    assert "month 2025-01" in progress[0]
    assert any("month 2025-02" in message for message in progress)
    assert result["download_order"]=="month_first"


def test_dry_run_and_unattended_cache_miss_never_prompt_or_touch_network(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.delenv(TEST_TOKEN_ENV, raising=False)
    monkeypatch.setattr(api.getpass, "getpass", lambda *_: pytest.fail("Unattended download prompted"))
    session = StubSession([])
    planned = api.download_from_config(config, dry_run=True, transport=transport(session))
    assert planned["status"] == "planned"
    assert planned["items"][0]["cached"] is False
    with pytest.raises(api.DataAPIError, match=TEST_TOKEN_ENV):
        api.download_from_config(config, transport=transport(session))
    assert not session.calls


def test_downloaded_cache_preserves_raw_native_data_and_reuses_without_auth(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    session = StubSession([StubResponse(content=price_xml())])
    first = api.download_from_config(config, transport=transport(session))
    assert first["downloaded_chunks"] == 1
    assert first["items"][0]["historically_eligible_records"] == 0
    directory = Path(first["items"][0]["directory"])
    frame = pd.read_parquet(directory / "records.parquet")
    assert len(frame) == 24
    assert frame.value.eq(-12.5).all()
    assert (frame.product_end - frame.product_start).eq(pd.Timedelta(hours=1)).all()
    assert frame.historical_available_at.isna().all()
    assert (directory / "response.xml").read_bytes() == price_xml()
    assert FAKE_TOKEN not in (directory / "manifest.json").read_text()
    assert FAKE_TOKEN not in Path(first["manifest"]).read_text()
    monkeypatch.delenv(TEST_TOKEN_ENV)
    monkeypatch.setattr(api, "_secret", lambda *_args, **_kwargs: pytest.fail("Cache hit accessed credentials"))
    offline = StubSession([])
    cached = api.download_from_config(config, transport=transport(offline))
    assert cached["cached_chunks"] == 1
    assert cached["downloaded_chunks"] == 0
    assert not offline.calls


def test_partial_daily_failure_resumes_with_only_missing_day(tmp_path, monkeypatch):
    config = configuration(tmp_path, end="2025-01-03")
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    broken = StubSession([StubResponse(content=price_xml()), StubResponse(503)])
    with pytest.raises(api.DataAPIError, match="503"):
        api.download_from_config(config, transport=transport(broken, attempts=1))
    resumed = StubSession([StubResponse(content=price_xml("2025-01-01T23:00Z", "2025-01-02T23:00Z"))])
    result = api.download_from_config(config, transport=transport(resumed))
    assert result["cached_chunks"] == 1
    assert result["downloaded_chunks"] == 1
    assert len(resumed.calls) == 1
    assert resumed.calls[0]["params"]["periodStart"] == "202501012300"


def test_cache_keys_include_provider_dataset_zone_product_window_and_vintage(tmp_path):
    config = configuration(tmp_path)
    identity = api.plan_downloads(config)[0]["identity"]
    cache = api.DownloadCache(tmp_path / "cache")
    baseline = cache.location(identity)
    overrides = [{"provider": "terna"}, {"dataset": "load_forecast"}, {"zone": "IT"},
                 {"start": "2025-01-02T00:00:00+00:00"}, {"end": "2025-01-02T01:00:00+00:00"},
                 {"vintage_revision": "separate-snapshot"}, {"schema_version": "new-schema"},
                 {"request": {**identity["request"], "params": {"documentType": "A65"}}}]
    assert all(cache.location({**deepcopy(identity), **change}) != baseline for change in overrides)


def test_raw_cache_corruption_is_rejected_before_returning_records(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    result = api.download_from_config(config, transport=transport(StubSession([StubResponse(content=price_xml())])))
    directory = Path(result["items"][0]["directory"])
    (directory / "response.xml").write_bytes(b"altered-raw-payload")
    with pytest.raises(api.DataAPIError, match="checksum"):
        api.DownloadCache(config["api"]["cache_directory"]).load(api.plan_downloads(config)[0]["identity"])


def test_unverified_auction_rejected_during_planning_before_secrets(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config["api"]["requests"][0]["dataset"] = "IDA1"
    monkeypatch.setattr(api, "_secret", lambda *_args, **_kwargs: pytest.fail("Invalid dataset accessed credentials"))
    with pytest.raises(ValueError, match="Unsupported"):
        api.download_from_config(config, transport=transport(StubSession([])))


@pytest.mark.parametrize("options", [{"securityToken": FAKE_TOKEN},
                                     {"extra": {"Client-Secret": FAKE_TOKEN}},
                                     {"unused_filter": "anything"}])
def test_entsoe_unknown_or_secret_options_cannot_enter_cache_identity(tmp_path, options):
    config = configuration(tmp_path)
    config["api"]["requests"][0]["options"].update(options)
    with pytest.raises(api.DataAPIError) as caught:
        api.plan_downloads(config)
    assert FAKE_TOKEN not in str(caught.value)


def test_nested_secret_cannot_bypass_public_request_validation():
    with pytest.raises(api.DataAPIError):
        api.ApiRequest(ENTSOE_ENDPOINT, params={"nested": [{"security_token": FAKE_TOKEN}]})


def test_normalized_cache_corruption_is_detected_with_raw_unchanged(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    result = api.download_from_config(config, transport=transport(StubSession([StubResponse(content=price_xml())])))
    directory = Path(result["items"][0]["directory"])
    frame = pd.read_parquet(directory / "records.parquet")
    frame["value"] = 1_000_000
    frame.to_parquet(directory / "records.parquet", index=False)
    with pytest.raises(api.DataAPIError, match="checksum"):
        api.DownloadCache(config["api"]["cache_directory"]).load(api.plan_downloads(config)[0]["identity"])


def test_cache_manifest_cannot_redirect_raw_reads_outside_its_directory(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    result = api.download_from_config(config, transport=transport(StubSession([StubResponse(content=price_xml())])))
    directory = Path(result["items"][0]["directory"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_file"] = "../outside.xml"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(api.DataAPIError, match="filename"):
        api.DownloadCache(config["api"]["cache_directory"]).load(api.plan_downloads(config)[0]["identity"])


def terna_configuration(tmp_path):
    """Configure only invented test credential variable names for Terna calls."""
    config = configuration(tmp_path)
    config["api"]["credentials"] = {"terna_client_id_env": "BESS_UNIT_TERNA_CLIENT_ID",
                                     "terna_client_secret_env": "BESS_UNIT_TERNA_CLIENT_SECRET"}
    config["api"]["requests"] = [{"provider": "terna", "dataset": "total_load",
                                  "options": {"zone": "NORD", "bidding_zones": ["North"]}}]
    return config


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("stage", ["authentication", "data"])
def test_terna_auth_errors_identify_request_stage_without_exposing_secrets(tmp_path, monkeypatch, status, stage):
    """Distinguish a rejected credential exchange from a rejected authenticated GET."""
    config = terna_configuration(tmp_path)
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", FAKE_TOKEN)
    denied = StubResponse(status=status, content=f"secret={FAKE_TOKEN}".encode())
    outcomes = [denied]
    if stage == "data":
        outcomes.insert(0, StubResponse(content=json.dumps({"access_token": FAKE_TOKEN,
            "token_type": "bearer", "expires_in": 300}).encode()))
    session = StubSession(outcomes)
    with pytest.raises(api.DataAPIError) as caught:
        api.download_from_config(config, transport=transport(session))
    message = str(caught.value)
    assert f"HTTP {status}" in message
    if stage == "authentication":
        assert "authentication/token request" in message
        assert len(session.calls) == 1
    else:
        assert "data request for total_load" in message
        assert "after obtaining an access token" in message
        assert len(session.calls) == 2
    assert FAKE_TOKEN not in "".join(traceback.format_exception(caught.value))


def terna_response():
    """One source record contains both actual and forecast load fields."""
    return json.dumps({"result": {"status": "Completed", "message": "OK"},
                       "total_load": [{"date": "2025-01-01 12:00:00", "date_tz": "Europe/Rome",
                                       "date_offset": "+01:00", "bidding_zone": "North",
                                       "total_load_MW": "123", "forecast_total_load_MW": "234"}]}).encode()


@pytest.mark.parametrize("raw_works", [True, False])
def test_terna_diagnostic_compares_only_documented_headers_without_saving_secrets(tmp_path, monkeypatch, raw_works):
    """A bounded comparison changes only Authorization and a support correlation ID."""
    from bess_pv_trading.data.terna_diagnostics import diagnose_terna
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", FAKE_TOKEN)
    token = json.dumps({"access_token": FAKE_TOKEN, "token_type": "bearer", "expires_in": 300}).encode()
    session = StubSession([StubResponse(content=token), StubResponse(403, FAKE_TOKEN.encode()),
        StubResponse(content=terna_response()) if raw_works else StubResponse(403, FAKE_TOKEN.encode())])
    result = diagnose_terna(terna_configuration(tmp_path), transport=transport(session))
    assert result["working_authorization_schemes"] == (["raw"] if raw_works else [])
    assert result["checks"][0]["http_status"] == 403
    assert len(session.calls) == 3
    bearer, raw = session.calls[1:]
    assert bearer["url"] == raw["url"] and bearer["params"] == raw["params"]
    assert bearer["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert raw["headers"]["Authorization"] == FAKE_TOKEN
    assert bearer["headers"]["businessID"] != raw["headers"]["businessID"]
    assert FAKE_TOKEN not in json.dumps(result)
    assert not list(tmp_path.rglob("*"))


def test_terna_diagnostic_stops_before_data_on_token_rejection(tmp_path, monkeypatch):
    from bess_pv_trading.data.terna_diagnostics import diagnose_terna
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", FAKE_TOKEN)
    session = StubSession([StubResponse(403, FAKE_TOKEN.encode())])
    with pytest.raises(api.DataAPIError, match="authentication/token diagnostic"):
        diagnose_terna(terna_configuration(tmp_path), transport=transport(session))
    assert len(session.calls) == 1


def test_opt_in_raw_terna_authorization_preserves_cache_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", FAKE_TOKEN)
    config = terna_configuration(tmp_path)
    previous_identity = api.plan_downloads(config)[0]["identity"]
    config["api"]["terna_authorization_scheme"] = "raw"
    assert api.plan_downloads(config)[0]["identity"] == previous_identity
    token = json.dumps({"access_token": FAKE_TOKEN, "token_type": "bearer", "expires_in": 300}).encode()
    session = StubSession([StubResponse(content=token), StubResponse(content=terna_response())])
    result = api.download_from_config(config, transport=transport(session))
    assert result["downloaded_chunks"] == 1
    assert session.calls[1]["headers"]["Authorization"] == FAKE_TOKEN
    config["api"]["terna_authorization_scheme"] = "bearer"
    assert api.download_from_config(config, transport=transport(StubSession([])))["cached_chunks"] == 1


def test_invalid_authorization_setting_fails_before_network(tmp_path):
    config = terna_configuration(tmp_path)
    config["api"]["terna_authorization_scheme"] = "unsupported"
    with pytest.raises(api.DataAPIError, match="terna_authorization_scheme"):
        api.plan_downloads(config)
    session = StubSession([])
    with pytest.raises(api.DataAPIError, match="only.*Terna"):
        transport(session).get(api.ApiRequest(ENTSOE_ENDPOINT), bearer_token=FAKE_TOKEN,
                               authorization_scheme="raw")
    assert not session.calls


def test_terna_plan_uses_inclusive_local_dates_without_double_requesting_dst_days(tmp_path):
    config = terna_configuration(tmp_path)
    config["api"].update(start="2025-03-29", end="2025-04-01")
    plans = api.plan_downloads(config)
    assert len(plans) == 3
    assert [plan["specification"].params["dateFrom"] for plan in plans] == ["29/03/2025", "30/03/2025", "31/03/2025"]
    assert all(plan["specification"].params["dateFrom"] == plan["specification"].params["dateTo"] for plan in plans)


@pytest.mark.parametrize("timezone,start", [("UTC", "2025-01-01"),
                                            ("Europe/Rome", "2025-01-01T12:00:00+01:00")])
def test_terna_plan_rejects_partial_italian_days(tmp_path, timezone, start):
    config = terna_configuration(tmp_path)
    config["api"].update(timezone=timezone, start=start)
    with pytest.raises(api.DataAPIError, match="calendar days"):
        api.plan_downloads(config)


def test_terna_sibling_fields_share_one_auth_and_one_http_payload(tmp_path, monkeypatch):
    config = terna_configuration(tmp_path)
    config["api"]["requests"].append({**deepcopy(config["api"]["requests"][0]), "dataset": "total_load_forecast"})
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", "synthetic-client-secret")
    session = StubSession([StubResponse(content=b'{"access_token":"synthetic-bearer","token_type":"bearer","expires_in":3600}'),
                           StubResponse(content=terna_response())])
    result = api.download_from_config(config, transport=transport(session))
    assert [call["method"] for call in session.calls] == ["POST", "GET"]
    assert len(result["items"]) == 2
    actual, forecast = [pd.read_parquet(Path(item["directory"]) / "records.parquet") for item in result["items"]]
    assert actual.value.iloc[0] == 123
    assert forecast.value.iloc[0] == 234
    assert actual.retrieved_at.iloc[0] == forecast.retrieved_at.iloc[0]
    for item in result["items"]:
        manifest = (Path(item["directory"]) / "manifest.json").read_text()
        assert "synthetic-bearer" not in manifest
        assert "synthetic-client-secret" not in manifest


def test_cached_terna_raw_can_derive_a_missing_sibling_without_credentials(tmp_path, monkeypatch):
    config = terna_configuration(tmp_path)
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", "synthetic-client-secret")
    session = StubSession([StubResponse(content=b'{"access_token":"synthetic-bearer","token_type":"bearer","expires_in":3600}'),
                           StubResponse(content=terna_response())])
    api.download_from_config(config, transport=transport(session))
    config["api"]["requests"].append({**deepcopy(config["api"]["requests"][0]), "dataset": "total_load_forecast"})
    monkeypatch.setattr(api, "_secret", lambda *_args, **_kwargs: pytest.fail("Cached raw sibling accessed credentials"))
    offline = StubSession([])
    result = api.download_from_config(config, transport=transport(offline))
    assert result["cached_chunks"] == 1
    assert not offline.calls
    forecast = pd.read_parquet(Path(result["items"][1]["directory"]) / "records.parquet")
    assert forecast.value.iloc[0] == 234


def test_raw_reuse_does_not_cross_distinct_requested_vintages(tmp_path, monkeypatch):
    config = terna_configuration(tmp_path)
    config["api"]["requests"][0]["vintage_revision"] = "first-snapshot"
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", "synthetic-client-secret")
    token_response = b'{"access_token":"synthetic-bearer","token_type":"bearer","expires_in":3600}'
    first_session = StubSession([StubResponse(content=token_response), StubResponse(content=terna_response())])
    api.download_from_config(config, transport=transport(first_session))
    config["api"]["requests"].append({**deepcopy(config["api"]["requests"][0]), "dataset": "total_load_forecast", "vintage_revision": "second-snapshot"})
    second_session = StubSession([StubResponse(content=token_response), StubResponse(content=terna_response())])
    result = api.download_from_config(config, transport=transport(second_session))
    assert result["cached_chunks"] == 1
    assert [call["method"] for call in second_session.calls] == ["POST", "GET"]


def test_hidden_credential_is_resolved_once_across_daily_chunks(tmp_path,monkeypatch):
    config=configuration(tmp_path,end="2025-01-03")
    resolved=[]
    def fake_secret(name,**kwargs):
        resolved.append(name)
        return FAKE_TOKEN
    monkeypatch.setattr(api,"_secret",fake_secret)
    session=StubSession([StubResponse(content=price_xml()),
        StubResponse(content=price_xml("2025-01-01T23:00Z","2025-01-02T23:00Z"))])
    result=api.download_from_config(config,transport=transport(session))
    assert result["downloaded_chunks"]==2
    assert resolved==[TEST_TOKEN_ENV]


def test_refresh_preserves_previous_snapshot_and_invalidates_plot(tmp_path,monkeypatch):
    config=configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV,FAKE_TOKEN)
    first=api.download_from_config(config,transport=transport(StubSession([StubResponse(content=price_xml())])))
    directory=Path(first["items"][0]["directory"])
    (directory/"values.png").write_bytes(b"old plot")
    changed=price_xml().replace(b"-12.5",b"-3.0")
    api.download_from_config(config,refresh_cache=True,
        transport=transport(StubSession([StubResponse(content=changed)])))
    archived=list((directory/"revisions").glob("*/records.parquet"))
    assert len(archived)==1
    assert pd.read_parquet(archived[0]).value.eq(-12.5).all()
    assert pd.read_parquet(directory/"records.parquet").value.eq(-3).all()
    assert not (directory/"values.png").exists()


def test_no_data_is_reported_then_reused_without_zero_values_or_credentials(tmp_path,monkeypatch):
    config=configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV,FAKE_TOKEN)
    response=b'<Acknowledgement_MarketDocument><Reason><code>999</code><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>'
    session=StubSession([StubResponse(400,response)])
    first=api.download_from_config(config,transport=transport(session))
    assert first["status"]=="complete-with-gaps" and first["no_data_chunks"]==1
    directory=Path(first["items"][0]["directory"])
    assert (directory/"no_data.json").is_file()
    assert not (directory/"records.parquet").exists()
    monkeypatch.setattr(api,"_secret",lambda *_args,**_kwargs:pytest.fail("No-data cache accessed credentials"))
    cached=api.download_from_config(config,transport=transport(StubSession([])))
    assert cached["items"][0]["status"]=="no-data-cached"
    assert cached["items"][0]["records"]==0


def test_no_data_day_does_not_prevent_next_day_download(tmp_path,monkeypatch):
    config=configuration(tmp_path,end="2025-01-03")
    monkeypatch.setenv(TEST_TOKEN_ENV,FAKE_TOKEN)
    missing=b'<Acknowledgement_MarketDocument><Reason><code>999</code><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>'
    session=StubSession([StubResponse(400,missing),
        StubResponse(content=price_xml("2025-01-01T23:00Z","2025-01-02T23:00Z"))])
    result=api.download_from_config(config,transport=transport(session))
    assert result["no_data_chunks"]==1 and result["downloaded_chunks"]==1
    assert len(session.calls)==2


def test_unparsed_terna_data_is_retained_and_reused_after_parser_fix_without_credentials(tmp_path,monkeypatch):
    """A real schema mismatch must not force another HTTP request to inspect/reparse."""
    from bess_pv_trading.data import terna
    config=terna_configuration(tmp_path)
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID","synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET",FAKE_TOKEN)
    payload=b'{"result":{"status":"Completed"},"changed_array":[]}'
    token=json.dumps({"access_token":FAKE_TOKEN,"token_type":"bearer","expires_in":300}).encode()
    session=StubSession([StubResponse(content=token),StubResponse(content=payload)])
    with pytest.raises(api.DataAPIError,match="Unparsed data retained"):
        api.download_from_config(config,transport=transport(session))
    cache=api.DownloadCache(config["api"]["cache_directory"])
    identity=api.plan_downloads(config)[0]["identity"]
    retained,receipt=cache.load_unparsed(identity)
    assert retained==payload
    assert cache.load(identity) is None
    assert not (cache.location(identity)/"no_data.json").exists()
    frame=terna.parse_terna_json(terna_response(),"total_load",identity["options"],receipt["retrieved_at"])
    def corrected_parser(data,**kwargs):
        assert data==payload and kwargs["retrieved_at"]==pd.Timestamp(receipt["retrieved_at"])
        return frame
    monkeypatch.setattr(terna,"parse_terna_json",corrected_parser)
    monkeypatch.setattr(api,"_secret",lambda *_a,**_k:pytest.fail("Reparsing accessed credentials"))
    result=api.download_from_config(config,transport=transport(StubSession([])))
    assert result["data_http_operations"]==result["authentication_operations"]==0
    assert cache.load(identity)[0].value.tolist()==[123]


@pytest.mark.parametrize("payload", [b'raw synthetic-unit-test-token',
    b'{"echo":"synthetic-unit-test-toke\\u006e"}'])
def test_unparsed_data_never_retains_echoed_authentication_material(tmp_path,payload):
    config=terna_configuration(tmp_path)
    cache=api.DownloadCache(config["api"]["cache_directory"])
    identity=api.plan_downloads(config)[0]["identity"]
    with pytest.raises(api.DataAPIError,match="authentication material"):
        cache.save_unparsed(identity,payload,"2026-10-16T12:00:00Z",[FAKE_TOKEN])
    assert not list(tmp_path.rglob("*"))


def test_retained_unparsed_response_checksum_is_verified(tmp_path):
    config=terna_configuration(tmp_path)
    cache=api.DownloadCache(config["api"]["cache_directory"])
    identity=api.plan_downloads(config)[0]["identity"]
    path=cache.save_unparsed(identity,b'{"unexpected":[]}',"2026-10-16T12:00:00Z")
    path.write_bytes(b'changed')
    with pytest.raises(api.DataAPIError,match="Invalid retained API response"):
        cache.load_unparsed(identity)


@pytest.mark.parametrize("body,category", [(b'Quota exceeded',"rate_limit_or_quota"),
    (b'{"error":"invalid_token"}',"invalid_or_expired_token"),
    (b'{"error":"insufficient_scope"}',"insufficient_scope_or_subscription"),
    (b'<html>Access denied</html>',"gateway_or_policy_block"),
    (b'Forbidden',"unspecified")])
def test_http_error_hints_are_fixed_categories_without_provider_text(body,category):
    session=StubSession([StubResponse(403,body+FAKE_TOKEN.encode())])
    with pytest.raises(api.DataAPIError) as caught:
        transport(session).get(api.ApiRequest("https://api.terna.it/load/v2.0/total-load"),bearer_token=FAKE_TOKEN)
    assert f"provider_hint={category}" in str(caught.value)
    assert FAKE_TOKEN not in str(caught.value)
    assert len(session.calls)==1
