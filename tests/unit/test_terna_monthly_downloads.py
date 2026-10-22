"""Monthly transport must preserve source days, values, DST and daily resumes."""
import json

import pandas as pd
import pytest

from bess_pv_trading.data import api, terna
from bess_pv_trading.config import load_config


def configuration(tmp_path, start, end, backend="duckdb"):
    """Use isolated stores and actual Terna parsing, without external requests."""
    return {"api": {"start": start, "end": end, "timezone": "Europe/Rome",
        "storage_backend": backend, "database_path": str(tmp_path / "market.duckdb"),
        "cache_directory": str(tmp_path / "files"), "plots": False,
        "output_directory": str(tmp_path / "reports"), "requests": [
            {"provider": "terna", "dataset": dataset,
             "options": {"zone": "NORD", "bidding_zones": ["North"]}}
            for dataset in ("total_load", "total_load_forecast")]}}


class MonthlyTransport:
    """Return native quarter-hours with explicit offsets across both DST changes."""
    def __init__(self, omit_day=None):
        self.requests = []
        self.auth_calls = 0
        self.omit_day = omit_day

    def post_form(self, url, fields):
        self.auth_calls += 1
        return b'{"access_token":"fixture-token","token_type":"Bearer","expires_in":300}'

    def get(self, specification, **kwargs):
        self.requests.append(specification)
        start = pd.Timestamp(pd.to_datetime(specification.params["dateFrom"], format="%d/%m/%Y"), tz="Europe/Rome")
        end = pd.Timestamp(pd.to_datetime(specification.params["dateTo"], format="%d/%m/%Y"), tz="Europe/Rome") + pd.DateOffset(days=1)
        rows = []
        for stamp in pd.date_range(start, end, freq="15min", inclusive="left"):
            if stamp.strftime("%Y-%m-%d") == self.omit_day:
                continue
            value = 1000 + (stamp.value // 900_000_000_000) % 1000
            rows.append({"date": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date_tz": "Europe/Rome", "date_offset": stamp.strftime("%z")[:3] + ":" + stamp.strftime("%z")[3:],
                "bidding_zone": "North", "total_load_MW": value,
                "forecast_total_load_MW": value + 20})
        return json.dumps({"result": {"status": "Completed"}, "total_load": rows}).encode()


@pytest.fixture(autouse=True)
def fixture_credentials(monkeypatch):
    monkeypatch.setattr(api, "_secret", lambda *args, **kwargs: "fixture-private")


@pytest.mark.parametrize("backend", ["duckdb", "files"])
@pytest.mark.parametrize("start,end,expected_calls", [
    ("2025-01-31", "2025-02-03", 2),
    ("2025-03-29", "2025-04-01", 1),
    ("2025-10-25", "2025-10-28", 1),
])
def test_monthly_transport_keeps_exact_daily_grid_values_and_offsets(tmp_path, backend, start, end, expected_calls):
    config = configuration(tmp_path, start, end, backend)
    transport = MonthlyTransport()
    result = api.download_from_config(config, transport=transport)
    assert result["status"] == "complete"
    assert len(transport.requests) == expected_calls
    assert result["data_http_operations"] == expected_calls
    assert transport.auth_calls == 1
    cache = api.cache_from_settings(config["api"])
    plans = api.plan_downloads(config)
    for plan in plans:
        identity = plan["identity"]
        frame, manifest = cache.load(identity)
        expected = pd.date_range(pd.Timestamp(identity["start"]), pd.Timestamp(identity["end"]),
                                 freq="15min", inclusive="left")
        assert list(frame.reference_time) == list(expected)
        assert set(frame.reference_time.dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d")) == {identity["local_day"]}
        expected_values = 1000 + (expected.as_unit("ns").asi8 // 900_000_000_000) % 1000
        if identity["dataset"] == "total_load_forecast":
            expected_values += 20
        assert frame.value.tolist() == expected_values.tolist()
        assert manifest["metadata"]["transport_request"] == api.asdict(plan["transport_specification"])
        assert not frame.reference_time.duplicated().any()
        if identity["local_day"] == "2025-03-30":
            assert len(frame) == 92
        if identity["local_day"] == "2025-10-26":
            assert len(frame) == 100
            repeated = frame.reference_time.dt.tz_convert("Europe/Rome").dt.hour == 2
            assert repeated.sum() == 8
            assert set(frame.loc[repeated, "source_offset"]) == {"+02:00", "+01:00"}
    resumed = MonthlyTransport()
    assert api.download_from_config(config, transport=resumed)["cached_chunks"] == len(plans)
    assert not resumed.requests and resumed.auth_calls == 0


def test_full_month_uses_one_response_for_both_load_fields(tmp_path):
    config = configuration(tmp_path, "2025-02-01", "2025-03-01")
    transport = MonthlyTransport()
    planned = api.download_from_config(config, dry_run=True)
    assert planned["requests"] == 56 and planned["data_http_operations_upper_bound"] == 1
    result = api.download_from_config(config, transport=transport)
    assert len(transport.requests) == 1
    assert transport.requests[0].params["dateFrom"] == "01/02/2025"
    assert transport.requests[0].params["dateTo"] == "28/02/2025"
    assert result["downloaded_chunks"] == 56


def test_invalid_period_rejected_before_any_cache_or_credentials(tmp_path):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    config["api"]["terna_request_period"] = "year"
    with pytest.raises(api.DataAPIError, match="terna_request_period"):
        api.plan_downloads(config)


def test_five_series_smoke_configuration_plans_four_data_calls_per_month(tmp_path):
    config = load_config("configs/api_experiments/terna_api.yaml")
    config["api"].update(start="2025-02-01", end="2025-03-01",
        database_path=str(tmp_path / "market.duckdb"))
    planned = api.download_from_config(config, dry_run=True)
    assert planned["requests"] == 5 * 28
    assert planned["data_http_operations_upper_bound"] == 4


def test_cli_distinguishes_daily_cache_chunks_from_monthly_http_calls(tmp_path, monkeypatch, capsys):
    from bess_pv_trading import cli
    config = configuration(tmp_path, "2025-02-01", "2025-03-01")
    planned = api.download_from_config(config, dry_run=True)
    monkeypatch.setattr(api, "download_from_config", lambda *args, **kwargs: planned)
    assert cli.main(["download-data", "--config", "configs/api_experiments/terna_api.yaml", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["requests"] == 56
    assert result["data_http_operations_upper_bound"] == 1
    assert result["terna_request_period"] == "month"


def test_monthly_parser_failure_retains_exact_batch_for_offline_recovery(tmp_path, monkeypatch):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    original = terna.parse_terna_json

    def reject_schema(*args, **kwargs):
        raise ValueError("fixture schema failure")

    monkeypatch.setattr(terna, "parse_terna_json", reject_schema)
    first = MonthlyTransport()
    with pytest.raises(api.DataAPIError, match="Unparsed data retained"):
        api.download_from_config(config, transport=first)
    assert len(first.requests) == 1
    monkeypatch.setattr(terna, "parse_terna_json", original)
    resumed = MonthlyTransport()
    result = api.download_from_config(config, transport=resumed)
    assert result["downloaded_chunks"] == 6
    assert not resumed.requests and resumed.auth_calls == 0


def test_legacy_retained_daily_response_remains_reusable(tmp_path):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    plan = api.plan_downloads(config)[0]
    transport = MonthlyTransport()
    payload = transport.get(plan["specification"])
    cache = api.cache_from_settings(config["api"])
    cache.save_unparsed(plan["identity"], payload, pd.Timestamp.now(tz="UTC"))
    transport = MonthlyTransport()
    result = api.download_from_config(config, transport=transport)
    assert result["downloaded_chunks"] == 6 and len(transport.requests) == 1
    first, manifest = cache.load(plan["identity"])
    assert len(first) == 96
    assert manifest["metadata"]["transport_last_day"] == "2025-01-01"


def test_old_daily_caches_are_reused_without_mistaking_them_for_monthly_payloads(tmp_path):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    config["api"]["terna_request_period"] = "day"
    daily = api.plan_downloads(config)[0]
    cache = api.cache_from_settings(config["api"])
    transport = MonthlyTransport()
    payload = transport.get(daily["specification"])
    retrieved = pd.Timestamp.now(tz="UTC")
    frame = terna.parse_terna_json(payload, "total_load", daily["identity"]["options"], retrieved)
    cache.save(daily["identity"], payload, frame, {"retrieved_at": retrieved.isoformat()})
    config["api"]["terna_request_period"] = "month"
    monthly = MonthlyTransport()
    result = api.download_from_config(config, transport=monthly)
    assert result["cached_chunks"] == 1 and result["downloaded_chunks"] == 5
    assert len(monthly.requests) == 1
    assert monthly.requests[0].params["dateTo"] == "03/01/2025"


def test_cached_sibling_load_response_is_reparsed_without_network(tmp_path):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    config["api"]["requests"] = config["api"]["requests"][1:]
    api.download_from_config(config, transport=MonthlyTransport())
    config["api"]["requests"] = configuration(tmp_path, "2025-01-01", "2025-01-04")["api"]["requests"]
    transport = MonthlyTransport()
    result = api.download_from_config(config, transport=transport)
    assert result["cached_chunks"] == 3 and result["downloaded_chunks"] == 3
    assert not transport.requests and transport.auth_calls == 0


def test_missing_day_remains_a_gap_instead_of_shifting_the_next_days_values(tmp_path):
    config = configuration(tmp_path, "2025-01-01", "2025-01-04")
    transport = MonthlyTransport(omit_day="2025-01-02")
    result = api.download_from_config(config, transport=transport)
    assert len(transport.requests) == 1
    assert result["status"] == "complete-with-gaps"
    assert result["no_data_chunks"] == 2
    cache = api.cache_from_settings(config["api"])
    for plan in api.plan_downloads(config):
        identity = plan["identity"]
        if identity["local_day"] == "2025-01-02":
            assert cache.load(identity) is None
        else:
            assert len(cache.load(identity)[0]) == 96


def test_forecast_sessions_and_months_are_distinct_transport_requests(tmp_path):
    config = configuration(tmp_path, "2025-01-31", "2025-02-02")
    config["api"]["requests"] = [{"provider": "terna", "dataset": "wind_forecast",
        "options": {"zone": "NORD", "session_type": session}} for session in ("MSD1", "MSD2")]
    plans = api.plan_downloads(config)
    assert len({_key(plan) for plan in plans}) == 4


def _key(plan):
    return json.dumps(api.asdict(plan["transport_specification"]), sort_keys=True)


def test_range_checks_and_dated_no_results():
    retrieved = pd.Timestamp.now(tz="UTC")
    payload = json.dumps({"result": {"status": "Completed", "message":
        "No results found for the following input. Date from: 01/01/2025, date to: 31/01/2025"}}).encode()
    assert terna.parse_terna_json(payload, "total_load", retrieved_at=retrieved,
        requested_day="2025-01-01", requested_end_day="2025-01-31").empty
    with pytest.raises(ValueError, match="no-results dates"):
        terna.parse_terna_json(payload, "total_load", retrieved_at=retrieved,
            requested_day="2025-01-01", requested_end_day="2025-01-30")
    payload = MonthlyTransport().get(api.ApiRequest(terna.API_BASE + "/load/v2.0/total-load",
        {"dateFrom": "01/01/2025", "dateTo": "02/01/2025"}))
    with pytest.raises(ValueError, match="outside the requested"):
        terna.parse_terna_json(payload, "total_load", retrieved_at=retrieved,
            requested_day="2025-01-01", requested_end_day="2025-01-01")
