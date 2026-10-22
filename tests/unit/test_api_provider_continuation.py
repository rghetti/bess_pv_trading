"""Terna denials pause that provider while remaining ENTSO-E months complete."""
from pathlib import Path
import json

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import api, entsoe, terna


class FakeTransport:
    """Return public fixture payloads and a structured, sanitized Terna failure."""

    def __init__(self, *, status=403, phase="data"):
        self.status = status
        self.phase = phase
        self.entsoe_calls = 0
        self.terna_calls = 0
        self.auth_calls = 0

    def fail(self):
        if self.status is not None:
            raise api.DataAPIHTTPError(f"api.terna.it returned HTTP {self.status}; response omitted",
                http_status=self.status, provider_host="api.terna.it", provider_hint="unspecified")

    def post_form(self, url, fields):
        self.auth_calls += 1
        if self.phase == "authentication":
            self.fail()
        return b"public-test-auth-placeholder"

    def get(self, specification, **kwargs):
        if "entsoe" in specification.url:
            self.entsoe_calls += 1
            return b"<public-fixture/>"
        self.terna_calls += 1
        self.fail()
        return b'{"public_fixture":true}'


@pytest.fixture
def setup_download(tmp_path, monkeypatch):
    config = load_config("tests/fixtures/configs/api_pause_cases.yaml")
    config["api"].update(database_path=str(tmp_path / "market.duckdb"),
        cache_directory=str(tmp_path / "legacy"), output_directory=str(tmp_path / "reports"),
        terna_request_period="day")
    private = config["test_case"]["private_marker"]
    monkeypatch.setattr(api, "_secret", lambda *args, **kwargs: private)
    monkeypatch.setattr(entsoe, "parse_entsoe_xml", lambda *args, **kwargs: pd.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(terna, "parse_terna_json", lambda *args, **kwargs: pd.DataFrame({"value": [1.0],
        "reference_time": [pd.Timestamp(kwargs["requested_day"], tz="Europe/Rome").tz_convert("UTC")]}))
    monkeypatch.setattr(terna, "parse_token", lambda payload: (private, 3600))
    return config


@pytest.mark.parametrize("backend", ["duckdb", "files"])
@pytest.mark.parametrize("phase", ["data", "authentication"])
def test_terna_403_continues_entsoe_in_later_months_reuses_cache_and_resumes(setup_download, backend, phase):
    config = setup_download
    config["api"]["storage_backend"] = backend
    plans = api.plan_downloads(config)
    cache = api.cache_from_settings(config["api"])
    cached = next(plan for plan in plans if plan["identity"]["provider"] == "terna"
        and plan["identity"]["dataset"] == "total_load"
        and plan["identity"]["local_day"] == config["test_case"]["cached_terna_day"])
    cache.save(cached["identity"], b"{}", pd.DataFrame({"value": [1.0]}),
        {"retrieved_at": config["test_case"]["retrieved_at"]})
    progress = []
    transport = FakeTransport(phase=phase)
    result = api.download_from_config(config, transport=transport, progress=progress.append)
    expected_entsoe = sum(plan["identity"]["provider"] == "entsoe" for plan in plans)
    expected_terna = len(plans) - expected_entsoe
    assert transport.entsoe_calls == expected_entsoe
    assert transport.terna_calls == (phase == "data")
    assert transport.auth_calls == 1
    assert result["status"] == "partial"
    assert result["failed_chunks"] == 1
    assert result["deferred_chunks"] == expected_terna - 2  # one failure and one cached chunk
    assert result["cached_chunks"] == 1
    assert result["provider_summaries"]["entsoe"]["status"] == "complete"
    assert result["provider_summaries"]["entsoe"]["downloaded_chunks"] == expected_entsoe
    assert result["paused_providers"]["terna"]["phase"] == phase
    assert result["data_http_operations"] == transport.entsoe_calls + transport.terna_calls
    assert any("Continuing other providers" in message for message in progress)
    manifest = Path(result["manifest"])
    assert json.loads(manifest.read_text())["status"] == "partial"
    assert config["test_case"]["private_marker"] not in manifest.read_text() + "\n".join(progress)
    failed = next(plan for plan in plans if plan["identity"]["provider"] == "terna")
    assert cache.load(failed["identity"]) is None
    assert cache.load_no_data(failed["identity"], config["api"]["no_data_cache_hours"]) is None
    recovered = FakeTransport(status=None)
    resumed = api.download_from_config(config, transport=recovered)
    assert resumed["status"] == "complete"
    assert recovered.entsoe_calls == 0
    assert resumed["cached_chunks"] == expected_entsoe + 1
    assert resumed["downloaded_chunks"] == expected_terna - 1
    assert not resumed["paused_providers"]


def test_strict_403_policy_still_fails_immediately(setup_download):
    setup_download["api"]["terna_http_403_policy"] = "error"
    transport = FakeTransport()
    with pytest.raises(api.DataAPIError, match="HTTP 403"):
        api.download_from_config(setup_download, transport=transport)
    january_entsoe = sum(plan["identity"]["provider"] == "entsoe" and plan["identity"]["local_day"].startswith("2025-01")
        for plan in api.plan_downloads(setup_download))
    assert transport.entsoe_calls == january_entsoe


@pytest.mark.parametrize("status", load_config("tests/fixtures/configs/api_pause_cases.yaml")["test_case"]["non_pause_statuses"])
def test_other_terna_errors_are_not_silently_deferred(setup_download, status):
    with pytest.raises(api.DataAPIError, match=f"HTTP {status}"):
        api.download_from_config(setup_download, transport=FakeTransport(status=status))


def test_transport_exposes_safe_structured_403_without_retry_or_response_body():
    private = load_config("tests/fixtures/configs/api_pause_cases.yaml")["test_case"]["private_marker"]

    class Response:
        status_code = 403
        content = f"quota exceeded: {private}".encode()
        headers = {}
        closed = False

        def close(self):
            self.closed = True

    class Session:
        calls = 0
        response = Response()

        def request(self, *args, **kwargs):
            self.calls += 1
            return self.response

    session = Session()
    transport = api.SecureTransport(session=session, minimum_request_interval_seconds=0)
    with pytest.raises(api.DataAPIHTTPError) as caught:
        transport.get(api.ApiRequest("https://api.terna.it/load/v2.0/total-load"), bearer_token=private)
    assert caught.value.http_status == 403
    assert caught.value.provider_host == "api.terna.it"
    assert caught.value.provider_hint == "rate_limit_or_quota"
    assert private not in str(caught.value)
    assert session.calls == 1 and session.response.closed


def test_cli_reports_partial_after_other_providers_finish(monkeypatch, capsys):
    from bess_pv_trading import cli
    monkeypatch.setattr(api, "download_from_config", lambda *args, **kwargs: {
        "status": "partial", "items": [], "paused_providers": {"terna": {"http_status": 403}}})
    assert cli.main(["download-data", "--config", "tests/fixtures/configs/api_pause_cases.yaml"]) == 2
    assert json.loads(capsys.readouterr().out)["paused_providers"]["terna"]["http_status"] == 403


def test_invalid_policy_is_rejected_before_opening_cache(setup_download):
    setup_download["api"]["terna_http_403_policy"] = setup_download["test_case"]["invalid_policy"]
    with pytest.raises(api.DataAPIError, match="terna_http_403_policy"):
        api.download_from_config(setup_download, transport=FakeTransport())
