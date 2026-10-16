"""GME acquisition/cache contracts with documented synthetic JSON ZIP payloads."""
import base64
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.data import api, gme


LOGIN = "synthetic-gme-login"
PASSWORD = "synthetic-gme-password"
TOKEN = "synthetic-gme-jwt"


class Response:
    """Represent one isolated HTTP response without actual credentials/network."""
    def __init__(self, body, status=200):
        self.content, self.status_code, self.headers = body, status, {}
        self.closed = False

    def close(self):
        self.closed = True


class Session:
    """Inspect HTTP body/authentication boundaries and reject unplanned calls."""
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected HTTP operation")
        return self.responses.pop(0)


def payload(dataset="ida1_prices", *, zones=("NORD",), empty=False, missing_field=False, secret=None):
    """Encode explicitly invented quarter-hour prices in the documented envelope."""
    rows = []
    if not empty:
        for zone in zones:
            for period in range(49 if dataset == "ida3_prices" else 1, 97):
                row = {"FlowDate": 20251001, "Hour": (period - 1) // 4 + 1,
                    "Period": str(period), "Market": gme.DATASETS[dataset]["segment"],
                    "Zone": zone, "Price": -12.5 + period / 10, "Notes": secret}
                if missing_field:
                    row.pop("Price")
                rows.append(row)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prices.json", json.dumps(rows))
    return json.dumps({"RequestId": "synthetic-request", "FormatType": ".json.zip",
        "ResultRequest": "Success", "ContentResponse": base64.b64encode(buffer.getvalue()).decode()}).encode()


def configuration(tmp_path, *, backend="files", datasets=("ida1_prices",)):
    """Use task-specific roots and synthetic secret environment variable names."""
    return {"api": {"start": "2025-10-01", "end": "2025-10-02", "timezone": "Europe/Rome",
        "storage_backend": backend, "cache_directory": str(tmp_path / "cache"),
        "database_path": str(tmp_path / "market.duckdb"), "output_directory": str(tmp_path / "outputs"),
        "plots": False, "credentials": {"gme_username_env": "BESS_TEST_GME_LOGIN",
                                         "gme_password_env": "BESS_TEST_GME_PASSWORD"},
        "requests": [{"provider": "gme", "dataset": dataset, "options": {"zone": "NORD"}}
                     for dataset in datasets]}}


def transport(responses):
    """Disable pacing only for offline protocol verification."""
    session = Session(responses)
    return session, api.SecureTransport(session=session, minimum_request_interval_seconds=0,
                                        retry_delay_seconds=0)


@pytest.fixture
def credentials(monkeypatch):
    """Populate synthetic credentials for this test process only."""
    monkeypatch.setenv("BESS_TEST_GME_LOGIN", LOGIN)
    monkeypatch.setenv("BESS_TEST_GME_PASSWORD", PASSWORD)


@pytest.mark.parametrize("backend", ["files", "duckdb"])
def test_all_four_auctions_authenticate_once_post_json_and_resume_without_credentials(tmp_path, credentials, monkeypatch, backend):
    datasets = tuple(gme.DATASETS)
    config = configuration(tmp_path, backend=backend, datasets=datasets)
    session, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()),
                              *[Response(payload(dataset)) for dataset in datasets]])
    result = api.download_from_config(config, transport=http)
    assert result["authentication_operations"] == 1 and result["data_http_operations"] == 4
    assert [item["records"] for item in result["items"]] == [96, 96, 96, 48]
    assert session.calls[0]["json"] == {"Login": LOGIN, "Password": PASSWORD}
    assert all(call["params"] == {} and call["data"] is None for call in session.calls)
    assert all(call["headers"].get("Authorization") == f"Bearer {TOKEN}" for call in session.calls[1:])
    assert all(call["json"]["DataName"] == "ME_ZonalPrices" for call in session.calls[1:])
    assert all(call["allow_redirects"] is False for call in session.calls)
    for plan in api.plan_downloads(config):
        frame, manifest = api.cache_from_settings(config["api"]).load(plan["identity"])
        assert frame.historical_feature_eligible.eq(False).all()
        assert frame.historical_available_at.isna().all()
        assert frame.price_eur_mwh.min() < 0 or plan["identity"]["dataset"] == "ida3_prices"
        assert not any(secret in json.dumps(manifest) for secret in (LOGIN, PASSWORD, TOKEN))
    monkeypatch.delenv("BESS_TEST_GME_LOGIN")
    monkeypatch.delenv("BESS_TEST_GME_PASSWORD")
    _, unused_http = transport([])
    resumed = api.download_from_config(config, transport=unused_http)
    assert resumed["cached_chunks"] == 4 and resumed["authentication_operations"] == 0
    assert resumed["data_http_operations"] == 0


def test_dry_run_needs_no_credentials_and_preserves_existing_provider_identity(tmp_path):
    config = configuration(tmp_path)
    planned = api.download_from_config(config, dry_run=True)
    assert planned["requests"] == 1 and not planned["items"][0]["cached"]
    request = api.ApiRequest("https://web-api.tp.entsoe.eu/api", {"documentType": "A44"})
    assert asdict(request) == {"url": request.url, "params": {"documentType": "A44"}, "headers": {}}
    config["api"]["timezone"] = "UTC"
    with pytest.raises(ValueError, match="complete Europe/Rome"):
        api.plan_downloads(config)


def test_cached_all_zone_payload_derives_another_zone_without_authentication(tmp_path, credentials):
    config = configuration(tmp_path)
    session, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()),
                               Response(payload(zones=("NORD", "CNOR")))])
    api.download_from_config(config, transport=http)
    config["api"]["requests"].append({"provider": "gme", "dataset": "ida1_prices", "options": {"zone": "CNOR"}})
    _, offline = transport([])
    result = api.download_from_config(config, transport=offline)
    assert result["cached_chunks"] == 1 and result["downloaded_chunks"] == 1
    assert result["authentication_operations"] == result["data_http_operations"] == 0


def test_empty_gme_data_is_a_bounded_absence_receipt(tmp_path, credentials):
    config = configuration(tmp_path)
    _, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()), Response(payload(empty=True))])
    result = api.download_from_config(config, transport=http)
    assert result["status"] == "complete-with-gaps" and result["no_data_chunks"] == 1
    _, offline = transport([])
    again = api.download_from_config(config, transport=offline)
    assert again["items"][0]["status"] == "no-data-cached"


def test_safe_schema_failure_can_be_reparsed_without_another_auth_or_data_request(tmp_path, credentials, monkeypatch):
    config = configuration(tmp_path)
    _, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()), Response(payload(missing_field=True))])
    with pytest.raises(api.DataAPIError, match="Unparsed data retained"):
        api.download_from_config(config, transport=http)
    identity = api.plan_downloads(config)[0]["identity"]
    assert api.cache_from_settings(config["api"]).load_unparsed(identity) is not None
    parser = gme.parse_gme_response
    monkeypatch.setattr(gme, "parse_gme_response", lambda value, **kwargs: parser(payload(), **kwargs))
    monkeypatch.delenv("BESS_TEST_GME_LOGIN")
    monkeypatch.delenv("BESS_TEST_GME_PASSWORD")
    _, offline = transport([])
    result = api.download_from_config(config, transport=offline)
    assert result["downloaded_chunks"] == 1 and result["data_http_operations"] == result["authentication_operations"] == 0


def test_credential_echo_inside_compressed_data_is_never_retained(tmp_path, credentials):
    config = configuration(tmp_path)
    _, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()), Response(payload(secret=PASSWORD))])
    with pytest.raises(api.DataAPIError, match="could not be safely decoded") as caught:
        api.download_from_config(config, transport=http)
    assert PASSWORD not in str(caught.value)
    assert not list((tmp_path / "cache").rglob("*"))


def test_authentication_and_http_failures_do_not_echo_provider_text_or_cycle_credentials(tmp_path, credentials):
    config = configuration(tmp_path)
    session, http = transport([Response(json.dumps({"Success": False, "Reason": PASSWORD}).encode())])
    with pytest.raises(api.DataAPIError, match="GME authentication failed") as caught:
        api.download_from_config(config, transport=http)
    assert PASSWORD not in str(caught.value) and len(session.calls) == 1
    session, http = transport([Response(json.dumps({"Success": True, "token": TOKEN}).encode()), Response(PASSWORD.encode(), status=403)])
    with pytest.raises(api.DataAPIError, match="HTTP 403") as caught:
        api.download_from_config(config, transport=http)
    assert PASSWORD not in str(caught.value) and len(session.calls) == 2


def test_optional_jwt_expiry_only_shortens_the_configured_refresh_bound(monkeypatch):
    monkeypatch.setattr(api.time, "time", lambda: 1000.)
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    assert api._token_refresh_age(f"header.{encode({'exp': 1100})}.signature", 900.) == 70.
    assert api._token_refresh_age(f"header.{encode({'exp': 5000})}.signature", 900.) == 900.
    assert api._token_refresh_age(f"header.{encode([])}.signature", 900.) == 900.
    assert api._token_refresh_age("opaque-token", 900.) == 900.
