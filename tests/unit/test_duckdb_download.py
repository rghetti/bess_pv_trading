"""Exercise acquisition, failure recovery and source reuse against DuckDB offline."""

from pathlib import Path
import json

import pytest

from bess_pv_trading.data import api
from bess_pv_trading.data.duckdb_store import DuckDBCache
from test_api_download import (
    FAKE_TOKEN, TEST_TOKEN_ENV, StubResponse, StubSession, configuration,
    price_xml, terna_configuration, terna_response, transport,
)


def database_configuration(tmp_path, **kwargs):
    """Keep all state in an isolated database and disable acquisition plots."""
    config = configuration(tmp_path, **kwargs)
    config["api"].update(storage_backend="duckdb",
                         database_path=str(tmp_path / "market.duckdb"))
    return config


def test_database_download_reuses_source_without_files_or_credentials(tmp_path, monkeypatch):
    config = database_configuration(tmp_path)
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    first = api.download_from_config(config, transport=transport(StubSession([
        StubResponse(content=price_xml())])))
    assert first["downloaded_chunks"] == 1
    assert first["storage_backend"] == "duckdb"
    cache = DuckDBCache(config["api"]["database_path"])
    identity = api.plan_downloads(config)[0]["identity"]
    frame, manifest = cache.load(identity)
    assert frame.value.eq(-12.5).all() and frame.historical_available_at.isna().all()
    assert cache.load_raw(identity) == price_xml()
    assert not Path(config["api"]["cache_directory"]).exists()
    assert not list(tmp_path.rglob("records.csv"))
    monkeypatch.delenv(TEST_TOKEN_ENV)
    monkeypatch.setattr(api, "_secret", lambda *a, **k: pytest.fail("Cache hit asked for credentials"))
    session = StubSession([])
    second = api.download_from_config(config, transport=transport(session))
    assert second["cached_chunks"] == 1 and not session.calls


def test_database_outage_keeps_completed_day_and_resume_only_requests_missing(tmp_path, monkeypatch):
    config = database_configuration(tmp_path, end="2025-01-03")
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    with pytest.raises(api.DataAPIError, match="503"):
        api.download_from_config(config, transport=transport(StubSession([
            StubResponse(content=price_xml()), StubResponse(503)]), attempts=1))
    session = StubSession([StubResponse(content=price_xml("2025-01-01T23:00Z", "2025-01-02T23:00Z"))])
    resumed = api.download_from_config(config, transport=transport(session))
    assert resumed["cached_chunks"] == resumed["downloaded_chunks"] == 1
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["periodStart"] == "202501012300"


def test_database_terna_shared_response_survives_cache_hit_without_raw_file(tmp_path, monkeypatch):
    config = terna_configuration(tmp_path)
    config["api"].update(storage_backend="duckdb", database_path=str(tmp_path / "market.duckdb"))
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_ID", "synthetic-client")
    monkeypatch.setenv("BESS_UNIT_TERNA_CLIENT_SECRET", FAKE_TOKEN)
    session = StubSession([StubResponse(content=(
        '{"access_token":"' + FAKE_TOKEN + '","token_type":"bearer","expires_in":300}').encode()),
        StubResponse(content=terna_response())])
    api.download_from_config(config, transport=transport(session))
    config["api"]["requests"].append({"provider": "terna", "dataset": "total_load_forecast",
                                     "options": {"zone": "NORD", "bidding_zones": ["North"]}})
    monkeypatch.setattr(api, "_secret", lambda *a, **k: pytest.fail("Shared cached response asked for credentials"))
    offline = StubSession([])
    result = api.download_from_config(config, transport=transport(offline))
    assert result["cached_chunks"] == result["downloaded_chunks"] == 1
    assert result["data_http_operations"] == 0 and not offline.calls


def test_database_dry_run_is_read_only_and_plots_are_monthly(tmp_path, monkeypatch):
    config = database_configuration(tmp_path)
    planned = api.download_from_config(config, dry_run=True)
    assert not planned["items"][0]["cached"]
    assert not Path(config["api"]["database_path"]).exists()
    monkeypatch.setenv(TEST_TOKEN_ENV, FAKE_TOKEN)
    config["api"]["plots"] = True
    plots = []
    monkeypatch.setattr(api, "_plot_records", lambda frame, path, title: plots.append(Path(path)))
    result = api.download_from_config(config, transport=transport(StubSession([
        StubResponse(content=price_xml())])))
    assert len(plots) == 1 and plots[0].name == "2025-01.png"
    assert "outputs" in plots[0].parts
    assert result["plots"] == [str(plots[0])]


def test_storage_configuration_rejects_unknown_backend(tmp_path):
    config = database_configuration(tmp_path)
    config["api"]["storage_backend"] = "misspelled"
    with pytest.raises(api.DataAPIError, match="storage_backend"):
        api.download_from_config(config, dry_run=True)


def test_retained_terna_no_results_reuses_duckdb_without_credentials_or_http(tmp_path, monkeypatch):
    config = terna_configuration(tmp_path)
    config["api"].update(storage_backend="duckdb", database_path=str(tmp_path / "market.duckdb"),
                         start="2025-03-26", end="2025-03-27")
    config["api"]["requests"] = [{"provider": "terna", "dataset": "wind_forecast",
                                  "options": {"zone": "NORD", "session_type": "MSD1"}}]
    identity = api.plan_downloads(config)[0]["identity"]
    cache = DuckDBCache(config["api"]["database_path"])
    payload = json.dumps({"result": {"status": "Completed", "message":
        "No results found for the following input. Date from: 26/03/2025, date to: 26/03/2025"}}).encode()
    cache.save_unparsed(identity, payload, "2026-01-01T12:00:00Z")
    monkeypatch.setattr(api, "_secret", lambda *a, **k: pytest.fail("Retained response requested credentials"))
    session = StubSession([])
    result = api.download_from_config(config, transport=transport(session))
    assert result["no_data_chunks"] == 1
    assert result["status"] == "complete-with-gaps"
    assert result["authentication_operations"] == result["data_http_operations"] == 0
    assert not session.calls and cache.load(identity) is None
    assert cache.load_unparsed(identity)[0] == payload
    assert api.download_from_config(config, transport=transport(session))["items"][0]["status"] == "no-data-cached"
