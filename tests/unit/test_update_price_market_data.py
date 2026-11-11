"""Verify terminal-only token prompting and noninteractive operation."""

import pandas as pd
import pytest

from scripts import update_price_market_data as updater
from bess_pv_trading.data.swissgrid_energy import discover_energy_workbook


def test_entsoe_token_is_prompted_without_echo_or_logging(monkeypatch, capsys):
    """A pasted token is available to the child downloader without being printed."""
    name = "TEST_ENTSOE_TOKEN"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(updater.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(updater.getpass, "getpass", lambda prompt: "temporary-token")
    assert updater._load_entsoe_token(name, interactive=True)
    assert updater.os.environ[name] == "temporary-token"
    assert "temporary-token" not in capsys.readouterr().out


def test_noninteractive_run_does_not_prompt_for_absent_token(monkeypatch):
    """Scheduled updates must report missing credentials instead of blocking."""
    name = "TEST_ENTSOE_TOKEN"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(updater.getpass, "getpass", lambda prompt: (_ for _ in ()).throw(AssertionError()))
    assert not updater._load_entsoe_token(name, interactive=False)


def test_previous_swiss_year_is_refreshed_during_delayed_publication_window():
    """Keep requesting the preceding year's delayed data after New Year."""
    assert updater._swiss_refresh_years(pd.Timestamp("2027-01-02"), 7) == [2026, 2027]
    assert updater._swiss_refresh_years(pd.Timestamp("2027-08-02"), 7) == [2027]
    assert updater._swiss_refresh_years(pd.Timestamp("2027-02-02"), 1) == [2027]


def test_swiss_energy_link_discovery_uses_only_official_matching_year():
    """A current provider link is selected; third-party and ambiguous links are rejected."""
    page = b'''<a href="/dam/jcr%3Aabc/EnergieUebersichtCH-2026.xlsx">current</a>
      <a href="https://evil.example/EnergieUebersichtCH-2026.xlsx">wrong host</a>
      <a href="/dam/jcr%3Aold/EnergieUebersichtCH-2025.xlsx">old</a>'''
    page_url = "https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html"
    assert discover_energy_workbook(page, page_url, 2026) == (
        "https://www.swissgrid.ch/dam/jcr%3Aabc/EnergieUebersichtCH-2026.xlsx")
    with pytest.raises(ValueError, match="found 0"):
        discover_energy_workbook(page, page_url, 2027)
    with pytest.raises(ValueError, match="found 2"):
        discover_energy_workbook(page + b'<a href="/other/EnergieUebersichtCH-2026.xlsx">duplicate</a>',
                                 page_url, 2026)
