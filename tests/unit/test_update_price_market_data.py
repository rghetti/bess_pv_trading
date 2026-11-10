"""Verify terminal-only token prompting and noninteractive operation."""

from scripts import update_price_market_data as updater


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
