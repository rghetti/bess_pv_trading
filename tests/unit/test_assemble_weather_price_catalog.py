"""Check that isolated weather-catalog assembly preserves its input boundaries."""

import pandas as pd
import pytest

from bess_pv_trading.data.duckdb_store import DuckDBCache
from scripts.assemble_weather_price_catalog import assemble


def _write_weather(path, day):
    """Create one checksummed catalog snapshot for the requested cycle."""
    identity = {"provider": "open_meteo", "dataset": "weather_forecast", "zone": "NORD", "day": day}
    frame = pd.DataFrame({"provider": ["open_meteo"], "dataset": ["weather_forecast"],
                          "cycle_time": pd.to_datetime([day], utc=True), "value": [1.0]})
    return DuckDBCache(path, threads=1).save(identity, b"{}", frame, {})


def test_assemble_retains_snapshot_and_protects_existing_output(tmp_path):
    """A later cycle is imported once, and an existing copy is never replaced."""
    base, weather = tmp_path / "base.duckdb", tmp_path / "weather.duckdb"
    _write_weather(base, "2026-08-30")
    snapshot = _write_weather(weather, "2026-09-01")
    destination, report = tmp_path / "combined.duckdb", tmp_path / "report.json"
    evidence = assemble(base, weather, destination, report)
    assert evidence["snapshot_ids"] == [snapshot["snapshot_id"]]
    assert report.exists()
    with pytest.raises(ValueError, match="outputs must not exist"):
        assemble(base, weather, destination, tmp_path / "other.json")


def test_assemble_rejects_overlapping_cycles(tmp_path):
    """Overlapping runs cannot silently introduce duplicate weather values."""
    base, weather = tmp_path / "base.duckdb", tmp_path / "weather.duckdb"
    _write_weather(base, "2026-09-01")
    _write_weather(weather, "2026-09-01")
    destination = tmp_path / "combined.duckdb"
    with pytest.raises(ValueError, match="overlap"):
        assemble(base, weather, destination, tmp_path / "report.json")
    assert not destination.exists()
