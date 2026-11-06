"""Repeatable DuckDB-only source preparation across revisions and machine paths."""

from pathlib import Path
import shutil

import pandas as pd

from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.price_search_inputs import build_price_search_inputs


def test_pinned_sources_survive_new_revisions_and_database_relocation(tmp_path):
    """A manifest selects the original observations after copying an updated catalog."""
    database = tmp_path / "original.duckdb"
    store = DuckDBCache(database, threads=1)
    identity = {"provider": "gme", "dataset": "day_ahead_prices", "zone": "NORD",
                "start": "2025-01-01", "end": "2025-01-02", "options": {}}
    starts = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    frame = pd.DataFrame({"provider": "gme", "dataset": "day_ahead_prices", "zone": "NORD",
                          "product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
                          "value": [100.0, 110.0]})
    store.save(identity, b"original values", frame, {"source_kind": "public_source"})
    settings = {"database_path": str(database), "database_threads": 1,
                "output_directory": str(tmp_path / "first"), "start": "2025-01-01",
                "end_exclusive": "2025-01-02", "timezone": "Europe/Rome", "sources": [
                    {"name": "prices", "provider": "gme", "zone": "NORD", "datasets": ["day_ahead_prices"],
                     "columns": ["product_start", "product_end", "value"]}
                ]}
    original = build_price_search_inputs({"price_search_inputs": settings})
    manifest = Path(settings["output_directory"]) / "manifest.json"
    # Publish a newer revision, then move the entire data dependency to another machine path.
    revised_frame = frame.assign(value=[200.0, 210.0])
    store.save(identity, b"revised values", revised_frame, {"source_kind": "public_source"})
    copied_database = tmp_path / "relocated.duckdb"
    shutil.copy2(database, copied_database)
    settings.update(database_path=str(copied_database), database_threads=2,
                    output_directory=str(tmp_path / "reproduced"))
    reproduced = build_price_search_inputs({"price_search_inputs": settings}, pinned_manifest=str(manifest))
    result = pd.read_parquet(Path(settings["output_directory"]) / "prices.parquet")
    assert result.value.tolist() == [100.0, 110.0]
    assert reproduced["selection_sha256"] == original["selection_sha256"]
    assert reproduced["tables"]["prices"]["snapshot_sources"] == original["tables"]["prices"]["snapshot_sources"]
    assert reproduced["tables"]["prices"]["selected_row_sha256"] == original["tables"]["prices"]["selected_row_sha256"]
    assert reproduced["farm_files_required"] is False
    assert reproduced["tables"]["prices"]["native_interval_minutes"] == {"15.0": 2}
