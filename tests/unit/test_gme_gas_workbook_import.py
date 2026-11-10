"""Verify atomic GME gas imports and retention before source deletion."""

from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.data.duckdb_store import DuckDBCache
from scripts.import_gme_gas_workbooks import merge, parse_gas_workbook, stage
from test_terna_workbooks import xlsx_bytes


def test_gas_workbooks_merge_and_repeat_without_duplicate_days(tmp_path):
    """Validated workbook bytes remain in DuckDB before their files are removed."""
    path = tmp_path / "IGIndex_Giornaliero_2026_Settembre.xlsx"
    raw = xlsx_bytes([["Giorno Gas", "Prezzo (€/MWh)"], ["1/9/2026", "70,08"], ["2/9/2026", "77,14"]])
    path.write_bytes(raw)
    payload, frame, _ = parse_gas_workbook(path)
    assert frame.value.tolist() == [70.08, 77.14]
    assert frame.historical_available_at.isna().all()
    target_path = tmp_path / "market.duckdb"
    target = DuckDBCache(target_path)
    old = frame.iloc[:1].copy()
    old["gas_delivery_day"] = "2026-08-31"
    target.save({"provider": "gme", "dataset": "gas_igi", "zone": "PSV", "seed": 1}, b"seed", old, {})
    stage([path], tmp_path / "stage.duckdb")
    result = merge(tmp_path / "stage.duckdb", target_path, delete_verified=True)
    assert result["inserted_days"] == 2
    assert not path.exists()
    assert len(target.entries(provider="gme", dataset="gas_igi", zone="PSV")) == 2
    path.write_bytes(payload)
    stage([path], tmp_path / "stage_again.duckdb")
    result = merge(tmp_path / "stage_again.duckdb", target_path, delete_verified=False)
    assert result["inserted_days"] == 0
    assert result["identical_days_reused"] == 2
    assert len(target.entries(provider="gme", dataset="gas_igi", zone="PSV")) == 2


def test_conflicting_gas_day_does_not_delete_source(tmp_path):
    """A conflicting existing value rolls back and leaves the workbook intact."""
    path = tmp_path / "IGIndex_Giornaliero_2026_Settembre.xlsx"
    path.write_bytes(xlsx_bytes([["Giorno Gas", "Prezzo (€/MWh)"], ["1/9/2026", "70,08"]]))
    _, frame, _ = parse_gas_workbook(path)
    frame["value"] = 999.0
    target_path = tmp_path / "market.duckdb"
    DuckDBCache(target_path).save({"provider": "gme", "dataset": "gas_igi", "zone": "PSV", "seed": 1},
                                  b"seed", frame, {})
    stage([path], tmp_path / "stage.duckdb")
    with pytest.raises(ValueError, match="Conflicting imported gas prices"):
        merge(tmp_path / "stage.duckdb", target_path, delete_verified=True)
    assert path.is_file()
