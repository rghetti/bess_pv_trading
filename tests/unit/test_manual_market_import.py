"""Offline workbook staging and future transactional merge/postcheck contracts."""

from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.manual_market_import import (
    _novel_records, check_market_history, merge_market_workbooks, stage_market_workbooks,
)
from bess_pv_trading.data.terna_workbooks import parse_terna_workbook
from test_terna_workbooks import xlsx_bytes, workbook_case


ROOT = Path(__file__).resolve().parents[2]


def configuration(tmp_path):
    """Use separate isolated source, staging and live paths with a short DST-safe window."""
    config = load_config(ROOT / "configs/api_experiments/manual_market_workbooks.yaml")
    settings = config["manual_market_import"]
    settings.update(source_directory=str(tmp_path / "sources"), stage_directory=str(tmp_path / "staging"),
        target_database=str(tmp_path / "live.duckdb"), date_start="2025-09-30", date_end_exclusive="2025-10-03",
        check_report_path=str(tmp_path / "postcheck.json"),
        datasets=[{"provider": "gme", "dataset": "day_ahead_prices"}])
    Path(settings["source_directory"]).mkdir()
    return config


def workbook(path, first, stop, minutes):
    """Generate native price values, including negative and zero observations."""
    rows = [["Data", "Ora", "Periodo", "€/MWh"]] if minutes == 15 else [["Data", "Ora", "€/MWh"]]
    for day in pd.date_range(first, stop, inclusive="left"):
        times = pd.date_range(day.tz_localize("Europe/Rome"),
            (day+pd.Timedelta(days=1)).tz_localize("Europe/Rome"), freq=f"{minutes}min", inclusive="left")
        for index, stamp in enumerate(times):
            values = [stamp.strftime("%d/%m/%Y"), index*minutes//60+1]
            if minutes == 15:
                values.append(index+1)
            values.append(str(index-2).replace(".", ","))
            rows.append(values)
    path.write_bytes(xlsx_bytes(rows))


def stage_case(tmp_path):
    """Exercise native MGP resolution transition and an exact overlapping export day."""
    config = configuration(tmp_path)
    source = Path(config["manual_market_import"]["source_directory"])
    workbook(source/"20250930_20250930_MGP_PrezziZonali_Nord.xlsx", "2025-09-30", "2025-10-01", 60)
    workbook(source/"20251001_20251001_MGP_PrezziZonali_Nord.xlsx", "2025-10-01", "2025-10-02", 15)
    workbook(source/"20251001_20251002_MGP_PrezziZonali_Nord.xlsx", "2025-10-01", "2025-10-03", 15)
    report = stage_market_workbooks(config)
    config["manual_market_import"]["stage_path"] = report["database"]
    return config, report


def seed_existing(config, *, conflict=False):
    """Represent existing native prices without needing the real live API database."""
    settings = config["manual_market_import"]
    stage = DuckDBCache(settings["stage_path"])
    entry = next(entry for entry in stage.entries()
                 if "20250930_20250930" in entry["identity"]["source_file"])
    frame, _ = stage.load(entry["identity"])
    if conflict:
        frame.loc[0, "price_eur_mwh"] += 1
        frame.loc[0, "value"] += 1
    target = DuckDBCache(settings["target_database"])
    identity = {"provider": "gme", "dataset": "day_ahead_prices", "zone": "NORD",
                "start": entry["identity"]["start"], "end": entry["identity"]["end"]}
    target.save(identity, b"older verified source", frame, {})
    return target


def test_stage_does_not_open_live_database_and_preserves_native_values(tmp_path, monkeypatch):
    original = DuckDBCache._connect

    def guard(self, **kwargs):
        if self.directory == tmp_path/"live.duckdb":
            pytest.fail("Staging opened the live downloader database")
        return original(self, **kwargs)

    monkeypatch.setattr(DuckDBCache, "_connect", guard)
    config, report = stage_case(tmp_path)
    assert report["live_database_opened"] is False
    assert report["records"] == 24+96+96
    assert report["identical_duplicate_rows_removed"] == 96
    assert not Path(config["manual_market_import"]["target_database"]).exists()
    cache = DuckDBCache(report["database"])
    frames = [cache.load(entry["identity"])[0] for entry in cache.entries()]
    records = pd.concat(frames, ignore_index=True)
    assert not records.duplicated(["provider", "dataset", "zone", "product_start", "product_end"]).any()
    assert set(records.native_interval_minutes) == {15, 60}
    assert records.price_eur_mwh.min() == -2
    assert records.price_eur_mwh.eq(0).any()
    assert records.published_at.isna().all() and not records.historical_feature_eligible.any()
    assert Path(report["import_config"]).is_file()
    assert Path(report["report"]).with_name("NEXT_STEPS.txt").is_file()
    assert report["final_schema_native_parity_verified_snapshots"] == 3
    assert Path(report["report"]).with_name("monthly_coverage.png").is_file()


def test_explicit_merge_reuses_existing_prices_and_is_idempotent(tmp_path):
    config, report = stage_case(tmp_path)
    target = seed_existing(config)
    protected_identity = {"provider": "entsoe", "dataset": "solar_forecast", "zone": "NORD",
                          "start": "2025-09-30T00:00:00Z", "end": "2025-10-01T00:00:00Z"}
    protected = target.save(protected_identity, b"independent forecast source",
                            pd.DataFrame({"value": [123.0]}), {})
    first = merge_market_workbooks(config)
    assert first["new_records"] == 192
    assert first["identical_records_reused"] == 24
    assert first["postcheck"]["status"] == "passed"
    assert first["postcheck"]["staged_value_and_interval_parity"]["status"] == "identical"
    assert first["unrelated_snapshots_preserved"] == 1
    assert target.manifest(protected_identity)["snapshot_id"] == protected["snapshot_id"]
    count = len(target.entries())
    second = merge_market_workbooks(config)
    assert second["new_records"] == 0
    assert len(target.entries()) == count
    assert second["postcheck"]["duplicate_products"] == 0
    assert second["postcheck"]["missing_prices"] == 0


def test_existing_value_conflict_rolls_back_every_staged_snapshot(tmp_path):
    config, _ = stage_case(tmp_path)
    target = seed_existing(config, conflict=True)
    before = target.entries()
    with pytest.raises(ValueError, match="Conflicting"):
        merge_market_workbooks(config)
    assert [entry["manifest"]["snapshot_id"] for entry in target.entries()] == [entry["manifest"]["snapshot_id"] for entry in before]


def test_overlap_conflict_between_workbooks_stops_before_stage_creation(tmp_path):
    config = configuration(tmp_path)
    source = Path(config["manual_market_import"]["source_directory"])
    workbook(source/"20251001_20251001_MGP_PrezziZonali_Nord.xlsx", "2025-10-01", "2025-10-02", 15)
    rows = [["Data", "Ora", "Periodo", "€/MWh"]]
    for day in ("01/10/2025", "02/10/2025"):
        for index in range(96):
            rows.append([day, index//4+1, index+1, str(index-1)])
    (source/"20251001_20251002_MGP_PrezziZonali_Nord.xlsx").write_bytes(xlsx_bytes(rows))
    with pytest.raises(ValueError, match="Conflicting"):
        stage_market_workbooks(config)
    assert not Path(config["manual_market_import"]["stage_directory"]).exists()


def test_terna_clock_change_is_quarantined_without_rounding_source_labels(tmp_path):
    config, _, rows = workbook_case(tmp_path, "autumn_day")
    settings = config["terna_workbook_import"]
    settings["ambiguous_time_policy"] = "quarantine_day"
    # Represent the +1-minute labels observed in the actual monthly export.
    for index, row in enumerate(rows[2:], start=2):
        if "02:" in str(row[0]) and index % 4 >= 2:
            text = str(row[0])
            minute = int(text[14:16])
            row[0] = text[:14] + f"{minute+1:02d}" + text[16:]
    result = parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])
    assert result.empty
    assert len(result.attrs["quarantined_source_rows"]) == 100
    assert any("02:01" in row["Data Riferimento"] for row in result.attrs["quarantined_source_rows"])
    assert result.attrs["excluded_other_zone_records"] == 100


def test_ordinary_off_grid_terna_labels_are_rejected(tmp_path):
    config, _, rows = workbook_case(tmp_path)
    settings = config["terna_workbook_import"]
    settings["ambiguous_time_policy"] = "quarantine_day"
    row = next(row for row in rows[2:] if row[1] == "NORD")
    row[0] = row[0].replace("00:00:00", "00:01:00")
    with pytest.raises(ValueError, match="outside an autumn"):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


def test_different_native_boundaries_never_silently_replace_existing_prices():
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    common = {"provider": ["gme"], "dataset": ["day_ahead_prices"], "zone": ["NORD"],
              "product_start": [start], "price_eur_mwh": [10.0], "unit": ["EUR/MWh"]}
    old = pd.DataFrame({**common, "product_end": [start+pd.Timedelta(hours=1)]})
    new = pd.DataFrame({**common, "product_end": [start+pd.Timedelta(minutes=15)]})
    with pytest.raises(ValueError, match="different durations"):
        _novel_records(old, new)


def test_postcheck_covers_preexisting_history_before_the_staged_files(tmp_path):
    config, _ = stage_case(tmp_path)
    seed_existing(config)
    merge_market_workbooks(config)
    settings = config["manual_market_import"]
    settings["date_start"] = "2025-01-01"
    report = check_market_history(config)
    assert report["status"] == "gaps_or_duplicates"
    assert report["missing_prices"] > 0
    assert report["datasets"][0]["first_start_utc"] == pd.Timestamp("2025-09-30", tz="Europe/Rome").tz_convert("UTC").isoformat()


def test_postcheck_validates_existing_january_ida_history_and_reports_missing_days(tmp_path):
    config = configuration(tmp_path)
    settings = config["manual_market_import"]
    settings.update(date_start="2025-01-01", date_end_exclusive="2025-01-03",
                    datasets=[{"provider": "gme", "dataset": "ida1_prices"}])
    target = DuckDBCache(settings["target_database"])
    for index, day in enumerate(("2025-01-01", "2025-01-02")):
        left = pd.Timestamp(day, tz="Europe/Rome")
        stop = left + pd.Timedelta(days=1)
        starts = pd.date_range(left, stop, freq="15min", inclusive="left").tz_convert("UTC")
        frame = pd.DataFrame({"provider": "gme", "dataset": "ida1_prices", "zone": "NORD",
            "product_start": starts, "product_end": starts+pd.Timedelta(minutes=15),
            "price_eur_mwh": -12.25, "value": -12.25, "unit": "EUR/MWh", "native_interval_minutes": 15})
        identity = {"provider": "gme", "dataset": "ida1_prices", "zone": "NORD",
                    "start": left.isoformat(), "end": stop.isoformat()}
        target.save(identity, b"preexisting January data", frame, {})
        report = check_market_history(config)
        assert report["missing_prices"] == (96 if index == 0 else 0)
    assert report["status"] == "passed"
    assert report["datasets"][0]["unique_products"] == 192


def test_postcheck_detects_native_sql_tampering(tmp_path):
    config, _ = stage_case(tmp_path)
    target = seed_existing(config)
    merge_market_workbooks(config)
    with target._connect(write=True) as connection:
        connection.execute("UPDATE api_records SET price_eur_mwh = price_eur_mwh + 1 WHERE _row_id = 0")
    with pytest.raises(ValueError, match="parity mismatch"):
        check_market_history(config)


def test_postcheck_cli_reports_a_nonzero_exit_for_gaps(monkeypatch, capsys):
    from bess_pv_trading import cli
    monkeypatch.setattr(cli, "dispatch", lambda *args: {"status": "gaps_or_duplicates", "missing_prices": 96})
    assert cli.main(["check-market-history", "--config", str(ROOT/"configs/api_experiments/manual_market_workbooks.yaml")]) == 2
    assert "gaps_or_duplicates" in capsys.readouterr().out


def test_new_configuration_imports_standard_directly():
    config = load_config(ROOT/"configs/api_experiments/manual_market_workbooks.yaml")
    settings = config["manual_market_import"]
    assert settings["target_database"] == config["api"]["database_path"]
    assert settings["stage_path"] is None
    assert settings["terna_ambiguous_time_policy"] == "second_occurrence_minute_marker"
