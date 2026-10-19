"""Native GME workbook periods, cancellations and transactional replacement."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.api import DataAPIError
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.data.gme import native_price_resolution_minutes
from bess_pv_trading.data.gme_workbooks import GMEWorkbookError, _parse_workbook, import_gme_workbooks
from test_terna_workbooks import xlsx_bytes

ROOT = Path(__file__).resolve().parents[2]
CASES = load_config(ROOT / "tests/fixtures/configs/gme_workbook_cases.yaml")


def workbook_case(tmp_path, dataset="ida1_prices", day="regular"):
    """Create native chronological periods, retaining whole-day IDA3 numbering."""
    config = load_config(ROOT / "configs/api_experiments/gme_workbook_import.yaml")
    settings = config["gme_workbook_import"]
    case = CASES["days"][day]
    minutes = native_price_resolution_minutes(dataset, pd.Timestamp(case["start"]).date())
    settings.update(database_path=str(tmp_path / "prices.duckdb"),
        retired_snapshot_directory=str(tmp_path / "archive"), report_path=str(tmp_path / "report.json"),
        date_start=case["start"], date_end_exclusive=case["stop"])
    item = {"dataset": dataset, "zone": "NORD", "path": str(tmp_path / f"{dataset}.xlsx"),
            "native_interval_minutes": minutes, "columns": {"period": "Periodo", "hour": "Ora"}}
    if dataset == "ida3_prices":
        item["eligible_start_clock"] = "12:00"
    settings["workbooks"] = [item]
    effective = {**settings, **item, "columns": {**settings["columns"], **item["columns"]}}
    times = pd.date_range(pd.Timestamp(case["start"], tz=settings["timezone"]),
        pd.Timestamp(case["stop"], tz=settings["timezone"]), freq=f"{minutes}min", inclusive="left")
    rows = [["Data", "Ora", "Periodo", "€/MWh", "Note"]]
    for index, stamp in enumerate(times):
        if dataset == "ida3_prices" and stamp.strftime("%H:%M") < item["eligible_start_clock"]:
            continue
        price = CASES["prices_eur_mwh"][index % len(CASES["prices_eur_mwh"])]
        rows.append([stamp.strftime(settings["date_format"]), index*minutes//60+1,
                     index+1, str(price).replace(".", ","), ""])
    return config, effective, rows


@pytest.mark.parametrize("dataset", ["day_ahead_prices", "ida1_prices", "ida2_prices", "ida3_prices"])
@pytest.mark.parametrize("day", list(CASES["days"]))
def test_native_periods_keep_dst_and_da_ida_resolutions_separate(tmp_path, dataset, day):
    _, settings, rows = workbook_case(tmp_path, dataset, day)
    frame = _parse_workbook(xlsx_bytes(rows), "prices.xlsx", dataset, settings)
    key = "da_products" if dataset == "day_ahead_prices" else "ida3_products" if dataset == "ida3_prices" else "ida_products"
    assert len(frame) == CASES["days"][day][key]
    assert frame.native_interval_minutes.eq(settings["native_interval_minutes"]).all()
    assert frame.product_end.iloc[-1] == pd.Timestamp(settings["date_end_exclusive"], tz=settings["timezone"]).tz_convert("UTC")
    if dataset == "ida3_prices":
        assert frame.product_start.iloc[0].tz_convert(settings["timezone"]).hour == 12
    assert frame.price_eur_mwh.iloc[0] == float(str(rows[1][3]).replace(",", "."))


@pytest.mark.parametrize("damage", ["hourly", "wrong_period_column", "missing_quarter", "wrong_year"])
def test_aggregates_wrong_dates_and_partial_days_are_rejected(tmp_path, damage):
    _, settings, rows = workbook_case(tmp_path)
    if damage == "hourly":
        settings["native_interval_minutes"] = 60
    elif damage == "wrong_period_column":
        settings["columns"]["period"] = "Ora"
    elif damage == "missing_quarter":
        rows.pop()
    else:
        rows[1][0] = pd.Timestamp(CASES["wrong_year"]).strftime(settings["date_format"])
    with pytest.raises(GMEWorkbookError):
        _parse_workbook(xlsx_bytes(rows), "prices.xlsx", "ida1_prices", settings)


def test_explicit_cancellation_is_retained_without_inventing_prices(tmp_path):
    _, settings, rows = workbook_case(tmp_path)
    settings["date_end_exclusive"] = CASES["canceled_stop"]
    rows.append([pd.Timestamp(CASES["canceled_day"]).strftime(settings["date_format"]),
                 "", "", "", "Asta cancellata"])
    frame = _parse_workbook(xlsx_bytes(rows), "prices.xlsx", "ida1_prices", settings)
    assert frame.attrs["cancelled_delivery_days"] == [CASES["canceled_day"]]
    assert len(frame) == CASES["days"]["regular"]["ida_products"]


def replacement_case(tmp_path):
    """Seed a legacy hourly IDA snapshot beside valid native MGP data."""
    config, effective, rows = workbook_case(tmp_path)
    item = config["gme_workbook_import"]["workbooks"][0]
    payload = xlsx_bytes(rows)
    Path(item["path"]).write_bytes(payload)
    frame = _parse_workbook(payload, Path(item["path"]).name, item["dataset"], effective)
    identity = {"provider": "gme", "dataset": item["dataset"], "zone": item["zone"],
        "source_kind": "manual_gme_xlsx",
        "start": frame.product_start.iloc[0].isoformat(), "end": frame.product_end.iloc[-1].isoformat()}
    cache = DuckDBCache(config["gme_workbook_import"]["database_path"])
    old = frame.iloc[::4].copy()
    old["native_interval_minutes"], old["resolution"] = 60, "PT60M"
    old["product_end"] = old.product_start + pd.Timedelta(hours=1)
    metadata = {"source_kind": "manual_gme_xlsx", "source_file": "old_prices.xlsx",
                "native_interval_minutes": 60}
    old_manifest = cache.save(identity, payload, old, metadata)
    da_identity = {**identity, "dataset": "day_ahead_prices"}
    da = old.assign(dataset="day_ahead_prices", auction="DA")
    da_manifest = cache.save(da_identity, payload, da, metadata)
    return config, cache, identity, old_manifest, da_identity, da_manifest


def test_replacement_removes_hourly_ida_from_all_tables_and_preserves_mgp(tmp_path):
    config, cache, identity, old, da_identity, da = replacement_case(tmp_path)
    result = import_gme_workbooks(config)
    assert len(result["retired_hourly_ida_snapshots"]) == 1
    archive = Path(result["retired_hourly_ida_snapshots"][0]["archive"])
    assert hashlib.sha256((archive / "records.parquet").read_bytes()).hexdigest() == old["normalized_sha256"]
    assert cache.manifest(da_identity)["snapshot_id"] == da["snapshot_id"]
    frame, new = cache.load(identity)
    assert frame.native_interval_minutes.eq(15).all()
    with duckdb.connect(str(cache.directory), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM api_snapshots WHERE snapshot_id = ?", [old["snapshot_id"]]).fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM api_records WHERE _snapshot_id = ?", [old["snapshot_id"]]).fetchone()[0] == 0
    repeated = import_gme_workbooks(config)
    assert repeated["datasets"][0]["status"] == "cached"
    assert not repeated["retired_hourly_ida_snapshots"]
    assert cache.manifest(identity)["snapshot_id"] == new["snapshot_id"]


def test_invalid_last_workbook_and_failed_archive_leave_database_unchanged(tmp_path):
    config, cache, identity, old, _, _ = replacement_case(tmp_path)
    invalid = deepcopy(config)
    item = deepcopy(invalid["gme_workbook_import"]["workbooks"][0])
    item["path"] = str(tmp_path / "not_here.xlsx")
    invalid["gme_workbook_import"]["workbooks"].append(item)
    with pytest.raises(FileNotFoundError):
        import_gme_workbooks(invalid)
    assert cache.manifest(identity)["snapshot_id"] == old["snapshot_id"]
    blocked = tmp_path / "archive_file"
    blocked.write_text("A file cannot serve as the archive directory")
    config["gme_workbook_import"]["retired_snapshot_directory"] = str(blocked)
    with pytest.raises(DataAPIError):
        import_gme_workbooks(config)
    assert cache.manifest(identity)["snapshot_id"] == old["snapshot_id"]
    assert len(cache.snapshots(identity)) == 1
