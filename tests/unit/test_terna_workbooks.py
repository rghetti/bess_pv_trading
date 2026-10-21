"""Terna native-price import, DST coverage, NORD filtering and ex-post alignment."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.terna_workbooks import import_terna_workbooks, parse_terna_workbook
from bess_pv_trading.data.api_catalog import load_api_cache_records
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.domain import DeliveryGrid
from bess_pv_trading.simulation.settlement import load_realized_imbalance_prices

ROOT = Path(__file__).resolve().parents[2]


def workbook_case(tmp_path, name="spring_day", *, encode_second_occurrence=False):
    config = load_config(ROOT / "configs/api_experiments/terna_workbook_import.yaml")
    settings = config["terna_workbook_import"]
    cases = load_config(ROOT / "tests/fixtures/configs/terna_workbook_cases.yaml")
    values = {**cases["spring_day"], **cases[name]}
    start = pd.Timestamp(values["date_start"]).tz_localize(settings["timezone"])
    stop = pd.Timestamp(values["date_end_exclusive"]).tz_localize(settings["timezone"])
    times = pd.date_range(start, stop, freq=f"{settings['native_interval_minutes']}min", inclusive="left")
    rows = [["Terna monthly summary"] * len(settings["columns"]), list(settings["columns"].values())]
    for i, stamp in enumerate(times):
        label = stamp.tz_localize(None)
        if encode_second_occurrence and stamp.fold:
            label += pd.Timedelta(minutes=1)
        for zone in ("SUD", "NORD"):
            price = values["nord_prices"][i % len(values["nord_prices"])] if zone == "NORD" else values["sud_price_eur_mwh"]
            incentive = values["nonzero_incentive_eur_mwh"] if price else 0
            rows.append([label.strftime(settings["date_format"]), zone, values["aggregate_imbalance_mwh"],
                "+", values["exchanges_mwh"], values["foreign_mwh"], price-incentive, incentive, price])
    settings["database_path"] = str(tmp_path / "market.duckdb")
    settings["report_path"] = str(tmp_path / "manifest.json")
    settings["workbooks"] = [{"path": str(tmp_path / "prices.xlsx"),
        "date_start": values["date_start"], "date_end_exclusive": values["date_end_exclusive"]}]
    return config, values, rows


def xlsx_bytes(rows):
    """Build a small value-only worksheet in the same format as provider exports."""
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ET.Element("worksheet", xmlns=namespace)
    sheet = ET.SubElement(root, "sheetData")
    for row_number, values in enumerate(rows, start=1):
        row = ET.SubElement(sheet, "row", r=str(row_number))
        for col, value in enumerate(values):
            cell = ET.SubElement(row, "c", r=f"{chr(ord('A')+col)}{row_number}", t="inlineStr")
            ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(root))
    return buffer.getvalue()


def test_spring_import_keeps_only_nord_and_native_utc_boundaries(tmp_path):
    config, case, rows = workbook_case(tmp_path)
    settings = config["terna_workbook_import"]
    frame = parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])
    assert len(frame) == case["expected_records"]
    assert frame.zone.eq(settings["zone"]).all()
    assert frame.product_start.iloc[0] == pd.Timestamp(case["first_utc"])
    assert frame.product_end.iloc[-1] == pd.Timestamp(case["end_utc_exclusive"])
    assert set(frame.price_eur_mwh) == set(case["nord_prices"])
    assert not frame.historical_feature_eligible.any()
    assert frame.attrs["excluded_other_zone_records"] == case["expected_records"]


def test_ambiguous_autumn_labels_are_rejected_instead_of_guessed(tmp_path):
    config, _, rows = workbook_case(tmp_path, "autumn_day")
    settings = config["terna_workbook_import"]
    settings["ambiguous_time_policy"] = "reject"
    with pytest.raises(Exception, match="[Aa]mbiguous|[Cc]annot infer"):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


@pytest.mark.parametrize("name", ["autumn_day", "autumn_2026"])
def test_second_occurrence_markers_keep_prices_and_sort_both_folds(tmp_path, name):
    config, _, rows = workbook_case(tmp_path, name, encode_second_occurrence=True)
    settings = config["terna_workbook_import"]
    # Distinct prices expose any accidental swapping of the two occurrences.
    for index, row in enumerate(rows[2:]):
        row[6], row[7], row[8] = index - 50, 0, index - 50
    source_prices = {row[0]: float(row[8]) for row in rows[2:] if row[1] == "NORD"}
    # Actual exports interleave 02:00, 02:01, 02:15, 02:16, etc.
    rows[2:] = sorted(rows[2:], key=lambda row: row[0])
    frame = parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])
    assert len(frame) == 100
    assert frame.product_start.is_monotonic_increasing and frame.product_start.is_unique
    assert frame.attrs["decoded_second_occurrence_records"] == 4
    assert not frame.attrs["quarantined_delivery_days"]
    np.testing.assert_array_equal(frame.price_eur_mwh, frame.source_timestamp_text.map(source_prices))
    repeated = frame.loc[frame.product_start.dt.tz_convert(settings["timezone"]).dt.hour.eq(2)]
    assert repeated.source_timestamp_text.str[-8:].tolist() == [
        "02:00:00", "02:15:00", "02:30:00", "02:45:00",
        "02:01:00", "02:16:00", "02:31:00", "02:46:00"]
    assert repeated.local_time_fold.tolist() == [0]*4 + [1]*4
    assert ((frame.product_end-frame.product_start) == pd.Timedelta(minutes=15)).all()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unmarked", "outside_hour"])
def test_invalid_second_occurrence_markers_are_rejected(tmp_path, mutation):
    config, _, rows = workbook_case(tmp_path, "autumn_day", encode_second_occurrence=True)
    settings = config["terna_workbook_import"]
    index = next(i for i, row in enumerate(rows) if row[1] == "NORD" and "02:01:00" in row[0])
    if mutation == "missing":
        rows.pop(index)
    elif mutation == "duplicate":
        rows.append(rows[index].copy())
    elif mutation == "unmarked":
        rows[index][0] = rows[index][0].replace("02:01:00", "02:00:00")
    else:
        rows[index][0] = rows[index][0].replace("02:01:00", "03:01:00")
    with pytest.raises(ValueError, match="exactly once|outside the repeated"):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


def test_spring_skips_missing_hour_and_rejects_invented_local_times(tmp_path):
    config, _, rows = workbook_case(tmp_path)
    settings = config["terna_workbook_import"]
    frame = parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])
    local = frame.product_start.dt.tz_convert(settings["timezone"])
    assert not local.dt.hour.eq(2).any()
    assert local.iloc[7].strftime("%H:%M%z") == "01:45+0100"
    assert local.iloc[8].strftime("%H:%M%z") == "03:00+0200"
    row = next(row for row in rows[2:] if row[1] == "NORD" and "03:00:00" in row[0])
    row[0] = row[0].replace("03:00:00", "02:00:00")
    with pytest.raises(Exception, match="2025-03-30 02:00|nonexistent|does not exist"):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


@pytest.mark.parametrize("name", ["spring_day", "autumn_day"])
def test_duckdb_import_is_idempotent_checksummed_and_loads_for_settlement(tmp_path, name):
    config, case, rows = workbook_case(tmp_path, name, encode_second_occurrence=True)
    settings = config["terna_workbook_import"]
    Path(settings["workbooks"][0]["path"]).write_bytes(xlsx_bytes(rows))
    first = import_terna_workbooks(config)
    second = import_terna_workbooks(config)
    assert first["records"] == (100 if name == "autumn_day" else case["expected_records"])
    assert second["months"][0]["status"] == "cached"
    assert first["months"][0]["snapshot_id"] == second["months"][0]["snapshot_id"]
    frame, _ = load_api_cache_records(settings["database_path"], provider="terna",
        dataset=settings["dataset"], zone=settings["zone"])
    # Manual staging uses ns Parquet clocks; the execution grid uses Python clocks.
    # Identical UTC instants must align across those representations on both DST days.
    cache = DuckDBCache(settings["database_path"])
    entry = cache.entries()[0]
    identity = entry["identity"]
    for column in ("product_start", "product_end"):
        frame[column] = frame[column].dt.as_unit("ns")
    cache.save(identity, xlsx_bytes(rows), frame, entry["manifest"]["metadata"])
    starts = tuple(stamp.to_pydatetime() for stamp in frame.product_start)
    ends = tuple(stamp.to_pydatetime() for stamp in frame.product_end)
    grid = DeliveryGrid(tuple(str(s) for s in starts), starts, ends)
    experiment = {"settlement": {"enabled": True, "database_path": settings["database_path"],
        "provider": "terna", "dataset": settings["dataset"], "zone": settings["zone"],
        "native_interval_minutes": settings["native_interval_minutes"],
        "price_column": "price_eur_mwh", "missing_price_policy": "error"}}
    prices, provenance = load_realized_imbalance_prices(experiment, grid)
    np.testing.assert_allclose(prices.settlement_price_eur_mwh, frame.price_eur_mwh)
    assert provenance["supplied_to_optimizer"] is False
    offset = pd.Timedelta(minutes=case["shifted_minutes"])
    shifted = DeliveryGrid(grid.interval_ids, tuple(s+offset for s in starts), tuple(s+offset for s in ends))
    with pytest.raises(ValueError, match="match|unique"):
        load_realized_imbalance_prices(experiment, shifted)
