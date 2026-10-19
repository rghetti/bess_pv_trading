"""Terna native-price import, DST coverage, NORD filtering and ex-post alignment."""
from copy import deepcopy
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
from bess_pv_trading.domain import DeliveryGrid
from bess_pv_trading.simulation.settlement import load_realized_imbalance_prices

ROOT = Path(__file__).resolve().parents[2]


def workbook_case(tmp_path, name="spring_day"):
    config = load_config(ROOT / "configs/api_experiments/terna_workbook_import.yaml")
    settings = config["terna_workbook_import"]
    cases = load_config(ROOT / "tests/fixtures/configs/terna_workbook_cases.yaml")
    values = {**cases["spring_day"], **cases[name]}
    start = pd.Timestamp(values["date_start"]).tz_localize(settings["timezone"])
    stop = pd.Timestamp(values["date_end_exclusive"]).tz_localize(settings["timezone"])
    times = pd.date_range(start, stop, freq=f"{settings['native_interval_minutes']}min", inclusive="left")
    rows = [["Terna monthly summary"] * len(settings["columns"]), list(settings["columns"].values())]
    for i, stamp in enumerate(times):
        for zone in ("SUD", "NORD"):
            price = values["nord_prices"][i % len(values["nord_prices"])] if zone == "NORD" else values["sud_price_eur_mwh"]
            incentive = values["nonzero_incentive_eur_mwh"] if price else 0
            rows.append([stamp.strftime(settings["date_format"]), zone, values["aggregate_imbalance_mwh"],
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


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite", "wrong_components"])
def test_terna_import_rejects_gaps_duplicates_and_bad_prices(tmp_path, damage):
    config, case, rows = workbook_case(tmp_path)
    settings = config["terna_workbook_import"]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif damage == "nonfinite":
        rows[-1][-1] = case["unparseable_price"]
    else:
        rows[-1][-1] = case["sud_price_eur_mwh"]
    with pytest.raises(ValueError):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


def test_ambiguous_autumn_labels_are_rejected_instead_of_guessed(tmp_path):
    config, _, rows = workbook_case(tmp_path, "autumn_day")
    settings = config["terna_workbook_import"]
    with pytest.raises(Exception, match="[Aa]mbiguous|[Cc]annot infer"):
        parse_terna_workbook(xlsx_bytes(rows), settings, settings["workbooks"][0])


def test_duckdb_import_is_idempotent_checksummed_and_loads_for_settlement(tmp_path):
    config, case, rows = workbook_case(tmp_path)
    settings = config["terna_workbook_import"]
    Path(settings["workbooks"][0]["path"]).write_bytes(xlsx_bytes(rows))
    first = import_terna_workbooks(config)
    second = import_terna_workbooks(config)
    assert first["records"] == case["expected_records"]
    assert second["months"][0]["status"] == "cached"
    assert first["months"][0]["snapshot_id"] == second["months"][0]["snapshot_id"]
    frame, _ = load_api_cache_records(settings["database_path"], provider="terna",
        dataset=settings["dataset"], zone=settings["zone"])
    starts, ends = tuple(frame.product_start), tuple(frame.product_end)
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
