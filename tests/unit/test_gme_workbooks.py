"""Native GME price products, DST boundaries and cancelled auctions."""
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.gme import native_price_resolution_minutes
from bess_pv_trading.data.gme_workbooks import GMEWorkbookError, _parse_workbook
from test_terna_workbooks import xlsx_bytes

ROOT = Path(__file__).resolve().parents[2]
CASES = load_config(ROOT / "tests/fixtures/configs/gme_workbook_cases.yaml")


def workbook_case(dataset="ida1_prices", day="regular"):
    """Create native chronological periods, retaining whole-day IDA3 numbering."""
    config = load_config(ROOT / "configs/api_experiments/gme_workbook_import.yaml")
    settings = config["gme_workbook_import"]
    case = CASES["days"][day]
    minutes = native_price_resolution_minutes(dataset, pd.Timestamp(case["start"]).date())
    settings.update(date_start=case["start"], date_end_exclusive=case["stop"])
    item = {"dataset": dataset, "zone": "NORD",
            "native_interval_minutes": minutes, "columns": {"period": "Periodo", "hour": "Ora"}}
    if dataset == "ida3_prices":
        item["eligible_start_clock"] = "12:00"
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
    return effective, rows


@pytest.mark.parametrize("dataset", ["day_ahead_prices", "ida1_prices", "ida2_prices", "ida3_prices"])
@pytest.mark.parametrize("day", ["regular", "spring", "autumn"])
def test_native_periods_keep_dst_and_da_ida_resolutions_separate(dataset, day):
    settings, rows = workbook_case(dataset, day)
    frame = _parse_workbook(xlsx_bytes(rows), "prices.xlsx", dataset, settings)
    key = "da_products" if dataset == "day_ahead_prices" else "ida3_products" if dataset == "ida3_prices" else "ida_products"
    assert len(frame) == CASES["days"][day][key]
    assert frame.native_interval_minutes.eq(settings["native_interval_minutes"]).all()
    assert frame.product_end.iloc[-1] == pd.Timestamp(settings["date_end_exclusive"], tz=settings["timezone"]).tz_convert("UTC")
    if dataset == "ida3_prices":
        assert frame.product_start.iloc[0].tz_convert(settings["timezone"]).hour == 12
    assert frame.price_eur_mwh.iloc[0] == float(str(rows[1][3]).replace(",", "."))
    assert frame.price_eur_mwh.tolist() == [float(str(row[3]).replace(",", ".")) for row in rows[1:]]


def test_partial_price_curve_is_rejected():
    settings, rows = workbook_case()
    rows.pop()
    with pytest.raises(GMEWorkbookError):
        _parse_workbook(xlsx_bytes(rows), "prices.xlsx", "ida1_prices", settings)


def test_explicit_cancellation_is_retained_without_inventing_prices():
    settings, rows = workbook_case()
    settings["date_end_exclusive"] = CASES["canceled_stop"]
    rows.append([pd.Timestamp(CASES["canceled_day"]).strftime(settings["date_format"]),
                 "", "", "", "Asta cancellata"])
    frame = _parse_workbook(xlsx_bytes(rows), "prices.xlsx", "ida1_prices", settings)
    assert frame.attrs["cancelled_delivery_days"] == [CASES["canceled_day"]]
    assert len(frame) == CASES["days"]["regular"]["ida_products"]
