"""Ensure later-auction training handles documented cancellations without inventing gaps."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_cancellation_features import build_cancellation_aware_base, cancellation_days
from bess_pv_trading.forecasting.studies.price_feature_data import _base_matrix


def _histories(settings):
    """Create complete quarter-hour price curves with two IDA1 gaps."""
    histories = {}
    for name, auction in settings["auctions"].items():
        rows = []
        for day in pd.date_range("2025-11-20", "2025-12-18"):
            label = day.date().isoformat()
            if name == "IDA1" and label in {"2025-12-10", "2025-12-14"}:
                continue
            start = pd.Timestamp(f"{label} {auction['delivery_start_hour']:02d}:00", tz="Europe/Rome")
            periods = (auction["delivery_end_hour"] - auction["delivery_start_hour"]) * 4
            starts = pd.date_range(start, periods=periods, freq="15min").tz_convert("UTC").as_unit("ns")
            available_day = day + pd.Timedelta(days=auction["publication_day_offset"])
            published = pd.Timestamp(f"{available_day.date()} {auction['publication_time']}", tz="Europe/Rome").tz_convert("UTC")
            level = {"DA": 100., "IDA1": 110., "IDA2": 120., "IDA3": 130.}[name]
            rows.append(pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
                "price_eur_per_mwh": level, "available_at": published}))
        histories[name] = pd.concat(rows, ignore_index=True)
    return histories


def test_documented_cancellation_falls_back_for_later_markets_and_lags():
    """Keep IDA2/3 targets and subsequent own-profile and other-market lags."""
    root = Path(__file__).resolve().parents[2]
    settings = load_config(root / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    histories = _histories(settings)
    cancellations = {"DA": set(), "IDA1": {"2025-12-10"}, "IDA2": set(), "IDA3": set()}
    for market, target_day, expected_anchor in [("IDA2", "2025-12-10", 100.),
                                                ("IDA3", "2025-12-10", 120.),
                                                ("IDA1", "2025-12-11", 100.),
                                                ("DA", "2025-12-11", 0.)]:
        history = histories[market]
        mask = pd.DatetimeIndex(history.product_start).tz_convert("Europe/Rome").strftime("%Y-%m-%d") == target_day
        starts = pd.DatetimeIndex(history.loc[mask, "product_start"]).as_unit("ns")
        group = {"market": market, "native_minutes": 15}
        matrix, names, anchor, _, _, _ = build_cancellation_aware_base(
            _base_matrix, histories, starts, settings, group, cancellations, 14)
        assert np.isfinite(matrix).all(), market
        assert np.all(anchor == expected_anchor)
        if market in ("IDA2", "IDA3"):
            assert np.all(matrix[:, names.index("earlier_IDA1_cancelled")] == 1)
            assert np.all(matrix[:, names.index("cleared_IDA1")] == 100.)
        if market == "IDA1":
            assert np.all(matrix[:, names.index("own_curve_1d_cancelled")] == 1)
        if market == "DA":
            assert np.all(matrix[:, names.index("past_IDA1_24h_cancelled")] == 1)


def test_unexplained_gap_is_not_treated_as_cancelled():
    """A missing IDA1 day without a notice must remain missing for IDA2."""
    root = Path(__file__).resolve().parents[2]
    settings = load_config(root / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    histories = _histories(settings)
    history = histories["IDA2"]
    mask = pd.DatetimeIndex(history.product_start).tz_convert("Europe/Rome").strftime("%Y-%m-%d") == "2025-12-14"
    starts = pd.DatetimeIndex(history.loc[mask, "product_start"]).as_unit("ns")
    cancellations = {name: set() for name in settings["auctions"]}
    matrix, names, anchor, _, _, _ = build_cancellation_aware_base(
        _base_matrix, histories, starts, settings, {"market": "IDA2", "native_minutes": 15}, cancellations, 14)
    assert np.isnan(anchor).all()
    assert np.all(matrix[:, names.index("earlier_IDA1_cancelled")] == 0)


def test_consecutive_cancellations_fall_back_to_da():
    """IDA3 uses DA when both intervening auctions were documented cancelled."""
    root = Path(__file__).resolve().parents[2]
    settings = load_config(root / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    histories = _histories(settings)
    ida2_days = pd.DatetimeIndex(histories["IDA2"].product_start).tz_convert("Europe/Rome").strftime("%Y-%m-%d")
    histories["IDA2"] = histories["IDA2"].loc[ida2_days != "2025-12-10"].reset_index(drop=True)
    history = histories["IDA3"]
    mask = pd.DatetimeIndex(history.product_start).tz_convert("Europe/Rome").strftime("%Y-%m-%d") == "2025-12-10"
    starts = pd.DatetimeIndex(history.loc[mask, "product_start"]).as_unit("ns")
    cancellations = {"DA": set(), "IDA1": {"2025-12-10"}, "IDA2": {"2025-12-10"}, "IDA3": set()}
    matrix, names, anchor, _, _, _ = build_cancellation_aware_base(
        _base_matrix, histories, starts, settings, {"market": "IDA3", "native_minutes": 15}, cancellations, 14)
    assert np.isfinite(matrix).all()
    assert np.all(anchor == 100.)
    assert np.all(matrix[:, names.index("earlier_IDA2_cancelled")] == 1)


def test_cancellation_notice_conflicting_with_prices_is_rejected():
    """A conflicting snapshot is not silently rewritten as a cancellation."""
    auctions = {"IDA1": {"dataset": "ida1_prices"}}
    frame = pd.DataFrame({"dataset": ["ida1_prices"], "product_start": [pd.Timestamp("2025-12-10T12:00Z")]})
    records = [{"dataset": "ida1_prices", "delivery_day": "2025-12-10"}]
    with pytest.raises(ValueError, match="conflicts"):
        cancellation_days(records, auctions, frame, "Europe/Rome")
