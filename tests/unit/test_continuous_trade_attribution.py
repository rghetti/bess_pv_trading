"""Accounting and auction-gate checks for immutable continuous trade reports."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.reporting.continuous_trade_attribution import _financial_daily, _native_products


def _case():
    """Create a quarter-hour ledger with four actual configured submission gates."""
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="15min")
    dispatch = pd.DataFrame({"interval_id": [value.isoformat() for value in starts],
        "start": starts, "delivery_day": "2026-01-01"})
    gates = {"DA": pd.Timestamp("2025-12-31T10:55:00Z"),
        "IDA1": pd.Timestamp("2025-12-31T13:55:00Z"),
        "IDA2": pd.Timestamp("2025-12-31T20:55:00Z"),
        "IDA3": pd.Timestamp("2026-01-01T08:55:00Z")}
    by_auction = {"DA": (range(4), [1.0, -0.5, 0.25, 0.5], [100.0, 101.0, 102.0, 103.0]),
        "IDA1": (range(4), [-0.25, 0.25, 0.0, 0.5], [105.0, 103.0, 104.0, 106.0]),
        "IDA2": (range(4), [0.0, 0.25, -0.5, 0.0], [99.0, 102.0, 98.0, 104.0]),
        "IDA3": (range(44, 48), [0.1, -0.1, 0.2, -0.2], [110.0, 109.0, 111.0, 108.0])}
    rows = []
    da_prices = {}
    for auction, (indices, quantities, prices) in by_auction.items():
        for index, quantity, price in zip(indices, quantities, prices):
            start = starts[index]
            rows.append({"auction": auction, "delivery_day": "2026-01-01",
                "interval_id": start.isoformat(), "quantity_mwh": quantity,
                "price_eur_per_mwh": price, "cashflow_eur": quantity * price,
                "submission_time": gates[auction]})
            if auction == "DA":
                da_prices[start] = price
    config = {"market": {"timezone": "Europe/Rome", "decision_buffer_minutes": 5,
        "gates": {"DA": "12:00", "IDA1": "15:00", "IDA2": "22:00", "IDA3": "10:00"},
        "gate_day_offsets": {"DA": -1, "IDA1": -1, "IDA2": -1, "IDA3": 0},
        "delivery_start_times": {"IDA3": "12:00"}}, "execution": {}}
    for index, start in enumerate(starts):
        da_prices.setdefault(start, 100.0 + index % 12)
    return config, dispatch, pd.DataFrame(rows), da_prices


def test_native_product_revenue_decomposition_and_actual_gates_reconcile():
    """Actual gate timestamps and hourly/QH accounting preserve exact signed cash."""
    config, dispatch, trades, da_prices = _case()
    products, positions, _ = _native_products(trades, dispatch, config, da_prices)
    assert products.da_reference_available.all()
    assert set(products.loc[products.auction.eq("DA"), "native_resolution_minutes"]) == {15}
    assert set(products.loc[products.auction.ne("DA"), "native_resolution_minutes"]) == {15}
    assert np.isclose(products.trade_cashflow_eur.sum(), trades.cashflow_eur.sum())
    da_value = products.loc[products.auction.eq("DA"), "final_position_da_value_eur"].sum()
    spreads = products.loc[products.auction.ne("DA"), "da_referenced_spread_eur"].sum()
    assert np.isclose(da_value + spreads, products.trade_cashflow_eur.sum())
    assert positions.final_position_mwh.notna().all()
    assert (products.position_increasing_mwh >= 0).all()
    assert (products.position_reducing_mwh >= 0).all()
    assert products.loc[products.auction.eq("IDA3"), "product_start_utc"].min() >= pd.Timestamp("2026-01-01T11:00:00Z")


def test_native_product_report_rejects_trade_outside_its_gate():
    """A valid cash quantity cannot conceal a submission at the wrong gate."""
    config, dispatch, trades, da_prices = _case()
    trades.loc[0, "submission_time"] += timedelta(minutes=15)
    with pytest.raises(ValueError, match="actual eligible auction gate"):
        _native_products(trades, dispatch, config, da_prices)


def test_missing_da_reference_remains_explicitly_unavailable():
    """No missing DA curve is replaced with an inferred price."""
    config, dispatch, trades, _ = _case()
    trades = trades.loc[~trades.auction.eq("DA")].copy()
    products, _, _ = _native_products(trades, dispatch, config, {})
    assert not products.da_reference_available.any()
    assert products.da_referenced_spread_eur.isna().all()


def test_hourly_da_products_aggregate_exactly_while_intraday_stays_quarter_hourly():
    """Equal hourly DA prices aggregate positions without losing signed cash flow."""
    config, dispatch, trades, da_prices = _case()
    da_mask = trades.auction.eq("DA")
    trades.loc[da_mask, "price_eur_per_mwh"] = 100.0
    trades.loc[da_mask, "cashflow_eur"] = trades.loc[da_mask, "quantity_mwh"] * 100.0
    da_prices = {stamp: 100.0 for stamp in da_prices}
    products, _, _ = _native_products(trades, dispatch, config, da_prices)
    da = products.loc[products.auction.eq("DA")]
    ida = products.loc[products.auction.eq("IDA1")]
    assert da.participated.sum() == 1
    assert set(da.native_resolution_minutes) == {60}
    assert len(ida) == 4
    assert set(ida.native_resolution_minutes) == {15}
    assert np.isclose(products.trade_cashflow_eur.sum(), trades.cashflow_eur.sum())


def test_realized_profit_reconciles_to_auction_settlement_and_costs(tmp_path):
    """Daily realized net profit includes settlement, degradation, fees and penalties."""
    strategy = tmp_path / "strategies" / "DA" / "csv"
    strategy.mkdir(parents=True)
    pd.DataFrame([{"delivery_day": "2026-01-01", "auction_revenue_eur": 100.0,
        "imbalance_cashflow_eur": -10.0, "degradation_eur": 5.0,
        "import_fee_eur": 2.0, "transaction_fee_eur": 1.0,
        "imbalance_penalty_eur": 3.0, "net_profit_eur": 79.0}]).to_csv(
            strategy / "realized_dispatch.csv", index=False)
    config = tmp_path / "strategies" / "DA" / "json"
    config.mkdir()
    (config / "study_config.json").write_text('{"execution": {}}')
    daily = _financial_daily(tmp_path / "strategies" / "DA", "strategies")
    assert daily.net_profit_eur.sum() == pytest.approx(79.0)
    pd.read_csv(strategy / "realized_dispatch.csv").assign(net_profit_eur=80.0).to_csv(
        strategy / "realized_dispatch.csv", index=False)
    with pytest.raises(ValueError, match="does not reconcile"):
        _financial_daily(tmp_path / "strategies" / "DA", "strategies")
