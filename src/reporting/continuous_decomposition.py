"""Reconcile continuous cash flows and PV-first physical energy by delivery day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .continuous_outputs import _financial_dispatch
from .plots import pv_power_allocation


AUCTIONS = ("DA", "IDA1", "IDA2", "IDA3")


def continuous_daily_decomposition(config, dispatch, ledger, days_data):
    """Attribute actual ledger trades and physical execution to aligned products."""
    frame = _financial_dispatch(config, dispatch)
    trades = pd.DataFrame(ledger)
    if trades.empty:
        trades = pd.DataFrame(columns=["interval_id", "auction", "quantity_mwh", "cashflow_eur", "price_eur_per_mwh"])
    if not set(trades.auction).issubset(AUCTIONS):
        raise ValueError("Unknown auction in continuous ledger")
    if not set(trades.interval_id).issubset(set(frame.interval_id)):
        raise ValueError("Ledger contains products outside the execution grid")
    rows = []
    for day, daily in frame.groupby("delivery_day", sort=True):
        product_ids = daily.interval_id.to_list()
        product_trades = trades.loc[trades.interval_id.isin(product_ids)].copy()
        by_product = product_trades.groupby("interval_id", sort=False)
        quantities = by_product.quantity_mwh.sum().reindex(product_ids, fill_value=0).to_numpy(float)
        cash = by_product.cashflow_eur.sum().reindex(product_ids, fill_value=0).to_numpy(float)
        if not np.allclose(quantities, daily.net_position_mwh, rtol=1e-10, atol=1e-7):
            raise ValueError(f"Ledger quantities do not reconcile on {day}")
        if not np.allclose(cash, daily.auction_revenue_eur, rtol=1e-10, atol=1e-7):
            raise ValueError(f"Ledger auction cash flow does not reconcile on {day}")
        if len(product_trades) and not np.allclose(product_trades.cashflow_eur,
                product_trades.quantity_mwh * product_trades.price_eur_per_mwh, rtol=1e-10, atol=1e-7):
            raise ValueError(f"Ledger price and quantity do not reconcile on {day}")
        quantity = product_trades.quantity_mwh.to_numpy(dtype=float)
        purchased = float(np.maximum(-quantity, 0).sum())
        sold = float(np.maximum(quantity, 0).sum())
        # Offsets are matched separately by delivery product, not across hours.
        purchases = product_trades.assign(amount=(-product_trades.quantity_mwh).clip(lower=0)).groupby("interval_id").amount.sum()
        sales = product_trades.assign(amount=product_trades.quantity_mwh.clip(lower=0)).groupby("interval_id").amount.sum()
        offsets = float(np.minimum(purchases.reindex(product_ids, fill_value=0), sales.reindex(product_ids, fill_value=0)).sum())
        row = {"delivery_day": day, "purchased_mwh": purchased, "sold_mwh": sold,
               "gross_traded_mwh": purchased + sold, "offsetting_mwh": offsets}
        for auction in AUCTIONS:
            row[f"auction_revenue_{auction.lower()}_eur"] = float(product_trades.loc[product_trades.auction.eq(auction), "cashflow_eur"].sum())
        da = days_data[day]["prices"].get("DA")
        if da is None:
            row.update(da_reference_value_eur=np.nan, inter_auction_spread_revenue_eur=np.nan,
                       da_reference_available=False)
        else:
            price_by_product = dict(zip(days_data[day]["grid"].interval_ids, np.asarray(da, dtype=float)))
            reference = np.asarray([price_by_product[product] for product in product_ids])
            value = float(np.dot(quantities, reference))
            spread = 0.0
            for trade in product_trades.itertuples():
                if trade.auction != "DA":
                    spread += (float(trade.price_eur_per_mwh) - price_by_product[trade.interval_id]) * float(trade.quantity_mwh)
            if not np.isclose(value + spread, cash.sum(), rtol=1e-10, atol=1e-6):
                raise ValueError(f"DA reference and intraday spread do not reconcile on {day}")
            row.update(da_reference_value_eur=value, inter_auction_spread_revenue_eur=spread,
                       da_reference_available=True)
        allocation = pv_power_allocation(daily)
        duration = daily.duration_hours.to_numpy(float)
        pv_battery = allocation.pv_to_battery_mw.to_numpy(float)
        row.update(pv_available_mwh=float(np.dot(daily.pv_ac_mw, duration)),
            pv_exported_mwh=float(np.dot(allocation.pv_to_grid_mw, duration)),
            pv_to_battery_mwh=float(np.dot(pv_battery, duration)),
            pv_curtailed_mwh=float(np.dot(allocation.pv_curtailed_mw, duration)),
            grid_to_battery_mwh=float(np.dot(daily.charge_mw.to_numpy(float) - pv_battery, duration)),
            battery_charge_mwh=float(np.dot(daily.charge_mw, duration)),
            battery_discharge_mwh=float(np.dot(daily.discharge_mw, duration)),
            initial_energy_mwh=float(daily.energy_start_mwh.iloc[0]),
            final_energy_mwh=float(daily.energy_end_mwh.iloc[-1]))
        row["battery_throughput_mwh"] = row["battery_charge_mwh"] + row["battery_discharge_mwh"]
        if not np.isclose(row["pv_available_mwh"], row["pv_exported_mwh"] + row["pv_to_battery_mwh"] + row["pv_curtailed_mwh"], rtol=1e-10, atol=1e-7):
            raise ValueError(f"PV energy allocation does not reconcile on {day}")
        grid_energy = float(np.dot(daily.grid_mw, duration))
        allocated_grid_energy = row["pv_exported_mwh"] + row["battery_discharge_mwh"] - row["grid_to_battery_mwh"]
        if not np.isclose(grid_energy, allocated_grid_energy, rtol=1e-10, atol=1e-7):
            raise ValueError(f"Grid energy allocation does not reconcile on {day}")
        for field in ("auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur", "transaction_fee_eur", "import_fee_eur", "imbalance_penalty_eur", "net_profit_eur"):
            row[field] = float(daily[field].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_continuous_decomposition(config, dispatch, ledger, days_data, destination):
    """Save daily and monthly ledger and energy accounting alongside the replay."""
    daily = continuous_daily_decomposition(config, dispatch, ledger, days_data)
    daily.to_csv(destination / "csv/revenue_energy_daily.csv", index=False)
    daily["delivery_month"] = daily.delivery_day.str[:7]
    sum_columns = [column for column in daily.select_dtypes(include="number").columns
                   if column not in ("initial_energy_mwh", "final_energy_mwh")]
    monthly = daily.groupby("delivery_month", sort=True)[sum_columns].sum(min_count=1)
    for column in ("da_reference_value_eur", "inter_auction_spread_revenue_eur"):
        monthly.loc[daily.groupby("delivery_month").da_reference_available.all().eq(False), column] = np.nan
    monthly["initial_energy_mwh"] = daily.groupby("delivery_month").initial_energy_mwh.first()
    monthly["final_energy_mwh"] = daily.groupby("delivery_month").final_energy_mwh.last()
    monthly.to_csv(destination / "csv/revenue_energy_monthly.csv")
