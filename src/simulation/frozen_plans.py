"""Realized profit from auction snapshots without later planning or forecast updates."""
from __future__ import annotations

import ast
from dataclasses import asdict
from time import perf_counter

import numpy as np
import pandas as pd

from ..domain import PhysicalState
from ..optimization.degradation import degradation_settings_from_config
from ..transaction_fees import account_trade_fees, transaction_fees_from_config
from .replay import execute_interval
from .settlement import settle_realized_dispatch


def plot_frozen_plan_comparison(summaries, destination):
    """Show whole-day frozen profits and signed gains from later reoptimization."""
    if summaries.empty or summaries.reoptimization_gain_eur.isna().any():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"Freeze after {auction}" for auction in summaries.frozen_after_auction]
    gains = summaries.reoptimization_gain_eur.to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    bars = axes[0].bar(labels, summaries.frozen_plan_profit_eur, color="tab:blue")
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].axhline(summaries.reoptimized_reference_profit_eur.iloc[0],
        color="black", linestyle="--", label="Reference with all reoptimizations")
    axes[0].set_ylabel("Whole-day net profit (EUR)")
    axes[0].legend(loc="lower left")
    bars = axes[1].bar(labels, gains, color=np.where(gains >= 0, "tab:green", "tab:red"))
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Reference minus frozen profit (EUR)")
    bars = axes[2].bar(labels, summaries.final_energy_mwh, color="tab:purple")
    axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].axhline(summaries.reference_final_energy_mwh.iloc[0], color="black", linestyle="--",
                    label="Reference final stored energy")
    axes[2].legend(loc="lower left")
    axes[2].set_ylabel("Final stored energy (MWh)")
    axes[2].set_xlabel("Last auction whose plan is allowed to change")
    axes[0].set_title("Value of later auction reoptimization with the same physical feedback")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.margins(y=0.15)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return True


def _actual_boundary(config, grid, dispatch, first):
    """Carry actual state and aging history; do not reset to forecast snapshot SOC."""
    prefix = dispatch.iloc[:first]
    segments = config["state"].get("cycle_segment_energy_mwh")
    if first and "cycle_segment_energy_end_mwh" in prefix:
        segments = prefix.iloc[-1].cycle_segment_energy_end_mwh
        if isinstance(segments, str):
            segments = ast.literal_eval(segments)
    return PhysicalState(
        timestamp=grid.starts[first],
        energy_mwh=float(dispatch.iloc[first].energy_start_mwh),
        cumulative_throughput_mwh=float(prefix.throughput_mwh.sum()),
        cumulative_degradation_eur=float(prefix.degradation_eur.sum()),
        origin=("frozen_plan_carried_actual_boundary" if first else
                config["state"].get("origin", "known_delivery_start_assumption")),
        cycle_segment_energy_mwh=None if segments is None else tuple(segments))


def simulate_frozen_auction_plans(config, grid, plant, schedule, actual_prices,
                                  plans, dispatch, ledger, reference_summary):
    """Freeze every snapshot except the last active auction and execute its suffix.

    Already submitted orders and executed history are identical to the reference.
    Future orders are taken from the frozen snapshot, submitted at their original
    gates, and priced at actual clearing prices. The same feedback and financial
    helpers handle measured PV, SOC protection, executed wear and gross trade fees.
    No solver, forecast provider, API, or database is called by this simulation.
    """
    if len(schedule) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    dispatch = dispatch.reset_index(drop=True)
    if list(dispatch.interval_id) != list(grid.interval_ids):
        raise ValueError("Frozen-plan reference dispatch must match the complete delivery grid")
    if dispatch.interval_id.duplicated().any() or ledger.duplicated(["auction", "interval_id"]).any():
        raise ValueError("Frozen-plan reference dispatch and ledger require unique product keys")
    required = {"interval_id", "interval_start_utc", "interval_end_utc", "optimizer_cutoff_utc",
                "charge_mw", "discharge_mw", "curtailment_mw", "snapshot_scenario_count",
                "planned_net_position_mwh", *(f"planned_trade_{event.auction}_mwh" for event in schedule)}
    if not required.issubset(plans.columns):
        raise ValueError("Frozen-plan baselines require complete auction trade snapshots; rerun the reference experiment")
    if not plans.snapshot_scenario_count.eq(1).all():
        raise ValueError("Frozen-plan baselines require deterministic snapshot plans, not a selected stochastic leaf")
    interval_index = {product: index for index, product in enumerate(grid.interval_ids)}
    reference_trades = ledger.set_index(["auction", "interval_id"])
    aging = degradation_settings_from_config(config)
    fees = transaction_fees_from_config(config)
    policy = config.get("execution", {}).get("dispatch_policy", "follow_battery_plan")
    import_fee = float(config["trading"]["grid_import_fee_eur_per_mwh"])
    priced = reference_summary["imbalance_settlement_status"] == "priced"
    positive_prices = dispatch.imbalance_price_eur_mwh.to_numpy(float) if priced else None
    negative_prices = (dispatch.imbalance_negative_price_eur_mwh.to_numpy(float)
        if priced and "imbalance_negative_price_eur_mwh" in dispatch else positive_prices)
    reference_profit = reference_summary.get("realized_profit_after_imbalance_eur")
    summaries, trajectories, all_trades = [], [], []
    selected = config.get("reoptimization_baseline", {}).get("freeze_after_auctions")

    for frozen_index, event in enumerate(schedule[:-1]):
        if selected is not None and event.auction not in selected:
            continue
        started = perf_counter()
        snapshot = plans.loc[plans.auction == event.auction].copy()
        snapshot = snapshot.sort_values("interval_start_utc")
        if snapshot.empty or snapshot.interval_id.duplicated().any():
            raise ValueError(f"Frozen-plan snapshot for {event.auction} is missing or duplicated")
        first = interval_index[snapshot.interval_id.iloc[0]]
        if list(snapshot.interval_id) != list(grid.interval_ids[first:]):
            raise ValueError(f"Frozen-plan snapshot for {event.auction} must cover the entire residual suffix")
        if (not pd.DatetimeIndex(pd.to_datetime(snapshot.interval_start_utc, utc=True)).equals(
                pd.DatetimeIndex(grid.starts[first:]))
            or not pd.DatetimeIndex(pd.to_datetime(snapshot.interval_end_utc, utc=True)).equals(
                pd.DatetimeIndex(grid.ends[first:]))):
            raise ValueError("Frozen-plan snapshot interval boundaries do not match delivery")
        snapshot_by_product = snapshot.set_index("interval_id")
        rows = []
        for auction_index, auction_event in enumerate(schedule):
            auction = auction_event.auction
            for index in auction_event.eligible_intervals:
                product = grid.interval_ids[index]
                if auction_index <= frozen_index:
                    if (auction, product) not in reference_trades.index:
                        raise ValueError(f"Missing fixed {auction} commitment for {product}")
                    quantity = float(reference_trades.loc[(auction, product), "quantity_mwh"])
                else:
                    if product not in snapshot_by_product.index:
                        raise ValueError("A future frozen order is absent from its snapshot")
                    quantity = float(snapshot_by_product.loc[product, f"planned_trade_{auction}_mwh"])
                if auction_event.submission_at > pd.Timestamp(grid.starts[index]):
                    raise ValueError("A frozen trade cannot affect a product before its submission")
                rows.append({"auction": auction, "interval_id": product,
                    "submission_time": auction_event.submission_at.isoformat(),
                    "publication_time": auction_event.results_at.isoformat(),
                    "quantity_mwh": quantity, "price_eur_mwh": float(actual_prices[auction][index]),
                    "cashflow_eur": quantity * float(actual_prices[auction][index]),
                    "trade_id": f"{auction}:{product}",
                    "quantity_source": "fixed_commitment" if auction_index <= frozen_index else "frozen_future_plan"})
        _, fee_rows, fee_summary = account_trade_fees(fees, fees.initial_state, rows)
        frozen_ledger = pd.DataFrame(fee_rows)
        frozen_positions = frozen_ledger.groupby("interval_id").quantity_mwh.sum().reindex(grid.interval_ids).fillna(0)
        # Prefix delivery cannot change after a freeze, including financial positions.
        if not np.allclose(frozen_positions.iloc[:first], dispatch.net_position_mwh.iloc[:first]):
            raise ValueError("Frozen future trades changed an already executed position")
        if not np.allclose(frozen_positions.iloc[first:], snapshot.planned_net_position_mwh):
            raise ValueError("Frozen auction trades do not reconcile to the snapshot's planned net position")
        frozen_dispatch = dispatch.copy()
        state = _actual_boundary(config, grid, dispatch, first)
        for offset, row in enumerate(snapshot.itertuples(index=False)):
            index = first + offset
            target = float(frozen_positions.iloc[index] / grid.delta_hours[index])
            state, executed = execute_interval(state=state, start=grid.starts[index], end=grid.ends[index],
                pv_ac_mw=float(dispatch.iloc[index].pv_ac_mw),
                requested_charge_mw=float(row.charge_mw), requested_discharge_mw=float(row.discharge_mw),
                requested_curtailment_mw=float(row.curtailment_mw), plant=plant,
                degradation_settings=aging, dispatch_policy=policy, grid_target_mw=target)
            for key, value in executed.items():
                if key == "cycle_segment_energy_end_mwh":
                    value = str(list(value))
                frozen_dispatch.loc[index, key] = value
            frozen_dispatch.loc[index, "import_fee_eur"] = max(0.0, -executed["grid_mw"]) * grid.delta_hours[index] * import_fee
            for source, destination in (("pv_ac_forecast_mw", "pv_ac_forecast_mw"),
                    ("pv_forecast_vintage_utc", "pv_forecast_vintage_utc"),
                    ("pv_forecast_available_at_utc", "pv_forecast_available_at_utc"),
                    ("soc_end_mwh", "planned_energy_end_mwh"),
                    ("optimizer_cutoff_utc", "plan_cutoff_utc")):
                if hasattr(row, source):
                    frozen_dispatch.loc[index, destination] = getattr(row, source)
            frozen_dispatch.loc[index, "active_plan_auction"] = event.auction
        frozen_dispatch["net_position_mwh"] = frozen_positions.to_numpy(float)
        delivery_fees = frozen_ledger.groupby("interval_id").transaction_fee_eur.sum()
        frozen_dispatch["transaction_fee_eur"] = delivery_fees.reindex(grid.interval_ids).fillna(0).to_numpy(float)
        revenue = float(frozen_ledger.cashflow_eur.sum())
        frozen_dispatch, financials = settle_realized_dispatch(frozen_dispatch,
            auction_revenue_eur=revenue, transaction_fees_eur=fee_summary["transaction_fee_eur"],
            imbalance_prices=positive_prices, imbalance_negative_prices=negative_prices)
        frozen_dispatch["imbalance_cashflow_eur"] = frozen_dispatch.realized_imbalance_cashflow_eur
        profit = financials["realized_profit_after_imbalance_eur"]
        gain = None if reference_profit is None or profit is None else float(reference_profit - profit)
        frozen_dispatch["frozen_after_auction"] = event.auction
        frozen_dispatch["replayed_from_frozen_plan"] = np.arange(len(dispatch)) >= first
        frozen_dispatch["reference_grid_mw"] = dispatch.grid_mw.to_numpy(float)
        frozen_dispatch["reference_energy_end_mwh"] = dispatch.energy_end_mwh.to_numpy(float)
        frozen_dispatch["reference_imbalance_mwh"] = dispatch.imbalance_mwh.to_numpy(float)
        frozen_ledger["frozen_after_auction"] = event.auction
        summaries.append({"frozen_after_auction": event.auction,
            "optimizer_cutoff_utc": snapshot.optimizer_cutoff_utc.iloc[0],
            "pv_forecast_vintage_utc": snapshot.iloc[0].get("pv_forecast_vintage_utc"),
            "replay_start_utc": grid.starts[first].isoformat(),
            "delivery_start_utc": grid.starts[0].isoformat(), "delivery_end_utc": grid.ends[-1].isoformat(),
            "common_prefix_intervals": first, "frozen_plan_intervals": len(snapshot),
            "later_auctions_not_reoptimized": [item.auction for item in schedule[frozen_index + 1:]],
            "dispatch_policy": policy, "additional_optimizer_solves": 0,
            "auction_revenue_eur": revenue, "transaction_fee_eur": fee_summary["transaction_fee_eur"],
            "transaction_fees_by_auction": fee_summary["by_auction"],
            "transaction_fees_by_month": fee_summary["by_month"],
            "degradation_cost_eur": float(frozen_dispatch.degradation_eur.sum()),
            "grid_import_fee_eur": float(frozen_dispatch.import_fee_eur.sum()),
            **financials, "frozen_plan_profit_eur": profit,
            "reoptimized_reference_profit_eur": reference_profit, "reoptimization_gain_eur": gain,
            "reoptimization_gain_percent": None if gain is None or profit == 0 else 100 * gain / abs(profit),
            "final_energy_mwh": float(state.energy_mwh),
            "reference_final_energy_mwh": float(dispatch.energy_end_mwh.iloc[-1]),
            "final_energy_difference_vs_reference_mwh": float(state.energy_mwh - dispatch.energy_end_mwh.iloc[-1]),
            "final_soc_percent": None if plant.battery.capacity_mwh == 0 else 100 * state.energy_mwh / plant.battery.capacity_mwh,
            "curtailment_mwh": float(np.dot(frozen_dispatch.curtailment_mw, grid.delta_hours)),
            "throughput_mwh": float(frozen_dispatch.throughput_mwh.sum()),
            "final_state": asdict(state), "simulation_seconds": perf_counter() - started,
            "comparison_scope": "Full delivery day; identical executed prefix and prior commitments, remaining auction quantities and battery commands frozen at this snapshot, same realized PV and physical feedback; no residual-energy valuation"})
        trajectories.append(frozen_dispatch)
        all_trades.append(frozen_ledger)
    if not summaries:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(summaries), pd.concat(trajectories, ignore_index=True), pd.concat(all_trades, ignore_index=True)
