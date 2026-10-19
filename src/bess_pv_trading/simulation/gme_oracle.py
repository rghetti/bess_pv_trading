"""Offline one-day GME oracle replay using manual prices and one PV vintage."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data import (align_forecast, config_path, load_metadata, load_observations,
                    open_store, select_forecast_vintage)
from ..data.api_catalog import load_api_cache_records
from ..data.gme import native_price_resolution_minutes
from ..domain import (BatteryParameters, DegradationSettings, PhysicalState, PlantParameters,
                      battery_parameters_from_config)
from ..forecasting.pv import forecast_pv
from ..markets import AuctionCalendar
from ..optimization.degradation import degradation_settings_from_config
from ..reporting.artifacts import organize_run_outputs
from ..reporting.plots import plot_boundary_series, plot_interval_series, plot_pv_allocation
from ..scenarios import synthetic_price_tree
from .synthetic import _replay_strategy, write_json
from .replay import execute_interval
from .settlement import expected_realized_kpis, load_realized_imbalance_prices, settle_realized_dispatch


def _apply_actual_settlement(config, grid, dispatch, summary, destination):
    """Price completed execution without exposing actual settlement prices to planning."""
    prices, provenance = load_realized_imbalance_prices(config, grid)
    if prices is None:
        return dispatch, summary, provenance
    settled, kpis = settle_realized_dispatch(dispatch,
        auction_revenue_eur=summary["components"]["auction_revenue_eur"],
        imbalance_prices=prices.settlement_price_eur_mwh.to_numpy(float))
    settled["imbalance_cashflow_eur_assumed"] = settled.imbalance_cashflow_eur
    settled["imbalance_cashflow_eur"] = settled.realized_imbalance_cashflow_eur
    components = {**summary["components"],
        "imbalance_cashflow_eur_assumed": summary["components"]["imbalance_cashflow_eur"],
        "imbalance_cashflow_eur": kpis["realized_imbalance_cashflow_eur"]}
    summary = {**summary, **kpis, "components": components,
        "realized_operating_profit_eur": kpis["realized_profit_after_imbalance_eur"],
        "realized_operating_profit_status": "actual_single_price_settlement_minus_executed_wear_and_import_fees",
        "settlement_price_provenance": provenance}
    summary.update(expected_realized_kpis(
        initial_expected_profit_eur=summary["initial_expected_profit_eur"],
        initial_expected_auction_revenue_eur=summary["initial_expected_auction_revenue_eur"],
        realized_profit_after_imbalance_eur=summary["realized_profit_after_imbalance_eur"],
        auction_revenue_eur=components["auction_revenue_eur"]))
    prices.to_csv(destination / "imbalance_prices.csv", index=False)
    return settled, summary, provenance


def _counterfactual_reoptimization_settlement(plans, dispatch, realized_pv,
        grid, plant, auctions):
    """Price each auction snapshot as an open-loop plan against measured PV.

    Battery commands and the snapshot's contracted grid target stay fixed;
    only realized PV replaces forecast PV. The execution helper clips planned
    commands to SOC, power, PV-curtailment, and POI limits. All steps are settled
    with the same ex-post single imbalance-price curve as actual dispatch.
    """
    required = {"expected_grid_mw", "charge_mw", "discharge_mw", "curtailment_mw",
                "soc_start_mwh", "interval_start_utc", "interval_end_utc", "interval_id"}
    if plans.empty or not required.issubset(plans.columns):
        return pd.DataFrame(), pd.DataFrame()
    if not {"realized_imbalance_cashflow_eur", "net_position_mwh"}.issubset(dispatch.columns):
        return pd.DataFrame(), pd.DataFrame()
    indexed = dispatch.set_index("interval_id")
    interval_index = {product: i for i, product in enumerate(grid.interval_ids)}
    groups = {auction: plans.loc[plans.auction == auction].sort_values("interval_start_utc")
              for auction in auctions}
    groups = {auction: frame for auction, frame in groups.items() if not frame.empty}
    if not groups:
        return pd.DataFrame(), pd.DataFrame()
    common_ids = set.intersection(*(set(frame.interval_id) for frame in groups.values()))
    detailed = []
    aging = DegradationSettings(0.0)
    for auction, frame in groups.items():
        first = frame.iloc[0]
        start = pd.Timestamp(first.interval_start_utc).to_pydatetime()
        state = PhysicalState(start, float(first.soc_start_mwh),
                              origin="counterfactual_snapshot_soc")
        for row in frame.itertuples(index=False):
            index = interval_index[row.interval_id]
            pv = float(realized_pv[index])
            end = pd.Timestamp(row.interval_end_utc).to_pydatetime()
            state, executed = execute_interval(state=state, start=state.timestamp,
                end=end, pv_ac_mw=pv, requested_charge_mw=float(row.charge_mw),
                requested_discharge_mw=float(row.discharge_mw),
                requested_curtailment_mw=float(row.curtailment_mw), plant=plant,
                degradation_settings=aging, dispatch_policy="follow_battery_plan")
            duration = grid.delta_hours[index]
            target = float(row.expected_grid_mw)
            imbalance = (executed["grid_mw"] - target) * duration
            price = float(indexed.loc[row.interval_id, "imbalance_price_eur_mwh"])
            settlement = imbalance * price
            final_position = float(indexed.loc[row.interval_id, "net_position_mwh"])
            final_position_imbalance = executed["grid_mw"] * duration - final_position
            detailed.append({"auction": auction, "interval_id": row.interval_id,
                "interval_start_utc": pd.Timestamp(row.interval_start_utc).isoformat(),
                "interval_end_utc": pd.Timestamp(row.interval_end_utc).isoformat(),
                "optimizer_cutoff_utc": row.optimizer_cutoff_utc,
                "common_comparison_horizon": row.interval_id in common_ids,
                "pv_forecast_mw": float(row.pv_ac_forecast_mw), "pv_realized_mw": pv,
                "planned_charge_mw": float(row.charge_mw),
                "planned_discharge_mw": float(row.discharge_mw),
                "counterfactual_charge_mw": executed["charge_mw"],
                "counterfactual_discharge_mw": executed["discharge_mw"],
                "counterfactual_pv_curtailment_mw": executed["curtailment_mw"],
                "planned_grid_mw": target, "counterfactual_grid_mw": executed["grid_mw"],
                "imbalance_mwh": imbalance, "imbalance_price_eur_mwh": price,
                "imbalance_cashflow_eur": settlement,
                "final_contract_position_mwh": final_position,
                "final_contract_imbalance_mwh": final_position_imbalance,
                "final_contract_imbalance_cashflow_eur": final_position_imbalance * price})
    details = pd.DataFrame(detailed)
    summaries = []
    for auction, frame in details.groupby("auction", sort=False):
        comparable = frame.loc[frame.common_comparison_horizon]
        summaries.append({"auction": auction,
            "optimizer_cutoff_utc": frame.optimizer_cutoff_utc.iloc[0],
            "delivery_window_start_utc": frame.interval_start_utc.min(),
            "delivery_window_end_utc": frame.interval_end_utc.max(),
            "intervals": len(frame),
            "absolute_imbalance_mwh": float(frame.imbalance_mwh.abs().sum()),
            "net_imbalance_mwh": float(frame.imbalance_mwh.sum()),
            "imbalance_cashflow_eur": float(frame.imbalance_cashflow_eur.sum()),
            "final_contract_absolute_imbalance_mwh": float(
                frame.final_contract_imbalance_mwh.abs().sum()),
            "final_contract_net_imbalance_mwh": float(
                frame.final_contract_imbalance_mwh.sum()),
            "final_contract_imbalance_cashflow_eur": float(
                frame.final_contract_imbalance_cashflow_eur.sum()),
            "pv_forecast_mae_mw": float((frame.pv_realized_mw-frame.pv_forecast_mw).abs().mean()),
            "common_horizon_start_utc": comparable.interval_start_utc.min() if len(comparable) else None,
            "common_horizon_end_utc": comparable.interval_end_utc.max() if len(comparable) else None,
            "common_horizon_intervals": len(comparable),
            "common_horizon_absolute_imbalance_mwh": float(comparable.imbalance_mwh.abs().sum()),
            "common_horizon_net_imbalance_mwh": float(comparable.imbalance_mwh.sum()),
            "common_horizon_imbalance_cashflow_eur": float(comparable.imbalance_cashflow_eur.sum()),
            "common_horizon_final_contract_absolute_imbalance_mwh": float(
                comparable.final_contract_imbalance_mwh.abs().sum()),
            "common_horizon_final_contract_net_imbalance_mwh": float(
                comparable.final_contract_imbalance_mwh.sum()),
            "common_horizon_final_contract_cashflow_eur": float(
                comparable.final_contract_imbalance_cashflow_eur.sum())})
    return pd.DataFrame(summaries), details


def _plot_settlement(destination, dispatch, summary, timezone):
    """Show native quarter-hour prices, signed energy errors and cumulative cash flow."""
    if summary["imbalance_settlement_status"] != "priced":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = pd.to_datetime(dispatch.start, utc=True).dt.tz_convert(timezone)
    ends = pd.to_datetime(dispatch.end, utc=True).dt.tz_convert(timezone)
    widths = (pd.to_datetime(dispatch.end, utc=True) - pd.to_datetime(dispatch.start, utc=True)).dt.total_seconds() / 86400
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    plot_interval_series(axes[0], times, ends, dispatch.imbalance_price_eur_mwh)
    axes[0].set_ylabel("Settlement price\nEUR/MWh")
    axes[1].bar(times, dispatch.imbalance_mwh, width=widths, align="edge",
        color=np.where(dispatch.imbalance_mwh >= 0, "tab:green", "tab:red"))
    axes[1].set_ylabel("Executed imbalance\nMWh / quarter-hour")
    cumulative = dispatch.realized_imbalance_cashflow_eur.cumsum()
    plot_boundary_series(axes[2], times, ends,
        cumulative - dispatch.realized_imbalance_cashflow_eur, cumulative, drawstyle="steps-post")
    axes[2].set_ylabel("Cumulative imbalance\nsettlement EUR")
    axes[2].set_xlabel(f"Delivery time ({timezone})")
    axes[0].set_title("Actual Terna NORD prices applied only to executed quarter-hour deviations")
    axes[0].set_xlim(times.iloc[0], ends.iloc[-1])
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.6)
        axis.grid(alpha=0.2)
    fig.savefig(destination / "imbalance_settlement.png", dpi=140)
    plt.close(fig)
    values = [summary["components"]["auction_revenue_eur"], summary["realized_imbalance_cashflow_eur"],
        -summary["components"]["degradation_eur"], -summary["components"]["import_fee_eur"],
        summary["realized_profit_after_imbalance_eur"]]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = axis.bar(["Auction revenue", "Imbalance settlement", "Battery wear", "Import fees", "Net operating profit"],
        values, color=["tab:blue", "tab:green" if values[1] >= 0 else "tab:red", "tab:red", "tab:red", "tab:purple"])
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylabel("EUR")
    axis.set_title("Executed daily operating profit with actual imbalance settlement")
    axis.axhline(0, color="black", linewidth=0.6)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(destination / "realized_profit_breakdown.png", dpi=140)
    plt.close(fig)


def _expected_product_starts(grid, eligible: tuple[int, ...], resolution: int,
                             timezone: str) -> pd.DatetimeIndex:
    """Return native product starts represented by this auction's eligibility."""
    from ..optimization.market_regime import market_product_groups
    selected = set(eligible)
    indexes = [group[0] for group in market_product_groups(grid, resolution, timezone)
               if any(index in selected for index in group)]
    return pd.DatetimeIndex([grid.starts[index] for index in indexes]).tz_convert("UTC")


def _load_auction_prices(config: Mapping[str, Any], calendar: AuctionCalendar,
                         grid, events) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict]]:
    """Load exact native prices; skip documented cancellations and reject gaps."""
    study = config["gme_oracle_experiment"]
    timezone = str(study["timezone"])
    first, stop = pd.Timestamp(grid.starts[0]), pd.Timestamp(grid.ends[-1])
    actual: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    skipped: list[dict] = []
    policy = study.get("missing_auction_policy")
    if policy != "skip_if_no_products_fail_if_partial":
        raise ValueError("Unsupported missing_auction_policy; choose skip_if_no_products_fail_if_partial")
    canceled = set(config.get("market", {}).get("cancelled_auctions", []))
    event_by_name = {event.auction: event for event in events}
    delivery_day = first.tz_convert(timezone).date()
    for auction, item in study["auction_datasets"].items():
        if auction not in event_by_name:
            raise ValueError(f"No market schedule exists for configured auction {auction}")
        event = event_by_name[auction]
        expected_resolution = int(item["native_interval_minutes"])
        native_resolution = native_price_resolution_minutes(item["dataset"], delivery_day)
        if expected_resolution != native_resolution:
            raise ValueError(f"{auction} requires native {native_resolution}-minute prices on {delivery_day}; update the experiment YAML")
        if auction in canceled:
            skipped.append({"auction": auction, "reason": "configured_cancelled_auction"})
            continue
        frame, sources = load_api_cache_records(study["price_database"],
            provider=study["provider"], dataset=item["dataset"], zone=study["zone"],
            start=first, end=stop, verify_checksums=True)
        if frame.empty:
            if any(delivery_day.isoformat() in source.get("cancelled_delivery_days", [])
                   for source in sources):
                skipped.append({"auction": auction, "dataset": item["dataset"],
                                "reason": "documented_cancelled_auction"})
                continue
            raise ValueError(f"{auction} has no native prices for {delivery_day}; missing data do not establish auction cancellation")
        if "price_eur_mwh" not in frame:
            raise ValueError(f"{auction} price snapshot lacks price_eur_mwh")
        frame = frame.copy()
        frame["product_start"] = pd.to_datetime(frame["product_start"], utc=True)
        frame["product_end"] = pd.to_datetime(frame["product_end"], utc=True)
        frame = frame.loc[(frame.product_start >= first) & (frame.product_end <= stop)]
        if frame.duplicated("product_start").any():
            raise ValueError(f"{auction} has duplicate native delivery products")
        declared = pd.to_numeric(frame.get("native_interval_minutes", expected_resolution), errors="coerce")
        if np.any(pd.Series(declared).dropna().to_numpy(dtype=int) != expected_resolution):
            raise ValueError(f"{auction} cached native resolution differs from the experiment YAML")
        expected = _expected_product_starts(grid, event.eligible_intervals,
                                            expected_resolution, timezone)
        observed_starts = pd.DatetimeIndex(frame.product_start).sort_values()
        if not len(observed_starts):
            raise ValueError(f"{auction} has no native prices contained in {delivery_day}; check product alignment")
        if not observed_starts.equals(expected.sort_values()):
            absent = expected.difference(observed_starts)
            extra = observed_starts.difference(expected)
            raise ValueError(f"{auction} price curve is partial or misaligned: missing={len(absent)}, unexpected={len(extra)}; no interpolation is allowed")
        prices = pd.to_numeric(frame.set_index("product_start").reindex(expected).price_eur_mwh,
                                errors="coerce")
        if prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
            raise ValueError(f"{auction} contains missing/nonfinite prices in an otherwise present auction")
        curve = np.full(len(grid.interval_ids), np.nan, dtype=float)
        for product_start, price in prices.items():
            product_end = product_start + pd.Timedelta(minutes=expected_resolution)
            mask = np.asarray([(start >= product_start and start < product_end)
                               for start in grid.starts])
            curve[mask] = float(price)
        eligible = np.zeros(len(grid.interval_ids), dtype=bool)
        eligible[list(event.eligible_intervals)] = True
        if not np.isfinite(curve[eligible]).all():
            raise ValueError(f"{auction} did not map every eligible physical interval")
        # Noneligible intervals are never traded; finite placeholders are required
        # by the common scenario-array contract and remain outside the objective.
        curve[~np.isfinite(curve)] = 0.0
        actual[auction] = curve
        provenance[auction] = {"dataset": item["dataset"], "native_interval_minutes": expected_resolution,
            "eligible_products": len(expected), "price_min_eur_per_mwh": float(prices.min()),
            "price_mean_eur_per_mwh": float(prices.mean()),
            "price_max_eur_per_mwh": float(prices.max()), "snapshots": sources}
    if not actual:
        raise ValueError("No auction price curves are available for this delivery day")
    return actual, provenance, skipped


def _forecast_pv_at_cutoff(config, calendar, dataset, metadata, cutoff, starts):
    """Convert the latest eligible weather cycle on the requested remaining grid."""
    forecast_settings = config["forecast"]
    farm_id = int(forecast_settings["farm_id"])
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("PV decision cutoff must include a timezone")
    starts = pd.DatetimeIndex(starts)
    weather = select_forecast_vintage(dataset, farm_id, cutoff, config["forecast_mapping"],
        latency_hours=float(config.get("source", {}).get("availability_lag_hours", 4)),
        explicit_availability_variable=config.get("source", {}).get("explicit_availability_variable"),
        numeric_lead_units=config.get("source", {}).get("numeric_lead_units", "hours"))
    weather = align_forecast(weather, starts, config["forecast_semantics"],
        config.get("policy", {}).get("nonnegative_forecasts", []))
    pv = forecast_pv(weather, metadata, config["pv"], config.get("policy", {}).get("night", {}))
    forecast = pd.to_numeric(pv.values["pv_ac_mw"], errors="coerce").to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(forecast)
    night = np.zeros(len(forecast), dtype=bool)
    if config.get("policy", {}).get("night", {}).get("enabled", True):
        local = starts.tz_convert(config["policy"]["night"].get("timezone", calendar.timezone))
        minute = local.hour * 60 + local.minute
        start_h, start_m = map(int, config["policy"]["night"].get("start", "21:30").split(":"))
        end_h, end_m = map(int, config["policy"]["night"].get("end", "05:00").split(":"))
        start, end = start_h * 60 + start_m, end_h * 60 + end_m
        night = (minute >= start) | (minute < end) if start > end else (minute >= start) & (minute < end)
    if missing.any():
        if forecast_settings["missing_forecast_policy"] != "zero_only_when_policy_marks_night" or np.any(missing & ~night):
            raise ValueError(f"PV forecast has {int(missing.sum())} unresolved intervals; the configured policy only permits explicit nighttime zero fill")
        forecast[missing] = 0.0
    if not np.isfinite(forecast).all() or np.any(forecast < 0):
        raise ValueError("PV forecast must be a finite nonnegative AC trajectory")
    pv.values["pv_ac_mw"] = forecast
    pv.provenance["forecast_missing_filled_night_zero_intervals"] = int(missing.sum())
    return pv


def _load_pv_forecast(config, calendar, cutoff, starts):
    """Read a causal weather/PV update without consulting realized production."""
    farm_id = int(config["forecast"]["farm_id"])
    with open_store(config) as dataset:
        metadata = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
        return _forecast_pv_at_cutoff(config, calendar, dataset, metadata, cutoff, starts)


def _load_pv(config: dict[str, Any], calendar: AuctionCalendar, grid, *, decision_cutoff=None):
    """Create the initial PV forecast and the separate realized execution series."""
    forecast_settings = config["forecast"]
    farm_id = int(forecast_settings["farm_id"])
    cutoff = pd.Timestamp(decision_cutoff if decision_cutoff is not None
                          else forecast_settings["initial_decision_cutoff"])
    dataset = open_store(config)
    try:
        metadata = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
        pv = _forecast_pv_at_cutoff(config, calendar, dataset, metadata, cutoff, grid.starts)
        forecast = pd.to_numeric(pv.values["pv_ac_mw"], errors="coerce").to_numpy(dtype=float, copy=True)
        observed = load_observations(dataset, [farm_id], variables=(forecast_settings.get("realized_power_variable", "power"),),
            start=grid.starts[0], end=grid.starts[-1],
            telemetry_latency_minutes=float(config.get("source", {}).get("telemetry_latency_minutes", 0)))
    finally:
        dataset.close()
    variable = forecast_settings.get("realized_power_variable", "power")
    rows = observed.frame.loc[observed.frame.variable == variable]
    raw = pd.Series(pd.to_numeric(rows.raw_value, errors="coerce").to_numpy(dtype=float),
                    index=pd.DatetimeIndex(pd.to_datetime(rows.time, utc=True)))
    raw = raw[~raw.index.duplicated(keep="last")].reindex(pd.DatetimeIndex(grid.starts))
    scale = float(forecast_settings["realized_power_scale_to_mw"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("realized_power_scale_to_mw must be finite and positive")
    actual = raw.to_numpy(dtype=float) * scale
    actual_missing = ~np.isfinite(actual)
    missing_actual_policy = forecast_settings["missing_realized_power_policy"]
    maximum_missing = int(forecast_settings["maximum_missing_realized_intervals"])
    if int(actual_missing.sum()) > maximum_missing:
        raise ValueError(f"Realized PV has {int(actual_missing.sum())} missing intervals, exceeding configured maximum {maximum_missing}")
    if actual_missing.any():
        if missing_actual_policy != "use_frozen_pv_forecast":
            raise ValueError("Missing measured PV execution inputs require an explicit configured fallback")
        actual[actual_missing] = forecast[actual_missing]
    if not np.isfinite(actual).all() or np.any(actual < -1e-9):
        raise ValueError("Realized AC PV must be finite and nonnegative after configured fallback")
    actual = np.maximum(actual, 0.0)
    return forecast, actual, pv, {"farm_id": farm_id,
        "farm_metadata": asdict(metadata),
        "installed_dc_capacity_mw": metadata.dc_capacity_mw,
        "forecast_cutoff_utc": cutoff.isoformat(),
        "forecast_vintage_utc": pv.forecast_vintage.isoformat(),
        "forecast_available_at_utc": pv.available_at.isoformat(),
        "forecast_member_count": len(pv.members), "forecast_member_origin": pv.member_origin,
        "forecast_missing_filled_night_zero_intervals": pv.provenance["forecast_missing_filled_night_zero_intervals"],
        "realized_power_variable": variable, "realized_power_scale_to_mw": scale,
        "realized_power_source_basis": observed.provenance.get("power_basis", "AC"),
        "realized_power_missing_fallback_intervals": int(actual_missing.sum()),
        "realized_power_fallback_interval_ids": [grid.interval_ids[i] for i in np.flatnonzero(actual_missing)],
        "pv_conversion_provenance": pv.provenance}


def _pv_forecast_curves(plans: pd.DataFrame, timezone: str):
    """Group auction snapshots by weather cycle, keeping only forecasted intervals."""
    if "pv_ac_forecast_mw" not in plans or plans.empty:
        return []
    rows = plans if "pv_forecast_vintage_utc" in plans else plans.assign(pv_forecast_vintage_utc=None)
    curves = []
    for vintage, group in rows.groupby("pv_forecast_vintage_utc", sort=False, dropna=False):
        auctions = " / ".join(dict.fromkeys(group.auction))
        cycle = (pd.Timestamp(vintage).tz_convert(timezone).strftime("%d/%m %H:%M %Z")
                 if pd.notna(vintage) else "initial cycle")
        selected = group.drop_duplicates("interval_start_utc").sort_values("interval_start_utc")
        curves.append((f"PV {auctions} — {cycle}", selected))
    return curves


def _plot_experiment(destination: Path, grid, prices: Mapping[str, np.ndarray],
                     plans: pd.DataFrame, dispatch: pd.DataFrame, timezone: str,
                     eligibility: Mapping[str, np.ndarray], *,
                     battery_capacity_mwh: float, soc_bounds_percent: tuple[float, float],
                     auction_events=()) -> None:
    """Save the auction-price panels, four decision trajectories, and realized SOC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = pd.DatetimeIndex(grid.starts).tz_convert(timezone)
    local_ends = pd.DatetimeIndex(grid.ends).tz_convert(timezone)
    if len(plans) and "interval_end_utc" not in plans:
        # Older saved plans still have exact ends available from the delivery grid.
        end_by_start = dict(zip(pd.DatetimeIndex(grid.starts), grid.ends))
        plans = plans.assign(interval_end_utc=pd.to_datetime(
            plans.interval_start_utc, utc=True).map(end_by_start))
    forecast_curves = _pv_forecast_curves(plans, timezone)

    def plot_pv_forecasts(axis):
        """Use identical cycle labels and colors in all PV comparison figures."""
        for index, (label, selected) in enumerate(forecast_curves):
            times = pd.to_datetime(selected.interval_start_utc, utc=True).dt.tz_convert(timezone)
            ends = pd.to_datetime(selected.interval_end_utc, utc=True).dt.tz_convert(timezone)
            plot_interval_series(axis, times, ends, selected.pv_ac_forecast_mw, linestyle="--",
                      color=f"C{index}", linewidth=1.5, label=label)

    fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    for auction in ("DA", "IDA1", "IDA2"):
        if auction in prices:
            plot_interval_series(axis, local, local_ends, prices[auction], label=auction)
    axis.set_title("MGP, MI-A1 and MI-A2 clearing prices")
    axis.set_ylabel("EUR/MWh")
    axis.set_xlabel(f"Delivery time ({timezone})")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3)
    axis.set_xlim(local[0], local_ends[-1])
    fig.savefig(destination / "prices_da_ida1_ida2.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    if "IDA3" in prices:
        values = np.where(eligibility["IDA3"], prices["IDA3"], np.nan)
        plot_interval_series(axis, local, local_ends, values, label="MI-A3")
    axis.set_title("MI-A3 clearing prices")
    axis.set_ylabel("EUR/MWh")
    axis.set_xlabel(f"Delivery time ({timezone})")
    axis.grid(alpha=0.2)
    axis.legend()
    axis.set_xlim(local[0], local_ends[-1])
    fig.savefig(destination / "prices_ida3.png", dpi=140)
    plt.close(fig)

    allocation = plot_pv_allocation(dispatch, destination / "pv_power_allocation.png",
                                    timezone=timezone)
    allocation.to_csv(destination / "pv_power_allocation.csv", index=False)
    executed_ends = pd.to_datetime(dispatch.end, utc=True).dt.tz_convert(timezone)

    # Keep each reoptimization's information set and schedule together. The
    # executed trajectory is common to all panels, so deviations from each
    # auction's plan are directly visible on the same delivery-time axis.
    realized_pv_times = pd.to_datetime(dispatch.start, utc=True).dt.tz_convert(timezone)
    realized_battery = dispatch.discharge_mw - dispatch.charge_mw
    auction_plot_names = {
        "DA": "reoptimization_DA.png",
        "IDA1": "reoptimization_IDA1.png",
        "IDA2": "reoptimization_IDA2.png",
        "IDA3": "reoptimization_IDA3.png",
    }
    event_by_auction = {event.auction: event for event in auction_events}
    for auction, filename in auction_plot_names.items():
        selected = plans.loc[plans.auction == auction]
        if selected.empty:
            continue  # Canceled or unavailable auctions have no decision to plot.
        plan_times = pd.to_datetime(selected.interval_start_utc, utc=True).dt.tz_convert(timezone)
        plan_ends = pd.to_datetime(selected.interval_end_utc, utc=True).dt.tz_convert(timezone)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]},
                                 layout="constrained")
        plot_interval_series(axes[0], plan_times, plan_ends,
            selected.pv_ac_forecast_mw, linestyle="--", label=f"PV forecast used at {auction}")
        plot_interval_series(axes[0], realized_pv_times,
            pd.to_datetime(dispatch.end, utc=True).dt.tz_convert(timezone),
            dispatch.pv_ac_mw, color="black", linewidth=2, label="Realized PV")
        axes[0].set_ylabel("PV power (MW AC)")
        event = event_by_auction.get(auction)
        if event is not None:
            gate_local = event.gate_at.tz_convert(timezone)
            if local[0] <= gate_local <= local_ends[-1]:
                axes[0].axvline(gate_local, color="tab:red", linestyle=":",
                    linewidth=1.2, label=f"{auction} gate closes {gate_local:%H:%M}")
            eligible = np.flatnonzero(eligibility.get(auction, np.zeros(len(local), dtype=bool)))
            if len(eligible):
                delivery_start = local[eligible[0]]
                if local[0] < delivery_start <= local_ends[-1]:
                    axes[0].axvline(delivery_start, color="tab:purple", linestyle="-.",
                        linewidth=1.2, label=f"{auction} products start {delivery_start:%H:%M}")
                    plan_start = plan_times.min()
                    if plan_start < delivery_start:
                        axes[0].axvspan(plan_start, delivery_start, color="tab:orange",
                            alpha=0.08, label="Physical plan before this auction's products")
            cutoff = pd.Timestamp(selected.optimizer_cutoff_utc.iloc[0]).tz_convert(timezone)
            axes[0].set_title(f"{auction} plan: cutoff {cutoff:%H:%M}, gate {gate_local:%H:%M}")
        else:
            axes[0].set_title(f"{auction} reoptimization: PV information and battery schedule")
        plot_interval_series(axes[1], plan_times, plan_ends,
            selected.net_battery_dispatch_mw, linestyle="--",
            label=f"Battery dispatch scheduled at {auction}")
        plot_interval_series(axes[1], realized_pv_times,
            pd.to_datetime(dispatch.end, utc=True).dt.tz_convert(timezone),
            realized_battery, color="black", linewidth=2, label="Realized battery dispatch")
        axes[1].axhline(0, color="grey", linewidth=0.7)
        axes[1].set_ylabel("Battery power (MW AC)\npositive = discharge")
        axes[1].set_xlabel(f"Delivery time ({timezone})")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend()
            axis.set_xlim(local[0], local_ends[-1])
        fig.savefig(destination / filename, dpi=140)
        plt.close(fig)

    if forecast_curves:
        fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
        plot_pv_forecasts(axis)
        times = pd.to_datetime(dispatch.start, utc=True).dt.tz_convert(timezone)
        plot_interval_series(axis, times, executed_ends, dispatch.pv_ac_mw, color="black", linewidth=2,
                  label="Realized PV (execution input)")
        axis.set_title("PV forecasts used at each auction and realized generation")
        axis.set_ylabel("PV power (MW AC)")
        axis.set_xlabel(f"Delivery time ({timezone})")
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_xlim(local[0], local_ends[-1])
        fig.savefig(destination / "pv_forecast_and_realized.png", dpi=140)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    starts = pd.to_datetime(dispatch.start, utc=True).dt.tz_convert(timezone)
    ends = pd.to_datetime(dispatch.end, utc=True).dt.tz_convert(timezone)
    scale = 100 / battery_capacity_mwh if battery_capacity_mwh > 0 else 0.0
    axis.axhspan(*soc_bounds_percent, color="grey", alpha=0.08, label="Configured SOC box")
    plot_boundary_series(axis, starts, ends, dispatch.energy_start_mwh * scale,
              dispatch.energy_end_mwh * scale, color="black", linewidth=2.2,
              label="Executed SOC")
    if len(plans):
        for auction in ("DA", "IDA1", "IDA2", "IDA3"):
            selected = plans.loc[plans.auction == auction]
            if len(selected):
                times = pd.to_datetime(selected.interval_start_utc, utc=True).dt.tz_convert(timezone)
                plan_ends = pd.to_datetime(selected.interval_end_utc, utc=True).dt.tz_convert(timezone)
                plot_boundary_series(axis, times, plan_ends, selected.soc_start_mwh * scale,
                          selected.soc_end_mwh * scale, alpha=0.65, linewidth=0.9,
                          label=f"{auction} expected SOC")
    axis.set_title("Battery state of charge: executed trajectory and auction plans")
    axis.set_ylabel("SOC (% of nominal capacity)")
    axis.set_ylim(0, 100)
    axis.set_xlabel(f"Delivery time ({timezone})")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3)
    axis.set_xlim(local[0], local_ends[-1])
    fig.savefig(destination / "soc_evolution.png", dpi=140)
    plt.close(fig)

    if "grid_target_mw" in dispatch and "requested_charge_mw" in dispatch:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
        plot_pv_forecasts(axes[0])
        plot_interval_series(axes[0], starts, ends, dispatch.pv_ac_mw, color="black", linewidth=2,
                     label="Realized PV")
        axes[0].set_ylabel("PV MW")
        plot_interval_series(axes[1], starts, ends, dispatch.grid_target_mw, linestyle="--", label="Committed grid target")
        plot_interval_series(axes[1], starts, ends, dispatch.grid_mw, label="Executed grid power")
        axes[1].set_ylabel("Grid MW")
        plot_interval_series(axes[2], starts, ends, dispatch.requested_discharge_mw-dispatch.requested_charge_mw,
                     linestyle="--", label="Active auction battery plan")
        plot_interval_series(axes[2], starts, ends, dispatch.discharge_mw-dispatch.charge_mw, label="Executed battery power")
        axes[2].set_ylabel("BESS MW\npositive = discharge")
        axes[2].set_xlabel(f"Delivery time ({timezone})")
        axes[0].set_title("Physical execution feedback using realized interval-average PV")
        for item in axes:
            item.grid(alpha=0.2)
            item.legend()
        axes[0].set_xlim(local[0], local_ends[-1])
        fig.savefig(destination / "execution_tracking.png", dpi=140)
        plt.close(fig)


def _soc_energy_targets(state: Mapping[str, Any], battery: BatteryParameters
                        ) -> tuple[float, float | None]:
    """Translate configured SOC percentages; an explicit null disables the target."""
    initial = float(state["initial_soc_percent"])
    terminal = state["terminal_soc_percent"]
    percentages = [initial] + ([] if terminal is None else [float(terminal)])
    if any(not np.isfinite(value) or not 0 <= value <= 100 for value in percentages):
        raise ValueError("Initial/terminal SOC percentages must be finite and between 0 and 100")
    initial_energy = battery.capacity_mwh * initial / 100.0
    terminal_energy = None if terminal is None else battery.capacity_mwh * float(terminal) / 100.0
    targets = [initial_energy] + ([] if terminal_energy is None else [terminal_energy])
    if any(not battery.energy_min_mwh <= value <= battery.maximum_energy for value in targets):
        raise ValueError("Configured initial/terminal SOC lies outside battery energy bounds")
    return initial_energy, terminal_energy


def run_gme_oracle_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the single-day GME price oracle with causally available PV updates."""
    study = config["gme_oracle_experiment"]
    day = str(study["delivery_day"])
    calendar = AuctionCalendar.from_config(config)
    market_grid = calendar.delivery_grid(day)
    grid = market_grid.to_domain()
    scheduled = calendar.schedule(day)
    actual_prices, price_provenance, skipped = _load_auction_prices(config, calendar, grid, scheduled)
    schedule = tuple(event for event in scheduled if event.auction in actual_prices)
    if not schedule:
        raise ValueError("All configured auctions were skipped; cannot build a price-taking study")
    pv_policy = config["forecast"]["reoptimization_pv_policy"]
    rolling_pv = pv_policy == "latest_available_forecast_at_each_reoptimization"
    forecast, realized_pv, pv_bundle, pv_provenance = _load_pv(config, calendar, grid,
        decision_cutoff=schedule[0].submission_at if rolling_pv else None)
    if pv_bundle.available_at > min(pd.Timestamp(grid.starts[0]),
                                    min(event.submission_at for event in schedule)):
        raise ValueError("The initial PV vintage is not available before initial delivery control/auction submission")

    battery = battery_parameters_from_config(config["battery"])
    plant_fields = {"farm_id", "poi_import_mw", "poi_export_mw"}
    plant = PlantParameters(**{key: config["plant"][key] for key in plant_fields}, battery=battery)
    if plant.farm_id != int(config["forecast"]["farm_id"]):
        raise ValueError("Plant and PV forecast must identify the same farm")
    study_state = config["state"]
    initial_energy, terminal_energy = _soc_energy_targets(study_state, battery)
    config["state"] = {**study_state, "initial_energy_mwh": initial_energy,
        "terminal_target_mwh": terminal_energy,
        "origin": study_state.get("origin", "known_delivery_start_assumption")}
    trade = config["trading"]
    config["decision_resolution_minutes"] = int(study["decision_resolution_minutes"])
    config["native_interval_minutes_by_auction"] = {
        auction: int(item["native_interval_minutes"])
        for auction, item in study["auction_datasets"].items() if auction in actual_prices}
    config["execution"] = {**config.get("execution", {}), "reoptimize_physical_controls": False}
    config["outputs"] = {"plots": bool(config["experiment_outputs"].get("plots", True)),
        "save_model_files": bool(config["experiment_outputs"].get("save_model_files", True))}

    auction_order = [event.auction for event in schedule]
    first_auction = auction_order[0]
    spreads = {current: actual_prices[current] - actual_prices[previous]
               for previous, current in zip(auction_order, auction_order[1:])}
    kwargs = {"base_prices": actual_prices[first_auction], "stage_spreads": spreads,
        "residual_scale": 0.0, "branch_values": (0.0,), "branch_probabilities": (1.0,),
        "imbalance_price": float(config["trading"].get("imbalance_price_eur_per_mwh", 0.0)),
        "seed": int(config["solver"].get("seed", 0))}
    scenarios, tree = synthetic_price_tree(grid, forecast, schedule, **kwargs)
    fixture = {"calendar": calendar, "grid": grid, "schedule": schedule, "pv": forecast,
        "realized_pv": realized_pv, "base": actual_prices[first_auction], "plant": plant,
        "tree": tree, "scenarios": scenarios, "realized": 0, "actual_prices": actual_prices,
        "kwargs": kwargs, "forecast_available_at": pv_bundle.available_at,
        "pv_forecast_vintage": pv_bundle.forecast_vintage, "pv_forecast_policy": pv_policy,
        "execution_pv_assumption": "measured AC generation after each interval closes; frozen forecast fills only configured missing measured intervals"}
    if rolling_pv:
        def latest_pv_forecast(cutoff, starts):
            """Reuse the initial conversion only for the identical initial request."""
            if (pd.Timestamp(cutoff) == pd.Timestamp(pv_provenance["forecast_cutoff_utc"])
                    and starts.equals(pd.DatetimeIndex(grid.starts))):
                return pv_bundle
            return _load_pv_forecast(config, calendar, cutoff, starts)
        fixture["pv_forecast_provider"] = latest_pv_forecast

    root = config_path(config, config["experiment_outputs"]["root"])
    run_id = str(config["experiment_outputs"].get("run_id") or
                 datetime.now(timezone.utc).strftime("gme_oracle_%Y%m%dT%H%M%SZ"))
    day_output = root / run_id / f"farm_{plant.farm_id}" / day
    if day_output.exists():
        raise FileExistsError(f"Experiment output exists; choose a fresh run_id: {day_output}")
    day_output.mkdir(parents=True, exist_ok=False)
    oracle_dir = day_output / "oracle"
    summary = _replay_strategy(config, fixture, "oracle", oracle_dir)
    dispatch = pd.read_csv(oracle_dir / "dispatch.csv")
    # Source actual imbalance prices only once the entire physical replay is over.
    dispatch, summary, settlement_provenance = _apply_actual_settlement(
        config, grid, dispatch, summary, day_output)
    dispatch.to_csv(oracle_dir / "dispatch.csv", index=False)
    write_json(oracle_dir / "summary.json", summary)
    if config["outputs"]["plots"]:
        _plot_settlement(day_output, dispatch, summary, study["timezone"])
        if summary["imbalance_settlement_status"] == "priced":
            from ..reporting.plots import plot_dispatch
            plot_dispatch(dispatch, actual_prices, summary, oracle_dir / "dispatch.png", timezone=study["timezone"])
    plans = pd.read_csv(oracle_dir / "expected_dispatch_plans.csv")
    reoptimization_settlements, reoptimization_settlement_intervals = (
        _counterfactual_reoptimization_settlement(plans, dispatch, realized_pv,
            grid, plant, tuple(actual_prices)))
    if not reoptimization_settlements.empty:
        reoptimization_settlements.to_csv(
            day_output / "reoptimization_imbalance_settlements.csv", index=False)
        reoptimization_settlement_intervals.to_csv(
            day_output / "reoptimization_imbalance_intervals.csv", index=False)
    eligibility = {}
    for event in schedule:
        mask = np.zeros(len(grid.interval_ids), dtype=bool)
        mask[list(event.eligible_intervals)] = True
        eligibility[event.auction] = mask
    prices_frame = pd.DataFrame({"interval_id": grid.interval_ids,
        "interval_start_utc": [stamp.isoformat() for stamp in grid.starts],
        "interval_start_local": pd.DatetimeIndex(grid.starts).tz_convert(study["timezone"]).astype(str),
        **{auction: np.where(eligibility[auction], curve, np.nan)
           for auction, curve in actual_prices.items()}})
    prices_frame.to_csv(day_output / "auction_prices_expanded.csv", index=False)
    pv_frame = pv_bundle.values.copy()
    active_forecast = dispatch.pv_ac_forecast_mw.to_numpy(float)
    pv_frame["pv_initial_forecast_ac_mw"] = forecast
    pv_frame["pv_forecast_used_ac_mw"] = active_forecast
    pv_frame["pv_forecast_used_vintage_utc"] = dispatch.pv_forecast_vintage_utc.to_numpy()
    pv_frame["pv_forecast_used_available_at_utc"] = dispatch.pv_forecast_available_at_utc.to_numpy()
    pv_frame["pv_realized_execution_ac_mw"] = realized_pv
    pv_frame["realized_minus_forecast_mw"] = realized_pv - active_forecast
    pv_frame["interval_id"] = grid.interval_ids
    pv_frame.to_csv(day_output / "pv_forecast_and_realized.csv", index_label="interval_start_utc")
    _plot_experiment(day_output, grid, actual_prices, plans, dispatch, study["timezone"], eligibility,
        battery_capacity_mwh=battery.capacity_mwh,
        soc_bounds_percent=(float(config["battery"]["soc_min_percent"]),
                            float(config["battery"]["soc_max_percent"])),
        auction_events=schedule)

    delta = grid.delta_hours
    capacity = battery.capacity_mwh
    solve_data = json_load(oracle_dir / "solves.json")
    forecast_updates = [{key: row.get(key) for key in (
        "current_auction", "cutoff", "planning_start_utc", "planning_end_utc",
        "pv_forecast_vintage_utc", "pv_forecast_available_at_utc",
        "pv_forecast_updated_from_utc", "pv_forecast_updated_intervals")}
        for row in solve_data if row["solve_trigger"] == "auction_submission"]
    write_json(day_output / "pv_forecast_updates.json", forecast_updates)
    plans[["auction", "optimizer_cutoff_utc", "interval_start_utc", "pv_ac_forecast_mw",
           "pv_forecast_vintage_utc", "pv_forecast_available_at_utc"]].to_csv(
        day_output / "pv_forecasts_by_auction.csv", index=False)
    model_variables = [int(item["variables"]) for item in solve_data if item.get("variables") is not None]
    model_constraints = [int(item["constraints"]) for item in solve_data if item.get("constraints") is not None]
    trade_volumes = {}
    ledger = pd.read_csv(oracle_dir / "ledger.csv")
    for auction in actual_prices:
        selected = ledger.loc[ledger.auction == auction] if "auction" in ledger else pd.DataFrame()
        trade_volumes[auction] = {"accepted_volume_mwh": float(pd.to_numeric(selected.get("quantity_mwh", pd.Series(dtype=float)), errors="coerce").abs().sum()),
            "cashflow_eur": float(summary["auction_cashflows_eur"].get(auction, 0.0))}
    fallback_ids = set(pv_provenance["realized_power_fallback_interval_ids"])
    measured_mask = np.asarray([interval_id not in fallback_ids for interval_id in grid.interval_ids], dtype=bool)
    pv_errors = realized_pv[measured_mask] - active_forecast[measured_mask]
    pv_mae = float(np.mean(np.abs(pv_errors))) if len(pv_errors) else None
    pv_rmse = float(np.sqrt(np.mean(pv_errors ** 2))) if len(pv_errors) else None
    command_error = ((dispatch.discharge_mw-dispatch.charge_mw)
                     -(dispatch.requested_discharge_mw-dispatch.requested_charge_mw))
    soc_error = dispatch.energy_end_mwh-dispatch.planned_energy_end_mwh
    kpis = {"study_id": study["study_id"], "study_date": day, "farm_id": plant.farm_id,
        "pv_installed_dc_mw": pv_provenance["installed_dc_capacity_mw"],
        "pv_tracking_type_id": pv_provenance["farm_metadata"]["tracking_type_id"],
        "market_zone": study["zone"], "strategy": "perfect_price_foresight_oracle_not_executable",
        "forecast_member_count": pv_provenance["forecast_member_count"],
        "forecast_vintage_utc": pv_provenance["forecast_vintage_utc"],
        "forecast_available_at_utc": pv_provenance["forecast_available_at_utc"],
        "forecast_initial_cutoff_utc": pv_provenance["forecast_cutoff_utc"],
        "pv_forecast_policy": pv_policy, "pv_forecast_updates": forecast_updates,
        "pv_forecast_distinct_vintages": len({row["pv_forecast_vintage_utc"] for row in forecast_updates}),
        "initial_soc_percent": float(study_state["initial_soc_percent"]),
        "terminal_soc_percent": (None if study_state["terminal_soc_percent"] is None
                                 else float(study_state["terminal_soc_percent"])),
        "terminal_constraint_enabled": terminal_energy is not None,
        "final_soc_percent": float(summary["final_state"]["energy_mwh"]) / capacity * 100 if capacity else None,
        "initial_energy_mwh": initial_energy, "final_energy_mwh": float(summary["final_state"]["energy_mwh"]),
        "total_gain_before_unavailable_real_imbalance_settlement_eur": summary["profit_before_imbalance_settlement_eur"],
        "profit_before_imbalance_settlement_eur": summary["profit_before_imbalance_settlement_eur"],
        "realized_imbalance_cashflow_eur": summary["realized_imbalance_cashflow_eur"],
        "realized_profit_after_imbalance_eur": summary["realized_profit_after_imbalance_eur"],
        "total_gain_eur": summary["realized_profit_after_imbalance_eur"],
        "realized_imbalance_receipts_eur": summary.get("realized_imbalance_receipts_eur"),
        "realized_imbalance_payments_eur": summary.get("realized_imbalance_payments_eur"),
        "imbalance_settlement_status": summary["imbalance_settlement_status"],
        "optimization_imbalances_enabled": summary["optimization_imbalances_enabled"],
        "execution_dispatch_policy": summary["execution_dispatch_policy"],
        "max_feedback_adjustment_mw": summary["max_feedback_adjustment_mw"],
        "max_battery_dispatch_deviation_from_active_plan_mw": float(command_error.abs().max()),
        "max_soc_deviation_from_active_plan_percentage_points": float(soc_error.abs().max()*100/capacity) if capacity else 0.0,
        "battery_feedback_intervals": int(command_error.abs().gt(
            config.get("execution", {}).get("comparison_tolerance_mw", 1e-6)).sum()),
        "oracle_initial_objective_eur": float(summary["initial_expected_profit_eur"]),
        **{key: summary[key] for key in (
            "expectation_baseline", "initial_expected_profit_eur", "initial_expected_auction_revenue_eur",
            "realized_minus_expected_profit_eur", "realized_minus_expected_profit_percent",
            "realized_minus_expected_auction_revenue_eur", "realized_minus_expected_auction_revenue_percent")},
        "auction_revenue_eur": float(summary["components"]["auction_revenue_eur"]),
        "imbalance_cashflow_eur_assumed": float(summary["components"].get("imbalance_cashflow_eur_assumed",
            summary["components"]["imbalance_cashflow_eur"])),
        "imbalance_penalty_eur": float(summary["components"]["imbalance_penalty_eur"]),
        "degradation_cost_eur": float(summary["components"]["degradation_eur"]),
        "grid_import_fee_eur": float(summary["components"]["import_fee_eur"]),
        "pv_forecast_energy_mwh": float(np.dot(active_forecast, delta)),
        "pv_initial_forecast_energy_mwh": float(np.dot(forecast, delta)),
        "pv_realized_execution_energy_mwh": float(np.dot(realized_pv, delta)),
        "pv_forecast_peak_ac_mw": float(active_forecast.max()),
        "pv_realized_peak_ac_mw": float(realized_pv.max()),
        "grid_import_mwh": float(np.dot(np.maximum(-dispatch.grid_mw.to_numpy(dtype=float), 0), delta)),
        "grid_export_mwh": float(np.dot(np.maximum(dispatch.grid_mw.to_numpy(dtype=float), 0), delta)),
        "pv_power_mae_mw_on_measured_intervals": pv_mae,
        "pv_power_rmse_mw_on_measured_intervals": pv_rmse,
        "pv_power_bias_mw_on_measured_intervals": float(np.mean(pv_errors)) if len(pv_errors) else None,
        "pv_error_reference": "active_auction_plan_forecast_at_interval_start",
        "pv_forecast_missing_night_fill_intervals": pv_provenance["forecast_missing_filled_night_zero_intervals"],
        "pv_realized_fallback_intervals": pv_provenance["realized_power_missing_fallback_intervals"],
        "curtailment_mwh": float(np.dot(dispatch.curtailment_mw.to_numpy(dtype=float), delta)),
        "battery_charge_mwh": float(np.dot(dispatch.charge_mw.to_numpy(dtype=float), delta)),
        "battery_discharge_mwh": float(np.dot(dispatch.discharge_mw.to_numpy(dtype=float), delta)),
        "battery_throughput_mwh": float(dispatch.throughput_mwh.sum()),
        "effective_battery_charge_efficiency": battery.eta_charge,
        "effective_battery_discharge_efficiency": battery.eta_discharge,
        "battery_ac_round_trip_efficiency": battery.eta_charge * battery.eta_discharge,
        "battery_conversion_loss_mwh": float(np.dot(
            (1 - battery.eta_charge) * dispatch.charge_mw.to_numpy(dtype=float)
            + (1 / battery.eta_discharge - 1) * dispatch.discharge_mw.to_numpy(dtype=float), delta)),
        "battery_power_basis": "AC_bus_power; SOC_is_internal_stored_energy",
        "electrical_topology": "AC_coupled_PV_and_BESS_share_POI",
        "equivalent_full_cycles_discharge_basis": float(np.dot(dispatch.discharge_mw.to_numpy(dtype=float), delta) / capacity) if capacity else 0.0,
        "absolute_imbalance_mwh": float(np.abs(dispatch.imbalance_mwh.to_numpy(dtype=float)).sum()),
        "net_imbalance_mwh": summary["net_imbalance_mwh"],
        "surplus_imbalance_mwh": summary["surplus_imbalance_mwh"],
        "shortfall_imbalance_mwh": summary["shortfall_imbalance_mwh"],
        "max_absolute_imbalance_mwh_per_interval": float(np.abs(dispatch.imbalance_mwh.to_numpy(dtype=float)).max()),
        "soc_min_mwh": float(dispatch.energy_start_mwh.min()), "soc_max_mwh": float(dispatch.energy_end_mwh.max()),
        "soc_min_percent": float(dispatch.energy_start_mwh.min()) / capacity * 100 if capacity else None,
        "soc_max_percent": float(dispatch.energy_end_mwh.max()) / capacity * 100 if capacity else None,
        "dispatch_intervals": len(dispatch), "active_auctions": list(actual_prices),
        "skipped_auctions": skipped, "auction_trade_kpis": trade_volumes,
        "reoptimization_solves": len(solve_data), "optimizer_statuses": [item["status"] for item in solve_data],
        "auction_decision_solves": summary["auction_decision_solves"],
        "physical_control_solves": summary["physical_control_solves"],
        "price_publication_events": summary["price_publication_events"],
        "planned_delivery_balance_resolution_minutes": trade.get("planned_delivery_balance_resolution_minutes"),
        "optimizer_parameters": config["solver"],
        "model_variables_min": min(model_variables) if model_variables else None,
        "model_variables_max": max(model_variables) if model_variables else None,
        "model_constraints_min": min(model_constraints) if model_constraints else None,
        "model_constraints_max": max(model_constraints) if model_constraints else None,
        "degradation_parameters": config["degradation"], "battery_parameters": config["battery"],
        "resolved_degradation_parameters": asdict(degradation_settings_from_config(config)),
        "plant_parameters": config["plant"], "market_assumptions": config["market"],
        "trading_assumptions": trade,
        "settlement_status": trade["settlement_status"],
        "unpriced_unrestricted_imbalance": (trade.get("optimize_imbalances", True)
            and float(trade["imbalance_penalty_eur_per_mwh"]) == 0
            and float(trade.get("imbalance_price_eur_per_mwh", 0)) == 0
            and trade.get("imbalance_limit_mwh") is None
            and trade.get("planned_delivery_balance_resolution_minutes") is None),
        "price_provenance": price_provenance,
        "settlement_price_provenance": settlement_provenance,
        "reoptimization_imbalance_settlements": reoptimization_settlements.to_dict("records"),
        "pv_provenance": pv_provenance,
        "objective_status": "clearing-price oracle benchmark; not an executable forecast-based strategy"}
    write_json(day_output / "kpis.json", kpis)
    pd.DataFrame([{"kpi": key, "value": str(value)} for key, value in kpis.items()
                  if isinstance(value, (str, int, float, bool)) or value is None]).to_csv(day_output / "kpis.csv", index=False)
    write_json(day_output / "study_config.json", config)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    manifest = {"status": "complete", "study_id": study["study_id"], "run_id": run_id,
        "study_date": day, "code_revision": revision, "output_dir": str(day_output),
        "strategy": "perfect_price_foresight_oracle_not_executable", "scenario_count": 1,
        "active_auctions": list(actual_prices), "skipped_auctions": skipped,
        "price_provenance": price_provenance, "pv_provenance": pv_provenance,
        "settlement_price_provenance": settlement_provenance,
        "optimizer_solves": len(solve_data), "summary": summary,
        "artifacts": ["prices_da_ida1_ida2.png", "prices_ida3.png",
            "pv_power_allocation.png", "pv_power_allocation.csv", "soc_evolution.png", "pv_forecast_and_realized.png",
            "execution_tracking.png",
            "auction_prices_expanded.csv", "pv_forecast_and_realized.csv", "kpis.json", "kpis.csv",
            "oracle/summary.json",
            "oracle/dispatch.csv", "oracle/expected_dispatch_plans.csv", "oracle/ledger.csv",
            "oracle/solves.json", "oracle/price_publications.json"]}
    manifest["artifacts"].extend(
        f"reoptimization_{auction}.png" for auction in actual_prices
        if (day_output / f"reoptimization_{auction}.png").exists())
    if not reoptimization_settlements.empty:
        manifest["artifacts"].extend(["reoptimization_imbalance_settlements.csv",
                                      "reoptimization_imbalance_intervals.csv"])
    if summary["imbalance_settlement_status"] == "priced":
        manifest["artifacts"].extend(["imbalance_prices.csv", "imbalance_settlement.png",
                                      "realized_profit_breakdown.png"])
    write_json(day_output / "manifest.json", manifest)
    return organize_run_outputs(day_output)


def json_load(path: Path) -> Any:
    """Load an experiment-generated JSON artifact."""
    import json
    return json.loads(path.read_text())
