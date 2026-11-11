"""Paired real-time controller diagnostics, cash accounting and visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..numerical_tolerances import continuous_tolerances
from ..optimization.mpc import MpcSettings
from ..simulation.synthetic import write_json
from .continuous_outputs import _financial_dispatch, realtime_daily_plot_limit
from .plots import plot_boundary_series, plot_interval_series
from .mpc_history import validate_mpc_history_settings, write_history_manifest


def _records(frame):
    """Preserve undefined ratios and missing predictions as JSON null, never NaN."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def controller_metrics(config, frame, solves):
    """Aggregate physical MWh and signed EUR, without monetizing planning penalties."""
    hours = frame.duration_hours.to_numpy(float)
    deviation = frame.imbalance_mwh.to_numpy(float)
    positive, negative = np.maximum(deviation, 0), np.maximum(-deviation, 0)
    price = frame.imbalance_price_eur_mwh.to_numpy(float)
    negative_price = frame.get("imbalance_negative_price_eur_mwh", frame.imbalance_price_eur_mwh).to_numpy(float)
    cash = positive * price - negative * negative_price
    absolute = np.abs(deviation)
    capacity = config["battery"]["capacity_mwh"]
    tolerance = continuous_tolerances(config).physical_absolute
    nonzero = absolute > tolerance

    def ratio(numerator, denominator):
        return None if denominator == 0 else float(numerator / denominator)

    # Include both interval boundaries in extrema; average SOC is time weighted.
    energies = np.r_[frame.energy_start_mwh.iloc[0], frame.energy_end_mwh]
    soc_scale = 100 / capacity if capacity else None
    runtimes = [row["runtime_seconds"] for row in solves if row.get("runtime_seconds") is not None]
    battery_min = capacity * config["battery"]["soc_min_percent"] / 100
    battery_max = capacity * config["battery"]["soc_max_percent"] / 100
    power = config["battery"]["max_power_mw"]
    expected = frame.get("expected_absolute_imbalance_mwh", pd.Series(np.nan, index=frame.index))
    expected_shortfall = frame.get("expected_shortfall_mwh", pd.Series(np.nan, index=frame.index))
    result = {
        "interval_count": len(frame), "absolute_imbalance_mwh": float(absolute.sum()),
        "signed_net_imbalance_mwh": float(deviation.sum()),
        "positive_imbalance_mwh": float(positive.sum()), "negative_imbalance_mwh": float(negative.sum()),
        "nonzero_imbalance_intervals": int(nonzero.sum()), "nonzero_imbalance_fraction": float(nonzero.mean()),
        "max_positive_deviation_mwh": float(np.max(positive)),
        "max_negative_deviation_mwh": float(np.min(np.minimum(deviation, 0))),
        "maximum_cumulative_shortage_mwh": float(max(0, -np.min(np.cumsum(deviation)))),
        "imbalance_cashflow_eur": float(cash.sum()),
        "positive_deviation_cashflow_eur": float((positive * price).sum()),
        "negative_deviation_cashflow_eur": float((-negative * negative_price).sum()),
        "absolute_settlement_exposure_eur": float(np.abs(cash).sum()),
        "volume_weighted_absolute_price_eur_per_mwh": ratio(np.abs(cash).sum(), absolute.sum()),
        "positive_volume_weighted_price_eur_per_mwh": ratio((positive * price).sum(), positive.sum()),
        "negative_volume_weighted_price_eur_per_mwh": ratio((negative * negative_price).sum(), negative.sum()),
        "settlement_eur_per_absolute_imbalance_mwh": ratio(cash.sum(), absolute.sum()),
        "charge_mwh": float(np.dot(frame.charge_mw, hours)),
        "discharge_mwh": float(np.dot(frame.discharge_mw, hours)),
        "throughput_mwh": float(np.dot(frame.charge_mw + frame.discharge_mw, hours)),
        "degradation_eur": float(frame.degradation_eur.sum()),
        "pv_curtailment_mwh": float(np.dot(frame.curtailment_mw, hours)),
        "soc_min_percent": None if soc_scale is None else float(energies.min() * soc_scale),
        "soc_max_percent": None if soc_scale is None else float(energies.max() * soc_scale),
        "soc_mean_percent": None if soc_scale is None else float(np.dot(
            (frame.energy_start_mwh + frame.energy_end_mwh) / 2, hours) / hours.sum() * soc_scale),
        "soc_final_percent": None if soc_scale is None else float(energies[-1] * soc_scale),
        "final_energy_mwh": float(energies[-1]),
        "soc_boundary_events": int((np.isclose(energies[1:], battery_min, atol=tolerance, rtol=0)
                                    | np.isclose(energies[1:], battery_max, atol=tolerance, rtol=0)).sum()),
        "power_boundary_events": int((np.maximum(frame.charge_mw, frame.discharge_mw) >= power - tolerance).sum()) if power else 0,
        "physical_safety_overrides": int((frame.safety_adjustment_mw > tolerance).sum()),
        "auction_revenue_eur": float(frame.auction_revenue_eur.sum()),
        "import_cost_eur": float(frame.import_fee_eur.sum()),
        "trading_fees_eur": float(frame.transaction_fee_eur.sum()),
        "realized_operating_profit_eur": float(frame.net_profit_eur.sum()),
        "expected_absolute_imbalance_first_actions_mwh": None if not expected.notna().any() else float(expected.sum()),
        "expected_first_action_intervals": int(expected.notna().sum()),
        "expected_cumulative_shortfall_first_actions_mwh": None if not expected_shortfall.notna().any() else float(expected_shortfall.sum()),
        "realized_minus_expected_absolute_imbalance_mwh": None if not expected.notna().any() else float(
            frame.loc[expected.notna(), "imbalance_mwh"].abs().sum() - expected.sum()),
        "mpc_solves": len(solves), "mpc_fallbacks": sum(bool(row.get("fallback_applied")) for row in solves),
        "mpc_timeouts": sum(any("time_limit" in stage["status"] for stage in row.get("stages", [])) for row in solves),
        "mpc_infeasibilities": sum(any(stage["status"] == "infeasible" for stage in row.get("stages", [])) for row in solves),
        "price_oracle_enabled": config["execution"]["dispatch_policy"].endswith("price_oracle"),
        "auction_price_oracle_enabled": (not config.get("energy_price_forecast", {}).get("auction_enabled", False)
            and config.get("gme_oracle_experiment", {}).get("price_information") == "perfect_foresight_actual_clearing_curves"),
        "settlement_forecast_enabled": any(row.get("price_forecast_enabled", False) for row in solves),
        "solver_runtime_mean_seconds": float(np.mean(runtimes)) if runtimes else None,
        "solver_runtime_median_seconds": float(np.median(runtimes)) if runtimes else None,
        "solver_runtime_p95_seconds": float(np.percentile(runtimes, 95)) if runtimes else None,
        "solver_runtime_max_seconds": float(np.max(runtimes)) if runtimes else None,
        "solver_runtime_total_seconds": float(np.sum(runtimes)),
    }
    # Health-aware reports distinguish falling usable capacity from nominal
    # depth coordinates, which remain fixed in the cycle-memory banks.
    if "available_capacity_mwh_start" in frame:
        start_soc = 100 * frame.energy_start_mwh / frame.available_capacity_mwh_start
        end_soc = 100 * frame.energy_end_mwh / frame.available_capacity_mwh_end
        all_soc = np.r_[start_soc.iloc[0], end_soc]
        end_capacity = frame.available_capacity_mwh_end.to_numpy(float)
        lower = end_capacity * config["battery"]["soc_min_percent"] / 100
        upper = end_capacity * config["battery"]["soc_max_percent"] / 100
        result.update(soc_min_percent=float(all_soc.min()), soc_max_percent=float(all_soc.max()),
            soc_mean_percent=float(np.dot((start_soc + end_soc) / 2, hours) / hours.sum()),
            soc_final_percent=float(end_soc.iloc[-1]),
            soc_reporting_basis="committed_available_capacity",
            initial_soh=float(frame.battery_soh_start.iloc[0]),
            final_soh=float(frame.battery_soh_end.iloc[-1]),
            final_available_capacity_mwh=float(end_capacity[-1]),
            soc_boundary_events=int((np.isclose(energies[1:], lower, atol=tolerance, rtol=0)
                | np.isclose(energies[1:], upper, atol=tolerance, rtol=0)).sum()),
            power_boundary_events=int((((frame.charge_power_limit_mw > tolerance)
                & (frame.charge_mw >= frame.charge_power_limit_mw - tolerance))
                | ((frame.discharge_power_limit_mw > tolerance)
                   & (frame.discharge_mw >= frame.discharge_power_limit_mw - tolerance))).sum()))
    return result


def save_mpc_outputs(config, destination, dispatch, state, projections, *, history_archives=None):
    """Save additive reporting alongside unchanged continuous replay artifacts."""
    destination = Path(destination)
    frame = _financial_dispatch(config, dispatch)
    solves = [] if state is None else state.solves
    plans = [] if state is None else state.plans
    scenarios = [] if state is None else state.scenarios
    actions = [] if state is None else state.issued_actions
    frame["absolute_imbalance_mwh"] = frame.imbalance_mwh.abs()
    frame["absolute_settlement_exposure_eur"] = frame.imbalance_cashflow_eur.abs()
    frame["executed_battery_mw"] = frame.discharge_mw - frame.charge_mw
    frame["planned_battery_mw"] = frame.requested_discharge_mw - frame.requested_charge_mw
    daily = []
    for day, selected in frame.groupby("delivery_day", sort=False):
        daily.append({"delivery_day": str(day), **controller_metrics(config, selected,
            [row for row in solves if row["delivery_day"] == day])})
    summary = controller_metrics(config, frame, solves)
    matched_current_pv = MpcSettings.from_config(config).current_pv_information == "realized_interval_average"
    # Whole-run cumulative shortage is a daily statistic, not an interday inventory.
    summary["maximum_cumulative_shortage_mwh"] = max(row["maximum_cumulative_shortage_mwh"] for row in daily)
    summary.update(controller=config["execution"]["dispatch_policy"],
        inventory_adjustment_in_realized_profit=False,
        mpc_current_pv_information=MpcSettings.from_config(config).current_pv_information,
        current_interval_pv_access_is_retrospective=matched_current_pv,
        tracking_information_advantage=config["execution"]["dispatch_policy"] == "track_committed_grid" and not matched_current_pv,
        fallback_information_advantage=any(row.get("fallback_applied") for row in solves) and not matched_current_pv)
    outputs = [("mpc_actions", actions), ("mpc_solves", solves), ("controller_daily_kpis", daily)]
    if validate_mpc_history_settings(config) == "memory":
        outputs.extend([("mpc_plans", plans), ("mpc_scenarios", scenarios)])
    else:
        if plans or scenarios:
            raise ValueError("Completed MPC horizon history must be archived before final reporting")
        summary["horizon_history"] = write_history_manifest(destination, history_archives or [])
    for name, rows in outputs:
        pd.DataFrame(rows).to_csv(destination / "csv" / f"{name}.csv", index=False)
        write_json(destination / "json" / f"{name}.json", rows)
    frame.to_csv(destination / "csv/controller_intervals.csv", index=False)
    write_json(destination / "json/controller_intervals.json", _records(frame))
    write_json(destination / "json/controller_kpis.json", summary)
    return {"mpc_metrics": summary}


def save_controller_comparison(config, destination, summaries, mode):
    """Compare matched dates, preserving the two experiments' different interpretations."""
    destination = Path(destination)
    baseline = summaries["track_committed_grid"]["mpc_metrics"]
    daily_frames, interval_frames, comparison = [], {}, []
    for controller, summary in summaries.items():
        output = Path(summary["output_dir"])
        daily = pd.read_csv(output / "csv/controller_daily_kpis.csv")
        daily["controller"] = controller
        daily_frames.append(daily)
        interval_frames[controller] = pd.read_csv(output / "csv/controller_intervals.csv")
        metrics = summary["mpc_metrics"]
        comparison.append({"controller": controller, **metrics,
            "incremental_profit_eur": metrics["realized_operating_profit_eur"] - baseline["realized_operating_profit_eur"],
            "absolute_imbalance_reduction_mwh": baseline["absolute_imbalance_mwh"] - metrics["absolute_imbalance_mwh"],
            "settlement_exposure_difference_eur": metrics["absolute_settlement_exposure_eur"] - baseline["absolute_settlement_exposure_eur"],
            "final_inventory_difference_mwh": metrics["final_energy_mwh"] - baseline["final_energy_mwh"],
            "auction_revenue_difference_eur": metrics["auction_revenue_eur"] - baseline["auction_revenue_eur"],
            "trading_fee_difference_eur": metrics["trading_fees_eur"] - baseline["trading_fees_eur"]})
    daily = pd.concat(daily_frames, ignore_index=True)
    reference = daily[daily.controller == "track_committed_grid"].set_index("delivery_day")
    daily["profit_minus_tracking_eur"] = daily.realized_operating_profit_eur - daily.delivery_day.map(reference.realized_operating_profit_eur)
    oracle_values = []
    for controller in summaries:
        if controller.endswith("price_oracle"):
            ordinary = controller.removesuffix("_price_oracle")
            if ordinary in summaries:
                left = daily[daily.controller == controller].set_index("delivery_day")
                right = daily[daily.controller == ordinary].set_index("delivery_day")
                oracle_values.extend({"delivery_day": day, "controller": controller,
                    "nonoracle_controller": ordinary, "settlement_improvement_eur": float(
                        left.loc[day, "imbalance_cashflow_eur"] - right.loc[day, "imbalance_cashflow_eur"])}
                    for day in left.index)
    pd.DataFrame(comparison).to_csv(destination / "controller_comparison.csv", index=False)
    daily.to_csv(destination / "paired_daily_metrics.csv", index=False)
    pd.DataFrame(oracle_values).to_csv(destination / "oracle_values.csv", index=False)
    write_json(destination / "controller_comparison.json", {"mode": mode, "controllers": comparison,
        "daily": _records(daily), "oracle_values": oracle_values,
        "interpretation": "Physical control only; identical submissions" if mode == "frozen_commitments" else
            "Physical control and consequent changes in later auction submissions"})
    if config.get("experiment_outputs", {}).get("plots", True):
        _comparison_plots(config, destination, daily, interval_frames, summaries, oracle_values)
    return comparison


def _comparison_plots(config, destination, daily, frames, summaries, oracle_values):
    """Render the ten requested plot families from paired, already settled outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = destination / "plots"
    output.mkdir(exist_ok=True)
    timezone = config.get("market", {}).get("timezone", "Europe/Rome")

    def save(fig, name):
        known = MpcSettings.from_config(config).current_pv_information == "realized_interval_average"
        information = "Retrospective: current-interval realized PV; future PV forecasts" if known else "Interval-start PV forecasts"
        replacement = config.get("pv", {}).get("forecast_replacement", {})
        models = replacement.get("models_by_farm", {})
        farm = config.get("farm_id")
        model = models.get(str(farm), models.get(farm))
        label = model["name"] if replacement.get("enabled") and model else "pvlib"
        fig.suptitle(f"{information} | {label}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output / name, dpi=130)
        plt.close(fig)

    # Cumulative imbalance, daily settlement and paired operating-profit differences.
    for filename, column, ylabel in (
        ("cumulative_absolute_imbalance.png", "absolute_imbalance_mwh", "Cumulative absolute imbalance (MWh)"),
        ("daily_signed_settlement.png", "imbalance_cashflow_eur", "Signed settlement cash flow (EUR)"),
        ("daily_profit_differences.png", "profit_minus_tracking_eur", "Operating profit minus tracking (EUR)")):
        fig, axis = plt.subplots(figsize=(12, 5))
        for controller, frame in frames.items():
            selected = daily[daily.controller == controller]
            if column == "absolute_imbalance_mwh":
                boundaries = pd.DatetimeIndex([pd.Timestamp(frame.start.iloc[0]),
                    *pd.to_datetime(frame.end, utc=True)]).tz_convert(timezone)
                axis.step(boundaries, np.r_[0, frame.imbalance_mwh.abs().cumsum()], where="post", label=controller)
                axis.set_xlabel(f"Delivery time ({timezone})")
            else:
                axis.plot(selected.delivery_day, selected[column], marker="o", label=controller)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7)
        save(fig, filename)
    # Plot each smoke day; larger future runs use the configured representative-day cap.
    day_limit = realtime_daily_plot_limit(config)
    selected_days = daily.delivery_day.drop_duplicates().iloc[:day_limit]
    for previous in output.glob("soc_and_planned_executed_power_*.png"):
        previous_day = previous.stem.removeprefix("soc_and_planned_executed_power_")
        if previous_day not in set(selected_days):
            previous.unlink()
    for day in selected_days:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for controller, frame in frames.items():
            selected = frame[frame.delivery_day == day]
            times = pd.to_datetime(selected.start, utc=True).dt.tz_convert(timezone)
            ends = pd.to_datetime(selected.end, utc=True).dt.tz_convert(timezone)
            capacity = config["battery"]["capacity_mwh"]
            start_capacity = selected.get("available_capacity_mwh_start", capacity)
            end_capacity = selected.get("available_capacity_mwh_end", capacity)
            start_soc = 100 * selected.energy_start_mwh / start_capacity if capacity else selected.energy_start_mwh
            end_soc = 100 * selected.energy_end_mwh / end_capacity if capacity else selected.energy_end_mwh
            plot_boundary_series(axes[0], times, ends, start_soc, end_soc, label=controller)
            plot_interval_series(axes[1], times, ends, selected.executed_battery_mw, label=controller)
            plot_interval_series(axes[1], times, ends, selected.planned_battery_mw, linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Actual SOC (%)")
        axes[1].set_ylabel("Battery MW: executed / planned dashed")
        axes[1].set_xlabel(f"Delivery time ({timezone})")
        axes[0].legend(fontsize=7)
        save(fig, f"soc_and_planned_executed_power_{day}.png")
    fig, axis = plt.subplots(figsize=(10, 6))
    for controller, frame in frames.items():
        axis.scatter(frame.imbalance_price_eur_mwh, frame.imbalance_mwh, s=8, alpha=0.5, label=controller)
    axis.set(xlabel="Realized imbalance price (EUR/MWh)", ylabel="Signed imbalance (MWh)")
    axis.legend(fontsize=7)
    save(fig, "imbalance_volume_vs_price.png")
    fig, axis = plt.subplots(figsize=(10, 5))
    oracle = pd.DataFrame(oracle_values)
    if not oracle.empty:
        for controller, selected in oracle.groupby("controller"):
            axis.plot(selected.delivery_day, selected.settlement_improvement_eur, marker="o", label=controller)
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "No imbalance-price oracle in this comparison", ha="center", transform=axis.transAxes)
    axis.set_ylabel("Oracle minus nonoracle signed settlement (EUR)")
    save(fig, "price_oracle_value.png")
    fig, axis = plt.subplots(figsize=(12, 5))
    for controller, summary in summaries.items():
        projections = pd.read_csv(Path(summary["output_dir"]) / "csv/state_projections.csv")
        if not projections.empty:
            selected = projections[projections.boundary_kind == "delivery_midnight"].dropna(subset=["prediction_error_mwh"])
            axis.scatter(pd.to_datetime(selected.cutoff_utc, utc=True), selected.prediction_error_mwh, label=controller)
    axis.set_ylabel("Realized minus projected midnight energy (MWh)")
    if axis.has_data():
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "Frozen ledger: no auction boundary projections", ha="center", transform=axis.transAxes)
    save(fig, "midnight_projection_errors.png")
    fig, axis = plt.subplots(figsize=(10, 6))
    for controller, selected in daily.groupby("controller"):
        axis.scatter(selected.absolute_imbalance_mwh, selected.realized_operating_profit_eur, label=controller)
    axis.set(xlabel="Daily absolute imbalance (MWh)", ylabel="Daily realized operating profit (EUR)")
    axis.legend(fontsize=7)
    save(fig, "profit_vs_absolute_imbalance.png")
    fig, axis = plt.subplots(figsize=(12, 5))
    samples, labels = [], []
    for controller, summary in summaries.items():
        rows = pd.read_json(Path(summary["output_dir"]) / "json/mpc_solves.json")
        if not rows.empty:
            samples.append(rows.runtime_seconds.to_numpy())
            labels.append(controller)
    if samples:
        axis.boxplot(samples, tick_labels=labels, showfliers=True)
        axis.tick_params(axis="x", labelrotation=15)
    axis.set_ylabel("MPC Gurobi runtime per decision (s)")
    save(fig, "solver_runtime_comparison.png")
