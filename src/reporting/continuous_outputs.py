"""Saved accounting, diagnostics and plots for uninterrupted auction operation."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .optimization_timing import optimization_timing_report, solve_timing, timing_statistics
from .plots import plot_boundary_series, plot_interval_series, pv_power_allocation


def _json_value(value):
    """Serialize diagnostics without non-standard JSON NaN or leaked object reprs."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, document):
    """Keep this reporting module independent of the auction replay engine."""
    path.write_text(json.dumps(_json_value(document), indent=2, allow_nan=False) + "\n")


def _financial_dispatch(config, dispatch):
    """Validate a continuous execution ledger and calculate settled interval profit."""
    frame = pd.DataFrame(dispatch).copy()
    required = {"interval_id", "start", "end", "delivery_day", "energy_start_mwh",
        "energy_end_mwh", "pv_ac_mw", "charge_mw", "discharge_mw", "curtailment_mw",
        "grid_mw", "net_position_mwh", "imbalance_mwh", "imbalance_cashflow_eur",
        "degradation_eur", "import_fee_eur", "transaction_fee_eur", "auction_revenue_eur"}
    if frame.empty or not required.issubset(frame):
        raise ValueError(f"Continuous reporting requires execution rows and columns: {sorted(required - set(frame))}")
    frame["start"] = pd.to_datetime(frame.start, utc=True)
    frame["end"] = pd.to_datetime(frame.end, utc=True)
    frame = frame.sort_values("start").reset_index(drop=True)
    if frame.interval_id.duplicated().any() or (frame.end <= frame.start).any():
        raise ValueError("Continuous execution intervals must be unique with positive durations")
    if len(frame) > 1 and not np.array_equal(frame.end.iloc[:-1].to_numpy(), frame.start.iloc[1:].to_numpy()):
        raise ValueError("Continuous execution intervals must have no gaps or overlaps")
    timezone = config.get("market", {}).get("timezone", "Europe/Rome")
    frame["delivery_day"] = frame.delivery_day.astype(str)
    if not frame.start.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d").eq(frame.delivery_day).all():
        raise ValueError("Continuous delivery dates must follow the configured local market timezone")
    numeric = sorted(required - {"interval_id", "start", "end", "delivery_day"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Continuous realized accounting requires finite execution and settlement values")
    tolerance = max(float(config.get("solver", {}).get("feasibility_tolerance") or 1e-6), 1e-7)
    if len(frame) > 1 and not np.allclose(frame.energy_end_mwh.iloc[:-1],
            frame.energy_start_mwh.iloc[1:], rtol=0, atol=tolerance):
        raise ValueError("Continuous battery state must carry across every interval and midnight")
    frame["duration_hours"] = (frame.end - frame.start).dt.total_seconds() / 3600
    if "imbalance_penalty_eur" not in frame:
        frame["imbalance_penalty_eur"] = 0.0
    if not np.isfinite(frame.imbalance_penalty_eur).all():
        raise ValueError("Continuous imbalance penalties must be finite")
    frame["net_profit_eur"] = (frame.auction_revenue_eur + frame.imbalance_cashflow_eur
        - frame.degradation_eur - frame.import_fee_eur - frame.transaction_fee_eur
        - frame.imbalance_penalty_eur)
    frame["cumulative_net_profit_eur"] = frame.net_profit_eur.cumsum()
    return frame


def _daily_kpis(config, strategy, frame, solves, initial_expected):
    """Attribute market cash and physical costs to delivery dates, without resets."""
    capacity = float(config["battery"]["capacity_mwh"])
    rows = []
    for day, daily in frame.groupby("delivery_day", sort=True):
        duration = daily.duration_hours.to_numpy(float)
        solved = [row for row in solves if str(row.get("delivery_day")) == day]
        initial, final = float(daily.energy_start_mwh.iloc[0]), float(daily.energy_end_mwh.iloc[-1])
        profit = float(daily.net_profit_eur.sum())
        row = {"strategy": strategy, "delivery_day": day, "study_date": day,
            "interval_count": len(daily), "delivery_hours": float(duration.sum()),
            "initial_energy_mwh": initial, "final_energy_mwh": final,
            "initial_soc_percent": initial / capacity * 100,
            "final_soc_percent": final / capacity * 100,
            "inventory_change_mwh": final - initial,
            "net_profit_eur": profit, "total_gain_eur": profit,
            "auction_solves": len(solved),
            "fallback_count": sum(bool(item.get("fallback_applied")) for item in solved),
            "initial_expected_profit_eur": initial_expected.get(day),
            "realized_minus_expected_profit_eur": (None if initial_expected.get(day) is None
                else profit - float(initial_expected[day]))}
        row.update({column: float(daily[column].sum()) for column in (
            "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
            "import_fee_eur", "transaction_fee_eur", "imbalance_penalty_eur")})
        row.update({f"{name}_mwh": float(np.dot(daily[f"{name}_mw"], duration))
            for name in ("charge", "discharge", "curtailment")})
        row["throughput_mwh"] = row["charge_mwh"] + row["discharge_mwh"]
        row["degradation_cost_eur"] = row["degradation_eur"]
        row["pv_realized_energy_mwh"] = float(np.dot(daily.pv_ac_mw, duration))
        row["curtailment_percent"] = (row["curtailment_mwh"] / row["pv_realized_energy_mwh"] * 100
            if row["pv_realized_energy_mwh"] else 0.0)
        row["grid_export_mwh"] = float(np.dot(daily.grid_mw.clip(lower=0), duration))
        row["grid_import_mwh"] = float(np.dot(-daily.grid_mw.clip(upper=0), duration))
        row["net_position_mwh"] = float(daily.net_position_mwh.sum())
        row["signed_imbalance_mwh"] = float(daily.imbalance_mwh.sum())
        row["absolute_imbalance_mwh"] = float(daily.imbalance_mwh.abs().sum())
        row["positive_imbalance_mwh"] = float(daily.imbalance_mwh.clip(lower=0).sum())
        row["negative_imbalance_mwh"] = float(-daily.imbalance_mwh.clip(upper=0).sum())
        if "pv_ac_forecast_mw" in daily and np.isfinite(daily.pv_ac_forecast_mw).all():
            error = daily.pv_ac_mw - daily.pv_ac_forecast_mw
            row.update(pv_forecast_energy_mwh=float(np.dot(daily.pv_ac_forecast_mw, duration)),
                pv_forecast_mae_mw=float(error.abs().mean()),
                pv_forecast_rmse_mw=float(np.sqrt(np.mean(error ** 2))))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_strategy(config, destination, strategy, frame, plans):
    """Plot uninterrupted delivery and each daily PV/dispatch view through final midnight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timezone = config.get("market", {}).get("timezone", "Europe/Rome")
    starts, ends = frame.start.dt.tz_convert(timezone), frame.end.dt.tz_convert(timezone)
    capacity = float(config["battery"]["capacity_mwh"])
    specs = (
        ("continuous_soc.png", "Battery SOC (%)", "SOC evolution", "soc"),
        ("continuous_dispatch.png", "Power (MW AC)", "Continuous delivery", "dispatch"),
        ("cumulative_profit.png", "Cumulative profit (EUR)", "Settled operating profit", "profit"))
    for filename, ylabel, title, kind in specs:
        fig, axis = plt.subplots(figsize=(13, 4), layout="constrained")
        if kind == "soc":
            plot_boundary_series(axis, starts, ends, frame.energy_start_mwh / capacity * 100,
                frame.energy_end_mwh / capacity * 100, label="Realized SOC")
            for key in ("soc_min_percent", "soc_max_percent"):
                if key in config["battery"]:
                    axis.axhline(config["battery"][key], color="gray", linestyle=":", linewidth=0.7)
        elif kind == "dispatch":
            plot_interval_series(axis, starts, ends, frame.discharge_mw - frame.charge_mw,
                label="Battery (+ discharge)", linewidth=0.9)
            plot_interval_series(axis, starts, ends, frame.grid_mw, label="Realized grid", alpha=0.7)
            plot_interval_series(axis, starts, ends, frame.net_position_mwh / frame.duration_hours,
                label="Committed grid", linestyle="--", linewidth=0.7)
        else:
            plot_boundary_series(axis, starts, ends,
                frame.cumulative_net_profit_eur - frame.net_profit_eur,
                frame.cumulative_net_profit_eur, label="Realized profit")
        axis.set(title=f"{strategy}: {title}", ylabel=ylabel, xlabel=f"Delivery time ({timezone})")
        axis.set_xlim(starts.iloc[0], ends.iloc[-1])
        axis.grid(alpha=0.2)
        axis.legend()
        fig.savefig(destination / filename, dpi=140)
        plt.close(fig)
    daily_destination = destination / "daily"
    daily_destination.mkdir(exist_ok=True)
    for day, daily in frame.groupby("delivery_day", sort=True):
        starts, ends = daily.start.dt.tz_convert(timezone), daily.end.dt.tz_convert(timezone)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), layout="constrained")
        if not plans.empty and "delivery_day" in plans:
            for auction, selected in plans.loc[plans.delivery_day.astype(str).eq(day)].groupby("auction", sort=False):
                if {"interval_start_utc", "interval_end_utc", "pv_ac_forecast_mw"}.issubset(selected):
                    selected = selected.sort_values("interval_start_utc")
                    plot_interval_series(axes[0], pd.to_datetime(selected.interval_start_utc, utc=True).dt.tz_convert(timezone),
                        pd.to_datetime(selected.interval_end_utc, utc=True).dt.tz_convert(timezone),
                        selected.pv_ac_forecast_mw, linestyle="--", alpha=0.7, label=f"{auction} PV forecast")
        elif "pv_ac_forecast_mw" in daily:
            plot_interval_series(axes[0], starts, ends, daily.pv_ac_forecast_mw,
                linestyle="--", label="Active PV forecast")
        plot_interval_series(axes[0], starts, ends, daily.pv_ac_mw, color="black", label="Realized PV")
        plot_interval_series(axes[1], starts, ends, daily.discharge_mw - daily.charge_mw,
            color="C2", label="Realized battery (+ discharge)")
        if {"requested_charge_mw", "requested_discharge_mw"}.issubset(daily):
            plot_interval_series(axes[1], starts, ends, daily.requested_discharge_mw - daily.requested_charge_mw,
                linestyle="--", color="C1", label="Scheduled battery (active plan)")
        # Use the physical battery scale so floating-point roundoff around zero
        # cannot become an apparently substantial trajectory on an autoscaled axis.
        power = float(config["battery"]["max_power_mw"])
        if power > 0:
            axes[1].set_ylim(-power, power)
        for axis in axes:
            axis.set_ylabel("Power (MW AC)")
            axis.grid(alpha=0.2)
            axis.legend(ncol=2)
        axes[-1].set_xlabel(f"Delivery time ({timezone})")
        axes[-1].set_xlim(starts.iloc[0], ends.iloc[-1])
        fig.suptitle(f"{strategy}: {day}")
        fig.savefig(daily_destination / f"{day}.png", dpi=140)
        plt.close(fig)


def _projection_plot_rows(config, projections, boundary_kind):
    """Keep midnight and IDA3 noon predictions on distinct delivery boundaries."""
    frame = pd.DataFrame(projections).copy()
    required = {"planning_start_utc", "auction", "predicted_initial_energy_mwh",
        "actual_initial_energy_mwh"}
    if frame.empty or not required.issubset(frame):
        return pd.DataFrame()
    frame["boundary"] = pd.to_datetime(frame.planning_start_utc, utc=True)
    local = frame.boundary.dt.tz_convert(config.get("market", {}).get("timezone", "Europe/Rome"))
    if "boundary_kind" not in frame:
        frame["boundary_kind"] = np.where((local.dt.hour == 0) & (local.dt.minute == 0),
            "delivery_midnight", "other")
    selected = frame.loc[frame.boundary_kind.eq(boundary_kind)].copy()
    values = ["predicted_initial_energy_mwh", "actual_initial_energy_mwh"]
    if selected.empty:
        return selected
    selected[values] = selected[values].apply(pd.to_numeric, errors="coerce")
    selected = selected.loc[np.isfinite(selected[values].to_numpy(float)).all(axis=1)].copy()
    selected["prediction_error_mwh"] = (selected.actual_initial_energy_mwh
        - selected.predicted_initial_energy_mwh)
    return selected


def _plot_projection_errors(config, destination, projections, *, filename_suffix="", strategy_label=None):
    """Plot actual-minus-predicted SOC independently for midnight and noon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timezone = config.get("market", {}).get("timezone", "Europe/Rome")
    scale = 100 / float(config["battery"]["capacity_mwh"])
    generated = []
    for kind, filename, label in (
            ("delivery_midnight", "midnight_soc_prediction_errors.png", "Delivery midnight"),
            ("ida3_noon", "ida3_noon_soc_prediction_errors.png", "IDA3 noon")):
        frame = _projection_plot_rows(config, projections, kind)
        if frame.empty:
            continue
        if "strategy" not in frame:
            frame["strategy"] = ""
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 8), layout="constrained")
        for (strategy, auction), selected in frame.groupby(["strategy", "auction"], sort=False):
            selected = selected.sort_values("boundary")
            times = selected.boundary.dt.tz_convert(timezone)
            name = f"{strategy}: {auction}" if strategy else str(auction)
            axes[0].plot(times, selected.predicted_initial_energy_mwh * scale,
                marker=".", linestyle="--", linewidth=0.9, label=f"{name} prediction")
            axes[1].plot(times, selected.prediction_error_mwh,
                marker=".", linewidth=0.9, label=name)
        for strategy, selected in frame.groupby("strategy", sort=False):
            actual = selected.sort_values("boundary").drop_duplicates("boundary")
            axes[0].plot(actual.boundary.dt.tz_convert(timezone), actual.actual_initial_energy_mwh * scale,
                linewidth=1.6, label=f"{strategy}: realized" if strategy else "Realized SOC")
        axes[1].axhline(0, color="black", linewidth=0.6)
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, fontsize="small")
        axes[0].set_ylabel("Boundary SOC (%)")
        axes[1].set(ylabel="Realized minus predicted energy (MWh)",
            xlabel=f"{label} boundary ({timezone})")
        prefix = f"{strategy_label}: " if strategy_label else ""
        fig.suptitle(f"{prefix}{label}: forecast-based SOC predictions and realized state")
        path = destination / f"{Path(filename).stem}{filename_suffix}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        generated.append(path)
    return generated


def save_continuous_projection_plots(config, destination, summaries):
    """Refresh only separate strategy boundary plots from saved projection data."""
    destination = Path(destination) / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = ([{"strategy": key, **summary} for key, summary in summaries.items()]
        if isinstance(summaries, dict) else list(summaries))
    strategies = [summary["strategy"] for summary in summaries]
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("Projection plots require a nonempty unique set of solved strategies")
    generated = []
    for summary in summaries:
        strategy = summary["strategy"]
        if strategy not in {"integrated", "current_auction_only", "perfect_pv"}:
            raise ValueError(f"Unsupported continuous projection strategy {strategy}")
        path = Path(summary["output_dir"]) / "json/state_projections.json"
        projections = json.loads(path.read_text())
        generated.extend(_plot_projection_errors(config, destination, projections,
            filename_suffix=f"_{strategy}", strategy_label=strategy))
    # Keep old figures until every replacement has rendered successfully.
    for name in ("midnight_soc_prediction_errors", "ida3_noon_soc_prediction_errors"):
        previous = destination / f"{name}.png"
        if previous.is_file() and any(path.stem.startswith(f"{name}_") for path in generated):
            previous.unlink()
    return [str(path) for path in generated]


def save_continuous_strategy_outputs(config, destination, strategy, dispatch, plans,
        ledger, solves, projections, summary_extra=None):
    """Save one uninterrupted strategy and return its additive delivery-day KPIs."""
    destination = Path(destination)
    for kind in ("csv", "json"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    frame = _financial_dispatch(config, dispatch)
    plans = pd.DataFrame(plans).copy()
    if plans.empty and not len(plans.columns):
        plans = pd.DataFrame(columns=["auction", "delivery_day", "interval_id",
            "optimizer_cutoff_utc", "interval_start_utc", "interval_end_utc",
            "pv_ac_forecast_mw", "pv_forecast_vintage_utc", "pv_forecast_available_at_utc",
            "charge_mw", "discharge_mw", "curtailment_mw", "soc_start_mwh", "soc_end_mwh"])
    solves, ledger, projections = list(solves), list(ledger), list(projections)
    extra = dict(summary_extra or {})
    supplied_attempts = "optimization_attempts" in extra
    attempts = list(extra.pop("optimization_attempts", solves))
    initial_expected = extra.pop("initial_expected_profit_by_day", {})
    if ledger:
        for interval_column, ledger_column in (("auction_revenue_eur", "cashflow_eur"),
                ("transaction_fee_eur", "transaction_fee_eur")):
            expected = sum(float(row.get(ledger_column, 0.0)) for row in ledger)
            if not np.isclose(frame[interval_column].sum(), expected, rtol=1e-10, atol=1e-7):
                raise ValueError(f"Continuous {interval_column} does not reconcile with submitted trade ledger")
            for day, executed in frame.groupby("delivery_day", sort=False):
                booked = sum(float(row.get(ledger_column, 0.0)) for row in ledger
                    if str(row.get("delivery_day")) == day)
                if not np.isclose(executed[interval_column].sum(), booked, rtol=1e-10, atol=1e-7):
                    raise ValueError(f"Continuous {interval_column} does not reconcile by delivery day {day}")
    daily = _daily_kpis(config, strategy, frame, solves, initial_expected)
    category = "perfect_pv_reference" if strategy == "perfect_pv" else "replay"
    timing_rows = [solve_timing(row, category=category, strategy=strategy,
        auction=row.get("current_auction"), delivery_day=row.get("delivery_day"),
        cutoff_utc=row.get("cutoff_utc", row.get("cutoff")), status=row.get("status"),
        fallback_applied=bool(row.get("fallback_applied")),
        rolled_back_for_resume=bool(row.get("rolled_back_for_resume"))) for row in attempts]
    timing = optimization_timing_report(timing_rows)
    timing["by_delivery_day"] = {day: optimization_timing_report([
        row for row in timing_rows if str(row.get("delivery_day")) == day])
        for day in daily.delivery_day}
    timing["daily_total_runtime_seconds"] = timing_statistics(
        report["gurobi_runtime_seconds"]["total"] if report["missing_runtime_count"] == 0 else None
        for report in timing["by_delivery_day"].values())
    components = {key: float(daily[key].sum()) for key in (
        "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur", "import_fee_eur",
        "transaction_fee_eur", "imbalance_penalty_eur")}
    profits = daily.net_profit_eur.to_numpy(float)
    summary = {**extra, "strategy": strategy, "output_dir": str(destination),
        "delivery_days": len(daily), "interval_count": len(frame),
        "first_delivery_day": daily.delivery_day.iloc[0], "last_delivery_day": daily.delivery_day.iloc[-1],
        "components": components, "total_gain_eur": float(profits.sum()),
        "net_profit_eur": float(profits.sum()), "realized_operating_profit_eur": float(profits.sum()),
        "mean_daily_profit_eur": float(profits.mean()), "std_daily_profit_eur": float(profits.std(ddof=0)),
        "min_daily_profit_eur": float(profits.min()), "max_daily_profit_eur": float(profits.max()),
        "initial_energy_mwh": float(daily.initial_energy_mwh.iloc[0]),
        "final_energy_mwh": float(daily.final_energy_mwh.iloc[-1]),
        "initial_soc_percent": float(daily.initial_soc_percent.iloc[0]),
        "final_soc_percent": float(daily.final_soc_percent.iloc[-1]),
        "auction_solves": len(solves), "fallback_count": int(daily.fallback_count.sum()),
        "optimization_timing": timing, "daily_kpis": daily.to_dict("records"),
        "profit_attribution": "Delivery day; trade ledger preserves submission/publication timestamps and transaction-month fees",
        "std_ddof": 0}
    summary.update({key: float(daily[key].sum()) for key in (
        "charge_mwh", "discharge_mwh", "throughput_mwh", "curtailment_mwh",
        "absolute_imbalance_mwh", "signed_imbalance_mwh", "pv_realized_energy_mwh",
        "grid_import_mwh", "grid_export_mwh")})
    frame.to_csv(destination / "csv/realized_dispatch.csv", index=False)
    daily.to_csv(destination / "csv/daily_kpis.csv", index=False)
    plans.to_csv(destination / "csv/expected_dispatch_plans.csv", index=False)
    pd.DataFrame(ledger, columns=None if ledger else ["trade_id", "auction", "delivery_day",
        "interval_id", "submission_time", "publication_time", "quantity_mwh",
        "price_eur_per_mwh", "cashflow_eur", "transaction_fee_eur"]).to_csv(
            destination / "csv/accepted_trades.csv", index=False)
    pd.DataFrame(projections, columns=None if projections else ["delivery_day", "auction",
        "cutoff_utc", "planning_start_utc", "predicted_initial_energy_mwh",
        "actual_initial_energy_mwh", "prediction_error_mwh"]).to_csv(
            destination / "csv/state_projections.csv", index=False)
    pv_power_allocation(frame).to_csv(destination / "csv/pv_power_allocation.csv", index=False)
    _write_json(destination / "json/kpis.json", summary)
    _write_json(destination / "json/optimization_timing.json", timing)
    _write_json(destination / "json/solves.json", solves)
    if supplied_attempts:
        _write_json(destination / "json/optimization_attempts.json", attempts)
    _write_json(destination / "json/state_projections.json", projections)
    _write_json(destination / "json/accepted_trades.json", ledger)
    if config.get("experiment_outputs", {}).get("plots", True):
        (destination / "plots").mkdir(exist_ok=True)
        _plot_strategy(config, destination / "plots", strategy, frame, plans)
        _plot_projection_errors(config, destination / "plots", projections)
    return summary


def save_daily_imbalance_comparison(config, destination, daily):
    """Plot daily surplus and shortfall energy for each trading strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    daily = pd.DataFrame(daily)
    strategies = [(key, label) for key, label in (
        ("integrated", "Integrated"), ("current_auction_only", "Current auction only"))
        if key in set(daily.strategy)]
    if not strategies:
        return None
    destination = Path(destination) / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(strategies), 1, sharex=True, sharey=True,
        figsize=(13, 4 * len(strategies)), layout="constrained", squeeze=False)
    for axis, (strategy, label) in zip(axes[:, 0], strategies):
        selected = daily.loc[daily.strategy.eq(strategy)].sort_values("delivery_day")
        days = pd.to_datetime(selected.delivery_day)
        axis.bar(days, selected.positive_imbalance_mwh, width=0.75,
            color="teal", label="Surplus delivered")
        axis.bar(days, -selected.negative_imbalance_mwh, width=0.75,
            color="indianred", label="Delivery shortfall")
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(ylabel="Daily imbalance (MWh)", axisbelow=True)
        axis.set_title(label, loc="left")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1, 0].set_xlabel(f"Delivery date ({config.get('market', {}).get('timezone', 'Europe/Rome')})")
    fig.suptitle("Daily realized imbalances\nSurplus and shortfall summed separately over quarter-hour deliveries")
    path = destination / "daily_imbalances_by_strategy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_continuous_comparison(config, destination, summaries):
    """Compare matched continuous trajectories and preserve each strategy's timing."""
    destination = Path(destination)
    for kind in ("csv", "json"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    summaries = list(summaries.values()) if isinstance(summaries, dict) else list(summaries)
    by_strategy = {item["strategy"]: item for item in summaries}
    main_strategies = [key for key in by_strategy if key != "perfect_pv"]
    if len(by_strategy) != len(summaries) or not main_strategies:
        raise ValueError("Continuous comparison requires unique strategies and a trading strategy")
    fingerprints = [summary.get("input_fingerprint") for summary in summaries]
    supplied_fingerprints = [value for value in fingerprints if value is not None]
    if supplied_fingerprints and (len(supplied_fingerprints) != len(fingerprints)
            or any(value != supplied_fingerprints[0] for value in supplied_fingerprints)):
        raise ValueError("Continuous strategies require identical complete input fingerprints")
    common_fingerprint = supplied_fingerprints[0] if supplied_fingerprints else None
    reference_strategy = "integrated" if "integrated" in by_strategy else main_strategies[0]
    reference = pd.DataFrame(by_strategy[reference_strategy]["daily_kpis"]).set_index("delivery_day")
    comparisons, daily_rows, paired, timings = [], [], [], []
    common_grid = None
    for strategy, summary in by_strategy.items():
        daily = pd.DataFrame(summary["daily_kpis"])
        if daily.delivery_day.duplicated().any() or set(daily.delivery_day) != set(reference.index):
            raise ValueError("Continuous comparisons require identical delivery dates")
        if not np.isfinite(daily.net_profit_eur).all():
            raise ValueError("Continuous comparisons require finite settled profits")
        execution = pd.read_csv(Path(summary["output_dir"]) / "csv/realized_dispatch.csv")
        columns = ["interval_id", "start", "end", "pv_ac_mw"]
        if common_grid is None:
            common_grid = execution[columns].copy()
        elif not common_grid.equals(execution[columns]):
            raise ValueError("Continuous strategies must use the same delivery grid and realized PV")
        daily["strategy"] = strategy
        daily["reference_strategy"] = reference_strategy
        daily["reference_profit_eur"] = daily.delivery_day.map(reference.net_profit_eur)
        daily["profit_minus_reference_eur"] = daily.net_profit_eur - daily.reference_profit_eur
        paired_columns = ["strategy", "delivery_day", "net_profit_eur", "reference_strategy",
            "reference_profit_eur", "profit_minus_reference_eur"]
        if reference_strategy == "integrated":
            daily["integrated_profit_eur"] = daily.reference_profit_eur
            daily["profit_minus_integrated_eur"] = daily.profit_minus_reference_eur
            paired_columns += ["integrated_profit_eur", "profit_minus_integrated_eur"]
        daily_rows.extend(daily.to_dict("records"))
        paired.extend(daily[paired_columns].to_dict("records"))
        differences = daily.profit_minus_reference_eur.to_numpy(float)
        comparison_row = {key: summary[key] for key in (
            "strategy", "delivery_days", "total_gain_eur", "mean_daily_profit_eur", "std_daily_profit_eur",
            "min_daily_profit_eur", "max_daily_profit_eur", "initial_soc_percent", "final_soc_percent",
            "final_energy_mwh", "auction_solves", "fallback_count")}
        comparison_row.update(reference_strategy=reference_strategy,
            profit_minus_reference_eur=float(differences.sum()),
            mean_daily_profit_minus_reference_eur=float(differences.mean()),
            std_daily_profit_minus_reference_eur=float(differences.std(ddof=0)))
        if reference_strategy == "integrated":
            comparison_row.update(profit_minus_integrated_eur=float(differences.sum()),
                mean_daily_profit_minus_integrated_eur=float(differences.mean()),
                std_daily_profit_minus_integrated_eur=float(differences.std(ddof=0)))
        comparisons.append(comparison_row)
        timings.extend(summary["optimization_timing"]["solves"])
    daily = pd.DataFrame(daily_rows)
    comparison = pd.DataFrame(comparisons)
    daily.to_csv(destination / "csv/daily_strategy_comparison.csv", index=False)
    comparison.to_csv(destination / "csv/strategy_comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(destination / "csv/daily_profit_differences.csv", index=False)
    timing = optimization_timing_report(timings)
    timing["by_strategy"] = {key: value["optimization_timing"] for key, value in by_strategy.items()}
    report = {"reference_strategy": reference_strategy, "delivery_days": len(reference),
        "comparison_basis": "Continuous realized operating profit under the same daily planning horizon; no daily SOC reset or terminal inventory salvage",
        "common_delivery_grid_and_realized_pv_verified": True,
        "common_input_fingerprint": common_fingerprint,
        "common_prices_weather_settlement_and_assets_verified": common_fingerprint is not None,
        "strategies": comparisons, "daily": daily.to_dict("records"), "std_ddof": 0}
    if "perfect_pv" in by_strategy:
        report["same_horizon_perfect_pv_profit_eur"] = by_strategy["perfect_pv"]["total_gain_eur"]
    _write_json(destination / "json/comparison.json", report)
    _write_json(destination / "json/optimization_timing.json", timing)
    if config.get("experiment_outputs", {}).get("plots", True):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (destination / "plots").mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), layout="constrained")
        for strategy, selected in daily.groupby("strategy", sort=False):
            selected = selected.sort_values("delivery_day")
            days = pd.to_datetime(selected.delivery_day)
            axes[0].plot(days, selected.net_profit_eur.cumsum(), marker=".", label=strategy)
            axes[1].plot(days, selected.net_profit_eur, marker=".", label=strategy)
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0].set_ylabel("Cumulative profit (EUR)")
        axes[1].set(ylabel="Daily profit (EUR)", xlabel="Delivery date")
        fig.suptitle("Continuous optimization: strategy comparison")
        fig.savefig(destination / "plots/continuous_comparison.png", dpi=140)
        plt.close(fig)
        save_continuous_projection_plots(config, destination, by_strategy)
        save_daily_imbalance_comparison(config, destination, daily)
    return report
