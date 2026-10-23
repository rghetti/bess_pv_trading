"""Compare saved continuous strategy runs that differ only in PV forecasting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.domain import battery_parameters_from_config


def read_json(path: Path):
    """Read one saved JSON artifact with a useful path on failure."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def load_run(path: Path):
    """Load the realized summary, daily ledger and input fingerprints."""
    manifest = read_json(path / "json/manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Continuous run is not complete: {path}")
    return {
        "path": path,
        "manifest": manifest,
        "provenance": read_json(path / "json/input_provenance.json"),
        "config": read_json(path / "json/study_config.json"),
        "integrated": read_json(path / "strategies/integrated/json/kpis.json"),
        "integrated_daily": pd.read_csv(path / "strategies/integrated/csv/daily_kpis.csv"),
        "integrated_dispatch": pd.read_csv(path / "strategies/integrated/csv/realized_dispatch.csv"),
        "integrated_trades": pd.read_csv(path / "strategies/integrated/csv/accepted_trades.csv"),
        "integrated_forecasts": pd.read_csv(path / "strategies/integrated/csv/pv_forecast_inputs.csv"),
        "integrated_solves": read_json(path / "strategies/integrated/json/solves.json"),
        "perfect": read_json(path / "strategies/perfect_pv/json/kpis.json"),
        "perfect_daily": pd.read_csv(path / "strategies/perfect_pv/csv/daily_kpis.csv"),
        "perfect_dispatch": pd.read_csv(path / "strategies/perfect_pv/csv/realized_dispatch.csv"),
        "perfect_trades": pd.read_csv(path / "strategies/perfect_pv/csv/accepted_trades.csv"),
        "perfect_solves": read_json(path / "strategies/perfect_pv/json/solves.json"),
    }


def verify_shared_inputs(neural, pvlib):
    """Prove the two forecasts faced identical local observations and markets."""
    left, right = neural["provenance"], pvlib["provenance"]
    if set(left["days"]) != set(right["days"]):
        raise ValueError("Runs cover different delivery dates")
    mismatches = []
    for day in left["days"]:
        for key in ("realized_pv_sha256", "auction_price_sha256"):
            if left["days"][day][key] != right["days"][day][key]:
                mismatches.append(f"{day}/{key}")
    for key in ("settlement_values_sha256",):
        if left.get(key) != right.get(key):
            mismatches.append(key)
    def weather_views(provenance):
        """Compare pinned source weather before forecast replacement is applied."""
        return {(row["cutoff_utc"], row["first_interval_utc"], row["last_interval_utc"]):
            row["aligned_weather_pv_sha256"] for row in provenance["weather_forecasts"]}

    if weather_views(left) != weather_views(right):
        mismatches.append("weather_forecasts")
    settings = ("plant", "battery", "state", "degradation", "transaction_fees", "trading", "execution")
    for key in settings:
        if left["asset_input_settings"].get(key) != right["asset_input_settings"].get(key):
            mismatches.append(f"asset_input_settings.{key}")
    if neural["config"]["solver"] != pvlib["config"]["solver"]:
        mismatches.append("solver")
    if mismatches:
        raise ValueError(f"Runs do not share inputs/settings: {mismatches}")
    return {
        "delivery_days": len(left["days"]),
        "realized_ac_pv_hashes_match": True,
        "auction_price_hashes_match": True,
        "quarter_hour_settlement_hash_matches": True,
        "source_weather_views_match": True,
        "solver_settings_match": True,
        "plant_battery_soc_degradation_fees_trading_execution_match": True,
        "neural_input_fingerprint": neural["manifest"].get("input_fingerprint"),
        "pvlib_input_fingerprint": pvlib["manifest"].get("input_fingerprint"),
        "input_fingerprints_differ_due_to_model_configuration": (
            neural["manifest"].get("input_fingerprint") != pvlib["manifest"].get("input_fingerprint")),
    }


def verify_perfect_reference(neural, pvlib):
    """Require duplicated perfect-PV paths to agree before keeping one row."""
    columns = ("net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur",
        "degradation_eur", "transaction_fee_eur", "absolute_imbalance_mwh",
        "signed_imbalance_mwh", "throughput_mwh", "curtailment_mwh", "final_soc_percent")
    left = neural["perfect_daily"].sort_values("delivery_day").reset_index(drop=True)
    right = pvlib["perfect_daily"].sort_values("delivery_day").reset_index(drop=True)
    if list(left.delivery_day) != list(right.delivery_day):
        raise ValueError("Perfect-PV runs cover different delivery dates")
    for column in columns:
        if not np.allclose(left[column], right[column], rtol=0, atol=1e-7):
            raise ValueError(f"Perfect-PV reference differs between runs in {column}")
    return {"same_perfect_pv_reference": True, "daily_metrics_checked": list(columns)}


def battery_usage_metrics(config, daily):
    """Count executed internal-energy EFC separately from AC charge/discharge throughput.

    The two-sided index assigns half a cycle to a nominal charge or discharge;
    the discharge-only index counts withdrawals without charging initial stock.
    These are usage measures, not an estimate of physical capacity fade.
    """
    battery = battery_parameters_from_config(config["battery"])
    charge = float(daily.charge_mwh.sum())
    discharge = float(daily.discharge_mwh.sum())
    internal_charge = battery.eta_charge * charge
    internal_discharge = discharge / battery.eta_discharge
    return {
        "battery_capacity_mwh": battery.capacity_mwh,
        "battery_max_power_mw": battery.max_power_mw,
        "battery_ac_round_trip_efficiency": battery.eta_charge * battery.eta_discharge,
        "battery_charge_ac_mwh": charge,
        "battery_discharge_ac_mwh": discharge,
        "equivalent_full_cycles_internal_discharge": internal_discharge / battery.capacity_mwh,
        "equivalent_full_cycles_internal_two_sided": (
            internal_charge + internal_discharge) / (2 * battery.capacity_mwh),
    }


def strategy_row(label, run, strategy):
    """Flatten one KPI summary into the cross-model comparison table."""
    kpi = run[strategy]
    components = kpi["components"]
    forecasts = run["integrated_forecasts"]
    forecast_models = sorted(forecasts.loc[
        forecasts.get("forecast_role", pd.Series(index=forecasts.index, dtype=str)) == "target_delivery",
        "pv_forecast_model"].dropna().astype(str).unique().tolist()) if strategy == "integrated" else ["perfect actual PV"]
    row = {
        "strategy": label,
        "total_gain_eur": kpi["total_gain_eur"],
        "mean_daily_profit_eur": kpi["mean_daily_profit_eur"],
        "std_daily_profit_eur": kpi["std_daily_profit_eur"],
        "auction_revenue_eur": components["auction_revenue_eur"],
        "imbalance_cashflow_eur": components["imbalance_cashflow_eur"],
        "degradation_cost_eur": components["degradation_eur"],
        "transaction_fees_eur": components["transaction_fee_eur"],
        "import_fees_eur": components["import_fee_eur"],
        "imbalance_penalty_eur": components["imbalance_penalty_eur"],
        "absolute_imbalance_mwh": kpi["absolute_imbalance_mwh"],
        "signed_imbalance_mwh": kpi["signed_imbalance_mwh"],
        "positive_imbalance_mwh": float(run[f"{strategy}_daily"].positive_imbalance_mwh.sum()),
        "negative_imbalance_mwh": float(run[f"{strategy}_daily"].negative_imbalance_mwh.sum()),
        "gross_traded_energy_mwh": float(run[f"{strategy}_trades"].quantity_mwh.abs().sum()),
        "throughput_mwh": kpi["throughput_mwh"],
        "curtailment_mwh": kpi["curtailment_mwh"],
        "final_soc_percent": kpi["final_soc_percent"],
        "auction_solves": kpi["auction_solves"],
        "fallback_count": kpi["fallback_count"],
        "forecast_models": ";".join(forecast_models),
    }
    row.update(battery_usage_metrics(run["config"], run[f"{strategy}_daily"]))
    reconciled = (row["auction_revenue_eur"] + row["imbalance_cashflow_eur"]
        - row["degradation_cost_eur"] - row["transaction_fees_eur"]
        - row["import_fees_eur"] - row["imbalance_penalty_eur"])
    if not np.isclose(reconciled, row["total_gain_eur"], rtol=0, atol=1e-6):
        raise ValueError(f"Realized financial components do not reconcile for {label}")
    row["financial_reconciliation_error_eur"] = reconciled - row["total_gain_eur"]
    return row


def forecast_error_rows(run, label):
    """Score each submitted forecast view against the same realized AC series."""
    forecasts = run["integrated_forecasts"]
    forecasts = forecasts.loc[forecasts.forecast_role.eq("target_delivery")].copy()
    forecasts["interval_start_utc"] = pd.to_datetime(forecasts.interval_start_utc, utc=True)
    actual = run["integrated_dispatch"].copy()
    actual["start"] = pd.to_datetime(actual.start, utc=True)
    merged = forecasts.merge(actual[["start", "pv_ac_mw"]], left_on="interval_start_utc",
        right_on="start", how="left", validate="many_to_one")
    if merged.pv_ac_mw.isna().any():
        raise ValueError(f"Missing actual AC targets for {label} forecast rows")
    merged["forecast_error_mw"] = merged.pv_ac_forecast_mw - merged.pv_ac_mw
    farm_id = str(run["config"]["plant"]["farm_id"])
    ac_rating_mw = float(run["config"]["farms"][farm_id]["ac_capacity_kw"]) / 1000
    rows = []
    for auction, group in merged.groupby("auction", sort=False):
        error = group.forecast_error_mw.to_numpy(float)
        rows.append({"model": label, "auction": auction, "forecast_view_rows": len(group),
            "bias_mw_forecast_minus_actual": float(error.mean()),
            "mae_mw": float(np.abs(error).mean()),
            "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
            "ac_rating_mw": ac_rating_mw,
            "nmae_percent_ac_rating": float(np.abs(error).mean()) / ac_rating_mw * 100,
            "nrmse_percent_ac_rating": float(np.sqrt(np.mean(error ** 2))) / ac_rating_mw * 100,
            "bias_percent_ac_rating": float(error.mean()) / ac_rating_mw * 100,
            "overprediction_rows": int((error > 0).sum()),
            "underprediction_rows": int((error < 0).sum()),
            "overlapping_auction_views": True})
    merged["model"] = label
    return rows, merged


def verify_soc_floor(run):
    """Check every successful solve included and satisfied the configured floor."""
    config = run["config"]
    percentage = config["state"].get("terminal_soc_min_percent")
    if percentage is None:
        raise ValueError("Both continuous runs must configure terminal_soc_min_percent")
    minimum_mwh = float(config["battery"]["capacity_mwh"]) * float(percentage) / 100.0
    reports = {}
    for strategy in ("integrated", "perfect"):
        solves = run[f"{strategy}_solves"]
        if not solves:
            raise ValueError(f"No solve diagnostics for {strategy}")
        floors = np.asarray([row.get("terminal_minimum_mwh", np.nan) for row in solves], dtype=float)
        if not np.isfinite(floors).all() or not np.allclose(floors, minimum_mwh, rtol=0, atol=1e-8):
            raise ValueError(f"A {strategy} solve omitted the configured terminal SOC floor")
        if any(not row.get("validation", {}).get("valid", False) for row in solves):
            raise ValueError(f"A {strategy} solve did not pass physical solution validation")
        reports[strategy] = {"successful_solves_with_floor": len(solves),
            "terminal_minimum_mwh": minimum_mwh, "all_solve_validations_passed": True}
    return {"minimum_soc_percent": float(percentage), "minimum_energy_mwh": minimum_mwh,
        "at_every_solve_horizon_end": reports}


def make_plots(destination: Path, daily: pd.DataFrame, minimum_soc_percent: float):
    """Plot economics, imbalances, executed wear and midnight SOC through the final boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {daily.strategy.iloc[0]: "#2864a5", "Integrated PVLib": "#e07a22",
        "Perfect PV knowledge": "#25855a"}
    timezone = "Europe/Rome"
    origin = pd.Timestamp(daily.delivery_day.min()).tz_localize(timezone)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, layout="constrained")

    for name, group in daily.groupby("strategy", sort=False):
        group = group.sort_values("delivery_day")
        x = pd.to_datetime(group.delivery_day).dt.tz_localize(timezone) + pd.Timedelta(days=1)
        cumulative = group.net_profit_eur.cumsum().to_numpy(float)
        axes[0].step([origin, *x], [0.0, *cumulative], where="post", lw=2,
            color=colors[name], label=f"{name} (€{cumulative[-1]:,.2f})")
        axes[1].plot(x, group.absolute_imbalance_mwh, marker="o", ms=3,
            lw=1.5, color=colors[name], label=name)
        initial = float(group.initial_soc_percent.iloc[0])
        wear = group.degradation_eur.cumsum().to_numpy(float)
        axes[2].step([origin, *x], [0.0, *wear], where="post", lw=2,
            color=colors[name], label=name)
        axes[3].plot([origin, *x], [initial, *group.final_soc_percent.to_numpy(float)],
            marker="o", ms=3, lw=1.5, color=colors[name], label=name)

    axes[0].set_ylabel("Cumulative realized profit (€)")
    axes[1].set_ylabel("Daily absolute imbalance (MWh)")
    axes[2].set_ylabel("Cumulative executed wear cost (€)")
    axes[3].set_ylabel("Realized SOC (%)")
    axes[3].set_xlabel("Delivery boundary (Europe/Rome)")
    axes[3].axhline(minimum_soc_percent, color="black", ls="--", lw=1.3,
        label=f"{minimum_soc_percent:g}% planned end-of-day floor")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    fig.savefig(destination / "plots/model_comparison.png", dpi=150)
    plt.close(fig)


def replacement_labels(config):
    """Name linear replacements accurately while retaining existing neural labels."""
    farm = str(config["plant"]["farm_id"])
    model = config["pv"]["forecast_replacement"]["models_by_farm"][farm]
    name = "Regression" if model["kind"] == "linear" else "Neural"
    return f"Integrated {name.lower()}", name


def main():
    """Combine existing runs without rerunning forecasts or optimizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-run", type=Path, required=True)
    parser.add_argument("--pvlib-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    neural, pvlib = load_run(args.neural_run), load_run(args.pvlib_run)
    fitted_label, fitted_model = replacement_labels(neural["config"])
    shared = verify_shared_inputs(neural, pvlib)
    perfect = verify_perfect_reference(neural, pvlib)
    strategies = [
        strategy_row(fitted_label, neural, "integrated"),
        strategy_row("Integrated PVLib", pvlib, "integrated"),
        strategy_row("Perfect PV knowledge", neural, "perfect"),
    ]
    comparison = pd.DataFrame(strategies)
    daily = pd.concat([
        neural["integrated_daily"].assign(strategy=fitted_label),
        pvlib["integrated_daily"].assign(strategy="Integrated PVLib"),
        neural["perfect_daily"].assign(strategy="Perfect PV knowledge"),
    ], ignore_index=True)
    pivot = daily.pivot(index="delivery_day", columns="strategy", values="net_profit_eur")
    daily_paired = pivot.reset_index()
    daily_paired["pvlib_minus_neural_profit_eur"] = (
        daily_paired["Integrated PVLib"] - daily_paired[fitted_label])
    daily_paired["perfect_minus_neural_profit_eur"] = (
        daily_paired["Perfect PV knowledge"] - daily_paired[fitted_label])
    daily_paired["perfect_minus_pvlib_profit_eur"] = (
        daily_paired["Perfect PV knowledge"] - daily_paired["Integrated PVLib"])
    forecast_neural, forecast_pvlib = forecast_error_rows(neural, fitted_model), forecast_error_rows(pvlib, "PVLib")
    forecast_metrics = pd.DataFrame(forecast_neural[0] + forecast_pvlib[0])

    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("csv", "json", "plots"):
        (args.output / folder).mkdir(exist_ok=True)
    comparison.to_csv(args.output / "csv/strategy_comparison.csv", index=False)
    daily.to_csv(args.output / "csv/daily_strategy_comparison.csv", index=False)
    daily_paired.to_csv(args.output / "csv/daily_paired_profit_differences.csv", index=False)
    forecast_metrics.to_csv(args.output / "csv/forecast_metrics_by_auction.csv", index=False)
    pd.concat([forecast_neural[1], forecast_pvlib[1]], ignore_index=True).to_csv(
        args.output / "csv/forecast_view_errors.csv", index=False)
    minimum_soc = float(neural["config"]["state"]["terminal_soc_min_percent"])
    make_plots(args.output, daily, minimum_soc)
    floor_verification = {"neural": verify_soc_floor(neural),
        "pvlib": verify_soc_floor(pvlib)}
    realized_soc = {}
    for label, run, strategy in ((fitted_label, neural, "integrated"),
            ("Integrated PVLib", pvlib, "integrated"),
            ("Perfect PV knowledge", neural, "perfect")):
        rows = run[f"{strategy}_daily"]
        realized_soc[label] = {"days_below_configured_floor": int(
                (rows.final_soc_percent < minimum_soc - 1e-6).sum()),
            "lowest_realized_daily_end_soc_percent": float(rows.final_soc_percent.min()),
            "mean_realized_daily_end_soc_percent": float(rows.final_soc_percent.mean()),
            "final_realized_soc_percent": float(rows.final_soc_percent.iloc[-1]),
            "days_at_battery_minimum": int((rows.final_soc_percent <=
                float(neural["config"]["battery"]["soc_min_percent"]) + 1e-6).sum())}

    summary = {
        "status": "complete",
        "farm_id": int(neural["config"]["plant"]["farm_id"]),
        "delivery_period": [str(daily.delivery_day.min()), str(daily.delivery_day.max())],
        "neural_run": str(neural["path"]), "pvlib_run": str(pvlib["path"]),
        "replacement_label": fitted_label,
        "legacy_key_note": "Keys and CLI paths named neural identify the fitted replacement, which may be linear.",
        "shared_inputs": shared, "perfect_reference_verification": perfect,
        "terminal_soc_floor_verification": floor_verification,
        "realized_soc_floor_outcomes": realized_soc,
        "strategies": comparison.to_dict("records"),
        "forecast_metrics_by_auction": forecast_metrics.to_dict("records"),
        "financial_interpretation": {
            "price_policy": "Actual historical auction clearing curves are passed to the optimizer (price-oracle study), not predicted prices.",
            "neural_model_selection": "Retrospective; the selected model used January-April 2026 validation data.",
            "transaction_fees": "Configured GME volume-tier tariff; check realized total in the comparison table.",
        },
        "paired_profit_differences_eur": {
            "pvlib_minus_neural": float(comparison.loc[comparison.strategy.eq("Integrated PVLib"),
                "total_gain_eur"].iloc[0] - comparison.loc[comparison.strategy.eq(fitted_label),
                "total_gain_eur"].iloc[0]),
            "perfect_minus_neural": float(comparison.loc[comparison.strategy.eq("Perfect PV knowledge"),
                "total_gain_eur"].iloc[0] - comparison.loc[comparison.strategy.eq(fitted_label),
                "total_gain_eur"].iloc[0]),
            "perfect_minus_pvlib": float(comparison.loc[comparison.strategy.eq("Perfect PV knowledge"),
                "total_gain_eur"].iloc[0] - comparison.loc[comparison.strategy.eq("Integrated PVLib"),
                "total_gain_eur"].iloc[0]),
        },
        "forecast_metric_note": "Forecast rows overlap across auction views; errors score predictions, not additive energy totals.",
        "cycle_metric_note": "Executed charge/discharge are converted to internal energy using effective efficiencies; two-sided EFC divides their sum by twice nominal capacity, and discharge EFC divides internal withdrawals by nominal capacity. Neither metric models capacity fade.",
    }
    def json_default(value):
        """Normalize report values originating in NumPy and pandas."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    (args.output / "json/comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output),
        "paired_profit_differences_eur": summary["paired_profit_differences_eur"]}, indent=2))


if __name__ == "__main__":
    main()
