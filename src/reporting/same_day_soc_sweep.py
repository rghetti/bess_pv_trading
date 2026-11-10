"""Summarize same-day SOC target trajectories over the configured delivery period."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import numpy as np


def _daily(root, start, end):
    """Require one daily accounting row for every local delivery date."""
    daily = pd.read_csv(root / "csv/daily_kpis.csv")
    expected = pd.date_range(start, end).strftime("%Y-%m-%d").tolist()
    if sorted(daily.delivery_day.astype(str)) != expected:
        raise ValueError(f"Incomplete or duplicated delivery days in {root}")
    return daily.sort_values("delivery_day")


def _metrics(root, daily):
    """Sum physical and financial outcomes and solver diagnostics."""
    fields = ("net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
              "import_fee_eur", "transaction_fee_eur", "pv_realized_energy_mwh", "absolute_imbalance_mwh",
              "signed_imbalance_mwh", "curtailment_mwh", "charge_mwh", "discharge_mwh", "fallback_count")
    result = {field: float(daily[field].sum()) for field in fields}
    result["final_soc_percent"] = float(daily.iloc[-1].final_soc_percent)
    result["mean_midnight_soc_percent"] = float(daily.final_soc_percent.mean())
    attempts = json.loads((root / "json/optimization_attempts.json").read_text())
    result["auction_solve_attempts"] = len(attempts)
    result["auction_solver_seconds"] = sum(float(row.get("runtime_seconds") or 0) for row in attempts)
    result["auction_fallbacks"] = sum(bool(row.get("fallback_applied")) for row in attempts)
    return result


def _check_common_inputs(old, new):
    """Check matching PV, settlement and auction price products across targets."""
    columns = ["interval_id", "pv_ac_mw", "imbalance_price_eur_mwh"]
    left = pd.read_csv(old / "csv/realized_dispatch.csv", usecols=columns)
    right = pd.read_csv(new / "csv/realized_dispatch.csv", usecols=columns)
    paired = left.merge(right, on="interval_id", how="outer", validate="one_to_one", indicator=True)
    if paired.empty or not paired["_merge"].eq("both").all():
        raise ValueError("PV and settlement product coverage differs across SOC targets")
    for key in columns[1:]:
        if not np.allclose(paired[key + "_x"], paired[key + "_y"], rtol=0, atol=1e-9, equal_nan=False):
            raise ValueError(f"Common PV or settlement input differs: {key}")
    keys = ["auction", "interval_id"]
    left = pd.read_csv(old / "csv/accepted_trades.csv", usecols=keys + ["price_eur_per_mwh"])
    right = pd.read_csv(new / "csv/accepted_trades.csv", usecols=keys + ["price_eur_per_mwh"])
    prices = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
    if prices.empty or not prices["_merge"].eq("both").all() or not np.allclose(
            prices.price_eur_per_mwh_x, prices.price_eur_per_mwh_y, rtol=0, atol=1e-9, equal_nan=False):
        raise ValueError("Auction products or actual clearing prices differ across SOC targets")
    return {"realized_pv_and_settlement_products": len(paired), "auction_products": len(prices)}


def _trajectory_root(batch, farm, target):
    """Locate the immutable replay and summary for one farm and SOC target."""
    directory = Path(batch) / "runs" / f"farm_{farm}" / f"soc_{target}"
    return directory, directory / "controllers" / "track_committed_grid"


def _read_trajectory(batch, result):
    """Reconcile daily accounting, configuration and actual SOC against the run summary."""
    farm = str(result["farm"])
    target = int(result["target_soc_percent"])
    directory, root = _trajectory_root(batch, farm, target)
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "resolved_config.json").read_text())
    if (config["state"].get("terminal_soc_percent") != target
            or config["state"].get("terminal_soc_min_percent") is not None
            or config["execution"]["dispatch_policy"] != "track_committed_grid"
            or config["continuous_optimization"].get("ida3_next_day_horizon", False)):
        raise ValueError(f"SOC study configuration mismatch for farm {farm}, target {target}%")
    start = str(config["continuous_optimization"]["start_date"])
    end = str(config["continuous_optimization"]["end_date"])
    days = len(pd.date_range(start, end))
    if (summary.get("first_delivery_day"), summary.get("last_delivery_day"), summary.get("delivery_days")) != (
            start, end, days):
        raise ValueError(f"Incomplete delivery-period summary for farm {farm}, target {target}%")
    daily = _daily(root, start, end)
    values = _metrics(root, daily)
    # Totals are calculated from delivery-day records and checked against the
    # saved run summary so a stale or truncated artifact cannot be plotted.
    for key in ("net_profit_eur", "absolute_imbalance_mwh", "pv_realized_energy_mwh",
                "signed_imbalance_mwh", "curtailment_mwh", "charge_mwh", "discharge_mwh"):
        if not math.isclose(values[key], float(summary[key]), rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError(f"Daily and summary {key} differ for farm {farm}, target {target}%")
    for key in ("auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
                "import_fee_eur", "transaction_fee_eur"):
        if not math.isclose(values[key], float(summary["components"][key]), rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError(f"Daily and summary {key} differ for farm {farm}, target {target}%")
    values["delivery_days"] = days
    values["period_start"] = start
    values["period_end"] = end
    values["midnights_at_target"] = int((daily.final_soc_percent - target).abs().le(0.01).sum())
    values["initial_soc_percent"] = float(daily.iloc[0].initial_soc_percent)
    production = values["pv_realized_energy_mwh"]
    values["imbalance_percent_of_pv"] = (100 * values["absolute_imbalance_mwh"] / production
                                         if production else None)
    values["curtailment_percent_of_pv"] = (100 * values["curtailment_mwh"] / production
                                           if production else None)
    return {"farm": farm, "target_soc_percent": target, **values}, root


def _plot_farm(frame, farm, destination):
    """Show financial, physical and actual SOC effects versus the planning target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame.loc[frame.farm == farm].sort_values("target_soc_percent")
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout="constrained")
    choices = [
        ("net_profit_eur", "Operating profit (€)"),
        ("auction_revenue_eur", "Auction cash flow (€)"),
        ("absolute_imbalance_mwh", "Absolute imbalance (MWh)"),
        ("curtailment_mwh", "Curtailment (MWh)"),
        ("imbalance_cashflow_eur", "Imbalance settlement (€)"),
        ("mean_midnight_soc_percent", "Mean actual midnight SOC (%)"),
    ]
    for axis, (key, label) in zip(axes.flat, choices):
        axis.plot(data.target_soc_percent, data[key], marker="o", linewidth=1.4, markersize=3)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Planned midnight SOC target (%)")
    figure.suptitle(f"Farm {farm}: same-day SOC sweep")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_soc_sweep_report(batch, results, expected_farms, expected_targets):
    """Write auditable delivery-period comparisons for all successfully completed targets."""
    batch = Path(batch)
    expected = {(str(farm), int(target)) for farm in expected_farms for target in expected_targets}
    indexed = {}
    for result in results:
        key = (str(result["farm"]), int(result["target_soc_percent"]))
        if key in indexed or key not in expected:
            raise ValueError(f"Duplicate or unexpected sweep trajectory {key}")
        indexed[key] = result
    records, evidence = [], {}
    for farm in map(str, expected_farms):
        reference = None
        for target in expected_targets:
            result = indexed.get((farm, int(target)))
            if result is None or result.get("status") != "complete":
                continue
            row, root = _read_trajectory(batch, result)
            if reference is None:
                reference = root
            else:
                # Compare actual PV and settlement/auction prices by delivery
                # product; the input fingerprint itself includes the SOC policy.
                evidence[f"{farm}/{target}"] = _check_common_inputs(reference, root)
            records.append(row)
    if not records:
        raise ValueError("No completed SOC-sweep trajectories to report")
    frame = pd.DataFrame(records).sort_values(["farm", "target_soc_percent"])
    frame.to_csv(batch / "soc_sweep_results.csv", index=False)
    (batch / "soc_sweep_input_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    incomplete = sorted(expected - {(row["farm"], row["target_soc_percent"]) for row in records})
    periods = frame[["period_start", "period_end"]].drop_duplicates()
    if len(periods) != 1:
        raise ValueError("SOC sweep trajectories have differing delivery periods")
    first, last = periods.iloc[0]
    lines = [f"# Same-day SOC sweep: {first} to {last}", "",
        "Each trajectory uses the greedy grid tracker. All auctions, including IDA3, plan only through "
        "the delivery-day midnight and impose the stated hard SOC target there. "
        "The tracker does not enforce that target in real time.", "",
        "Actual auction clearing prices are an oracle. Neural models were "
        "selected retrospectively and curated PV includes synthetic values. Operating profit includes modeled "
        "degradation and does not value residual final battery inventory; terminal SOC differences can therefore "
        "affect apparent profitability. Percentages use gross realized PV before curtailment.", "",
        f"Completed {len(records)} of {len(expected)} requested farm/target trajectories.", ""]
    if incomplete:
        lines.extend(["## Incomplete or failed trajectories", ""])
        for farm, target in incomplete:
            result = indexed.get((farm, target), {})
            status = result.get("status", "not started")
            detail = str(result.get("error", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(f"- Farm {farm}, {target}%: {status}" + (f" — {detail}" if detail else ""))
        lines.append("")
    def percentage(value):
        """Format undefined ratios without inventing a percentage."""
        return "n/a" if value is None or pd.isna(value) else f"{value:.2f}"

    lines.extend(["Full numeric results: [CSV](soc_sweep_results.csv). Input matching: "
        "[JSON](soc_sweep_input_checks.json).", "", "## Farm reports", ""])
    for farm in map(str, expected_farms):
        if farm not in set(frame.farm):
            continue
        path = batch / f"farm_{farm}_soc_sweep.png"
        _plot_farm(frame, farm, path)
        selected = frame.loc[frame.farm == farm].sort_values("target_soc_percent").to_dict("records")
        report = [f"# Farm {farm}: same-day SOC targets, {first} to {last}", "",
            "All auction plans target the delivery-day midnight; the greedy tracker does not enforce it. "
            "Profit includes modeled degradation and excludes a final-inventory valuation; compare actual final "
            "SOC before interpreting differences. Forecast selection and auction-price oracle are retrospective.", "",
            "| Target SOC (%) | Profit (€) | Auction cash (€) | Imbalance cash (€) | Degradation (€) | "
            "Imbalance (MWh) | Imbalance / PV (%) | Curtailment (MWh) | Curtailment / PV (%) | "
            "Start SOC (%) | Mean midnight SOC (%) | Days at target | Final SOC (%) | Fallbacks |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in selected:
            report.append(f"| {row['target_soc_percent']} | {row['net_profit_eur']:,.2f} | "
                f"{row['auction_revenue_eur']:,.2f} | {row['imbalance_cashflow_eur']:,.2f} | "
                f"{row['degradation_eur']:,.2f} | {row['absolute_imbalance_mwh']:,.2f} | "
                f"{percentage(row['imbalance_percent_of_pv'])} | {row['curtailment_mwh']:,.2f} | "
                f"{percentage(row['curtailment_percent_of_pv'])} | {row['initial_soc_percent']:.2f} | "
                f"{row['mean_midnight_soc_percent']:.2f} | {row['midnights_at_target']}/{row['delivery_days']} | "
                f"{row['final_soc_percent']:.2f} | {row['auction_fallbacks']} |")
        missing_targets = [target for missing_farm, target in incomplete if missing_farm == farm]
        if missing_targets:
            report.extend(["", "Incomplete or failed targets (%): " + ", ".join(map(str, missing_targets)) + "."])
        report.extend(["", "Days at target count actual midnight SOC within 0.01 percentage points. "
            "Auction fallbacks reuse an earlier planned order after an infeasible solve.", "",
            f"![Farm {farm} SOC sweep](farm_{farm}_soc_sweep.png)", "",
            "[Combined report](summary.md)", ""])
        (batch / f"farm_{farm}_summary.md").write_text("\n".join(report))
        lines.append(f"- [Farm {farm}: full target table and plot](farm_{farm}_summary.md)")
    path = batch / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path
