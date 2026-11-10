"""Summarize matched January extended-IDA3 midnight SOC target trajectories."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .ida3_horizon_comparison import _check_common_inputs, _daily, _metrics


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
            or config["continuous_optimization"].get("ida3_enforce_intermediate_soc") is not False):
        raise ValueError(f"SOC study configuration mismatch for farm {farm}, target {target}%")
    if (summary.get("first_delivery_day"), summary.get("last_delivery_day"), summary.get("delivery_days")) != (
            "2026-01-01", "2026-01-31", 31):
        raise ValueError(f"Incomplete January summary for farm {farm}, target {target}%")
    daily = _daily(root)
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
    figure.suptitle(f"Farm {farm}: January 2026 final-midnight SOC sweep")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def write_soc_sweep_report(batch, results, expected_farms, expected_targets):
    """Write auditable January comparisons for all successfully completed targets."""
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
    lines = ["# January 2026: IDA3 final-midnight SOC sweep", "",
        "Each trajectory uses the greedy grid tracker. The extended IDA3 plan imposes the stated SOC target "
        "at the final next-day midnight, with no target at the intermediate midnight. "
        "DA, IDA1 and IDA2 retain their delivery-day terminal equality. The tracker does not enforce the target in real time. "
        "Rows cover January delivery products only; February 1 is planning context.", "",
        "Actual auction clearing prices, including future prices at IDA3, are an oracle. Neural models were "
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
        report = [f"# Farm {farm}: January final-midnight SOC targets", "",
            "The extended IDA3 planner targets the final next-day midnight, without an intermediate target. "
            "DA, IDA1 and IDA2 retain their delivery-day target; the greedy tracker does not enforce it. "
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
                f"{row['mean_midnight_soc_percent']:.2f} | {row['midnights_at_target']}/31 | "
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
