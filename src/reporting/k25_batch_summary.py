"""Write a readable combined summary from completed K25 controller trajectories."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


def _label(controller, case, config):
    """Identify the controller and priority without relying on a fixed farm list."""
    if controller == "track_committed_grid":
        return "Grid tracker"
    settings = config.get("realtime_mpc", {})
    if controller == "deterministic_mpc" and case.startswith("deterministic_"):
        if settings.get("first_interval_priority") == "weighted":
            return f"MPC K={settings.get('first_interval_weight', 1):g}"
        return "MPC equal weights"
    return f"{controller} ({case})"


def _load_rows(batch, jobs, results):
    """Read distinct completed trajectories and reconcile their saved accounting."""
    indexed = {(row["job_id"], row["controller"]): row for row in results}
    if len(indexed) != len(results):
        raise ValueError("Duplicate trajectory results in batch")
    rows = []
    expected = set()
    for job in jobs:
        farm = str(job["farm_id"])
        settings = job.get("config", {}).get("continuous_optimization", {})
        run_id = settings.get("run_id", job["job_id"])
        destination = batch / "runs" / f"farm_{farm}" / run_id / "continuous"
        config = json.loads((destination / "resolved_config.json").read_text())
        for controller in job.get("controllers", config["realtime_mpc_comparison"]["controllers"]):
            expected.add((job["job_id"], controller))
            result = indexed.get((job["job_id"], controller))
            if result is None or result.get("status") != "complete":
                raise ValueError(f"Incomplete trajectory {job['job_id']}/{controller}")
            # Reused tracking results already appear under their original case.
            if result.get("reused_from_job_id"):
                continue
            summary = json.loads((destination / f"summary__{controller}.json").read_text())
            daily = pd.DataFrame(summary["daily_kpis"])
            if daily.empty or daily.delivery_day.duplicated().any():
                raise ValueError(f"Missing or duplicated daily records for {job['job_id']}/{controller}")
            csv_dir = destination / "controllers" / controller / "csv"
            ledger = pd.read_csv(csv_dir / "accepted_trades.csv", usecols=["auction", "cashflow_eur"])
            auction_cash = ledger.groupby("auction").cashflow_eur.sum().to_dict()
            if not math.isclose(sum(auction_cash.values()), summary["components"]["auction_revenue_eur"],
                                rel_tol=1e-9, abs_tol=1e-5):
                raise ValueError(f"Auction ledger does not reconcile for {job['job_id']}/{controller}")
            for field in ("net_profit_eur", "absolute_imbalance_mwh", "pv_realized_energy_mwh", "curtailment_mwh"):
                if not math.isclose(float(daily[field].sum()), float(summary[field]), rel_tol=1e-9, abs_tol=1e-5):
                    raise ValueError(f"Daily {field} does not reconcile for {job['job_id']}/{controller}")
            metrics = summary["mpc_metrics"]
            rejected = 0
            if metrics.get("mpc_solves", 0):
                solves = pd.read_csv(csv_dir / "mpc_solves.csv", usecols=["status"])
                rejected = int(solves.status.astype(str).str.endswith("_tiebreak_rejected").sum())
            target = config.get("state", {}).get("terminal_soc_percent")
            if controller != "track_committed_grid":
                target = config.get("realtime_mpc", {}).get("terminal_soc_target_percent", target)
            minimum = config["battery"]["soc_min_percent"]
            end_soc = pd.to_numeric(daily.final_soc_percent, errors="raise")
            daily_controller = pd.read_csv(csv_dir / "controller_daily_kpis.csv",
                                           usecols=["delivery_day", "mpc_fallbacks"])
            combined = daily[["delivery_day", "final_soc_percent"]].merge(
                daily_controller, on="delivery_day", how="outer", validate="one_to_one", indicator=True)
            if not combined["_merge"].eq("both").all():
                raise ValueError(f"Daily controller records differ for {job['job_id']}/{controller}")
            combined["mpc_fallbacks"] = pd.to_numeric(combined.mpc_fallbacks, errors="raise")
            if combined.mpc_fallbacks.isna().any() or (combined.mpc_fallbacks < 0).any():
                raise ValueError(f"Invalid daily fallback counts for {job['job_id']}/{controller}")
            missed = 0 if target is None else int((combined.final_soc_percent - target).abs().gt(0.01).sum())
            missed_with_fallback = 0 if target is None else int(((combined.final_soc_percent - target).abs().gt(0.01)
                & (combined.mpc_fallbacks > 0)).sum())
            rows.append({"farm": farm, "case": job["case_id"], "window": job.get("window_id"),
                "baseline_case": job.get("baseline_case"), "controller": controller,
                "label": _label(controller, job["case_id"], config), "summary": summary,
                "config": config, "daily": daily, "auctions": auction_cash,
                "target": target, "days_at_target": None if target is None else int((end_soc - target).abs().le(0.01).sum()),
                "days_at_minimum": int((end_soc - minimum).abs().le(0.01).sum()),
                "mean_end_soc": float(end_soc.mean()), "rejected_refinements": rejected,
                "missed_days": missed, "missed_with_fallback": missed_with_fallback})
    if set(indexed) != expected:
        raise ValueError("Batch contains unexpected trajectory results")
    # Different non-reused trackers or windows need distinct labels in plots and tables.
    labels = Counter((row["farm"], row["label"]) for row in rows)
    for row in rows:
        if labels[row["farm"], row["label"]] > 1:
            row["label"] += f" ({row['case']}, {row['window']})"
    return rows


def _table(lines, headers, values):
    """Append one Markdown table with consistent right-aligned numeric columns."""
    lines.extend(["| " + " | ".join(headers) + " |",
                  "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |"])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in values)
    lines.append("")


def _plots(batch, rows):
    """Plot cumulative profit and imbalance and daily SOC for each farm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = batch / "summary_plots"
    output.mkdir(exist_ok=True)
    paths = []
    for farm in sorted({row["farm"] for row in rows}):
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
        for row in (item for item in rows if item["farm"] == farm):
            daily = row["daily"].sort_values("delivery_day")
            dates = pd.to_datetime(daily.delivery_day)
            axes[0].plot(dates, daily.net_profit_eur.cumsum(), label=row["label"])
            axes[1].plot(dates, daily.absolute_imbalance_mwh.cumsum(), label=row["label"])
            axes[2].plot(dates, daily.final_soc_percent, label=row["label"], alpha=0.8)
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("Cumulative profit (€)")
        axes[1].set_ylabel("Absolute imbalance (MWh)")
        axes[2].set_ylabel("End-of-day SOC (%)")
        axes[2].set_xlabel("Italian delivery day")
        figure.suptitle(f"Farm {farm}: continuous replay")
        path = output / f"farm_{farm}_main_results.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def write_k25_batch_summary(batch, jobs, results, *, plots=True):
    """Summarize complete trajectories without rerunning an optimization."""
    batch = Path(batch)
    rows = _load_rows(batch, jobs, results)
    if not rows:
        raise ValueError("No distinct completed trajectories to summarize")
    starts = {row["summary"]["first_delivery_day"] for row in rows}
    ends = {row["summary"]["last_delivery_day"] for row in rows}
    periods = ", ".join(sorted({f"{row['summary']['first_delivery_day']} to {row['summary']['last_delivery_day']} "
                                 f"({row['summary']['delivery_days']} days)" for row in rows}))
    lines = ["# Continuous K25 batch: main results", "", f"**Delivery periods:** {periods} (Italian local time).", "",
             "Tracker trajectories reused across cases are reported once. Absolute imbalance sums interval magnitudes; "
             "signed imbalance is positive for surplus export. PV percentages use total replay PV production, "
             "including any curated synthetic values.", "", "## Physical results", ""]
    physical, money, auction, settlement, battery = [], [], [], [], []
    for row in rows:
        s, m, c = row["summary"], row["summary"]["mpc_metrics"], row["summary"]["components"]
        production = s["pv_realized_energy_mwh"]
        percent = lambda value: "n/a" if production == 0 else f"{100 * value / production:.2f}%"
        base = [row["farm"], row["label"]]
        physical.append(base + [f"{production:,.2f}", f"{s['absolute_imbalance_mwh']:,.2f}",
            percent(s["absolute_imbalance_mwh"]), f"{s['signed_imbalance_mwh']:,.2f}",
            f"{s['curtailment_mwh']:,.2f}", percent(s["curtailment_mwh"])])
        money.append(base + [f"{c['auction_revenue_eur']:,.2f}", f"{c['imbalance_cashflow_eur']:,.2f}",
            f"{c['degradation_eur']:,.2f}", f"{c['import_fee_eur'] + c['transaction_fee_eur']:,.2f}",
            f"{s['net_profit_eur']:,.2f}"])
        auction.append(base + [f"{row['auctions'].get(name, 0.0):,.2f}" for name in ("DA", "IDA1", "IDA2", "IDA3")])
        settlement.append(base + [f"{m['positive_imbalance_mwh']:,.2f}", f"{m['negative_imbalance_mwh']:,.2f}",
            f"{m['positive_deviation_cashflow_eur']:,.2f}", f"{m['negative_deviation_cashflow_eur']:,.2f}",
            f"{m['absolute_settlement_exposure_eur']:,.2f}"])
        target_days = "n/a" if row["days_at_target"] is None else f"{row['days_at_target']}/{s['delivery_days']}"
        fallback_days = ("n/a" if row["target"] is None or row["controller"] == "track_committed_grid" else
                         f"{row['missed_with_fallback']}/{row['missed_days']}")
        battery.append(base + [f"{s['charge_mwh']:,.2f}", f"{s['discharge_mwh']:,.2f}",
            f"{row['mean_end_soc']:.2f}%", target_days, f"{row['days_at_minimum']}/{s['delivery_days']}",
            f"{s['final_soc_percent']:.2f}%", str(m['mpc_fallbacks']), fallback_days,
            str(row['rejected_refinements']), str(s['fallback_count'])])
    _table(lines, ["Farm", "Controller", "PV (MWh)", "Absolute imbalance (MWh)", "Imbalance / PV", "Signed imbalance (MWh)", "Curtailment (MWh)", "Curtailment / PV"], physical)
    lines.extend(["## Realized cash and degradation", "", "Auction cash includes purchases; settlement cash is signed. "
        "Profit includes auction cash and settlement, less degradation, fees and any realized imbalance penalty.", ""])
    _table(lines, ["Farm", "Controller", "Auction cash (€)", "Imbalance cash (€)", "Degradation (€)", "Fees and imports (€)", "Profit (€)"], money)
    lines.extend(["## Auction cash breakdown", ""])
    _table(lines, ["Farm", "Controller", "DA (€)", "IDA1 (€)", "IDA2 (€)", "IDA3 (€)"], auction)
    lines.extend(["## Imbalance direction and settlement", "", "Positive and negative volumes are nonnegative magnitudes. "
                  "Cash columns retain their signs; exposure sums absolute interval cash flows.", ""])
    _table(lines, ["Farm", "Controller", "Surplus (MWh)", "Shortfall (MWh)", "Surplus cash (€)", "Shortfall cash (€)", "Absolute exposure (€)"], settlement)
    lines.extend(["## Battery and solver", "", "Daily target and minimum counts use actual end-of-day SOC within "
        "0.01 percentage points of each configured value. The target count is descriptive for the tracker; "
        "it does not enforce the MPC's terminal constraint.", ""])
    _table(lines, ["Farm", "Controller", "Charge (MWh)", "Discharge (MWh)", "Mean daily end SOC", "Days at target", "Days at minimum", "Final SOC", "MPC fallbacks", "Missed days with fallback", "Rejected cost tie-breaks", "Auction fallbacks"], battery)
    lines.extend(["MPC fallbacks use the configured fallback controller. A rejected cost refinement retains a "
        "validated primary plan; it is counted separately from fallback. A hard MPC target cannot be guaranteed "
        "on a fallback interval. 'Missed days with fallback' counts days with any fallback among days outside "
        "the target tolerance; it does not establish causation. Auction fallbacks retain an earlier auction plan.", ""])
    # Compare only declared baseline cases with the same farm, dates and input fingerprint.
    differences = []
    for row in rows:
        if not row["baseline_case"]:
            continue
        matches = [base for base in rows if base["farm"] == row["farm"] and base["window"] == row["window"]
                   and base["case"] == row["baseline_case"] and base["controller"] == row["controller"]]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous baseline for {row['farm']}/{row['case']}")
        base = matches[0]
        left, right = base["summary"], row["summary"]
        if (not left.get("input_fingerprint") or left.get("input_fingerprint") != right.get("input_fingerprint")
                or set(base["daily"].delivery_day) != set(row["daily"].delivery_day)):
            raise ValueError(f"Baseline inputs or delivery dates differ for {row['farm']}/{row['case']}")
        differences.append([row["farm"], f"{row['label']} − {base['label']}",
            f"{right['absolute_imbalance_mwh'] - left['absolute_imbalance_mwh']:,.2f}",
            f"{right['curtailment_mwh'] - left['curtailment_mwh']:,.2f}",
            f"{right['components']['imbalance_cashflow_eur'] - left['components']['imbalance_cashflow_eur']:,.2f}",
            f"{right['components']['degradation_eur'] - left['components']['degradation_eur']:,.2f}",
            f"{right['net_profit_eur'] - left['net_profit_eur']:,.2f}"])
    if differences:
        lines.extend(["## Controller case differences", "", "Differences are case minus its declared baseline; "
                      "continuous trajectories may submit different auction positions.", ""])
        _table(lines, ["Farm", "Comparison", "Imbalance change (MWh)", "Curtailment change (MWh)",
            "Settlement change (€)", "Degradation change (€)", "Profit change (€)"], differences)
    lines.extend(["## Configured formulation", "", "The auction terminal scope applies to planning; "
                  "the MPC terminal mode applies only to the MPC trajectory.", ""])
    formulations = []
    for row in rows:
        config = row["config"]
        mpc = config.get("realtime_mpc", {})
        tracking = row["controller"] == "track_committed_grid"
        formulations.append([row["farm"], row["label"],
            config.get("continuous_optimization", {}).get("terminal_soc_scope", "final_replay_day"),
            "n/a" if tracking else mpc.get("terminal_inventory_mode", "not specified"),
            "n/a" if tracking or row["target"] is None else f"{row['target']:g}%",
            "n/a" if tracking else mpc.get("planning_objective", "not specified"),
            "n/a" if tracking else str(mpc.get("first_interval_weight", 1))])
    _table(lines, ["Farm", "Controller", "Auction terminal scope", "MPC terminal mode", "MPC target", "MPC objective", "First-interval weight"], formulations)
    lines.extend(["## Interpretation", "", "- Realized profit has no end-of-replay inventory adjustment, so strategies "
        "with different final SOC are not fully adjusted for residual battery value.",
        "- Curtailed energy has no separate realized cash-flow line; effects appear through market and settlement outcomes.",
        "- Consult the resolved configurations and case comparison reports for the forecast-vintage assumptions, "
        "price information, terminal treatment and other study-specific limitations."])
    if len(starts) > 1 or len(ends) > 1:
        lines.append("- The rows cover different delivery periods; totals are not directly comparable.")
    if any(row["config"].get("pv", {}).get("forecast_replacement", {}).get("allow_retrospective_artifacts") for row in rows):
        lines.append("- The configuration permits retrospectively selected PV model artifacts; treat results as historical comparisons.")
    if any(row["summary"]["mpc_metrics"].get("auction_price_oracle_enabled") for row in rows):
        lines.append("- Auction planning uses actual clearing-price curves as a retrospective price oracle.")
    if any(row["summary"]["mpc_metrics"].get("current_interval_pv_access_is_retrospective") for row in rows):
        lines.append("- The MPC receives the realized current-interval PV average retrospectively.")
    if plots:
        paths = _plots(batch, rows)
        lines.extend(["", "## Plots", ""])
        lines.extend(f"- [Farm {path.stem.split('_')[1]}: profit, imbalance and daily SOC]({path.relative_to(batch)})" for path in paths)
    lines.append("")
    output = batch / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
