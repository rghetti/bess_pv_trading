"""Compare January grid-tracking trajectories against preserved historical runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


START = "2026-01-01"
END = "2026-01-31"


def _january(frame):
    """Select delivery products, rather than submission or publication dates."""
    return frame.loc[frame.delivery_day.astype(str).between(START, END)].copy()


def _check_common_inputs(old, new):
    """Require the same realized PV and actual prices on every scored product."""
    left = _january(pd.read_csv(old / "csv/realized_dispatch.csv"))
    right = _january(pd.read_csv(new / "csv/realized_dispatch.csv"))
    columns = ["interval_id", "pv_ac_mw", "imbalance_price_eur_mwh"]
    paired = left[columns].merge(right[columns], on="interval_id", how="outer", validate="one_to_one", indicator=True)
    if len(paired) != 31 * 96 or not paired["_merge"].eq("both").all():
        raise ValueError("January PV/settlement product coverage differs")
    for key in columns[1:]:
        if not np.allclose(paired[key + "_x"], paired[key + "_y"], rtol=0, atol=1e-9, equal_nan=False):
            raise ValueError(f"January common input differs: {key}")
    # Auction orders may differ; compare prices and products rather than quantities.
    keys = ["auction", "interval_id"]
    trades = []
    for root in (old, new):
        frame = _january(pd.read_csv(root / "csv/accepted_trades.csv"))
        trades.append(frame[keys + ["price_eur_per_mwh"]])
    prices = trades[0].merge(trades[1], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not prices["_merge"].eq("both").all() or not np.allclose(
            prices.price_eur_per_mwh_x, prices.price_eur_per_mwh_y, rtol=0, atol=1e-9, equal_nan=False):
        raise ValueError("January auction products or actual clearing prices differ")
    return {"realized_pv_and_settlement_products": len(paired), "auction_products": len(prices)}


def _daily(root):
    """Read January daily accounting and ensure no dates are missing or duplicated."""
    daily = _january(pd.read_csv(root / "csv/daily_kpis.csv"))
    expected = pd.date_range(START, END).strftime("%Y-%m-%d").tolist()
    if sorted(daily.delivery_day.astype(str)) != expected:
        raise ValueError(f"Incomplete January daily output in {root}")
    return daily.sort_values("delivery_day")


def _metrics(root, daily):
    """Summarize realized delivery-day accounting and auction solver attempts."""
    fields = ("net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
              "import_fee_eur", "transaction_fee_eur", "pv_realized_energy_mwh", "absolute_imbalance_mwh",
              "signed_imbalance_mwh", "curtailment_mwh", "charge_mwh", "discharge_mwh", "fallback_count")
    result = {field: float(daily[field].sum()) for field in fields}
    result["final_soc_percent"] = float(daily.iloc[-1].final_soc_percent)
    result["mean_midnight_soc_percent"] = float(daily.final_soc_percent.mean())
    result["midnights_below_40_percent"] = int((daily.final_soc_percent < 40 - 1e-6).sum())
    attempts = json.loads((root / "json/optimization_attempts.json").read_text())
    attempts = [row for row in attempts if START <= row["delivery_day"] <= END]
    result["auction_solve_attempts"] = len(attempts)
    result["auction_solver_seconds"] = sum(float(row.get("runtime_seconds") or 0) for row in attempts)
    result["auction_fallbacks"] = sum(bool(row.get("fallback_applied")) for row in attempts)
    return result


def _midnight_records(root, daily, capacity, label):
    """Compare each IDA3 plan's end-of-day SOC with the actual executed SOC."""
    plans = pd.read_csv(root / "csv/expected_dispatch_plans.csv")
    plans = plans.loc[(plans.auction == "IDA3") & plans.delivery_day.astype(str).between(START, END)].copy()
    # The previous day's extended solve also stores next-day IDA3 plans.
    # They are provisional and do not represent this day's submitted IDA3.
    if "plan_status" in plans:
        plans = plans.loc[plans.plan_status != "provisional_previous_day_ida3"].copy()
    if not plans.empty:
        plans["end"] = pd.to_datetime(plans.interval_end_utc, utc=True)
        # Extended snapshots can also contain the following delivery day; only
        # the midnight ending the stated delivery day matches the actual SOC.
        local_end = plans["end"].dt.tz_convert("Europe/Rome")
        plans = plans.loc[(local_end.dt.hour == 0) & (local_end.dt.minute == 0)
            & ((local_end - pd.Timedelta(seconds=1)).dt.strftime("%Y-%m-%d") == plans.delivery_day.astype(str))]
        plans = plans.sort_values("end").groupby("delivery_day", as_index=False).tail(1)
        plans = plans[["delivery_day", "soc_end_mwh"]].rename(columns={"soc_end_mwh": "ida3_planned_end_mwh"})
    else:
        plans = pd.DataFrame(columns=["delivery_day", "ida3_planned_end_mwh"])
    actual = daily[["delivery_day", "final_soc_percent"]].merge(plans, on="delivery_day", how="left", validate="one_to_one")
    actual["ida3_planned_end_soc_percent"] = 100 * actual.ida3_planned_end_mwh / capacity
    actual["framework"] = label
    return actual.drop(columns="ida3_planned_end_mwh")


def _all_planned_midnights(root, daily, capacity, farm):
    """Report every modeled midnight and its actual SOC when it was executed."""
    attempts = json.loads((root / "json/optimization_attempts.json").read_text())
    actual = dict(zip(daily.delivery_day.astype(str), daily.final_soc_percent))
    rows = []
    for attempt in attempts:
        for midnight, energy in attempt.get("planned_midnight_energy_mwh", {}).items():
            stamp = pd.Timestamp(midnight).tz_convert("Europe/Rome")
            day = (stamp - pd.Timedelta(seconds=1)).date().isoformat()
            rows.append({"farm": farm, "decision_day": attempt["delivery_day"],
                "auction": attempt["current_auction"], "cutoff_utc": attempt["cutoff_utc"],
                "midnight_utc": midnight, "ending_delivery_day": day,
                "planned_soc_percent": 100 * float(energy) / capacity,
                "actual_soc_percent": actual.get(day)})
    return rows


def write_comparison(batch, farms, historical):
    """Write paired metrics and SOC diagnostics without rerunning historical studies."""
    batch = Path(batch)
    rows = []
    soc = []
    planned_midnights = []
    daily_plot_data = {}
    evidence = {}
    for farm in farms:
        old = Path(historical[str(farm)])
        new = batch / "runs" / f"farm_{farm}" / "controllers/track_committed_grid"
        evidence[str(farm)] = _check_common_inputs(old, new)
        old_daily, new_daily = _daily(old), _daily(new)
        new_config = json.loads((new / "json/study_config.json").read_text())
        capacity = float(new_config["battery"]["capacity_mwh"])
        planned_midnights.extend(_all_planned_midnights(new, new_daily, capacity, str(farm)))
        daily_plot_data[str(farm)] = (old_daily, new_daily)
        for label, root, daily in (("Historical: same-day, 50% target", old, old_daily),
                                   ("Extended IDA3, 40% floor", new, new_daily)):
            rows.append({"farm": str(farm), "framework": label, **_metrics(root, daily)})
            soc.append(_midnight_records(root, daily, capacity, label).assign(farm=str(farm)))
    frame = pd.DataFrame(rows)
    frame.to_csv(batch / "january_comparison.csv", index=False)
    pd.concat(soc, ignore_index=True).to_csv(batch / "january_midnight_soc.csv", index=False)
    pd.DataFrame(planned_midnights, columns=["farm", "decision_day", "auction", "cutoff_utc", "midnight_utc",
        "ending_delivery_day", "planned_soc_percent", "actual_soc_percent"]).to_csv(
            batch / "all_planned_midnights.csv", index=False)
    (batch / "comparison_input_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    lines = ["# January 2026 grid-tracking comparison", "",
        "The extended IDA3 horizon also changes the planner from a 50% daily equality to a 40% midnight floor. "
        "Differences therefore combine both changes. The greedy tracker does not enforce the planning floor. "
        "Continuous trajectories can make different auction commitments.", "",
        "Only January delivery products are scored; February 1 is planning context. "
        "The input checks compare realized PV, settlement prices and auction-product prices. "
        "PV model selection and current-interval observations are retrospective, and future actual auction prices "
        "are an oracle, including February prices in the extended IDA3 horizon. "
        "Final inventory is not valued in profit.", "",
        "| Farm | Framework | Profit (€) | Auction (€) | Settlement (€) | Wear (€) | Curtailment (MWh) | Absolute imbalance (MWh) | PV (MWh) | Charge (MWh) | Discharge (MWh) | Mean midnight SOC (%) | Midnights below 40% | Final SOC (%) | Auction fallbacks | Solve time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['farm']} | {row['framework']} | {row['net_profit_eur']:,.2f} | "
            f"{row['auction_revenue_eur']:,.2f} | {row['imbalance_cashflow_eur']:,.2f} | "
            f"{row['degradation_eur']:,.2f} | {row['curtailment_mwh']:,.2f} | "
            f"{row['absolute_imbalance_mwh']:,.2f} | {row['pv_realized_energy_mwh']:,.2f} | "
            f"{row['charge_mwh']:,.2f} | {row['discharge_mwh']:,.2f} | "
            f"{row['mean_midnight_soc_percent']:.2f} | {row['midnights_below_40_percent']} | "
            f"{row['final_soc_percent']:.2f} | {row['auction_fallbacks']} | {row['auction_solver_seconds']:.2f} |")
    lines += ["", "Daily IDA3 planned and actual midnight SOC: [CSV](january_midnight_soc.csv). "
        "All modeled midnight plans: [CSV](all_planned_midnights.csv). "
        "Input matching counts: [JSON](comparison_input_checks.json).", "", "## Farm reports", ""]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for farm, (old_daily, new_daily) in daily_plot_data.items():
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
        for label, daily in (("Historical 50% target", old_daily), ("Extended 40% floor", new_daily)):
            dates = pd.to_datetime(daily.delivery_day)
            axes[0].plot(dates, daily.net_profit_eur.cumsum(), label=label)
            axes[1].plot(dates, daily.absolute_imbalance_mwh.cumsum(), label=label)
            axes[2].plot(dates, daily.final_soc_percent, label=label)
        axes[0].set_ylabel("Cumulative profit (€)")
        axes[1].set_ylabel("Absolute imbalance (MWh)")
        axes[2].set_ylabel("Midnight SOC (%)")
        axes[2].axhline(40, color="black", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Italian delivery day")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle(f"Farm {farm}: January continuous replay")
        figure.savefig(batch / f"farm_{farm}_comparison.png", dpi=150)
        plt.close(figure)
        farm_rows = [row for row in rows if row["farm"] == farm]
        report = [f"# Farm {farm}: January grid-tracking comparison", "",
            "Horizon length and planning SOC policy both change; actual tracker SOC may fall below the floor.", "",
            "| Framework | Profit (€) | Auction (€) | Settlement (€) | Wear (€) | Fees (€) | PV (MWh) | Imbalance (MWh) | Curtailment (MWh) | Charge (MWh) | Discharge (MWh) | Final SOC (%) | Below 40% (days) | Fallbacks | Solve time (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in farm_rows:
            report.append(f"| {row['framework']} | {row['net_profit_eur']:,.2f} | "
                f"{row['auction_revenue_eur']:,.2f} | {row['imbalance_cashflow_eur']:,.2f} | "
                f"{row['degradation_eur']:,.2f} | {row['import_fee_eur'] + row['transaction_fee_eur']:,.2f} | "
                f"{row['pv_realized_energy_mwh']:,.2f} | {row['absolute_imbalance_mwh']:,.2f} | "
                f"{row['curtailment_mwh']:,.2f} | {row['charge_mwh']:,.2f} | {row['discharge_mwh']:,.2f} | "
                f"{row['final_soc_percent']:.2f} | {row['midnights_below_40_percent']} | "
                f"{row['auction_fallbacks']} | {row['auction_solver_seconds']:.2f} |")
        report.extend(["", f"![Farm {farm} comparison](farm_{farm}_comparison.png)", "",
            "[Planned versus actual midnight SOC](all_planned_midnights.csv); "
            "[combined results and limitations](summary.md).", ""])
        (batch / f"farm_{farm}_summary.md").write_text("\n".join(report))
        lines.append(f"- [Farm {farm}](farm_{farm}_summary.md)")
    lines.append("")
    path = batch / "summary.md"
    path.write_text("\n".join(lines))
    return path
