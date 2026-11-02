"""Compare matched January PV replays with and without battery storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(path):
    return json.loads(path.read_text())


def compare(battery_deterministic, battery_scenarios, no_battery_deterministic,
            no_battery_scenarios, output):
    """Validate shared histories and save matched daily and monthly results."""
    roots = (battery_deterministic, battery_scenarios, no_battery_deterministic, no_battery_scenarios)
    manifests = [_read(root / "json/manifest.json") for root in roots]
    if any(item.get("status") != "complete" for item in manifests):
        raise ValueError("All four experiments must be complete")
    evidence = [_read(root / "json/input_provenance.json") for root in roots]
    reference = evidence[0]
    days = list(reference["days"])
    settings = {key: value for key, value in reference["asset_input_settings"].items()
                if key not in {"battery", "state", "degradation"}}
    plant = {key: value for key, value in reference["plant"].items() if key != "battery"}
    for item in evidence[1:]:
        if list(item["days"]) != days:
            raise ValueError("Delivery days differ across experiments")
        for day in days:
            for key in ("realized_pv_sha256", "auction_price_sha256"):
                if item["days"][day][key] != reference["days"][day][key]:
                    raise ValueError(f"{key} differs on {day}")
        other_settings = {key: value for key, value in item["asset_input_settings"].items()
                          if key not in {"battery", "state", "degradation"}}
        other_plant = {key: value for key, value in item["plant"].items() if key != "battery"}
        if (item["weather_forecasts"] != reference["weather_forecasts"] or
                item["settlement_values_sha256"] != reference["settlement_values_sha256"] or
                other_settings != settings or other_plant != plant):
            raise ValueError("Weather, market or non-battery assumptions differ")
    for day in days:
        if (evidence[1]["pv_ensemble"][day]["values_sha256"] !=
                evidence[3]["pv_ensemble"][day]["values_sha256"]):
            raise ValueError(f"PV ensemble differs on {day}")
    if any(evidence[i]["plant"]["battery"]["capacity_mwh"] != 0 or
           evidence[i]["plant"]["battery"]["max_power_mw"] != 0 for i in (2, 3)):
        raise ValueError("No-battery experiments must have zero capacity and power")
    sources = (("deterministic", "battery", roots[0], "integrated"),
               ("scenarios_51", "battery", roots[1], "integrated"),
               ("perfect_pv", "battery", roots[1], "perfect_pv"),
               ("deterministic", "no_battery", roots[2], "integrated"),
               ("scenarios_51", "no_battery", roots[3], "integrated"),
               ("perfect_pv", "no_battery", roots[3], "perfect_pv"))
    frames = []
    for policy, asset, root, strategy in sources:
        frame = pd.read_csv(root / "strategies" / strategy / "csv/daily_kpis.csv")
        if list(frame.delivery_day) != days:
            raise ValueError(f"Incomplete daily results for {policy}/{asset}")
        frame.insert(0, "policy", policy)
        frame.insert(1, "asset", asset)
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    metrics = ["net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
               "transaction_fee_eur", "signed_imbalance_mwh", "absolute_imbalance_mwh",
               "curtailment_mwh", "pv_realized_energy_mwh", "fallback_count"]
    monthly = daily.groupby(["policy", "asset"], sort=False)[metrics].sum().reset_index()
    with_battery = monthly.loc[monthly.asset.eq("battery")].set_index("policy")
    without_battery = monthly.loc[monthly.asset.eq("no_battery")].set_index("policy")
    deltas = (with_battery[metrics] - without_battery[metrics]).add_suffix("_battery_minus_no_battery")
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    monthly.to_csv(output / "monthly_comparison.csv", index=False)
    deltas.to_csv(output / "battery_effect_by_policy.csv")
    (output / "input_verification.json").write_text(json.dumps({"matched_delivery_days": len(days),
        "matched": ["measured PV", "auction prices", "settlement", "deterministic forecasts",
                    "ensemble values", "non-battery assumptions"],
        "with_battery_initial_energy_mwh": reference["plant"]["battery"]["capacity_mwh"] *
            float(reference["asset_input_settings"]["state"]["initial_soc_percent"]) / 100,
        "note": "Battery and no-battery runs start with different stored energy; no terminal salvage is included."},
        indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"deterministic": "Deterministic", "scenarios_51": "51 scenarios", "perfect_pv": "Perfect PV (DA only)"}
    colors = {"deterministic": "C0", "scenarios_51": "C1", "perfect_pv": "C2"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    for (policy, asset), frame in daily.groupby(["policy", "asset"], sort=False):
        dates = pd.to_datetime(frame.delivery_day)
        label = f"{labels[policy]} — {'battery' if asset == 'battery' else 'no battery'}"
        style = "-" if asset == "battery" else "--"
        for axis, values in zip(axes, (frame.net_profit_eur.cumsum(), frame.absolute_imbalance_mwh, frame.curtailment_mwh)):
            axis.plot(dates, values, linestyle=style, color=colors[policy], label=label)
    axes[0].set_ylabel("Cumulative realized profit (EUR)")
    axes[1].set_ylabel("Daily absolute imbalance (MWh)")
    axes[2].set_ylabel("Daily curtailment (MWh)")
    axes[2].set_xlabel("Local delivery day")
    axes[0].legend(ncol=2, fontsize="small")
    fig.savefig(output / "profit_imbalances_curtailment.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-deterministic", type=Path, required=True)
    parser.add_argument("--battery-scenarios", type=Path, required=True)
    parser.add_argument("--no-battery-deterministic", type=Path, required=True)
    parser.add_argument("--no-battery-scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.battery_deterministic, args.battery_scenarios,
            args.no_battery_deterministic, args.no_battery_scenarios, args.output)


if __name__ == "__main__":
    main()
