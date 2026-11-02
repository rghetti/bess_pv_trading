"""Compare matched deterministic, PV-scenario and perfect-PV continuous replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path):
    """Read an experiment artifact without modifying its pinned input evidence."""
    return json.loads(path.read_text())


def compare(deterministic: Path, scenarios: Path, output: Path) -> None:
    """Check common inputs and write matched daily, monthly and plotted results."""
    roots = (deterministic, scenarios)
    manifests = [_read_json(root / "json/manifest.json") for root in roots]
    if any(item.get("status") != "complete" for item in manifests):
        raise ValueError("Both experiments must be complete before comparison")
    inputs = [_read_json(root / "json/input_provenance.json") for root in roots]
    days = list(inputs[0]["days"])
    if days != list(inputs[1]["days"]):
        raise ValueError("Experiment delivery days do not match")
    for day in days:
        for key in ("realized_pv_sha256", "auction_price_sha256"):
            if inputs[0]["days"][day][key] != inputs[1]["days"][day][key]:
                raise ValueError(f"Experiment {key} differs on {day}")
    for key in ("settlement_values_sha256", "plant", "asset_input_settings", "weather_forecasts"):
        if inputs[0][key] != inputs[1][key]:
            raise ValueError(f"Experiment input {key} differs")
    sources = (("deterministic", deterministic, "integrated"),
               ("scenarios_51", scenarios, "integrated"),
               ("perfect_pv", scenarios, "perfect_pv"))
    frames = []
    for label, root, strategy in sources:
        frame = pd.read_csv(root / "strategies" / strategy / "csv/daily_kpis.csv")
        if list(frame.delivery_day) != days:
            raise ValueError(f"Daily results for {label} do not match the pinned dates")
        frame.insert(0, "comparison_label", label)
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    metrics = ("net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
               "import_fee_eur", "transaction_fee_eur", "signed_imbalance_mwh", "absolute_imbalance_mwh",
               "fallback_count")
    monthly = daily.groupby("comparison_label", sort=False)[list(metrics)].sum().reset_index()
    last_soc = daily.groupby("comparison_label", sort=False).final_soc_percent.last()
    monthly["final_soc_percent"] = monthly.comparison_label.map(last_soc)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    monthly.to_csv(output / "monthly_comparison.csv", index=False)
    (output / "input_verification.json").write_text(json.dumps({"matched_delivery_days": len(days),
        "matched": ["realized PV", "auction prices", "settlement prices", "plant", "asset settings",
                    "deterministic weather forecasts"],
        "perfect_pv_source": str(scenarios), "deterministic_source": str(deterministic)}, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"deterministic": "Updating deterministic forecast", "scenarios_51": "51 PV scenarios",
              "perfect_pv": "Perfect PV reference (DA only)"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for label, frame in daily.groupby("comparison_label", sort=False):
        dates = pd.to_datetime(frame.delivery_day)
        axes[0].plot(dates, frame.net_profit_eur.cumsum(), label=labels[label])
        axes[1].plot(dates, frame.absolute_imbalance_mwh, label=labels[label])
    axes[0].set_ylabel("Cumulative realized profit (EUR)")
    axes[1].set_ylabel("Daily absolute imbalance (MWh)")
    axes[1].set_xlabel("Local delivery day")
    axes[0].legend()
    fig.savefig(output / "profit_and_imbalances.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.deterministic, args.scenarios, args.output)


if __name__ == "__main__":
    main()
