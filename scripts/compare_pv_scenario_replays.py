"""Compare completed exact and penalty continuous PV scenario experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compare(exact_root: Path, penalty_root: Path, output: Path) -> None:
    """Create matched daily and monthly tables and a compact comparison figure."""
    frames = []
    for treatment, root in (("exact", exact_root), ("penalty", penalty_root)):
        manifest = json.loads((root / "json/manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Cannot compare an incomplete experiment: {root}")
        for strategy in ("integrated", "perfect_pv"):
            data = pd.read_csv(root / "strategies" / strategy / "csv/daily_kpis.csv")
            data.insert(0, "treatment", treatment)
            data["comparison_label"] = f"{treatment}_{strategy}"
            frames.append(data)
    dates = [tuple(frame.delivery_day) for frame in frames]
    if any(value != dates[0] for value in dates[1:]):
        raise ValueError("Experiments do not have identical delivery days")
    daily = pd.concat(frames, ignore_index=True)
    metrics = ["net_profit_eur", "auction_revenue_eur", "imbalance_cashflow_eur",
        "degradation_eur", "import_fee_eur", "transaction_fee_eur",
        "illustrative_imbalance_penalty_eur", "signed_imbalance_mwh", "absolute_imbalance_mwh",
        "fallback_count"]
    monthly = daily.groupby("comparison_label", sort=False)[metrics].sum().reset_index()
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    monthly.to_csv(output / "monthly_comparison.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for label, group in daily.groupby("comparison_label", sort=False):
        dates = pd.to_datetime(group.delivery_day)
        axes[0].plot(dates, group.net_profit_eur.cumsum(), label=label)
        axes[1].plot(dates, group.absolute_imbalance_mwh, label=label)
    axes[0].set_ylabel("Cumulative realized profit (EUR)")
    axes[1].set_ylabel("Daily absolute imbalance (MWh)")
    axes[1].set_xlabel("Local delivery day")
    axes[0].legend()
    fig.savefig(output / "profit_and_imbalances.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--penalty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.exact, args.penalty, args.output)


if __name__ == "__main__":
    main()
