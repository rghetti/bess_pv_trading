"""Run and compare deterministic and ensemble continuous participation studies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.participation import participation_profiles, profile_id
from bess_pv_trading.reporting.continuous_summary import aggregate_continuous_daily


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("sequential_all_auctions", "perfect_pv_all_auctions")


def _read_json(path):
    """Read the recorded configuration, status or input evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def _comparable_config(config):
    """Remove only the intended forecast variant and destination settings."""
    result = deepcopy(config)
    settings = result["continuous_optimization"]
    for field in ("pv_ensemble", "run_id", "resume"):
        settings.pop(field, None)
    return result


def compare(deterministic, ensemble, output):
    """Validate matched completed inputs and write paired monthly accounting."""
    roots = (Path(deterministic), Path(ensemble))
    manifests = [_read_json(root / "json/manifest.json") for root in roots]
    configs = [_read_json(root / "json/study_config.json") for root in roots]
    inputs = [_read_json(root / "json/input_provenance.json") for root in roots]
    if any(manifest.get("status") != "complete" for manifest in manifests):
        raise ValueError("Both participation studies must be complete")
    if _comparable_config(configs[0]) != _comparable_config(configs[1]):
        raise ValueError("Study configurations differ beyond forecast variant and run identity")
    if configs[0]["continuous_optimization"]["pv_ensemble"]["enabled"] or not configs[1]["continuous_optimization"]["pv_ensemble"]["enabled"]:
        raise ValueError("Expected deterministic followed by ensemble study")
    days = list(manifests[0]["delivery_days"])
    if not days or manifests[1]["delivery_days"] != days or any(list(item["days"]) != days for item in inputs):
        raise ValueError("Study delivery dates do not match")
    # Compare pinned actuals, prices, forecast inputs and physical assumptions.
    for key in ("plant", "asset_input_settings", "weather_forecasts", "settlement_values_sha256"):
        if inputs[0][key] != inputs[1][key]:
            raise ValueError(f"Study input {key} differs")
    for day in days:
        for key in ("realized_pv_sha256", "auction_price_sha256", "skipped_auctions"):
            if inputs[0]["days"][day][key] != inputs[1]["days"][day][key]:
                raise ValueError(f"Study {key} differs on {day}")
    profiles = [profile_id(profile) for profile in participation_profiles(configs[0]["continuous_participation_study"])]
    names = profiles + list(BASELINES)
    frames = []
    for variant, root, manifest in zip(("deterministic", "ensemble"), roots, manifests):
        if len(manifest["completed"]) != len(names) or set(manifest["completed"]) != set(names):
            raise ValueError(f"Incomplete or unexpected profiles in {variant} study")
        daily = pd.read_csv(root / "csv/daily_participation.csv", dtype={"delivery_day": str, "profile": str})
        expected = {(name, day) for name in names for day in days}
        if daily.duplicated(["profile", "delivery_day"]).any() or set(zip(daily.profile, daily.delivery_day)) != expected:
            raise ValueError(f"Incomplete or duplicate daily results in {variant} study")
        daily["forecast_variant"] = variant
        frames.append(daily)
    daily = pd.concat(frames, ignore_index=True)
    totals = []
    for variant, frame in daily.groupby("forecast_variant", sort=False):
        monthly = aggregate_continuous_daily(frame, strategy_column="profile")
        # Final inventory differs by trajectory and is relevant to cash-profit comparison.
        end_soc = frame.loc[frame.delivery_day.eq(days[-1])].set_index("profile").final_soc_percent
        monthly["final_soc_percent"] = monthly.profile.map(end_soc)
        if not np.isfinite(pd.to_numeric(monthly.final_soc_percent).to_numpy(float)).all():
            raise ValueError(f"Missing final SOC in {variant} study")
        monthly["forecast_variant"] = variant
        totals.append(monthly)
    combined = pd.concat(totals, ignore_index=True)
    left = combined[combined.forecast_variant.eq("deterministic")].set_index("profile")
    right = combined[combined.forecast_variant.eq("ensemble")].set_index("profile")
    paired = left.join(right, lsuffix="_deterministic", rsuffix="_ensemble").loc[names].reset_index()
    paired["profit_difference_ensemble_minus_deterministic_eur"] = paired.net_profit_eur_ensemble - paired.net_profit_eur_deterministic
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    paired.to_csv(output / "monthly_comparison.csv", index=False)
    labels = {"sequential_all_auctions": "Sequential (all auctions)",
              "perfect_pv_all_auctions": "Perfect PV (all auctions)"}
    farm_id = configs[0]["plant"]["farm_id"]
    lines = [f"# Farm {farm_id}: continuous participation forecast comparison", "",
        f"Local delivery dates: {days[0]} to {days[-1]} ({len(days)} days).", "",
        "| Auction participation / strategy | Profit deterministic (€) | Profit ensemble (€) | Profit difference (€) | Signed imbalance deterministic (MWh) | Signed imbalance ensemble (MWh) | Absolute imbalance deterministic (MWh) | Absolute imbalance ensemble (MWh) | Curtailment deterministic (MWh) | Curtailment ensemble (MWh) | Final SOC deterministic (%) | Final SOC ensemble (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in paired.to_dict("records"):
        fields = ("net_profit_eur_deterministic", "net_profit_eur_ensemble",
            "profit_difference_ensemble_minus_deterministic_eur", "signed_imbalance_mwh_deterministic",
            "signed_imbalance_mwh_ensemble", "absolute_imbalance_mwh_deterministic",
            "absolute_imbalance_mwh_ensemble", "curtailment_mwh_deterministic", "curtailment_mwh_ensemble",
            "final_soc_percent_deterministic", "final_soc_percent_ensemble")
        values = [f"{float(row[field]):,.2f}" for field in fields]
        lines.append("| " + " | ".join([labels.get(row["profile"], row["profile"]), *values]) + " |")
    lines.extend(["", "Profit is realized auction revenue plus imbalance settlement, less battery degradation, fees and any realized penalty. Profit difference is ensemble minus deterministic. Signed imbalance is net surplus; absolute imbalance sums interval magnitudes. Curtailment is executed energy.",
        "", "Each row is a separate continuous trajectory with its own carried SOC and commitments. Final SOC is shown because cash profits do not value remaining inventory. The perfect-PV reference is an information benchmark, not a guaranteed upper bound. Actual auction prices are known in planning.",
        "", "The ensemble includes the updated deterministic forecast and 50 weather members from the preceding day's 00 UTC cycle, using the configured availability assumption. The pointwise minimum is an optimization envelope, not a coherent weather trajectory.",
        "", f"Deterministic source: `{roots[0]}`", f"Ensemble source: `{roots[1]}`", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return paired


def main():
    """Run both variants sequentially, or regenerate a report from completed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-id", type=int, required=True, choices=(1, 345, 901, 910))
    parser.add_argument("--report-only", action="store_true", help="Compare completed runs without starting or resuming experiments")
    args = parser.parse_args()
    farm_prefix = "" if args.farm_id == 1 else f"farm{args.farm_id}_"
    paths = [ROOT / f"configs/full_experiments/continuous_participation_2026_01_{farm_prefix}{variant}.yaml"
             for variant in ("deterministic", "ensemble")]
    configs = [load_config(path) for path in paths]
    if _comparable_config(configs[0]) != _comparable_config(configs[1]):
        raise ValueError("Study configurations differ beyond forecast variant and run identity")
    # Sequential subprocesses avoid oversubscribing Gurobi and the shared database.
    if not args.report_only:
        for path in paths:
            subprocess.run([sys.executable, "-m", "bess_pv_trading", "experiment-continuous",
                            "--config", str(path)], cwd=ROOT, check=True)
    roots = [ROOT / config["continuous_optimization"]["output_root"] / f"farm_{args.farm_id}"
             / config["continuous_optimization"]["run_id"] for config in configs]
    output = ROOT / "outputs/continuous_optimization/comparisons" / f"farm_{args.farm_id}" / "2026_01_continuous_participation_forecast_comparison"
    compare(roots[0], roots[1], output)
    print(f"Comparison saved to {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
