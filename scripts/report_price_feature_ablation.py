"""Compare matched price forecasts with and without supplemental features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from scripts.report_price_search_monthly import _sha256


def paired_score(base, added, *, timezone, seed, bootstrap_samples):
    """Measure paired absolute-error changes and resample whole delivery days."""
    base_times = pd.DatetimeIndex(pd.to_datetime(base.product_start, utc=True))
    added_times = pd.DatetimeIndex(pd.to_datetime(added.product_start, utc=True))
    if base_times.has_duplicates or not base_times.is_monotonic_increasing or not base_times.equals(added_times):
        raise ValueError("Ablation predictions do not have identical ordered products")
    actual = base.actual.to_numpy(float)
    if not np.array_equal(actual, added.actual.to_numpy(float)):
        raise ValueError("Ablation predictions have different actual prices")
    base_error = np.abs(base.point.to_numpy(float) - actual)
    added_error = np.abs(added.point.to_numpy(float) - actual)
    if not (np.isfinite(actual).all() and np.isfinite(base_error).all() and np.isfinite(added_error).all()):
        raise ValueError("Ablation values must all be finite")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")

    # Resample days as blocks so intervals within a day remain together.
    days = base_times.tz_convert(timezone).strftime("%Y-%m-%d")
    daily = pd.DataFrame({"day": days, "difference": added_error - base_error}).groupby("day").agg(
        error_sum=("difference", "sum"), count=("difference", "size"))
    if daily.empty:
        raise ValueError("Ablation needs at least one observed day")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(daily), size=(bootstrap_samples, len(daily)))
    sums = daily.error_sum.to_numpy()[samples].sum(axis=1)
    counts = daily['count'].to_numpy()[samples].sum(axis=1)
    low, high = np.percentile(sums / counts, [2.5, 97.5])
    denominator = np.abs(actual).sum()
    return {"observations": len(actual), "days": len(daily),
            "base_mae": float(base_error.mean()), "added_mae": float(added_error.mean()),
            "mae_difference": float((added_error - base_error).mean()),
            "difference_day_bootstrap_low": float(low), "difference_day_bootstrap_high": float(high),
            "base_wape_percent": float(100 * base_error.sum() / denominator) if denominator else None,
            "added_wape_percent": float(100 * added_error.sum() / denominator) if denominator else None}


def run(config):
    """Verify full matched searches, then report all paired candidates."""
    settings = config["price_feature_ablation"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures") or manifest.get("selection"):
        raise ValueError("Ablation requires a complete, failure-free full search")
    if not manifest["config"].get("common_training_rows"):
        raise ValueError("Ablation requires matched training rows")
    candidates = {candidate["id"] for candidate in manifest["config"]["candidates"]}
    results = manifest["results"]
    rows = []
    for group in manifest["config"]["product_groups"]:
        for pair in settings["pairs"]:
            base_name, added_name = pair["base"], pair["added"]
            for feature in (base_name, added_name):
                records = [row for row in results if row["group"] == group and row["feature_set"] == feature]
                if len(records) != len(candidates) or {row["candidate"] for row in records} != candidates:
                    raise ValueError(f"Incomplete or duplicate candidate results for {group}/{feature}")
            # The common-row preparation is checked against stored training
            # labels as well as the exact validation products for each fit.
            base_data = source / f"{group}__{base_name}.npz"
            added_data = source / f"{group}__{added_name}.npz"
            with np.load(base_data) as left, np.load(added_data) as right:
                if not np.array_equal(left['train_y'], right['train_y']):
                    raise ValueError("Matched ablation training targets differ")
                if 'train_starts' in left and 'train_starts' in right and not np.array_equal(left['train_starts'], right['train_starts']):
                    raise ValueError("Matched ablation training products differ")
            for candidate in sorted(candidates):
                left = pd.read_csv(source / f"{group}__{base_name}__{candidate}.csv")
                right = pd.read_csv(source / f"{group}__{added_name}__{candidate}.csv")
                rows.append({"group": group, "base": base_name, "added": added_name, "candidate": candidate,
                             **paired_score(left, right, timezone=settings["timezone"], seed=settings["seed"],
                                            bootstrap_samples=settings["bootstrap_samples"])})
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "paired_scores.csv", index=False)
    lines = ["# Paired price-feature ablations", "",
             "Negative MAE differences mean the added features improved the score. Comparisons use "
             "matched training rows and identical evaluation products. Results are exploratory: "
             "models and features have been inspected on these same dates. Day-block bootstrap "
             "intervals describe sampling variation, not a corrected multiple-testing test.", "",
             "| Market | Base | Added | Candidates | Improved | Median MAE change EUR/MWh |",
             "|---|---|---|---:|---:|---:|"]
    for (group, base, added), scores in table.groupby(["group", "base", "added"], sort=True):
        lines.append(f"| {group} | {base} | {added} | {len(scores)} | "
                     f"{int((scores.mae_difference < 0).sum())} | {scores.mae_difference.median():.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "source_manifest_sha256": _sha256(manifest_path), "pairs": len(rows)}, indent=2) + "\n")
    return output


def main():
    """Run the configured comparison for an already-completed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
