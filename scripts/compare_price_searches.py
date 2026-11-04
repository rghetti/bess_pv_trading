"""Compare matching price-model fits on common evaluation products."""

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
from scripts.report_price_feature_ablation import paired_score
from scripts.report_price_search_monthly import _sha256


def align_predictions(left, right):
    """Join unique delivery products and verify their actual prices agree."""
    left = left.copy()
    right = right.copy()
    for frame in (left, right):
        frame["product_start"] = pd.to_datetime(frame.product_start, utc=True)
        if frame.product_start.duplicated().any():
            raise ValueError("Comparison requires unique delivery products")
    joined = left.merge(right, on="product_start", how="inner", suffixes=("_left", "_right"), validate="one_to_one")
    joined = joined.sort_values("product_start").reset_index(drop=True)
    if joined.empty or not np.array_equal(joined.actual_left.to_numpy(), joined.actual_right.to_numpy()):
        raise ValueError("Comparison requires common products with identical actual prices")
    baseline = joined.rename(columns={"actual_left": "actual", "point_left": "point"})
    comparison = joined.rename(columns={"actual_right": "actual", "point_right": "point"})
    return baseline, comparison


def run(config):
    """Verify complete searches and score each identical candidate on common products."""
    settings = config["price_search_comparison"]
    root = Path(config.get("_project_root", ROOT))
    sources = [root / settings["baseline_search"], root / settings["comparison_search"]]
    manifests = [json.loads((source / "manifest.json").read_text()) for source in sources]
    if any(manifest.get("status") != "complete" or manifest.get("failures") or manifest.get("selection")
           for manifest in manifests):
        raise ValueError("Comparison requires complete failure-free full searches")
    candidate_maps = [{item["id"]: item for item in manifest["config"]["candidates"]} for manifest in manifests]
    if candidate_maps[0] != candidate_maps[1]:
        raise ValueError("Comparison candidate definitions differ")
    result_maps = [{(row["group"], row["feature_set"], row["candidate"]): row for row in manifest["results"]}
                   for manifest in manifests]
    if any(len(mapping) != len(manifest["results"]) for mapping, manifest in zip(result_maps, manifests)):
        raise ValueError("Comparison source has duplicate candidate records")
    for mapping, manifest in zip(result_maps, manifests):
        study = manifest["config"]
        expected = {(group, feature, candidate) for group, split in study["product_groups"].items()
                    for feature in split.get("feature_sets", study["feature_sets"]) for candidate in candidate_maps[0]}
        if set(mapping) != expected:
            raise ValueError("Comparison source lacks a declared candidate result")
    if set(result_maps[0]) != set(result_maps[1]):
        raise ValueError("Comparison sources contain different candidate results")
    rows, training_checks = [], {}
    for group, feature, candidate in sorted(result_maps[0]):
        # Training-target equality is diagnostic: some feature policies change
        # which rows are eligible, even when evaluation products are matched.
        if (group, feature) not in training_checks:
            datasets = [source / f"{group}__{feature}.npz" for source in sources]
            with np.load(datasets[0], allow_pickle=False) as left, np.load(datasets[1], allow_pickle=False) as right:
                same_targets = bool(np.array_equal(left["train_y"], right["train_y"]))
                same_starts = (bool(np.array_equal(left["train_starts"], right["train_starts"]))
                               if "train_starts" in left and "train_starts" in right else None)
                training_checks[(group, feature)] = (same_targets, same_starts, len(left["train_y"]), len(right["train_y"]))
        same_targets, same_starts, left_count, right_count = training_checks[(group, feature)]
        name = f"{group}__{feature}__{candidate}.csv"
        left = pd.read_csv(sources[0] / name)
        right = pd.read_csv(sources[1] / name)
        baseline, comparison = align_predictions(left, right)
        rows.append({"group": group, "feature_set": feature, "candidate": candidate,
                     "baseline_products": len(left), "comparison_products": len(right),
                     "baseline_training_rows": left_count, "comparison_training_rows": right_count,
                     "same_training_targets": same_targets, "same_training_timestamps": same_starts,
                     **paired_score(baseline, comparison, timezone=settings["timezone"], seed=settings["seed"],
                                    bootstrap_samples=settings["bootstrap_samples"])})
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "paired_scores.csv", index=False)
    lines = ["# Price-search comparison on common products", "",
             "Negative MAE differences favor the comparison search. Each row uses matching model "
             "specifications and common actual delivery prices. The searches may use different "
             "training rows and multiple feature-policy changes; see the CSV before attributing differences. "
             "Scores and bootstrap intervals are exploratory on previously inspected dates.", "",
             "| Market | Feature set | Candidates | Improved | Median MAE change €/MWh | Same training targets |",
             "|---|---|---:|---:|---:|---|"]
    for (group, feature), values in table.groupby(["group", "feature_set"], sort=True):
        lines.append(f"| {group} | {feature} | {len(values)} | {int((values.mae_difference < 0).sum())} | "
                     f"{values.mae_difference.median():.3f} | {bool(values.same_training_targets.all())} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "source_manifest_sha256": [_sha256(source / "manifest.json") for source in sources],
        "candidate_pairs": len(rows)}, indent=2) + "\n")
    return output


def main():
    """Run the configured comparison of completed search outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
