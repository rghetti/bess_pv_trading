"""Audit completed price-search forecasts by month and observed price regime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.price_search import point_metrics


def _sha256(path):
    """Fingerprint the completed search manifest used to select files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monthly_rows(frame, *, timezone, near_zero_threshold):
    """Compute pooled errors and observed-price context for each local month."""
    if near_zero_threshold < 0 or not np.isfinite(near_zero_threshold):
        raise ValueError("Near-zero threshold must be finite and nonnegative")
    values = frame.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(values.product_start, utc=True))
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Monthly report requires unique increasing delivery products")
    values["month"] = timestamps.tz_convert(timezone).strftime("%Y-%m")
    rows = []
    for month, group in values.groupby("month", sort=True):
        actual = group.actual.to_numpy(float)
        scores = point_metrics(actual, group.point.to_numpy(float), group.seasonal_naive.to_numpy(float))
        rows.append({"month": month, **scores, "mean_absolute_actual_eur_per_mwh": float(np.abs(actual).mean()),
                     "negative_price_count": int((actual < 0).sum()),
                     "near_zero_price_count": int((np.abs(actual) <= near_zero_threshold).sum())})
    return rows


def run(config):
    """Check matched samples, retain every monthly score and render a compact report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = config["price_monthly_report"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Monthly report requires a completed search without failed fits")
    if manifest.get("selection"):
        raise ValueError("This diagnostic expects the full candidate search, not a selected-model evaluation")
    if {row["group"] for row in manifest["results"]} != set(manifest["config"]["product_groups"]):
        raise ValueError("Completed search does not contain every declared market group")
    if not np.isfinite(settings["target_wape_percent"]) or settings["target_wape_percent"] < 0:
        raise ValueError("Target WAPE must be finite and nonnegative")
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in sorted({record["group"] for record in manifest["results"]}):
        reference_starts, reference_actual = None, None
        records = sorted((record for record in manifest["results"] if record["group"] == group),
                         key=lambda record: (record["feature_set"], record["candidate"]))
        split = manifest["config"]["product_groups"][group]
        feature_sets = split.get("feature_sets", manifest["config"]["feature_sets"])
        expected = {(feature, candidate["id"]) for feature in feature_sets
                    for candidate in manifest["config"]["candidates"]}
        if len(records) != len(expected) or {(row["feature_set"], row["candidate"]) for row in records} != expected:
            raise ValueError(f"Completed search has incomplete or duplicate candidates for {group}")
        for record in records:
            path = source / f"{group}__{record['feature_set']}__{record['candidate']}.csv"
            frame = pd.read_csv(path)
            starts = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True))
            actual = frame.actual.to_numpy(float)
            if reference_starts is None:
                reference_starts, reference_actual = starts, actual
            elif not reference_starts.equals(starts) or not np.array_equal(reference_actual, actual):
                raise ValueError(f"Candidate {path.name} does not share the market's exact evaluation sample")
            rows.extend({"group": group, "feature_set": record["feature_set"], "candidate": record["candidate"],
                         **row} for row in monthly_rows(frame, timezone=settings["timezone"],
                                                       near_zero_threshold=settings["near_zero_eur_per_mwh"]))
    table = pd.DataFrame(rows)
    table.to_csv(output / "monthly_scores.csv", index=False)
    # The best whole-period model is shown only for diagnostic plots; picking a
    # model after looking at these months is not an independent performance test.
    lines = ["# Monthly price-search diagnostics", "",
             "All models in each market use identical evaluation intervals. The displayed best "
             "whole-period candidate was selected on the same period and is exploratory; "
             "all model/month scores are retained in `monthly_scores.csv`.", ""]
    for group in sorted(table.group.unique()):
        candidates = [row for row in manifest["results"] if row["group"] == group]
        best = min(candidates, key=lambda row: (row["mae_eur_per_mwh"], row["feature_set"], row["candidate"]))
        selected = table.loc[(table.group == group) & (table.feature_set == best["feature_set"])
                             & (table.candidate == best["candidate"])].sort_values("month")
        lines.extend([f"## {group}", "", f"Exploratory whole-period MAE leader: `{best['feature_set']}/{best['candidate']}`.", "",
                      "| Month | Products | MAE EUR/MWh | WAPE % | Mean absolute actual EUR/MWh | Negative | Near zero |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for row in selected.itertuples(index=False):
            relative = "n/a" if pd.isna(row.wape_percent) else f"{row.wape_percent:.2f}"
            lines.append(f"| {row.month} | {row.observations} | {row.mae_eur_per_mwh:.3f} | {relative} | "
                         f"{row.mean_absolute_actual_eur_per_mwh:.2f} | {row.negative_price_count} | {row.near_zero_price_count} |")
        lines.append("")
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(selected.month, selected.wape_percent)
        axis.axhline(settings["target_wape_percent"], color="red", linestyle="--", label="Study target")
        axis.set(ylabel="WAPE (%)", title=f"{group}: exploratory selected candidate")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{group}_monthly_wape.png", dpi=130)
        plt.close(figure)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    evidence = {"status": "complete", "source_manifest_sha256": _sha256(manifest_path),
                "source_search": str(source), "settings": settings, "rows": len(rows)}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    """Default to a plan; execute only when explicitly selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(run(config) if args.run else config["price_monthly_report"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
