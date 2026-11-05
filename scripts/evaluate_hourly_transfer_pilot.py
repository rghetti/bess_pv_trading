"""Evaluate a two-stage hourly-to-quarter-hourly price transfer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bess_pv_trading.config import load_config


def digest(path):
    """Identify the exact saved source arrays used by the pilot."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(hourly_path, quarter_path, hourly_cutoff, lag_count, hourly_alphas, quarter_alphas, hourly_weights):
    """Fit the hourly component before the resolution change and score later products."""
    cutoff = pd.Timestamp(hourly_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Hourly cutoff must be timezone-aware")
    cutoff_ns = cutoff.tz_convert("UTC").as_unit("ns").value
    if isinstance(lag_count, bool) or not isinstance(lag_count, int) or lag_count < 1:
        raise ValueError("Lag count must be a positive integer")
    columns = lag_count + 7
    # Retain hourly training labels only before the resolution boundary.
    with np.load(hourly_path, allow_pickle=False) as data:
        training_starts = pd.DatetimeIndex(pd.to_datetime(data["train_starts"], utc=True))
        validation_starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
        if training_starts.max().as_unit("ns").value >= cutoff_ns:
            raise ValueError("Hourly training extends beyond the requested cutoff")
        keep = ((data["available_ns"] <= cutoff_ns) & (data["product_end_ns"] <= cutoff_ns)
                & (validation_starts.as_unit("ns").asi8 < cutoff_ns))
        hourly_x = np.concatenate([data["train_x"][:, :columns], data["validation_x"][keep, :columns]])
        hourly_y = np.concatenate([data["train_y"], data["validation_y"][keep]])
    # Quarter-hourly training remains separate and strictly precedes evaluation.
    with np.load(quarter_path, allow_pickle=False) as data:
        quarter_starts = pd.DatetimeIndex(pd.to_datetime(data["train_starts"], utc=True))
        evaluation_starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
        if quarter_starts.min().as_unit("ns").value < cutoff_ns:
            raise ValueError("Quarter-hour training precedes the hourly training cutoff")
        if quarter_starts.max() >= evaluation_starts.min():
            raise ValueError("Quarter-hour training overlaps evaluation")
        quarter_x = data["train_x"][:, :columns]
        quarter_y = data["train_y"]
        evaluation_x = data["validation_x"][:, :columns]
        evaluation_y = data["validation_y"]
    if min(hourly_x.shape[1], quarter_x.shape[1], evaluation_x.shape[1]) != columns:
        raise ValueError("A source matrix lacks the required lag and calendar columns")
    results = []
    # A zero weight is the matched quarter-hourly-only reference.
    for hourly_alpha in hourly_alphas:
        hourly_model = make_pipeline(StandardScaler(), Ridge(alpha=hourly_alpha))
        hourly_model.fit(hourly_x, hourly_y)
        hourly_on_train = hourly_model.predict(quarter_x)
        hourly_on_test = hourly_model.predict(evaluation_x)
        for quarter_alpha in quarter_alphas:
            for weight in hourly_weights:
                residual = make_pipeline(StandardScaler(), Ridge(alpha=quarter_alpha))
                residual.fit(quarter_x, quarter_y - weight * hourly_on_train)
                prediction = weight * hourly_on_test + residual.predict(evaluation_x)
                error = np.abs(evaluation_y - prediction)
                results.append({"hourly_alpha": hourly_alpha, "quarter_alpha": quarter_alpha,
                                "hourly_weight": weight, "mae": float(error.mean()),
                                "wape_percent": float(100 * error.sum() / np.abs(evaluation_y).sum())})
    return {"hourly_sha256": digest(hourly_path), "quarter_sha256": digest(quarter_path),
            "hourly_rows": len(hourly_y), "quarter_rows": len(quarter_y),
            "evaluation_rows": len(evaluation_y), "results": results}


def main():
    """Read pinned sources and print the exploratory comparison as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    settings = load_config(args.config)["hourly_transfer_pilot"]
    # Require matching declared lags, timezone and separate native resolutions.
    hourly_path = args.project_root / settings["hourly_source"]
    quarter_path = args.project_root / settings["quarter_hourly_source"]
    hourly_manifest = json.loads((hourly_path.parent / "manifest.json").read_text())
    quarter_manifest = json.loads((quarter_path.parent / "manifest.json").read_text())
    if any(manifest.get("status") != "complete" or manifest.get("failures")
           for manifest in (hourly_manifest, quarter_manifest)):
        raise ValueError("Both source searches must be complete without failures")
    hourly_config = hourly_manifest["config"]
    quarter_config = quarter_manifest["config"]
    hourly_group = hourly_config["product_groups"]["hourly"]
    quarter_group = quarter_config["product_groups"]["quarter_hourly"]
    hourly_lags = hourly_group.get("lag_hours", hourly_config["lag_hours"])
    quarter_lags = quarter_group.get("lag_hours", quarter_config["lag_hours"])
    # The source training mask must not include labels after the transfer boundary.
    if pd.Timestamp(hourly_group["cutoff"]) > pd.Timestamp(settings["hourly_cutoff"]):
        raise ValueError("Hourly source cutoff exceeds the transfer boundary")
    if (hourly_lags != quarter_lags or len(hourly_lags) != settings["lag_count"]
            or hourly_config["timezone"] != quarter_config["timezone"]
            or hourly_group["native_minutes"] != 60 or quarter_group["native_minutes"] != 15):
        raise ValueError("Source lag layouts, timezones or native resolutions do not match")
    result = evaluate(hourly_path, quarter_path, settings["hourly_cutoff"], settings["lag_count"],
                      settings["hourly_alphas"], settings["quarter_alphas"], settings["hourly_weights"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
