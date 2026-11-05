"""Evaluate joint IDA spread curves against the latest-cleared-price baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bess_pv_trading.config import load_config
from scripts.evaluate_daily_price_curves import check_joint_cutoffs, checksum


def market_days(starts, features, targets, timezone, start_hour, interval_minutes):
    """Group exact ordinary-day delivery windows and report excluded days."""
    times = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if (not times.is_monotonic_increasing or times.has_duplicates or x.ndim != 2
            or len(times) != len(x) or len(times) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Market inputs must be finite, aligned and chronologically unique")
    if (isinstance(start_hour, bool) or not isinstance(start_hour, int) or not 0 <= start_hour < 24
            or isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int)
            or interval_minutes < 1 or 60 % interval_minutes):
        raise ValueError("Invalid market window or native interval")
    local = times.tz_convert(timezone)
    days = np.asarray(local.strftime("%Y-%m-%d"))
    expected_count = (24 - start_hour) * 60 // interval_minutes
    rows, excluded = [], []
    for day in np.unique(days):
        selected = np.flatnonzero(days == day)
        midnight = pd.Timestamp(day, tz=timezone)
        next_midnight = midnight + pd.DateOffset(days=1)
        full_grid = pd.date_range(midnight, next_midnight, freq=f"{interval_minutes}min", inclusive="left")
        window = full_grid[full_grid.hour >= start_hour].tz_convert("UTC")
        # Reject DST-length and partial windows rather than inventing products.
        if len(full_grid) != 1440 // interval_minutes or len(window) != expected_count or not times[selected].equals(window):
            excluded.append({"date": day, "rows": len(selected), "expected_window_rows": len(window)})
            continue
        rows.append(selected)
    if not rows:
        raise ValueError("No complete ordinary market windows are available")
    index = np.stack(rows)
    return x[index], y[index], index, excluded


def run(settings, root):
    """Fit training-only spread PCA and ridge on each market's native window."""
    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Source search must be complete without failures")
    ranks, alphas = settings["target_ranks"], settings["ridge_alphas"]
    if (not ranks or not alphas or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in ranks)
            or any(not np.isfinite(x) or x <= 0 for x in alphas)):
        raise ValueError("Target ranks and ridge penalties must be positive")
    results, provenance, selected_outputs = [], {}, {}
    for group, start_hour in settings["group_start_hours"].items():
        split = manifest["config"]["product_groups"][group]
        feature = settings["feature_set"]
        if (split["native_minutes"] != settings["interval_minutes"]
                or manifest["config"]["residual_target_features"].get(feature) != "latest_cleared_auction"):
            raise ValueError("Source must declare native resolution and latest-cleared anchor")
        markets = split.get("market_price_features", [])
        lags = split.get("lag_hours", manifest["config"]["lag_hours"])
        if not markets or feature not in manifest["config"]["add_market_price_features_to"]:
            raise ValueError("Source feature arm must contain the earlier auction prices")
        # Direct auction prices precede their neighboring-price features.
        anchor_index = len(lags) + 7 + len(markets) - 1
        array_path = source / f"{group}__{feature}.npz"
        baseline_path = source / f"{group}__{feature}__zero_spread.csv"
        baseline = pd.read_csv(baseline_path, parse_dates=["product_start"])
        with np.load(array_path, allow_pickle=False) as data:
            train_x, train_y, _, train_excluded = market_days(data["train_starts"], data["train_x"], data["train_y"],
                settings["timezone"], start_hour, settings["interval_minutes"])
            valid_x, valid_y, valid_rows, valid_excluded = market_days(data["starts"], data["validation_x"], data["validation_y"],
                settings["timezone"], start_hour, settings["interval_minutes"])
            starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
            check_joint_cutoffs(data["decisions_ns"], valid_rows)
            if (not starts.equals(pd.DatetimeIndex(baseline.product_start))
                    or not np.allclose(data["validation_y"], baseline.actual, rtol=0, atol=1e-10)
                    or not np.allclose(data["validation_x"][:, anchor_index], baseline.point, rtol=0, atol=1e-10)):
                raise ValueError("Source baseline, targets or anchor do not match saved arrays")
            if pd.to_datetime(data["train_starts"], utc=True).max() >= starts.min():
                raise ValueError("Training products must precede evaluation")
        if max(ranks) > min(train_y.shape):
            raise ValueError("Target rank exceeds complete training curves")
        train_anchor = train_x[:, :, anchor_index]
        valid_anchor = valid_x[:, :, anchor_index]
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_x.reshape(len(train_x), -1))
        scaled_valid = scaler.transform(valid_x.reshape(len(valid_x), -1))
        denominator = float(np.abs(valid_y).sum())
        if not denominator:
            raise ValueError("WAPE is undefined on zero absolute target sum")
        baseline_error = np.abs(valid_y - valid_anchor)
        best = None
        for rank in ranks:
            # Both target PCA and regression use only the training windows.
            pca = PCA(n_components=rank, svd_solver="full")
            latent = pca.fit_transform(train_y - train_anchor)
            for alpha in alphas:
                model = Ridge(alpha=alpha, solver="cholesky")
                model.fit(scaled_train, latent)
                prediction = valid_anchor + pca.inverse_transform(model.predict(scaled_valid))
                error = np.abs(valid_y - prediction)
                row = {"group": group, "target_rank": rank, "ridge_alpha": alpha,
                    "training_days": len(train_y), "validation_days": len(valid_y), "products": int(valid_y.size),
                    "mae_eur_per_mwh": float(error.mean()), "wape_percent": float(100 * error.sum() / denominator),
                    "baseline_mae_eur_per_mwh": float(baseline_error.mean()),
                    "baseline_wape_percent": float(100 * baseline_error.sum() / denominator)}
                results.append(row)
                if best is None or row["mae_eur_per_mwh"] < best["row"]["mae_eur_per_mwh"]:
                    best = {"row": row, "starts": starts[valid_rows.ravel()], "actual": valid_y.ravel(),
                            "point": prediction.ravel(), "baseline": valid_anchor.ravel()}
        selected_outputs[group] = best
        provenance[group] = {"dataset_sha256": checksum(array_path), "baseline_sha256": checksum(baseline_path),
            "anchor_index": anchor_index, "excluded_training_days": train_excluded,
            "excluded_validation_days": valid_excluded}
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(output / "candidate_scores.csv", index=False)
    for group, best in selected_outputs.items():
        pd.DataFrame({key: best[key] for key in ("starts", "actual", "point", "baseline")}).to_csv(
            output / f"{group}_selected_predictions.csv", index=False)
    evidence = {"status": "complete", "settings": settings, "source_manifest_sha256": checksum(manifest_path),
        "implementation_sha256": checksum(Path(__file__)), "provenance": provenance, "results": results,
        "selected": {group: best["row"] for group, best in selected_outputs.items()}}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Resolve the explicit IDA curve-study settings and output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)["daily_ida_spreads"], args.project_root.resolve()))


if __name__ == "__main__":
    main()
