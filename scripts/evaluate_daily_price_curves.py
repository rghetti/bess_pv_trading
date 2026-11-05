"""Evaluate low-rank joint daily DA curves on a saved native-product split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bess_pv_trading.config import load_config


def checksum(path):
    """Record the exact source file used by a curve experiment."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_local_days(starts, features, targets, timezone, interval_minutes):
    """Group only complete ordinary local days into ordered input and target curves."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if (not timestamps.is_monotonic_increasing or timestamps.has_duplicates or features.ndim != 2
            or len(timestamps) != len(features) or len(targets) != len(timestamps)
            or not np.isfinite(features).all() or not np.isfinite(targets).all()):
        raise ValueError("Curve inputs must be finite, aligned and chronologically unique")
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or 1440 % interval_minutes:
        raise ValueError("Native interval must divide one ordinary day")
    local = timestamps.tz_convert(timezone)
    dates = np.asarray(local.strftime("%Y-%m-%d"))
    expected = 1440 // interval_minutes
    indices, excluded = [], []
    for day in np.unique(dates):
        selected = np.flatnonzero(dates == day)
        # Require the exact local-day grid; DST and partial days are explicit gaps.
        midnight = pd.Timestamp(day, tz=timezone)
        next_midnight = midnight + pd.DateOffset(days=1)
        grid = pd.date_range(midnight, next_midnight, freq=f"{interval_minutes}min", inclusive="left").tz_convert("UTC")
        if len(grid) != expected or not timestamps[selected].equals(grid):
            excluded.append({"date": day, "rows": int(len(selected)), "expected_local_rows": len(grid)})
            continue
        indices.append(selected)
    if not indices:
        raise ValueError("No complete ordinary local days are available")
    rows = np.stack(indices)
    return features[rows].reshape(len(rows), -1), targets[rows], rows, excluded


def check_joint_cutoffs(decisions, rows):
    """Reject a daily feature curve assembled from different decision times."""
    if np.any(np.ptp(np.asarray(decisions, dtype=np.int64)[rows], axis=1) != 0):
        raise ValueError("Daily curve products must share one forecast decision time")


def curve_target_and_anchor(features, targets, mode):
    """Build either absolute targets or residuals from the first price lag."""
    if mode not in {"absolute", "first_lag_residual"}:
        raise ValueError("Unknown curve target mode")
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets) or features.shape[1] % targets.shape[1]:
        raise ValueError("Daily features and targets must have compatible shapes")
    anchor = features.reshape(len(features), targets.shape[1], -1)[:, :, 0]
    return (targets, np.zeros_like(anchor)) if mode == "absolute" else (targets - anchor, anchor)


def run(settings, root):
    """Fit target-curve PCA and ridge only on the saved early training days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Curve study requires a complete source search")
    group = settings["group"]
    split = manifest["config"]["product_groups"][group]
    if split["native_minutes"] != settings["interval_minutes"]:
        raise ValueError("Curve grid must use the source native product resolution")
    target_mode = settings.get("target_mode", "absolute")
    lag_hours = split.get("lag_hours", manifest["config"]["lag_hours"])
    if target_mode == "first_lag_residual" and (not lag_hours or lag_hours[0] != 24):
        raise ValueError("Residual curve anchor must be the source's first 24-hour price lag")
    ranks = settings["target_ranks"]
    alphas = settings["ridge_alphas"]
    if (not ranks or not alphas or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in ranks)
            or any(not np.isfinite(x) or x <= 0 for x in alphas)):
        raise ValueError("Target ranks and ridge penalties must be positive")
    results, provenance, best_predictions = [], {}, None
    for feature in settings["feature_sets"]:
        # Use the same prebuilt information sets as the existing source search.
        array_path = source / f"{group}__{feature}.npz"
        baseline_path = source / f"{group}__{feature}__{settings['baseline_candidate']}.csv"
        baseline = pd.read_csv(baseline_path, parse_dates=["product_start"])
        with np.load(array_path, allow_pickle=False) as data:
            train_x, train_y, _, train_excluded = full_local_days(
                data["train_starts"], data["train_x"], data["train_y"], settings["timezone"], settings["interval_minutes"])
            valid_x, valid_y, valid_rows, valid_excluded = full_local_days(
                data["starts"], data["validation_x"], data["validation_y"], settings["timezone"], settings["interval_minutes"])
            # A joint input is valid only when every delivery product shares one decision cutoff.
            check_joint_cutoffs(data["decisions_ns"], valid_rows)
            starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
            if (not starts.equals(pd.DatetimeIndex(baseline.product_start))
                    or not np.allclose(data["validation_y"], baseline.actual, rtol=0, atol=1e-10)):
                raise ValueError("Baseline must match saved evaluation products and targets")
            if pd.to_datetime(data["train_starts"], utc=True).max() >= starts.min():
                raise ValueError("Training products must strictly precede evaluation")
        if max(ranks) > min(train_y.shape):
            raise ValueError("Requested target rank exceeds available complete training curves")
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_x)
        scaled_valid = scaler.transform(valid_x)
        train_target, _ = curve_target_and_anchor(train_x, train_y, target_mode)
        _, validation_anchor = curve_target_and_anchor(valid_x, valid_y, target_mode)
        baseline_values = baseline.point.to_numpy()[valid_rows]
        baseline_error = np.abs(valid_y - baseline_values)
        denominator = float(np.abs(valid_y).sum())
        if not denominator:
            raise ValueError("WAPE is undefined on zero absolute target sum")
        for rank in ranks:
            # PCA and regression are both fitted exclusively on training days.
            pca = PCA(n_components=rank, svd_solver="full")
            latent = pca.fit_transform(train_target)
            for alpha in alphas:
                estimator = Ridge(alpha=alpha, solver="cholesky")
                estimator.fit(scaled_train, latent)
                predicted = pca.inverse_transform(estimator.predict(scaled_valid)) + validation_anchor
                error = np.abs(valid_y - predicted)
                row = {"feature_set": feature, "target_rank": rank, "ridge_alpha": alpha,
                    "train_days": len(train_y), "validation_days": len(valid_y), "products": int(valid_y.size),
                    "mae_eur_per_mwh": float(error.mean()), "wape_percent": float(100 * error.sum() / denominator),
                    "baseline_mae_eur_per_mwh": float(baseline_error.mean()),
                    "baseline_wape_percent": float(100 * baseline_error.sum() / denominator)}
                results.append(row)
                if best_predictions is None or row["mae_eur_per_mwh"] < best_predictions["row"]["mae_eur_per_mwh"]:
                    best_predictions = {"row": row, "starts": starts[valid_rows.ravel()], "actual": valid_y.ravel(),
                                        "point": predicted.ravel(), "baseline": baseline_values.ravel()}
        provenance[feature] = {"dataset_sha256": checksum(array_path), "baseline_sha256": checksum(baseline_path),
                               "excluded_training_days": train_excluded, "excluded_validation_days": valid_excluded}
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(output / "candidate_scores.csv", index=False)
    prediction = pd.DataFrame({name: best_predictions[name] for name in ("starts", "actual", "point", "baseline")})
    prediction.to_csv(output / "selected_predictions.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "source_manifest_sha256": checksum(manifest_path), "implementation_sha256": checksum(Path(__file__)),
        "provenance": provenance, "results": results,
        "selected": best_predictions["row"]}, indent=2, allow_nan=False) + "\n")
    # Display the first selected day as an example, not as a separate score.
    count = 1440 // settings["interval_minutes"]
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.plot(prediction.starts.iloc[:count], prediction.actual.iloc[:count], label="Actual")
    axis.plot(prediction.starts.iloc[:count], prediction.point.iloc[:count], label="Selected curve")
    axis.plot(prediction.starts.iloc[:count], prediction.baseline.iloc[:count], label="Interaction reference", alpha=0.8)
    axis.set(ylabel="Price (EUR/MWh)", title="First complete selection day (UTC timestamps)")
    axis.legend()
    figure.autofmt_xdate()
    figure.savefig(output / "first_selection_day.png", dpi=150)
    plt.close(figure)
    return output


def main():
    """Read the explicit curve-study configuration and project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)["daily_price_curves"], args.project_root.resolve()))


if __name__ == "__main__":
    main()
