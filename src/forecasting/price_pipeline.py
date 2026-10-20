"""Connect chronological linear-price studies to calibrated Gaussian paths.

Historical API revisions remain retrospective research inputs. This runner
exports reloadable uncertainty parameters and illustrative held-out paths;
operational use requires separately verified publication vintages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .gaussian_prices import calibrate_gaussian_prices, evaluate_gaussian_prices
from .price_experiment import _write_frame


def _write_json(path: Path, value: Any) -> None:
    """Save finite, reviewable evidence without altering source snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plot_paths(products: pd.DataFrame, paths: np.ndarray, mean: np.ndarray, path: Path) -> None:
    """Show a sample of coherent daily paths with the untouched realized curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(11, 4))
    starts = pd.DatetimeIndex(products.product_start)
    axes.plot(starts, paths[:min(20, len(paths))].T, color="steelblue", alpha=.14, linewidth=.8)
    axes.plot(starts, mean, label="Linear forecast + calibration bias", color="navy")
    axes.plot(starts, products.actual, label="Held-out actual price", color="black", linewidth=1.3)
    axes.set(xlabel="Delivery time (UTC)", ylabel="EUR/MWh",
             title=f"Retrospective Gaussian paths: {products.delivery_day.iloc[0]}")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _sample_day(artifact: Any, test: pd.DataFrame, scored: pd.DataFrame,
                count: int, seed: int) -> dict[str, Any]:
    """Validate and sample one complete held-out day before writing artifacts."""
    day = str(scored.delivery_day.min())
    products = test.loc[test.delivery_day.astype(str) == day].sort_values("product_start").copy()
    bias, covariance = artifact.parameters_for(products)
    point = products.prediction.to_numpy(float)
    innovations = artifact.sample_innovations(products, count, seed=seed)
    paths = artifact.sample_future(point, products, count,
        decision_cutoff=pd.to_datetime(products.forecast_cutoff, utc=True).max(), seed=seed,
        require_operational=False)
    return {"day": day, "products": products, "point": point, "bias": bias,
            "covariance": covariance, "innovations": innovations, "mean": point + bias, "paths": paths}


def attach_gaussian_study(study: Mapping[str, Any], config: Mapping[str, Any], *,
                          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze calibration errors, then score later test days without refitting.

    The study files contain one selected model's chronological predictions.
    Selection uses an earlier validation period; this function never selects
    models or changes covariance using the final test outcomes.
    """
    settings = dict(config.get("price_scenarios", {}))
    directory = Path(study["output_dir"])
    calibration_rows = pd.read_parquet(directory / "calibration_predictions.parquet")
    test_rows = pd.read_parquet(directory / "test_predictions.parquet")
    if calibration_rows.empty or test_rows.empty:
        raise ValueError("Gaussian calibration requires nonempty separate calibration and test predictions")
    minimum_days = settings.get("minimum_complete_days", 10)
    count = settings.get("sample_count", 100)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("price_scenarios.sample_count must be a positive integer")
    seed = settings.get("seed", 41)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("price_scenarios.seed must be a nonnegative integer")
    timezone = dict(config.get("price_experiment", {})) | dict(config.get("linear_price_study", {}))
    timezone_name = timezone.get("timezone", "Europe/Rome")
    output = directory / "uncertainty"
    results: list[dict[str, Any]] = []
    prepared = []
    for minutes, calibration_part in calibration_rows.groupby("native_interval_minutes", sort=True):
        test_part = test_rows.loc[test_rows.native_interval_minutes == minutes].copy()
        if test_part.empty:
            results.append({"native_interval_minutes": int(minutes), "status": "no_test_rows"})
            continue
        if progress:
            progress(f"Calibrating {int(minutes)}-minute Gaussian errors on {calibration_part.delivery_day.nunique()} days")
        # Delivery maturity is a conservative research boundary, never an
        # assertion about the original market publication time.
        cutoff = pd.to_datetime(calibration_part.product_end, utc=True).max()
        artifact = calibrate_gaussian_prices(calibration_part, calibration_cutoff=cutoff,
            timezone=timezone_name, covariance_shrinkage=settings.get("covariance_shrinkage", .5),
            minimum_complete_days=minimum_days,
            minimum_variance_eur2_mwh2=settings.get("minimum_variance_eur2_mwh2", 1e-6))
        scored = evaluate_gaussian_prices(artifact, test_part)
        sample = _sample_day(artifact, test_part, scored["scored_predictions"], count, seed)
        prepared.append((minutes, calibration_part, artifact, scored, sample))
    if not prepared:
        raise ValueError("No native regime has both calibration and test predictions")
    # Validate all regimes before creating an output directory, so malformed
    # chronology or parameters do not leave a misleading partial calibration.
    output.mkdir(parents=True, exist_ok=False)
    for minutes, calibration_part, artifact, scored, sample in prepared:
        regime_dir = output / f"{int(minutes)}min"
        regime_dir.mkdir()
        artifact.save(regime_dir / "gaussian_calibration.json")
        _write_frame(scored["scored_predictions"], regime_dir, "test_scores")
        _write_json(regime_dir / "metrics.json", {key: value for key, value in scored.items()
                                                if key != "scored_predictions"})
        day = sample["day"]
        np.savez_compressed(regime_dir / f"paths_{day}.npz", prices_eur_mwh=sample["paths"],
            innovations_eur_mwh=sample["innovations"], point_forecast_eur_mwh=sample["point"],
            bias_eur_mwh=sample["bias"], gaussian_mean_eur_mwh=sample["mean"],
            covariance_eur2_mwh2=sample["covariance"], probabilities=np.full(count, 1 / count))
        _write_frame(sample["products"], regime_dir, f"products_{day}")
        _plot_paths(sample["products"], sample["paths"], sample["mean"], regime_dir / f"paths_{day}.png")
        item = {"native_interval_minutes": int(minutes), "status": "evaluated",
                "calibration_cutoff": artifact.calibration_cutoff,
                "calibration_rows": len(calibration_part), "calibration_grids": {
                    key: {"days": grid.calibration_days, "products_per_day": len(grid.slot_labels),
                          "mean_error_variance_eur2_mwh2": float(np.diag(grid.covariance_eur2_mwh2).mean())}
                    for key, grid in artifact.grids.items()},
                "calibration_excluded_days": dict(artifact.excluded_days),
                "test_excluded_days": scored["excluded_days"], "test_metrics": scored["metrics"],
                "sample_day": day, "sample_count": count, "seed": seed,
                "artifact": str(regime_dir / "gaussian_calibration.json"),
                "sample_paths": str(regime_dir / f"paths_{day}.npz")}
        results.append(item)
    study_path = Path(study.get("manifest", directory / "manifest.json"))
    result = {"status": "complete", "study_manifest": str(study_path),
              "study_manifest_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
              "output_dir": str(output), "operational_eligible": False,
              "innovation_distribution": "multivariate Gaussian with zero mean and shrunk temporal error covariance",
              "variance_source": "chronological out-of-sample calibration errors, not raw price variance",
              "selection_policy": "earlier validation selects model; separate calibration freezes bias/covariance; later test excluded from both fits",
              "sample_scope": "illustrative held-out forecasts; each path is a complete native delivery-day price curve",
              "milp_adapter": "bess_pv_trading.forecasting.gaussian_prices.gaussian_price_tree",
              "regimes": results}
    _write_json(output / "manifest.json", result)
    return result


def run_linear_price_pipeline(config: dict[str, Any], *,
                              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run the offline mean-model study and its separate uncertainty evaluation."""
    from .linear_price_study import run_linear_price_study

    study = run_linear_price_study(config, progress=progress)
    if study.get("status") == "insufficient_data":
        return study
    if config.get("price_scenarios", {}).get("enabled", True):
        uncertainty = attach_gaussian_study(study, config, progress=progress)
        return {"status": "complete", "output_dir": study["output_dir"],
                "study_manifest": study.get("manifest", str(Path(study["output_dir"]) / "manifest.json")),
                "uncertainty_manifest": str(Path(uncertainty["output_dir"]) / "manifest.json"),
                "operational_eligible": False, "regimes": uncertainty["regimes"]}
    return study
