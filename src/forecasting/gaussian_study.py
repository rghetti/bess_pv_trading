"""Independent static/adaptive Gaussian studies from frozen point forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .gaussian_adaptive import evaluate_adaptive_gaussian, select_gaussian_policy
from .gaussian_prices import calibrate_gaussian_prices, evaluate_gaussian_prices
from .price_experiment import _write_frame
from .price_pipeline import _plot_paths, _sample_day, _write_json


def _input_file(source: Mapping[str, Any], directory: Path, field: str, default: str) -> Path:
    """Resolve declared prediction files relative to their owning manifest."""
    value = Path(source.get(field, default))
    return value if value.is_absolute() else directory / value


def run_gaussian_study(config: Mapping[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Select calibration-only policies and compare later static/online forecasts.

    Inputs come exclusively from the explicit existing point-study manifest;
    this command performs no API calls and never changes the original study.
    The adaptive policy's hyperparameters are frozen after inner calibration.
    During later online evaluation, earlier complete mature test errors may
    update its covariance and bias. The static comparator never uses test
    labels in calibration. Source byte snapshots and all resulting uncertainty
    parameters are saved in a new independent run directory.
    """
    settings = dict(config.get("price_scenarios", {}))
    if not settings.get("study_manifest"):
        raise ValueError("price_scenarios.study_manifest must identify an existing completed point-model study")
    source_path = Path(settings["study_manifest"]).resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("status") not in ("completed", "complete"):
        raise ValueError("Gaussian studies require a completed source point-model study")
    calibration_path = _input_file(source, source_path.parent, "calibration_predictions", "calibration_predictions.parquet")
    test_path = _input_file(source, source_path.parent, "test_predictions", "test_predictions.parquet")
    calibration_bytes, test_bytes = calibration_path.read_bytes(), test_path.read_bytes()
    calibration = pd.read_parquet(BytesIO(calibration_bytes))
    test = pd.read_parquet(BytesIO(test_bytes))
    if calibration.empty or test.empty:
        raise ValueError("Separate nonempty frozen calibration and test predictions are required")
    if "native_interval_minutes" not in calibration or "native_interval_minutes" not in test:
        raise ValueError("Frozen predictions must declare their native product duration")
    count, seed = settings.get("sample_count", 100), settings.get("seed", 41)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("price_scenarios.sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("price_scenarios.seed must be a nonnegative integer")
    timezone_name = str(settings.get("timezone", source.get("settings", {}).get("timezone", "Europe/Rome")))
    adaptive_settings = {"timezone": timezone_name, **dict(settings.get("adaptive", {}))}
    if adaptive_settings["timezone"] != timezone_name:
        raise ValueError("Static and adaptive study timezones must agree")
    prepared, skipped = [], []
    for minutes, calibration_part in calibration.groupby("native_interval_minutes", sort=True):
        test_part = test.loc[test.native_interval_minutes.eq(minutes)].copy()
        if test_part.empty:
            skipped.append({"native_interval_minutes": int(minutes), "status": "no_test_rows"})
            continue
        if progress:
            progress(f"Selecting {int(minutes)}-minute Gaussian policy using calibration predictions only")
        policy, candidates = select_gaussian_policy(calibration_part, adaptive_settings)
        if progress:
            progress(f"Evaluating frozen {int(minutes)}-minute policy online; only earlier mature errors may update covariance")
        adaptive = evaluate_adaptive_gaussian(calibration_part, test_part, policy)
        static_cutoff = pd.to_datetime(calibration_part.product_end, utc=True).max()
        static_artifact = calibrate_gaussian_prices(calibration_part, calibration_cutoff=static_cutoff,
            timezone=timezone_name, covariance_shrinkage=settings.get("covariance_shrinkage", .5),
            minimum_complete_days=settings.get("minimum_complete_days", 10),
            minimum_variance_eur2_mwh2=settings.get("minimum_variance_eur2_mwh2", 1e-6))
        static = evaluate_gaussian_prices(static_artifact, test_part)
        sample_day = str(adaptive["scored_predictions"].delivery_day.min())
        sample = _sample_day(adaptive["daily_calibrations"][sample_day], test_part,
                             adaptive["scored_predictions"], count, seed)
        prepared.append({"minutes": int(minutes), "policy": policy, "candidates": candidates,
            "adaptive": adaptive, "static": static, "static_artifact": static_artifact, "sample": sample})
    if not prepared:
        raise ValueError("No native product regime has both calibration and test predictions")
    # Numerical validation and path sampling happen before publication. A
    # pending sibling directory keeps incomplete I/O/plotting runs unpublished.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parent = Path(settings.get("output_directory", "outputs/price_uncertainty"))
    directory = parent / run_id
    pending = parent / f".{run_id}.pending"
    pending.mkdir(parents=True, exist_ok=False)
    (pending / "source_manifest.json").write_bytes(source_bytes)
    (pending / "calibration_predictions.parquet").write_bytes(calibration_bytes)
    (pending / "test_predictions.parquet").write_bytes(test_bytes)
    source_hashes = {"source_manifest.json": hashlib.sha256(source_bytes).hexdigest(),
        "calibration_predictions.parquet": hashlib.sha256(calibration_bytes).hexdigest(),
        "test_predictions.parquet": hashlib.sha256(test_bytes).hexdigest()}
    summaries = []
    for item in prepared:
        minutes, adaptive, static, sample = item["minutes"], item["adaptive"], item["static"], item["sample"]
        relative = Path(f"{minutes}min")
        destination = pending / relative
        destination.mkdir()
        _write_json(destination / "policy.json", item["policy"])
        _write_frame(item["candidates"], destination, "inner_candidate_scores")
        _write_frame(adaptive["scored_predictions"], destination, "test_scores")
        _write_json(destination / "metrics.json", adaptive["metrics"])
        _write_json(destination / "daily_audit.json", adaptive["daily_audit"])
        _write_json(destination / "excluded_days.json", adaptive["excluded_days"])
        for day_name, artifact in adaptive["daily_calibrations"].items():
            artifact.save(destination / "daily_calibrations" / f"{day_name}.json")
        item["static_artifact"].save(destination / "static" / "gaussian_calibration.json")
        _write_frame(static["scored_predictions"], destination / "static", "test_scores")
        _write_json(destination / "static" / "metrics.json", static["metrics"])
        _write_json(destination / "static" / "excluded_days.json", static["excluded_days"])
        day_name = sample["day"]
        np.savez_compressed(destination / f"paths_{day_name}.npz", prices_eur_mwh=sample["paths"],
            innovations_eur_mwh=sample["innovations"], point_forecast_eur_mwh=sample["point"],
            bias_eur_mwh=sample["bias"], gaussian_mean_eur_mwh=sample["mean"],
            covariance_eur2_mwh2=sample["covariance"], probabilities=np.full(count, 1 / count))
        _write_frame(sample["products"], destination, f"products_{day_name}")
        _plot_paths(sample["products"], sample["paths"], sample["mean"], destination / f"paths_{day_name}.png")
        summaries.append({"native_interval_minutes": minutes, "status": "evaluated", "selected_policy": item["policy"],
            "static_metrics": static["metrics"], "adaptive_metrics": adaptive["metrics"],
            "static_excluded_days": static["excluded_days"], "adaptive_excluded_days": adaptive["excluded_days"],
            "sample_day": day_name, "sample_count": count, "seed": seed,
            "policy": str((directory / relative / "policy.json").resolve()),
            "inner_candidate_scores": str((directory / relative / "inner_candidate_scores.parquet").resolve()),
            "daily_calibrations": str((directory / relative / "daily_calibrations").resolve()),
            "sample_paths": str((directory / relative / f"paths_{day_name}.npz").resolve())})
    summaries.extend(skipped)
    result = {"schema_version": "gaussian-price-study-v1", "status": "completed", "run_id": run_id,
        "output_dir": str(directory.resolve()), "run_directory": str(directory.resolve()),
        "manifest": str((directory / "manifest.json").resolve()), "source_study_manifest": str(source_path),
        "source_manifest_sha256": source_hashes["source_manifest.json"], "frozen_input_sha256": source_hashes,
        "timezone": timezone_name, "operational_eligible": False,
        "innovation_distribution": "Conditionally zero-mean multivariate Gaussian; separate rolling bias correction",
        "variance_source": "Chronological out-of-sample prediction errors; never raw price variance",
        "selection_policy": "Only inner calibration days select window/shrinkage; no final-test outcomes select hyperparameters",
        "static_evaluation": "Frozen calibration bias/covariance; no test labels used for updates",
        "adaptive_evaluation": "Frozen policy with online covariance/bias updates using only complete earlier mature test errors",
        "prior_evaluation_overlap": source.get("prior_evaluation_overlap", "Historical research comparison; prior inspection of these outcomes is not ruled out"),
        "limitation": "Latest-retrieved API revisions have unverified historical publication/vintage availability; research scenarios are not operational forecasts",
        "settings": settings, "regimes": summaries}
    _write_json(pending / "manifest.json", result)
    pending.rename(directory)
    if progress:
        progress(f"Published Gaussian uncertainty study to {directory}")
    return result
