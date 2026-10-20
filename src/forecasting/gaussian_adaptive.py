"""Chronologically selected rolling Gaussian price-error calibration.

The mean model remains fixed by the separate price study. Gaussian innovations
have zero mean, with a separate rolling error-bias correction. Hyperparameters
are selected using calibration predictions only. Online test evaluation may
update covariance from earlier test errors only after their full native
delivery day has matured before the new forecast issuance cutoff.
"""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gaussian_prices import (GaussianPriceCalibration, _cutoff, _validated_rows,
                             calibrate_gaussian_prices, evaluate_gaussian_prices)


POLICY_SCHEMA = "adaptive-gaussian-price-policy-v1"


def _integer(value: Any, label: str, minimum: int) -> int:
    """Validate count settings without treating booleans or floats as integers."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return int(value)


def _policy(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one candidate or a previously selected frozen policy."""
    window = _integer(settings.get("window_days", 42), "window_days", 2)
    minimum = _integer(settings.get("minimum_complete_days", 10), "minimum_complete_days", 2)
    if window < minimum:
        raise ValueError("window_days must cover at least minimum_complete_days")
    shrinkage = float(settings.get("covariance_shrinkage", .3))
    floor = float(settings.get("minimum_variance_eur2_mwh2", 1e-6))
    if not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("Adaptive covariance_shrinkage must lie in (0,1]")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("minimum_variance_eur2_mwh2 must be positive")
    if settings.get("native_interval_minutes") is not None and settings["native_interval_minutes"] not in (15, 30, 60):
        raise ValueError("Selected adaptive native interval must be 15, 30, or 60 minutes")
    return {**dict(settings), "schema": POLICY_SCHEMA, "window_days": window,
            "minimum_complete_days": minimum, "covariance_shrinkage": shrinkage,
            "minimum_variance_eur2_mwh2": floor, "timezone": str(settings.get("timezone", "Europe/Rome"))}


def _one_regime(rows: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Validate one native price market independently of other product regimes."""
    frame = _validated_rows(rows, timezone_name)
    if frame.native_interval_minutes.nunique() != 1:
        raise ValueError("Adaptive Gaussian policy selection/evaluation requires one native duration at a time")
    return frame


def calibrate_for_cutoff(error_rows: pd.DataFrame, cutoff: Any,
                         policy: Mapping[str, Any]) -> GaussianPriceCalibration:
    """Fit a reusable Gaussian artifact from earlier complete mature error days.

    The window is measured in calendar days, never a count of available rows.
    A partly matured day is excluded in full: at D−1 noon, errors from delivery
    D−1 are not used, even though some of its products have already ended.
    This conservative maturity boundary does not assert historical publication
    availability for the underlying latest-retrieved API revisions.
    """
    selected = _policy(policy)
    frame = _one_regime(error_rows, selected["timezone"])
    return _calibrate_validated(frame, cutoff, selected)


def _calibrate_validated(frame: pd.DataFrame, cutoff: Any,
                          selected: Mapping[str, Any]) -> GaussianPriceCalibration:
    """Reuse validated histories during bounded rolling policy comparisons."""
    boundary = _cutoff(cutoff, "forecast cutoff")
    if selected.get("selection_max_product_end") is not None:
        if _cutoff(selected["selection_max_product_end"], "policy selection cutoff") > boundary:
            raise ValueError("Adaptive policy was selected using information after the forecast cutoff")
    if (selected.get("native_interval_minutes") is not None
            and selected["native_interval_minutes"] != frame.native_interval_minutes.iloc[0]):
        raise ValueError("Selected adaptive policy and error history native durations disagree")
    local_boundary = boundary.tz_convert(selected["timezone"])
    last_possible_day = local_boundary.date() - timedelta(days=1)
    first_possible_day = local_boundary.date() - timedelta(days=selected["window_days"])
    keep = (frame.product_end.le(boundary)
            & frame.delivery_day.ge(first_possible_day.isoformat())
            & frame.delivery_day.le(last_possible_day.isoformat()))
    matured = frame.loc[keep].copy()
    if matured.empty:
        raise ValueError("No complete mature price-error history lies inside the requested calendar window")
    return calibrate_gaussian_prices(matured, calibration_cutoff=boundary,
        timezone=selected["timezone"], covariance_shrinkage=selected["covariance_shrinkage"],
        minimum_complete_days=selected["minimum_complete_days"],
        minimum_variance_eur2_mwh2=selected["minimum_variance_eur2_mwh2"])


def _aggregate(scored_days: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine already-scored native days without changing their calibrations."""
    if not scored_days:
        raise ValueError("No complete forecast days could be scored by this Gaussian policy")
    scored = pd.concat([day["scored_predictions"] for day in scored_days], ignore_index=True)
    weights = np.array([day["metrics"]["scored_rows"] for day in scored_days], float)
    actual_error = scored.actual - scored.gaussian_mean
    point_error = scored.actual - scored.prediction
    metrics = {"scored_rows": len(scored), "scored_days": int(scored.delivery_day.nunique()),
        "mae_eur_mwh": float(actual_error.abs().mean()),
        "rmse_eur_mwh": float(np.sqrt(np.square(actual_error).mean())),
        "uncorrected_point_mae_eur_mwh": float(point_error.abs().mean()),
        "uncorrected_point_rmse_eur_mwh": float(np.sqrt(np.square(point_error).mean())),
        "mean_error_eur_mwh": float(actual_error.mean()),
        "mean_crps_eur_mwh": float(scored.crps_eur_mwh.mean()),
        "mean_marginal_nll": float(scored.marginal_nll.mean()),
        "mean_joint_nll_per_product": float(np.average(
            [day["metrics"]["mean_joint_nll_per_product"] for day in scored_days], weights=weights))}
    levels = scored_days[0]["metrics"]["coverage"]
    metrics["coverage"] = {level: float(scored[f"covered_{level}"].mean()) for level in levels}
    metrics["mean_interval_width_eur_mwh"] = {
        level: float(scored[f"interval_width_{level}"].mean()) for level in levels}
    ramps = [day["metrics"]["consecutive_product_ramps"] for day in scored_days]
    ramp_weights = np.array([value["rows"] for value in ramps], float)
    metrics["consecutive_product_ramps"] = {"rows": int(ramp_weights.sum()),
        "mae_eur_mwh": float(np.average([value["mae_eur_mwh"] for value in ramps], weights=ramp_weights)),
        "rmse_eur_mwh": float(np.sqrt(np.average(
            [value["rmse_eur_mwh"] ** 2 for value in ramps], weights=ramp_weights))),
        "coverage_95": float(np.average([value["coverage_95"] for value in ramps], weights=ramp_weights)),
        "mean_sigma_eur_mwh": float(np.average([value["mean_sigma_eur_mwh"] for value in ramps], weights=ramp_weights))}
    return scored, metrics


def _score_one_day(history: pd.DataFrame, day: pd.DataFrame,
                   policy: Mapping[str, Any]) -> tuple[GaussianPriceCalibration, dict[str, Any]]:
    """Use all point predictions only when their complete curve was available."""
    cutoff = day.forecast_cutoff.max()
    if (day.product_start < cutoff).any():
        raise ValueError("A complete native day cannot use point forecasts issued after its first product starts")
    artifact = _calibrate_validated(history, cutoff, policy)
    return artifact, evaluate_gaussian_prices(artifact, day)


def select_gaussian_policy(calibration_rows: pd.DataFrame,
                           settings: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose a bounded window/shrinkage grid using calibration-only chronology.

    Defaults compare 14/28/42-day rolling windows and .1/.3/.5 shrinkages on
    the final 14 calibration days. Each scored inner day receives an artifact
    fitted exclusively from complete residual days matured by its own forecast
    cutoff. Every candidate is compared on the same successfully scored days.
    No final-test rows are accepted by this function or used for selection.
    """
    timezone_name = str(settings.get("timezone", "Europe/Rome"))
    calibration = _one_regime(calibration_rows, timezone_name)
    windows = settings.get("window_days_candidates", [14, 28, 42])
    shrinkages = settings.get("covariance_shrinkage_candidates", [.1, .3, .5])
    if not isinstance(windows, (list, tuple)) or not windows or not isinstance(shrinkages, (list, tuple)) or not shrinkages:
        raise ValueError("Adaptive policy candidates must be nonempty window and shrinkage lists")
    windows = [_integer(value, "window_days candidate", 2) for value in windows]
    shrinkages = [float(value) for value in shrinkages]
    if len(set(windows)) != len(windows) or len(set(shrinkages)) != len(shrinkages):
        raise ValueError("Adaptive policy candidate lists must not contain duplicates")
    inner_count = _integer(settings.get("inner_validation_days", 14), "inner_validation_days", 1)
    minimum_inner = _integer(settings.get("minimum_inner_scored_days", min(7, inner_count)), "minimum_inner_scored_days", 1)
    metric_aliases = {"joint_nll": "mean_joint_nll_per_product", "crps": "mean_crps_eur_mwh",
                      "mean_joint_nll_per_product": "mean_joint_nll_per_product", "mean_crps_eur_mwh": "mean_crps_eur_mwh"}
    requested_metric = str(settings.get("selection_metric", "joint_nll"))
    if requested_metric not in metric_aliases:
        raise ValueError("Adaptive selection_metric must be joint_nll or crps")
    metric = metric_aliases[requested_metric]
    last_day = pd.Timestamp(calibration.delivery_day.max()).date()
    first_inner = (last_day - timedelta(days=inner_count - 1)).isoformat()
    inner_days = {str(name): day for name, day in calibration.loc[calibration.delivery_day.ge(first_inner)].groupby("delivery_day", sort=True)}
    candidates = []
    for window in windows:
        for shrinkage in shrinkages:
            policy = _policy({"window_days": window, "covariance_shrinkage": shrinkage,
                "minimum_complete_days": settings.get("minimum_complete_days", 10),
                "minimum_variance_eur2_mwh2": settings.get("minimum_variance_eur2_mwh2", 1e-6),
                "timezone": timezone_name})
            scored, excluded = {}, {}
            for day_name, day in inner_days.items():
                try:
                    _, result = _score_one_day(calibration, day, policy)
                    scored[day_name] = result
                except ValueError as exc:
                    excluded[day_name] = str(exc)
            candidates.append((policy, scored, excluded))
    common_days = sorted(set.intersection(*(set(scores) for _, scores, _ in candidates)))
    if len(common_days) < minimum_inner:
        raise ValueError(f"Adaptive policy selection has {len(common_days)} common mature inner days; requires {minimum_inner}")
    summaries = []
    for policy, scores, excluded in candidates:
        _, metrics = _aggregate([scores[day] for day in common_days])
        summaries.append({"window_days": policy["window_days"], "covariance_shrinkage": policy["covariance_shrinkage"],
            "scored_days": len(common_days), "scored_rows": metrics["scored_rows"], "selection_metric": metric,
            "selection_score": metrics[metric], "mean_joint_nll_per_product": metrics["mean_joint_nll_per_product"],
            "mean_crps_eur_mwh": metrics["mean_crps_eur_mwh"], "coverage_50": metrics["coverage"]["50"],
            "coverage_80": metrics["coverage"]["80"], "coverage_95": metrics["coverage"]["95"],
            "ramp_coverage_95": metrics["consecutive_product_ramps"]["coverage_95"],
            "excluded_days": json.dumps(excluded, sort_keys=True)})
    # Stable configured-order ties avoid selecting a specification using any
    # later outcomes or incidental artifact ordering.
    best_index = min(range(len(summaries)), key=lambda index: summaries[index]["selection_score"])
    selected = {**candidates[best_index][0], "selection_metric": metric,
        "selection_score": summaries[best_index]["selection_score"],
        "selection_inner_days": common_days, "selection_source": "calibration_predictions_only",
        "selection_max_product_end": calibration.product_end.max().isoformat(),
        "native_interval_minutes": int(calibration.native_interval_minutes.iloc[0]),
        "fixed_after_selection": True,
        "update_policy": "Rolling covariance and bias may use only complete days matured before each forecast issuance"}
    scores = pd.DataFrame(summaries)
    scores["selected"] = np.arange(len(scores)) == best_index
    return selected, scores


def evaluate_adaptive_gaussian(calibration_rows: pd.DataFrame, test_rows: pd.DataFrame,
                               selected_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a frozen online policy with only previously matured test errors.

    This is online evaluation: earlier mature test observations can update
    bias/covariance under the already-selected policy. It is distinct from the
    static held-out evaluator whose error parameters never use test labels.
    Future test labels cannot influence any earlier forecast distribution.
    """
    policy = _policy(selected_policy)
    calibration = _one_regime(calibration_rows, policy["timezone"])
    test = _one_regime(test_rows, policy["timezone"])
    if calibration.native_interval_minutes.iloc[0] != test.native_interval_minutes.iloc[0]:
        raise ValueError("Calibration and test native product durations must agree")
    selection_end = _cutoff(policy.get("selection_max_product_end", calibration.product_end.max()), "policy selection cutoff")
    if calibration.product_end.max() > selection_end:
        raise ValueError("Calibration rows extend beyond the frozen policy selection history")
    if (test.product_start < selection_end).any() or (test.forecast_cutoff < selection_end).any():
        raise ValueError("Online test forecasts must follow the frozen calibration selection cutoff")
    if set(calibration.product_start).intersection(test.product_start):
        raise ValueError("Calibration and test products must be separate")
    scores, artifacts, excluded, audits = [], {}, {}, []
    history = pd.concat([calibration, test], ignore_index=True)
    for day_name, day in test.groupby("delivery_day", sort=True):
        try:
            artifact, result = _score_one_day(history, day, policy)
        except ValueError as exc:
            excluded[str(day_name)] = str(exc)
            continue
        scores.append(result)
        artifacts[str(day_name)] = artifact
        cutoff = day.forecast_cutoff.max()
        mature_test = test.loc[(test.product_end <= cutoff)
                              & (test.delivery_day < cutoff.tz_convert(policy["timezone"]).date().isoformat())]
        audits.append({"delivery_day": str(day_name), "forecast_cutoff": cutoff.isoformat(),
            "calibration_cutoff": artifact.calibration_cutoff,
            "latest_error_delivery_day": max(grid.last_delivery_day for grid in artifact.grids.values()),
            "mature_prior_test_days_available": int(mature_test.delivery_day.nunique())})
    scored, metrics = _aggregate(scores)
    metrics["unscored_rows"] = len(test) - len(scored)
    return {"scored_predictions": scored, "metrics": metrics, "excluded_days": excluded,
        "daily_calibrations": artifacts, "daily_audit": audits, "policy": policy,
        "evaluation_policy": "Frozen calibration-selected policy; online bias/covariance updates use earlier complete mature test errors only",
        "test_labels_used_for_selection": False, "earlier_mature_test_errors_used_for_updates": True,
        "operational_eligible": False,
        "availability_policy": "retrospective_latest_revision_unverified_publication"}
