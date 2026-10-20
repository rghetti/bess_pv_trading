"""Reusable standardized linear estimators without additional dependencies.

Every fitted estimator returns the existing scaler/coefficient artifact used by
price forecasting. Chronological row selection and publication eligibility stay
with the caller; only its supplied training slice enters scaling and fitting.
Method settings must be saved alongside the compatible coefficient artifact.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .prices import RidgePriceModel


METHODS = {"ols", "ridge", "weighted_ridge", "huber", "lasso", "elastic_net"}


class LinearEstimatorConvergenceError(RuntimeError):
    """The configured iterative estimator did not reach its stated tolerance."""


def validate_estimator(settings: Mapping[str, Any], *, method: str | None = None) -> dict[str, Any]:
    """Normalize explicit algorithm settings and reject ambiguous penalties.

    Ridge methods minimize weighted/unweighted squared-error sum plus
    ``alpha * ||beta||²``; their intercept is unpenalized. Lasso/elastic-net
    minimize mean squared error divided by two plus ``alpha*l1_ratio*||beta||1``
    and ``alpha*(1-l1_ratio)*||beta||²/2``. Huber uses summed Huber loss with an
    initial training-residual MAD scale and an unpenalized intercept.
    """
    values = dict(settings)
    unknown = set(values) - {"name", "method", "alpha", "half_life_days", "epsilon", "max_iter", "tolerance", "l1_ratio"}
    if unknown:
        raise ValueError(f"Unknown linear estimator settings: {sorted(unknown)}")
    chosen = str(method if method is not None else values.get("method", "ridge"))
    if chosen not in METHODS or (method is not None and values.get("method", chosen) != chosen):
        raise ValueError("Linear estimator method must be ols, ridge, weighted_ridge, huber, lasso or elastic_net")
    alpha = values.get("alpha", 0. if chosen == "ols" else 1.)
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("Linear estimator alpha must be nonnegative and finite")
    if chosen == "ols" and float(alpha) != 0:
        raise ValueError("OLS requires alpha=0; use ridge for an L2 penalty")
    result = {"method": chosen, "alpha": float(alpha)}
    if "name" in values:
        result["name"] = str(values["name"])
    if chosen == "weighted_ridge":
        half_life = values.get("half_life_days", 28.)
        if isinstance(half_life, bool) or not np.isfinite(float(half_life)) or float(half_life) <= 0:
            raise ValueError("half_life_days must be finite and positive")
        result["half_life_days"] = float(half_life)
    elif "half_life_days" in values:
        raise ValueError("half_life_days applies only to weighted_ridge")
    if chosen in {"huber", "lasso", "elastic_net"}:
        maximum = values.get("max_iter", 200 if chosen == "huber" else 10000)
        tolerance = values.get("tolerance", 1e-6)
        if isinstance(maximum, bool) or not isinstance(maximum, (int, np.integer)) or maximum < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(tolerance, bool) or not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
            raise ValueError("tolerance must be finite and positive")
        result.update(max_iter=int(maximum), tolerance=float(tolerance))
    elif {"max_iter", "tolerance"} & set(values):
        raise ValueError("max_iter and tolerance apply only to iterative estimators")
    if chosen == "huber":
        epsilon = values.get("epsilon", 1.35)
        if isinstance(epsilon, bool) or not np.isfinite(float(epsilon)) or float(epsilon) <= 0:
            raise ValueError("Huber epsilon must be finite and positive")
        result["epsilon"] = float(epsilon)
    elif "epsilon" in values:
        raise ValueError("epsilon applies only to huber")
    if chosen in {"lasso", "elastic_net"}:
        ratio = values.get("l1_ratio", 1. if chosen == "lasso" else .5)
        if isinstance(ratio, bool) or not np.isfinite(float(ratio)) or not 0 <= float(ratio) <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        if chosen == "lasso" and float(ratio) != 1:
            raise ValueError("Lasso requires l1_ratio=1; use elastic_net for mixed penalties")
        result["l1_ratio"] = float(ratio)
    elif "l1_ratio" in values:
        raise ValueError("l1_ratio applies only to lasso or elastic_net")
    return result


def _least_squares(values: np.ndarray, target: np.ndarray, alpha: float,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """Solve an augmented weighted system without squaring its condition number."""
    design, response = values, target
    if weights is not None:
        roots = np.sqrt(weights)
        design, response = values * roots[:, None], target * roots
    if alpha > 0:
        design = np.vstack([design, np.sqrt(alpha) * np.eye(values.shape[1])])
        response = np.concatenate([response, np.zeros(values.shape[1])])
    return np.linalg.lstsq(design, response, rcond=None)[0]


def _huber(values: np.ndarray, target: np.ndarray, settings: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Fit convex fixed-scale Huber loss by iteratively reweighted least squares."""
    design = np.column_stack([np.ones(len(values)), values])
    # Estimate the robust scale once from this training slice. A changing scale
    # would change the optimized objective between IRLS iterations.
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ theta
    mad = float(np.median(np.abs(residual - np.median(residual))))
    scale = 1.4826 * mad
    if scale < np.sqrt(np.finfo(float).eps):
        scale = max(float(np.median(np.abs(residual))),
                    np.sqrt(np.finfo(float).eps) * max(1., float(np.std(target))))
    threshold = settings["epsilon"] * scale
    # Only coefficient columns receive the L2 term; the intercept remains free.
    penalty = np.zeros((values.shape[1], values.shape[1] + 1))
    penalty[:, 1:] = np.sqrt(settings["alpha"]) * np.eye(values.shape[1])
    for _ in range(settings["max_iter"]):
        residual = target - design @ theta
        weights = np.minimum(1., threshold / np.maximum(np.abs(residual), np.finfo(float).tiny))
        roots = np.sqrt(weights)
        augmented = np.vstack([design * roots[:, None], penalty])
        response = np.concatenate([target * roots, np.zeros(values.shape[1])])
        updated = np.linalg.lstsq(augmented, response, rcond=None)[0]
        step = float(np.max(np.abs(updated - theta)))
        if step <= settings["tolerance"] * (1 + float(np.max(np.abs(updated)))):
            return updated[1:], float(updated[0])
        theta = updated
    raise LinearEstimatorConvergenceError(f"Huber IRLS did not converge within {settings['max_iter']} iterations")


def _coordinate_descent(values: np.ndarray, target: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    """Fit standardized elastic-net coefficients until the KKT conditions hold."""
    if settings["alpha"] == 0:
        return _least_squares(values, target, 0.)
    rows, columns = values.shape
    gram = values.T @ values / rows
    right = values.T @ target / rows
    l1 = settings["alpha"] * settings["l1_ratio"]
    l2 = settings["alpha"] * (1 - settings["l1_ratio"])
    coefficients = np.zeros(columns)
    tolerance = settings["tolerance"] * max(1., float(np.max(np.abs(right))))
    for _ in range(settings["max_iter"]):
        for index in range(columns):
            diagonal = gram[index, index]
            if diagonal < 1e-14:
                coefficients[index] = 0.
                continue
            partial = right[index] - gram[index] @ coefficients + diagonal * coefficients[index]
            coefficients[index] = np.sign(partial) * max(abs(partial) - l1, 0.) / (diagonal + l2)
        gradient = gram @ coefficients - right + l2 * coefficients
        active = coefficients != 0
        violation = np.maximum(np.abs(gradient) - l1, 0.)
        violation[active] = np.abs(gradient[active] + l1 * np.sign(coefficients[active]))
        if float(violation.max()) <= tolerance:
            return coefficients
    raise LinearEstimatorConvergenceError(
        f"{settings['method']} coordinate descent did not converge within {settings['max_iter']} iterations")


def fit_linear_estimator(train: pd.DataFrame, *, feature_names: Sequence[str], fit_cutoff: str,
                          method: str, settings: Mapping[str, Any]) -> RidgePriceModel:
    """Fit a compatible linear prediction artifact using supplied training rows.

    ``train.price_eur_mwh`` is the caller's level or already-transformed target.
    ``delivery_day`` is additionally required for recency weighting. Weighted
    ridge normalizes exponential day weights to sum to the training row count,
    preserving the existing ridge penalty convention. Its centering and scaling
    use those same training weights. No validation or prediction rows enter fit.
    """
    parameters = validate_estimator(settings, method=method)
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) for name in names):
        raise ValueError("Linear estimator feature_names must be nonempty unique column names")
    if train.empty or not {*names, "price_eur_mwh"}.issubset(train.columns):
        raise ValueError("Linear estimators require nonempty training predictors and price_eur_mwh targets")
    cutoff = pd.Timestamp(fit_cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Linear estimator fit_cutoff must be finite and timezone-aware")
    values = train.loc[:, list(names)].to_numpy(float)
    target = train.price_eur_mwh.to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Linear estimator training features and targets must be finite")
    if method == "ridge":
        # Reuse the exact established numerical kernel and penalty semantics.
        from .price_experiment import _fit_retrospective_ridge
        return _fit_retrospective_ridge(train, parameters["alpha"], cutoff.isoformat(), names)
    weights = None
    if method == "weighted_ridge":
        if "delivery_day" not in train:
            raise ValueError("weighted_ridge requires delivery_day for each training row")
        try:
            days = [date.fromisoformat(str(value)) for value in train.delivery_day]
        except (TypeError, ValueError) as exc:
            raise ValueError("weighted_ridge delivery_day must contain explicit calendar dates") from exc
        latest = max(days)
        ages = np.array([(latest - day).days for day in days], float)
        weights = np.exp2(-ages / parameters["half_life_days"])
        weights *= len(weights) / weights.sum()
    center = values.mean(axis=0) if weights is None else np.average(values, axis=0, weights=weights)
    variance = np.square(values - center).mean(axis=0) if weights is None else np.average(np.square(values - center), axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.
    normalized = (values - center) / scale
    intercept = float(target.mean() if weights is None else np.average(target, weights=weights))
    if method in {"ols", "weighted_ridge"}:
        coefficients = _least_squares(normalized, target - intercept, parameters["alpha"], weights)
    elif method == "huber":
        coefficients, intercept = _huber(normalized, target, parameters)
    else:
        coefficients = _coordinate_descent(normalized, target - intercept, parameters)
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise LinearEstimatorConvergenceError("Linear estimator produced nonfinite fitted parameters")
    return RidgePriceModel(names, center, scale, coefficients, intercept, parameters["alpha"],
        cutoff.isoformat(), len(train), f"retrospective_{method}_unverified_vintages")
