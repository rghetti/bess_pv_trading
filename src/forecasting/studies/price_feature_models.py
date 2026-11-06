"""Deterministic parameter sampling and explicitly single-thread CPU estimators."""

from __future__ import annotations

import hashlib
import math
import random

import numpy as np


def stable_seed(*parts):
    """Keep candidate seeds stable across process order, paths and worker counts."""
    encoded = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(encoded).hexdigest()[:8], 16) % (2**31 - 1)


def sample_parameters(space, seed, count):
    """Expand declared fixed values and bounded randomized distributions."""
    rng = random.Random(seed)
    candidates = []
    for _ in range(count):
        parameters = {}
        for name, specification in space.items():
            if not isinstance(specification, dict):
                value = specification
            elif set(specification) == {"choices"}:
                value = rng.choice(specification["choices"])
            elif set(specification) == {"uniform"}:
                value = rng.uniform(*specification["uniform"])
            elif set(specification) == {"log_uniform"}:
                low, high = specification["log_uniform"]
                if not 0 < low <= high:
                    raise ValueError(f"Invalid log-uniform range for {name}")
                value = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                raise ValueError(f"Unsupported parameter distribution for {name}")
            parameters[name] = value
        candidates.append(parameters)
    return candidates


def make_model(family, parameters, seed):
    """Create a CPU model with scaling learned exclusively from its training rows."""
    options = dict(parameters)
    if family in {"ridge", "elastic_net", "mlp"}:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if family == "ridge":
            from sklearn.linear_model import Ridge
            estimator = Ridge(**options)
        elif family == "elastic_net":
            from sklearn.linear_model import ElasticNet
            estimator = ElasticNet(random_state=seed, **options)
        else:
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.neural_network import MLPRegressor
            options["hidden_layer_sizes"] = tuple(options["hidden_layer_sizes"])
            estimator = TransformedTargetRegressor(
                regressor=MLPRegressor(random_state=seed, **options), transformer=StandardScaler())
        return make_pipeline(StandardScaler(), estimator)
    if family in {"random_forest", "extra_trees"}:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        constructor = RandomForestRegressor if family == "random_forest" else ExtraTreesRegressor
        return constructor(random_state=seed, n_jobs=1, **options)
    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **options)
    if family == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=1, verbosity=-1, **options)
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, thread_count=1, task_type="CPU", verbose=False,
                                 allow_writing_files=False, **options)
    if family == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=1, device="cpu", verbosity=0, **options)
    raise ValueError(f"Unknown model family {family}")


def training_objective(family, parameters):
    """Distinguish fit losses from the common absolute-price evaluation metrics."""
    if family == "ridge":
        return "squared error plus L2 coefficient penalty"
    if family == "elastic_net":
        return "squared error plus L1/L2 coefficient penalty"
    if family == "mlp":
        return "squared error on training-standardized target plus L2 penalty"
    return parameters.get("objective", parameters.get("loss_function", parameters.get("loss", parameters.get("criterion"))))


def point_metrics(actual, prediction, reference=None):
    """Use pooled absolute-price WAPE, including negative and near-zero prices."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not len(actual) or actual.shape != prediction.shape or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Scores require aligned, finite, nonempty price vectors")
    errors = np.abs(prediction - actual)
    denominator = float(np.abs(actual).sum())
    result = {"mae_eur_per_mwh": float(errors.mean()),
              "rmse_eur_per_mwh": float(np.sqrt(np.mean((prediction - actual) ** 2))),
              "wape_percent": None if denominator == 0 else float(100 * errors.sum() / denominator),
              "observations": len(actual)}
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != actual.shape or not np.isfinite(reference).all():
            raise ValueError("Reference predictions must match the scored rows")
        baseline = float(np.abs(actual - reference).sum())
        result["relative_mae_to_reference"] = None if baseline == 0 else float(errors.sum() / baseline)
    return result
