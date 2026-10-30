"""Bounded physical calibration and strictly historical model selection."""
from __future__ import annotations

from copy import deepcopy
import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from ..models.pv_physical import PhysicalPVModel, parameter_defaults


def available_methods(section: dict, farm) -> list[str]:
    """Do not construct tracker or rear-side calibration variables for fixed farms."""
    result = []
    # Do not construct tracker or rear-side calibration variables for fixed farms for each `name` in `section['methods']`.
    for name in section["methods"]:
        family = name.split("_")[0]
        # Skip the current item when `family == 'tracker' and farm.tracking_type_id not in {1, 3}`.
        if family == "tracker" and farm.tracking_type_id not in {1, 3}:
            continue
        # Skip the current item when `family == 'bifacial' and farm.tracking_type_id != 3`.
        if family == "bifacial" and farm.tracking_type_id != 3:
            continue
        result.append(name)
    return result


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every observed day/auction view equal mass regardless of its length."""
    counts = frame.groupby(["delivery_day", "auction"])["delivery_day"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights / weights.sum()


def loss_values(residual, loss: str, delta: float):
    """Use squared error or a continuous Huber loss in nameplate-normalized units."""
    # Return `residual ** 2` when `loss == 'mse'`.
    if loss == "mse":
        return residual ** 2
    absolute = np.abs(residual)
    return np.where(absolute <= delta, residual ** 2, 2 * delta * absolute - delta ** 2)


def fit_settings(settings: dict, method: str, transposition: str) -> dict:
    """Rear-side fits isolate the added irradiance model with declared row geometry.

    The separate tracker family fits rotation/GCR; rear-side thermal fits keep
    those geometry assumptions fixed to avoid fitting unmeasured rear geometry
    and aggregate output gain simultaneously.
    """
    result = deepcopy(settings)
    result["parameter_registry"] = None
    result["transposition_model"] = transposition
    result.setdefault("bifacial", {})["enabled"] = method.startswith("bifacial_")
    return result


def fit_one(frame: pd.DataFrame, farm, settings: dict, section: dict, method: str,
            transposition: str, regularization: float, seed: int) -> dict:
    """Fit the complete clipped physical forward model with deterministic starts."""
    started = time.perf_counter()
    family, loss = method.rsplit("_", 1)
    names = ["coefficient"]
    # When `family in {'thermal', 'tracker', 'bifacial'}`, set `names` using `['u0', 'u1']`.
    if family in {"thermal", "tracker", "bifacial"}:
        names += ["u0", "u1"]
    # When `family == 'tracker'`, set `names` using `['gcr', 'max_angle']`.
    if family == "tracker":
        names += ["gcr", "max_angle"]
    model_settings = fit_settings(settings, method, transposition)
    model = PhysicalPVModel(frame.set_index("interval_start_utc"), farm, model_settings, section["night_policy"])
    defaults = parameter_defaults(settings)
    bounds = np.asarray([section["parameter_bounds"][name] for name in names], dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    width = upper - lower
    prior = np.asarray([defaults[name] for name in names])
    weights = balanced_weights(frame)
    observed = frame.measured_ac_mw.to_numpy() / farm.ac_capacity_mw
    evaluations = 0

    def parameters(vector):
        return {**defaults, **dict(zip(names, map(float, vector)))}

    def objective(vector):
        nonlocal evaluations
        evaluations += 1
        residual = model.evaluate(parameters(vector)) / farm.ac_capacity_mw - observed
        # Return `1e+20` when `not np.isfinite(residual).all()`.
        if not np.isfinite(residual).all():
            return 1e20
        value = np.dot(weights, loss_values(residual, loss, float(section["huber_delta"])))
        return float(value + regularization * np.sum(((vector - prior) / width) ** 2))

    # Always include the configured assumptions; random starts explore only bounds.
    initial = np.clip(prior, lower, upper)
    trials = []
    # Update `answer` when this condition is met.
    if len(names) == 1:
        answer = minimize_scalar(lambda x: objective(np.array([x])), bounds=tuple(bounds[0]),
            method="bounded", options={"xatol": float(section["optimizer_ftol"]),
                                       "maxiter": int(section["max_function_evaluations"])})
        candidates = [(np.array([answer.x]), bool(answer.success), str(answer.message))]
    else:
        rng = np.random.default_rng(seed)
        starts = [initial] + [rng.uniform(lower, upper) for _ in range(int(section["multistarts"])-1)]
        candidates = []
        # Fit the complete clipped physical forward model with deterministic starts for each `start` in `starts`.
        for start in starts:
            # Unit-box coordinates prevent the output multiplier's gradient from
            # hiding the much smaller temperature/geometry gradients.
            answer = minimize(lambda value: objective(lower+value*width),
                (start-lower)/width, method="L-BFGS-B", bounds=[(0,1)]*len(names),
                options={"maxfun": int(section["max_function_evaluations"]),
                         "maxiter": int(section["max_function_evaluations"]),
                         "ftol": float(section["optimizer_ftol"]),
                         "gtol": float(section["optimizer_gtol"]), "maxls": 15})
            candidates.append((lower+answer.x*width, bool(answer.success), str(answer.message)))
    # A numerical budget stop is recorded, and cannot make the accepted iterate
    # worse than its configured starting point on the fitting objective.
    candidates.append((initial, False, "configured_start_reference"))
    # Fit the complete clipped physical forward model with deterministic starts for each `(vector, success, message)` in `candidates`.
    for vector, success, message in candidates:
        trials.append({"parameters": parameters(vector), "objective": objective(vector),
                       "converged": success, "message": message})
    best = min(trials, key=lambda item: item["objective"])
    # Enforce the input contract: Physical calibration produced no finite candidate.
    if best["objective"] >= 1e19:
        raise ValueError("Physical calibration produced no finite candidate")
    vector = np.array([best["parameters"][name] for name in names])
    base_prediction = model.evaluate(best["parameters"]) / farm.ac_capacity_mw
    columns = []
    # Fit the complete clipped physical forward model with deterministic starts for each `(position, name)` in `enumerate(names)`.
    for position, name in enumerate(names):
        changed = vector.copy()
        step = width[position] * 1e-5
        changed[position] += step
        columns.append((model.evaluate(parameters(changed)) / farm.ac_capacity_mw - base_prediction) / step * width[position])
    jacobian = np.asarray(columns).T * np.sqrt(weights[:, None])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(singular[0]/singular[-1]) if singular[-1] > np.finfo(float).eps else None
    return {"method": method, "parameters": best["parameters"], "fitted_parameters": names,
        "transposition_model": transposition, "bifacial_enabled": family == "bifacial",
        "bifacial_settings": model_settings["bifacial"], "albedo": model_settings["albedo"],
        "regularization": regularization, "loss": loss,
        "objective": best["objective"], "converged": best["converged"], "optimizer_message": best["message"],
        "function_evaluations": evaluations, "training_rows": len(frame),
        "training_days": int(frame.delivery_day.nunique()),
        "latest_target_available_at_utc": frame.target_available_at_utc.max().isoformat(),
        "bound_hits": [name for i, name in enumerate(names)
                       if min(abs(vector[i]-lower[i]), abs(vector[i]-upper[i])) <= width[i] * 1e-4],
        "jacobian_rank": rank, "jacobian_condition_number": condition,
        "weakly_identified": rank < len(names) or condition is None or condition > 1e6,
        "multistart_trials": trials, "runtime_seconds": time.perf_counter()-started}


def fixed_fit(settings: dict) -> dict:
    """Represent the unchanged baseline with the same portable artifact interface."""
    return {"method": "fixed", "parameters": parameter_defaults(settings),
        "fitted_parameters": [], "transposition_model": settings.get("transposition_model", "isotropic"),
        "bifacial_enabled": bool(settings["bifacial"]["enabled"]),
        "bifacial_settings": deepcopy(settings["bifacial"]), "albedo": settings["albedo"],
        "runtime_seconds": 0.0, "converged": True, "function_evaluations": 0,
        "bound_hits": [], "weakly_identified": False}


def predict_fit(frame: pd.DataFrame, farm, settings: dict, night_policy: dict, fit: dict) -> np.ndarray:
    """Use exactly the operational physical engine and preserve the frozen mask."""
    configured = deepcopy(settings)
    configured["transposition_model"] = fit["transposition_model"]
    configured["bifacial"] = fit["bifacial_settings"]
    configured["albedo"] = fit["albedo"]
    model = PhysicalPVModel(frame.set_index("interval_start_utc"), farm, configured, night_policy)
    prediction = model.evaluate(fit["parameters"])
    prediction[model.night & ~np.isfinite(prediction)] = 0.0
    # Enforce the input contract: Candidate forecast is missing on the frozen evaluation mask.
    if not np.isfinite(prediction[frame.eligible.to_numpy()]).all():
        raise ValueError("Candidate forecast is missing on the frozen evaluation mask")
    return prediction


def validation_score(frame, prediction, rating):
    """Rank candidates by balanced daylight nameplate-normalized RMSE."""
    mask = frame.eligible & frame.daylight
    chosen = frame.loc[mask]
    residual = (prediction[mask.to_numpy()] - chosen.measured_ac_mw.to_numpy()) / rating
    return float(np.sqrt(np.dot(balanced_weights(chosen), residual ** 2)))


def select_and_refit(corpus: pd.DataFrame, issue, farm, settings: dict, section: dict,
                     method: str) -> tuple[dict, dict]:
    """Select on an earlier holdout, refit, then expose an immutable issue-time fit."""
    issue = pd.Timestamp(issue)
    local_midnight = issue.tz_convert(section["delivery_timezone"]).normalize()
    last_day = local_midnight.date().isoformat()
    validation_day = (local_midnight-pd.DateOffset(days=int(section["validation_days"]))).date().isoformat()
    available = corpus.loc[(corpus.target_available_at_utc <= issue) & (corpus.delivery_day < last_day)]
    validation = available.loc[available.delivery_day >= validation_day]
    # Enforce the input contract: PV calibration validation has no eligible daylight observations.
    if validation.loc[validation.eligible & validation.daylight].empty:
        raise ValueError("PV calibration validation has no eligible daylight observations")
    # When `method == 'fixed'`, set `fit` using `fixed_fit(settings)`.
    if method == "fixed":
        fit = fixed_fit(settings)
        score = validation_score(validation, validation.baseline_mw.to_numpy(), farm.ac_capacity_mw)
        fit["validation_nrmse"] = score
        return fit, {"method": method, "validation_nrmse": score, "candidates": []}
    family = method.split("_")[0]
    models = ([settings.get("transposition_model", "isotropic")] if family == "scale"
              else [x for x in section["transposition_models"]
                    if family != "bifacial" or x in {"isotropic", "haydavies"}])
    regularizations = [0.0] if family == "scale" else section["regularization_values"]
    candidates, seen = [], set()
    # Process each `(window, transposition, regularization)` in this collection.
    for window, transposition, regularization in itertools.product(section["training_windows_days"], models, regularizations):
        history = available if window is None else available.loc[available.delivery_day >=
            (local_midnight-pd.DateOffset(days=int(window))).date().isoformat()]
        train = history.loc[(history.delivery_day < validation_day) & history.eligible & history.daylight]
        # Apply this branch only when its condition is met.
        if train.delivery_day.nunique() < int(section["minimum_training_days"]) or len(train) < int(section["minimum_pairs"]):
            continue
        key = (train.delivery_day.min(), train.delivery_day.max(), transposition, regularization)
        # Skip the current item when `key in seen`.
        if key in seen:
            continue
        seen.add(key)
        seed = int(section["seed"]) + farm.farm_id + issue.year * 12 + issue.month
        fitted = fit_one(train, farm, settings, section, method, transposition, float(regularization), seed)
        predicted = predict_fit(validation, farm, settings, section["night_policy"], fitted)
        score = validation_score(validation, predicted, farm.ac_capacity_mw)
        candidates.append({"window_days": window, "validation_nrmse": score, "fit": fitted})
    # Enforce the input contract: PV calibration `method` has insufficient earlier training history.
    if not candidates:
        raise ValueError(f"PV calibration {method} has insufficient earlier training history")
    best = min(candidates, key=lambda item: (item["validation_nrmse"], item["fit"]["regularization"] != 0))
    history = available if best["window_days"] is None else available.loc[available.delivery_day >=
        (local_midnight-pd.DateOffset(days=int(best["window_days"]))).date().isoformat()]
    full = history.loc[history.eligible & history.daylight]
    fitted = fit_one(full, farm, settings, section, method, best["fit"]["transposition_model"],
        best["fit"]["regularization"], int(section["seed"]) + farm.farm_id + issue.year * 12 + issue.month)
    fitted.update(validation_nrmse=best["validation_nrmse"], window_days=best["window_days"],
        training_start_day=full.delivery_day.min(), training_end_day=full.delivery_day.max())
    diagnostics = {"method": method, "validation_start_day": validation_day,
        "validation_end_day": validation.delivery_day.max(), "validation_nrmse": best["validation_nrmse"],
        "candidates": candidates, "refit": fitted,
        "selection_runtime_seconds": sum(x["fit"]["runtime_seconds"] for x in candidates)}
    return fitted, diagnostics
