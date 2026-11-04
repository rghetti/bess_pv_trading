"""Build chronology-safe price-search matrices and evaluate matched days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .energy_prices import _day_grid, _features, historical_decision_times, validate_price_history


def prepare_search_matrix(history, settings, exogenous, *, cutoff, validation_start, validation_end):
    """Compute historical decision features once, retaining explicit incomplete-day gaps."""
    history = validate_price_history(history)
    starts = pd.DatetimeIndex(history.product_start)
    decisions = historical_decision_times(starts, settings)
    matrix, fallback_count = _features(history, starts, decisions, settings, exogenous)
    finite = np.isfinite(matrix).all(axis=1)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Search cutoff must be timezone-aware")
    start = pd.Timestamp(validation_start, tz=settings.timezone).tz_convert("UTC")
    end = pd.Timestamp(validation_end, tz=settings.timezone).tz_convert("UTC")
    if cutoff > start or start >= end:
        raise ValueError("Search validation must follow its training cutoff")
    valid_start = historical_decision_times(pd.DatetimeIndex([start]), settings)[0]
    if cutoff > valid_start:
        raise ValueError("Training cutoff occurs after the first validation decision")
    training_clock = (history.available_at <= cutoff).to_numpy() & (history.product_end <= cutoff).to_numpy()
    validation_clock = (starts >= start) & (starts < end)
    training = finite & training_clock
    validation = finite & validation_clock
    days = starts.tz_convert(settings.timezone).strftime("%Y-%m-%d")
    return {"matrix": matrix, "target": history.price_eur_per_mwh.to_numpy(float),
            "starts": starts, "days": days, "training": training, "validation": validation,
            "training_clock": training_clock, "validation_clock": validation_clock,
            "fallback_count": fallback_count}


def project_search_matrix(dataset, columns):
    """Select one feature group, recomputing its missingness from the shared matrix."""
    matrix = dataset["matrix"][:, columns]
    finite = np.isfinite(matrix).all(axis=1)
    return {**dataset, "matrix": matrix, "training": finite & dataset["training_clock"],
            "validation": finite & dataset["validation_clock"]}


def complete_shared_days(datasets, *, start, end, settings):
    """Require every selected feature set to cover each complete delivery day."""
    days = []
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        label = day.date().isoformat()
        expected = _day_grid(label, settings.interval_minutes, settings.timezone,
                             settings.delivery_start_hour, settings.delivery_end_hour)
        if all(pd.DatetimeIndex(dataset["starts"][dataset["validation"] & (dataset["days"] == label)]).equals(expected)
               for dataset in datasets):
            days.append(label)
    return days


def point_metrics(actual, prediction, reference):
    """Return pooled scale-aware errors and the shared seasonal-naive error ratio."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not len(actual) or prediction.shape != actual.shape or reference.shape != actual.shape:
        raise ValueError("Search scores require aligned nonempty vectors")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Search scores require finite vectors")
    errors = np.abs(prediction - actual)
    scale = np.abs(actual).sum()
    baseline = np.abs(reference - actual).sum()
    return {"mae_eur_per_mwh": float(errors.mean()),
            "rmse_eur_per_mwh": float(np.sqrt(np.mean((prediction - actual) ** 2))),
            "wape_percent": None if scale == 0 else float(100 * errors.sum() / scale),
            "relative_mae_to_naive": None if baseline == 0 else float(errors.sum() / baseline),
            "observations": len(actual)}
