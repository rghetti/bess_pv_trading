"""Build chronology-safe price-search matrices and evaluate matched days."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .energy_prices import _day_grid, _features, historical_decision_times, validate_price_history


def daily_price_curve_features(history, starts, cutoffs, settings, lag_days):
    """Return previous local-day hourly price profiles with decision-time checks.

    Quarter-hour native prices are averaged within each hour only when every
    native product exists. DST transition days remain missing instead of being
    silently padded or averaging two distinct occurrences of the same hour.
    """
    if (not lag_days or len(set(lag_days)) != len(lag_days)
            or any(isinstance(day, bool) or not isinstance(day, int) or day < 1 for day in lag_days)):
        raise ValueError("Curve lags must be distinct positive calendar-day counts")
    if 60 % settings.interval_minutes:
        raise ValueError("Hourly curve features require a native resolution dividing one hour")
    frame = validate_price_history(history)
    times = pd.DatetimeIndex(frame.product_start)
    local = times.tz_convert(settings.timezone)
    local_dates = local.date
    hours = local.hour
    per_hour = 60 // settings.interval_minutes
    selected_hours = range(settings.delivery_start_hour, settings.delivery_end_hour)
    profiles = {}
    for day in np.unique(local_dates):
        positions = np.flatnonzero(local_dates == day)
        values = np.full(len(selected_hours), np.nan)
        availability = np.full(len(selected_hours), np.iinfo(np.int64).max, dtype=np.int64)
        for column, hour in enumerate(selected_hours):
            rows = positions[hours[positions] == hour]
            # Missing, duplicate or repeated DST-hour products do not form a
            # standard hourly feature; preserve the gap rather than imputing.
            if len(rows) != per_hour:
                continue
            hour_times = times[rows]
            hour_local = hour_times.tz_convert(settings.timezone)
            if (not np.array_equal(hour_local.minute.to_numpy(), np.arange(per_hour) * settings.interval_minutes)
                    or np.any(hour_local.second != 0)):
                continue
            if len(rows) > 1 and not np.all(np.diff(hour_times.asi8) == pd.Timedelta(minutes=settings.interval_minutes).value):
                continue
            values[column] = frame.price_eur_per_mwh.iloc[rows].mean()
            availability[column] = pd.DatetimeIndex(frame.available_at.iloc[rows]).asi8.max()
        profiles[day] = (values, availability)
    starts = pd.DatetimeIndex(starts)
    cutoffs = pd.DatetimeIndex(cutoffs).as_unit("ns")
    if len(starts) != len(cutoffs) or starts.tz is None or cutoffs.tz is None:
        raise ValueError("Curve requests require aligned timezone-aware starts and cutoffs")
    target_dates = starts.tz_convert(settings.timezone).date
    output = np.full((len(starts), len(lag_days) * len(selected_hours)), np.nan)
    for day in np.unique(target_dates):
        rows = np.flatnonzero(target_dates == day)
        for lag_index, lag in enumerate(lag_days):
            profile = profiles.get(day - timedelta(days=lag))
            if profile is None:
                continue
            values, availability = profile
            block = slice(lag_index * len(selected_hours), (lag_index + 1) * len(selected_hours))
            output[rows, block] = np.where(availability[None, :] <= cutoffs.asi8[rows, None], values[None, :], np.nan)
    return output


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
            "decisions": decisions, "label_available_at": pd.DatetimeIndex(history.available_at),
            "label_product_end": pd.DatetimeIndex(history.product_end),
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
