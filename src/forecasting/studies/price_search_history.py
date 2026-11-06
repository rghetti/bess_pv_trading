"""Reuse the established auction-gated price-history feature rules for the CPU search.

These functions are carried from the existing realtime price implementation.
The module imports no farm adapters, PV models or optimization solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _aware(value, label):
    """Normalize a declared clock without interpreting a naive timestamp as UTC."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _aware_series(values, label):
    """Check every clock before pandas can silently assign a timezone."""
    return pd.DatetimeIndex([_aware(value, label) for value in values]).as_unit("ns")


@dataclass(frozen=True)
class PriceForecastSettings:
    """All fitting choices are explicit and separate from evaluation metrics."""
    method: str = "ridge"
    lag_hours: tuple[float, ...] = (24.0, 168.0)
    forecast_lead_hours: float = 24.0
    ridge_alpha: float = 10.0
    min_training_rows: int = 192
    calibration_days: int = 7
    interval_minutes: int = 15
    timezone: str = "Europe/Rome"
    max_lag_age_hours: float = 336.0
    exogenous_features: tuple[str, ...] = ()
    random_seed: int = 0
    decision_local_time: str | None = None
    decision_day_offset: int = -1
    delivery_start_hour: int = 0
    delivery_end_hour: int = 24
    residual_dst_policy: str = "strict_day_length"
    daily_lag_fallback: str = "elapsed_clock"

    def __post_init__(self):
        # Reject ambiguous YAML values and normalize lists from JSON/YAML.
        object.__setattr__(self, "lag_hours", tuple(self.lag_hours))
        object.__setattr__(self, "exogenous_features", tuple(self.exogenous_features))
        if self.method not in {"ridge", "seasonal_persistence"}:
            raise ValueError("Price method must be ridge or seasonal_persistence")
        if (not self.lag_hours or len(set(self.lag_hours)) != len(self.lag_hours)
                or any(isinstance(value, bool) or not np.isfinite(value) or value <= 0
                       for value in self.lag_hours)):
            raise ValueError("Price lag_hours must contain distinct positive finite hours")
        for name in ("forecast_lead_hours", "ridge_alpha", "max_lag_age_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Price {name} must be finite and nonnegative")
        for name, minimum in (("min_training_rows", 1), ("calibration_days", 0),
                              ("interval_minutes", 1), ("random_seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Price {name} must be an integer >= {minimum}")
        if 1440 % self.interval_minutes:
            raise ValueError("Price interval_minutes must divide a calendar day")
        if len(set(self.exogenous_features)) != len(self.exogenous_features):
            raise ValueError("Price exogenous feature names must be unique")
        if any(not isinstance(name, str) or not name for name in self.exogenous_features):
            raise ValueError("Price exogenous features need nonempty names")
        pd.Timestamp("2026-01-01", tz=self.timezone)
        if self.decision_local_time is not None:
            datetime.strptime(self.decision_local_time, "%H:%M")
        if isinstance(self.decision_day_offset, bool) or not isinstance(self.decision_day_offset, int):
            raise ValueError("Price decision_day_offset must be an integer")
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in (self.delivery_start_hour, self.delivery_end_hour))
                or not 0 <= self.delivery_start_hour < self.delivery_end_hour <= 24):
            raise ValueError("Price delivery hours must define a nonempty local-day window")
        if self.residual_dst_policy not in {"strict_day_length", "same_clock_slots"}:
            raise ValueError("Unknown price residual_dst_policy")
        if self.daily_lag_fallback not in {"elapsed_clock", "target_clock"}:
            raise ValueError("Unknown price daily_lag_fallback")


def validate_price_history(history):
    """Retain raw gaps while rejecting duplicate products and uncertain clocks."""
    required = {"product_start", "product_end", "available_at", "price_eur_per_mwh"}
    if not required.issubset(history.columns):
        raise ValueError(f"Price history requires {sorted(required)}")
    frame = history[list(sorted(required))].copy()
    for name in ("product_start", "product_end", "available_at"):
        frame[name] = _aware_series(frame[name], name)
    frame["price_eur_per_mwh"] = pd.to_numeric(frame.price_eur_per_mwh, errors="raise")
    if not np.isfinite(frame.price_eur_per_mwh).all():
        raise ValueError("Price history contains missing or nonfinite values")
    if frame.product_start.duplicated().any() or (frame.product_end <= frame.product_start).any():
        raise ValueError("Price products must be unique with positive durations")
    frame = frame.sort_values("product_start", kind="stable").reset_index(drop=True)
    if len(frame) > 1 and (frame.product_start.iloc[1:].array < frame.product_end.iloc[:-1].array).any():
        raise ValueError("Price products overlap")
    return frame


@dataclass(frozen=True)
class PreparedPriceHistory:
    """Read-only numerical indices built once for repeated MPC cutoff queries."""
    starts: pd.DatetimeIndex
    availability: pd.DatetimeIndex
    prices: np.ndarray
    slot_positions: Mapping[int, np.ndarray]
    timezone: str


def prepare_price_history(history, timezone="Europe/Rome"):
    """Validate source rows and precompute same-clock seasonal fallback groups once."""
    if isinstance(history, PreparedPriceHistory):
        if history.timezone != timezone:
            raise ValueError("Prepared price history timezone differs from the fitted model")
        return history
    frame = validate_price_history(history)
    starts = pd.DatetimeIndex(frame.product_start)
    availability = pd.DatetimeIndex(frame.available_at)
    prices = frame.price_eur_per_mwh.to_numpy(float, copy=True)
    prices.setflags(write=False)
    local = starts.tz_convert(timezone)
    slots = local.hour * 60 + local.minute
    positions = {int(slot): np.flatnonzero(slots == slot) for slot in np.unique(slots)}
    for values in positions.values():
        values.setflags(write=False)
    return PreparedPriceHistory(starts, availability, prices, positions, timezone)


@dataclass(frozen=True)
class PreparedExogenous:
    """Validated feature tables indexed once for many cutoff-specific predictions."""
    groups: Mapping[str, pd.DataFrame]


def prepare_exogenous_features(exogenous):
    """Compile long feature vintages once without constructing a group per product."""
    if isinstance(exogenous, PreparedExogenous):
        return exogenous
    required = {"product_start", "available_at", "feature", "value"}
    if exogenous is None or not required.issubset(exogenous.columns):
        raise ValueError("Exogenous inputs require product_start/available_at/feature/value")
    frame = exogenous.copy()
    frame["product_start"] = _aware_series(frame.product_start, "exogenous product_start")
    frame["available_at"] = _aware_series(frame.available_at, "exogenous available_at")
    if frame.duplicated(["feature", "product_start", "available_at"]).any():
        raise ValueError("Duplicate exogenous feature vintages")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    if not np.isfinite(frame.value).all():
        raise ValueError("Exogenous feature values must be finite")
    groups = {name: group.sort_values(["product_start", "available_at"]).reset_index(drop=True)
              for name, group in frame.groupby("feature", sort=False)}
    return PreparedExogenous(groups)


def _exogenous_values(exogenous, names, starts, cutoffs):
    """Select the latest published vintage with vectorized as-of joins."""
    if not names:
        return np.empty((len(starts), 0))
    prepared = prepare_exogenous_features(exogenous)
    values = np.full((len(starts), len(names)), np.nan)
    left = pd.DataFrame({"product_start": pd.DatetimeIndex(starts).as_unit("ns"),
                         "decision": pd.DatetimeIndex(cutoffs).as_unit("ns"), "row": np.arange(len(starts))})
    left = left.sort_values("decision", kind="stable")
    # Restrict each indexed feature to target products before joining publication clocks.
    for column, name in enumerate(names):
        group = prepared.groups.get(name)
        if group is None:
            continue
        first = group.product_start.searchsorted(starts.min(), side="left")
        stop = group.product_start.searchsorted(starts.max(), side="right")
        right = group.iloc[first:stop].sort_values("available_at", kind="stable")
        if len(right):
            matched = pd.merge_asof(left, right, left_on="decision", right_on="available_at",
                                   by="product_start", direction="backward")
            values[matched.row.to_numpy(), column] = matched.value.to_numpy()
    return values


def _features(history, starts, cutoffs, settings, exogenous=None):
    """Build all features as of individual decision clocks, never target delivery."""
    prepared = prepare_price_history(history, settings.timezone)
    index = prepared.starts
    availability = prepared.availability
    prices = prepared.prices
    slot_positions = prepared.slot_positions
    columns = []
    fallback_count = 0
    for lag in settings.lag_hours:
        wanted = starts - pd.Timedelta(hours=lag)
        positions = index.get_indexer(wanted)
        valid = positions >= 0
        valid[valid] &= availability[positions[valid]] <= cutoffs[valid]
        values = np.full(len(starts), np.nan)
        values[valid] = prices[positions[valid]]
        # Retain the same local clock time when the nominal lag is not published.
        for row in np.flatnonzero(~valid):
            # Daily lags can explicitly fall back to the target clock after DST;
            # non-day lags always retain their requested elapsed-hour clock.
            use_target_clock = settings.daily_lag_fallback == "target_clock" and lag % 24 == 0
            local = (starts[row] if use_target_clock else wanted[row]).tz_convert(settings.timezone)
            candidates = slot_positions.get(local.hour * 60 + local.minute, np.array([], dtype=int))
            candidates = candidates[(index[candidates] <= wanted[row])
                                    & (availability[candidates] <= cutoffs[row])
                                    & (index[candidates] >= wanted[row] - pd.Timedelta(hours=settings.max_lag_age_hours))]
            if len(candidates):
                values[row] = prices[candidates[-1]]
                fallback_count += 1
        columns.append(values)
    # Calendar terms are deterministic at the historical decision cutoff.
    local = starts.tz_convert(settings.timezone)
    hour = local.hour + local.minute / 60
    weekday = local.dayofweek.to_numpy(float)
    annual = (local.dayofyear.to_numpy(float) - 1) / 365.25
    columns.extend([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
                    (weekday >= 5).astype(float), np.sin(2 * np.pi * annual),
                    np.cos(2 * np.pi * annual)])
    matrix = np.column_stack(columns)
    matrix = np.column_stack([matrix, _exogenous_values(exogenous, settings.exogenous_features, starts, cutoffs)])
    return matrix, fallback_count


def historical_decision_times(starts, settings):
    """Use one configured auction gate per local day or an explicit lead horizon."""
    if settings.decision_local_time is None:
        return starts - pd.Timedelta(hours=settings.forecast_lead_hours)
    dates = starts.tz_convert(settings.timezone).date
    days = {day: pd.Timestamp(f"{day + timedelta(days=settings.decision_day_offset)} {settings.decision_local_time}",
                             tz=settings.timezone).tz_convert("UTC") for day in np.unique(dates)}
    return pd.DatetimeIndex([days[day] for day in dates]).as_unit("ns")
