"""Small causal price models with explicit publication clocks and daily residuals.

This module does not infer historical availability from a download timestamp.
Model fitting is an explicit operation; loading or predicting never trains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
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


def _digest(value):
    """Hash the actual serialized artifact, including its input provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


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


@dataclass(frozen=True)
class PricePrediction:
    """Conditional point estimate and coherent sampled residual trajectories."""
    point_eur_per_mwh: np.ndarray
    scenarios_eur_per_mwh: np.ndarray
    available_at: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "available_at", _aware(self.available_at, "prediction availability"))
        for name in ("point_eur_per_mwh", "scenarios_eur_per_mwh"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(values).all():
                raise ValueError("Price predictions must be finite")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (self.point_eur_per_mwh.ndim != 1 or self.scenarios_eur_per_mwh.ndim != 2
                or self.scenarios_eur_per_mwh.shape[1] != len(self.point_eur_per_mwh)
                or not len(self.scenarios_eur_per_mwh)):
            raise ValueError("Price prediction shapes must be T and S x T")


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


def _fit_linear(matrix, targets, alpha):
    """Standardize using training data and leave the intercept unpenalized."""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    penalty = np.eye(design.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    # Augmented least squares avoids squaring the condition number in normal equations.
    coefficient = np.linalg.lstsq(np.vstack([design, penalty]),
        np.concatenate([targets, np.zeros(design.shape[1])]), rcond=None)[0]
    return coefficient, mean, scale


def _linear_predict(matrix, coefficient, mean, scale):
    """Reuse immutable training scaling at every subsequent decision cutoff."""
    return np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale]) @ coefficient


def _day_grid(day, interval_minutes, timezone, first_hour=0, last_hour=24):
    """Construct local-clock product windows without assuming 24-hour DST days."""
    first = pd.Timestamp(f"{day} {first_hour:02}:00", tz=timezone)
    if last_hour == 24:
        stop = pd.Timestamp((first.date() + timedelta(days=1)).isoformat(), tz=timezone)
    else:
        stop = pd.Timestamp(f"{day} {last_hour:02}:00", tz=timezone)
    return pd.date_range(first, stop, freq=f"{interval_minutes}min", inclusive="left").tz_convert("UTC")


def historical_decision_times(starts, settings):
    """Use one configured auction gate per local day or an explicit lead horizon."""
    if settings.decision_local_time is None:
        return starts - pd.Timedelta(hours=settings.forecast_lead_hours)
    dates = starts.tz_convert(settings.timezone).date
    days = {day: pd.Timestamp(f"{day + timedelta(days=settings.decision_day_offset)} {settings.decision_local_time}",
                             tz=settings.timezone).tz_convert("UTC") for day in np.unique(dates)}
    return pd.DatetimeIndex([days[day] for day in dates]).as_unit("ns")


def _daily_blocks(starts, residuals, interval_minutes, timezone, first_hour=0, last_hour=24):
    """Keep complete chronological local days, including 23/25-hour DST days."""
    frame = pd.DataFrame({"start": starts, "residual": residuals})
    dates = starts.tz_convert(timezone).strftime("%Y-%m-%d")
    blocks = []
    for day, group in frame.groupby(dates, sort=True):
        expected = _day_grid(day, interval_minutes, timezone, first_hour, last_hour)
        if pd.DatetimeIndex(group.start).equals(expected) and np.isfinite(group.residual).all():
            blocks.append({"day": day, "starts": [value.isoformat() for value in expected],
                           "residual_eur_per_mwh": group.residual.to_list()})
    return blocks


class CausalPriceForecaster:
    """A fitted price artifact; forecasts contain no realized future prices."""

    def __init__(self, settings, history, fit_cutoff, coefficient, mean, scale,
                 residual_blocks, provenance):
        self.settings = settings
        self._history = validate_price_history(history)
        self._prepared_history = prepare_price_history(self._history, settings.timezone)
        self.fit_cutoff = _aware(fit_cutoff, "fit cutoff")
        self.coefficient = np.array(coefficient, dtype=float, copy=True)
        self.mean = np.array(mean, dtype=float, copy=True)
        self.scale = np.array(scale, dtype=float, copy=True)
        self._residual_blocks = deepcopy(residual_blocks)
        self._provenance = deepcopy(dict(provenance))
        if (self._history.available_at > self.fit_cutoff).any() or (self._history.product_end > self.fit_cutoff).any():
            raise ValueError("Price artifact contains labels beyond its fit cutoff")
        expected = len(settings.lag_hours) + 7 + len(settings.exogenous_features)
        if (self.coefficient.shape != (expected + 1,) or self.mean.shape != (expected,)
                or self.scale.shape != (expected,) or not np.isfinite(self.coefficient).all()
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid price artifact model parameters")
        # Freeze fitting arrays and validate calibration paths before caching a digest.
        for values in (self.coefficient, self.mean, self.scale):
            values.setflags(write=False)
        for block in self._residual_blocks:
            starts = _aware_series(block["starts"], "calibration residual start")
            residuals = np.asarray(block["residual_eur_per_mwh"], dtype=float)
            if (residuals.shape != (len(starts),) or not np.isfinite(residuals).all()
                    or not starts.isin(self._history.product_start).all()
                    or len(_daily_blocks(starts, residuals, settings.interval_minutes, settings.timezone,
                                        settings.delivery_start_hour, settings.delivery_end_hour)) != 1
                    or starts[0].tz_convert(settings.timezone).date().isoformat() != block["day"]):
                raise ValueError("Price residual blocks require complete finite historical local days")
        self._artifact_sha256 = _digest(self._payload())

    @property
    def history(self):
        """Expose a copy so callers cannot alter the fitted artifact's retained data."""
        return self._history.copy()

    @property
    def residual_blocks(self):
        """Expose calibration evidence without allowing mutation of future draws."""
        return deepcopy(self._residual_blocks)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    @classmethod
    def fit(cls, history, *, cutoff, settings=None, exogenous=None, provenance=None):
        """Fit causal features and reserve a chronological calibration period.

        The calibration predictor is fitted before the first calibration decision.
        The deployed predictor is then refitted on all labels available at cutoff.
        Residuals remain out of sample for the predictor that generated them.
        """
        settings = settings or PriceForecastSettings()
        cutoff = _aware(cutoff, "fit cutoff")
        history = validate_price_history(history)
        history = history.loc[(history.available_at <= cutoff) & (history.product_end <= cutoff)].copy()
        if history.empty:
            raise ValueError("No price labels were available by the fit cutoff")
        starts = pd.DatetimeIndex(history.product_start)
        decisions = historical_decision_times(starts, settings)
        matrix, fallback_count = _features(history, starts, decisions, settings, exogenous)
        usable = np.isfinite(matrix).all(axis=1) & (decisions <= starts)
        targets = history.price_eur_per_mwh.to_numpy(float)
        if usable.sum() < settings.min_training_rows:
            raise ValueError("Insufficient causally available training rows after lag/vintage exclusions")
        # Only prior labels enter the predictor used to score the calibration tail.
        blocks = []
        calibration_score = None
        calibration_rows = 0
        if settings.calibration_days:
            local_end = starts[-1].tz_convert(settings.timezone).normalize() + pd.DateOffset(days=1)
            calibration_start = local_end - pd.DateOffset(days=settings.calibration_days)
            calibration_cutoff = historical_decision_times(pd.DatetimeIndex([calibration_start]), settings)[0]
            prefix = usable & (history.available_at <= calibration_cutoff).to_numpy() & (starts < calibration_start)
            calibration = usable & (starts >= calibration_start)
            if prefix.sum() < settings.min_training_rows or not calibration.any():
                raise ValueError("Insufficient pre-calibration history; reduce calibration_days or supply more past data")
            if settings.method == "ridge":
                coefficient, mean, scale = _fit_linear(matrix[prefix], targets[prefix], settings.ridge_alpha)
                predicted = _linear_predict(matrix[calibration], coefficient, mean, scale)
            else:
                predicted = matrix[calibration, 0]
            residuals = targets[calibration] - predicted
            calibration_rows = int(calibration.sum())
            calibration_score = float(np.mean(np.abs(residuals)))
            blocks = _daily_blocks(starts[calibration], residuals, settings.interval_minutes, settings.timezone,
                                   settings.delivery_start_hour, settings.delivery_end_hour)
        if settings.method == "ridge":
            coefficient, mean, scale = _fit_linear(matrix[usable], targets[usable], settings.ridge_alpha)
        else:
            coefficient = np.zeros(matrix.shape[1] + 1)
            mean = np.zeros(matrix.shape[1])
            scale = np.ones(matrix.shape[1])
        evidence = dict(provenance or {})
        evidence.update({"model": settings.method, "fit_cutoff_utc": cutoff.isoformat(),
                         "training_rows": int(usable.sum()), "excluded_feature_rows": int((~usable).sum()),
                         "training_label_max_available_at_utc": history.loc[usable, "available_at"].max().isoformat(),
                         "historical_feature_lead_hours": settings.forecast_lead_hours,
                         "historical_decision_local_time": settings.decision_local_time,
                         "historical_decision_day_offset": settings.decision_day_offset,
                         "lag_fallback_count": fallback_count, "calibration_rows": calibration_rows,
                         "complete_calibration_days": len(blocks),
                         "calibration_point_mae_eur_per_mwh": calibration_score,
                         "training_objective": "squared_error_plus_standardized_ridge_penalty"
                         if settings.method == "ridge" else "none_seasonal_persistence",
                         "residual_source": "chronological_pre_calibration_fit",
                         "historical_availability": "explicit_row_clocks_or_disclosed_source_assumption"})
        return cls(settings, history, cutoff, coefficient, mean, scale, blocks, evidence)

    def predict(self, starts, *, cutoff, scenario_count=1, history=None, exogenous=None, seed=None):
        """Predict only after artifact availability, optionally adding later past labels."""
        cutoff = _aware(cutoff, "prediction cutoff")
        starts = _aware_series(starts, "prediction product_start")
        if cutoff < self.fit_cutoff:
            raise ValueError("Price model is not available at the prediction cutoff")
        if not len(starts) or (starts < cutoff).any() or not starts.is_monotonic_increasing or starts.has_duplicates:
            raise ValueError("Prediction targets must be unique increasing intervals at or after cutoff")
        if isinstance(scenario_count, bool) or not isinstance(scenario_count, int) or scenario_count < 1:
            raise ValueError("Price scenario_count must be a positive integer")
        source = self._prepared_history if history is None else prepare_price_history(history, self.settings.timezone)
        cutoffs = pd.DatetimeIndex([cutoff] * len(starts)).as_unit("ns")
        matrix, fallback_count = _features(source, starts, cutoffs, self.settings, exogenous)
        if not np.isfinite(matrix).all():
            raise ValueError("Prediction lacks a causally available lag or exogenous feature; no zero filling")
        point = (matrix[:, 0] if self.settings.method == "seasonal_persistence" else
                 _linear_predict(matrix, self.coefficient, self.mean, self.scale))
        scenarios = np.tile(point, (scenario_count, 1))
        selected_blocks = []
        if scenario_count > 1:
            if not self._residual_blocks:
                raise ValueError("Residual scenarios require complete chronological calibration days")
            generator = np.random.default_rng(self.settings.random_seed if seed is None else seed)
            local = starts.tz_convert(self.settings.timezone)
            days = local.strftime("%Y-%m-%d")
            # Draw one complete residual trajectory per forecast day, never per interval.
            for day in np.unique(days):
                positions = np.flatnonzero(days == day)
                full_day = _day_grid(day, self.settings.interval_minutes, self.settings.timezone,
                                    self.settings.delivery_start_hour, self.settings.delivery_end_hour)
                offsets = full_day.get_indexer(starts[positions])
                if np.any(offsets < 0):
                    raise ValueError("Price targets do not align with configured interval_minutes")
                eligible = [block for block in self._residual_blocks if len(block["starts"]) == len(full_day)]
                if not eligible:
                    if self.settings.residual_dst_policy == "strict_day_length":
                        raise ValueError("No complete calibration residual day matches the target DST day length")
                    # A declared clock-slot approximation duplicates the repeated autumn hour.
                    target_local = starts[positions].tz_convert(self.settings.timezone)
                    target_slots = target_local.hour * 60 + target_local.minute
                    eligible = []
                    for block in self._residual_blocks:
                        source_local = pd.DatetimeIndex(block["starts"]).tz_convert(self.settings.timezone)
                        source_slots = source_local.hour * 60 + source_local.minute
                        first_positions = {int(slot): int(np.flatnonzero(source_slots == slot)[0])
                                           for slot in np.unique(source_slots)}
                        if all(int(slot) in first_positions for slot in target_slots):
                            selected = [first_positions[int(slot)] for slot in target_slots]
                            eligible.append({**block, "residual_eur_per_mwh": np.asarray(block["residual_eur_per_mwh"])[selected].tolist()})
                    if not eligible:
                        raise ValueError("Calibration residual days do not cover all requested local clock slots")
                    offsets = np.arange(len(positions))
                chosen = generator.integers(len(eligible), size=scenario_count)
                residuals = np.array([eligible[index]["residual_eur_per_mwh"] for index in chosen])
                scenarios[:, positions] += residuals[:, offsets]
                selected_blocks.append({"delivery_day": day, "sampled_residual_days": [eligible[index]["day"] for index in chosen]})
        # Detailed snapshot manifests live once in the checksummed model artifact.
        # Repeating them at every MPC cutoff makes diagnostics unnecessarily large.
        evidence = {name: deepcopy(self._provenance[name]) for name in (
            "target", "model", "fit_cutoff_utc", "training_rows", "calibration_rows",
            "complete_calibration_days", "historical_feature_lead_hours", "historical_decision_local_time",
            "historical_decision_day_offset", "training_objective", "residual_source",
            "historical_availability", "study_information", "split", "availability") if name in self._provenance}
        source = self._provenance.get("source", self._provenance)
        source = source if isinstance(source, Mapping) else {}
        evidence["source_availability"] = deepcopy(source.get("availability"))
        if source.get("availability") is not None:
            evidence["availability"] = deepcopy(source["availability"])
        evidence["full_snapshot_lineage"] = "retained_in_model_payload_identified_by_artifact_sha256"
        feature_evidence = self._provenance.get("features") or {}
        feature_evidence = feature_evidence if isinstance(feature_evidence, Mapping) else {}
        evidence["feature_availability"] = [{"name": row.get("name"), "availability": row.get("availability")}
                                            for row in feature_evidence.get("features", [])]
        evidence.update({"prediction_cutoff_utc": cutoff.isoformat(), "model_available_at_utc": self.fit_cutoff.isoformat(),
                         "artifact_sha256": self.artifact_sha256, "scenario_count": scenario_count,
                         "prediction_lag_fallback_count": fallback_count, "sampled_blocks": selected_blocks,
                         "cross_day_residual_dependence": "independent_daily_blocks",
                         "residual_dst_policy": self.settings.residual_dst_policy,
                         "forecast_is_oracle": False})
        return PricePrediction(point, scenarios, cutoff, evidence)

    def _payload(self):
        """Serialize parameters and only historically available retained labels."""
        history = self._history.copy()
        for name in ("product_start", "product_end", "available_at"):
            history[name] = history[name].map(lambda value: value.isoformat())
        return {"schema": "causal-energy-prices-v1", "settings": asdict(self.settings),
                "history": history.to_dict("records"), "fit_cutoff": self.fit_cutoff.isoformat(),
                "coefficient": self.coefficient.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "residual_blocks": self._residual_blocks, "provenance": self._provenance}

    @property
    def artifact_sha256(self):
        """Reuse a digest computed once instead of hashing history every MPC interval."""
        return self._artifact_sha256

    def save(self, path):
        """Write a complete checksummed artifact without Python pickle execution."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        document = {"sha256": _digest(payload), "payload": payload}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        self._artifact_sha256 = document["sha256"]

    @classmethod
    def load(cls, path):
        """Validate artifact content before accepting its historical availability."""
        document = json.loads(Path(path).read_text())
        payload = document["payload"]
        if document.get("sha256") != _digest(payload) or payload.get("schema") != "causal-energy-prices-v1":
            raise ValueError("Price artifact checksum or schema mismatch")
        model = cls(PriceForecastSettings(**payload["settings"]), pd.DataFrame(payload["history"]),
                    payload["fit_cutoff"], payload["coefficient"], payload["mean"], payload["scale"],
                    payload["residual_blocks"], payload["provenance"])
        # Added backward-compatible defaults must not rename an already verified artifact.
        model._artifact_sha256 = document["sha256"]
        return model


def price_forecast_scores(actual, scenarios, weights=None):
    """Report proper empirical CRPS separately from point MAE and trading profit."""
    actual = np.asarray(actual, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    if actual.ndim != 1 or scenarios.ndim != 2 or scenarios.shape[1] != len(actual) or not len(scenarios):
        raise ValueError("Price scores require actual T and scenarios S x T")
    if not np.isfinite(actual).all() or not np.isfinite(scenarios).all():
        raise ValueError("Price scores do not silently drop missing observations")
    weights = np.full(len(scenarios), 1 / len(scenarios)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(scenarios),) or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1):
        raise ValueError("Price scoring weights must be normalized and nonnegative")
    order = np.argsort(scenarios, axis=0, kind="stable")
    sorted_values = np.take_along_axis(scenarios, order, axis=0)
    sorted_weights = weights[order]
    cumulative_weight = np.cumsum(sorted_weights, axis=0) - sorted_weights
    cumulative_value = np.cumsum(sorted_values * sorted_weights, axis=0) - sorted_values * sorted_weights
    half_pairwise = np.sum(sorted_weights * (sorted_values * cumulative_weight - cumulative_value), axis=0)
    crps = weights @ np.abs(scenarios - actual) - half_pairwise
    return {"mae_mean_eur_per_mwh": float(np.mean(np.abs(weights @ scenarios - actual))),
            "crps_eur_per_mwh": float(np.mean(crps)), "observation_count": len(actual)}


def load_price_history(database, *, provider, dataset, zone, price_column="price_eur_mwh",
                       availability_column=None, publication_lag_hours=None,
                       publication_local_time=None, publication_day_offset=-1,
                       timezone="Europe/Rome", interval_minutes=15, start=None, end=None,
                       source_semantics=None, native_resolution_minutes=None):
    """Read verified DuckDB snapshots and apply one explicit availability policy.

    A named publication column is evidence. End-plus-lag and delivery-day clocks
    are assumptions recorded as such, not claims of historically observed release.
    Call only after coordinating with any active downloader; no connections persist.
    """
    from ..data.api_catalog import load_api_cache_records

    if Path(database).suffix != ".duckdb":
        raise ValueError("Price research workflows require the DuckDB snapshot catalog")
    choices = [availability_column is not None, publication_lag_hours is not None,
               publication_local_time is not None]
    if sum(choices) != 1:
        raise ValueError("Declare exactly one publication evidence column or explicit availability assumption")
    if availability_column in {"cached_at", "retrieved_at", "downloaded_at"}:
        raise ValueError("Retrieval time is not historical publication evidence")
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or interval_minutes < 1:
        raise ValueError("Price interval_minutes must be a positive integer")
    records, sources = load_api_cache_records(database, provider=provider, dataset=dataset, zone=zone,
                                               start=start, end=end, verify_checksums=True)
    if records.empty:
        raise ValueError("No price snapshots cover the requested history")
    if price_column not in records:
        raise ValueError(f"Price column {price_column} is missing from catalog records")
    records = records.copy()
    records["product_start"] = _aware_series(records.product_start, "catalog product_start")
    records["product_end"] = _aware_series(records.product_end, "catalog product_end")
    if native_resolution_minutes is not None:
        if (isinstance(native_resolution_minutes, bool) or not isinstance(native_resolution_minutes, int)
                or native_resolution_minutes < 1):
            raise ValueError("Native price resolution must be a positive integer in minutes")
        # Select source products before any expansion, so hourly prices cannot
        # masquerade as independent quarter-hour observations during model fitting.
        duration = (records.product_end - records.product_start).dt.total_seconds() / 60
        records = records.loc[duration == native_resolution_minutes].copy()
        if records.empty:
            raise ValueError("No native price products match the requested resolution")
    if availability_column is not None:
        if availability_column not in records:
            raise ValueError("Declared publication evidence is absent from catalog records")
        records["available_at"] = _aware_series(records[availability_column], "catalog publication")
        availability = {"kind": "explicit_publication_column", "column": availability_column}
    elif publication_lag_hours is not None:
        if isinstance(publication_lag_hours, bool) or not np.isfinite(publication_lag_hours) or publication_lag_hours < 0:
            raise ValueError("Publication lag must be finite and nonnegative")
        records["available_at"] = records.product_end + pd.Timedelta(hours=publication_lag_hours)
        availability = {"kind": "assumed_lag_after_product_end", "lag_hours": publication_lag_hours}
    else:
        if isinstance(publication_day_offset, bool) or not isinstance(publication_day_offset, int):
            raise ValueError("Publication day offset must be an integer")
        local = pd.DatetimeIndex(records.product_start).tz_convert(timezone)
        days = local.date + pd.Timedelta(days=publication_day_offset)
        records["available_at"] = pd.DatetimeIndex([pd.Timestamp(f"{day} {publication_local_time}", tz=timezone)
                                                    for day in days]).tz_convert("UTC")
        availability = {"kind": "assumed_delivery_day_publication_clock", "timezone": timezone,
                        "day_offset": publication_day_offset, "local_time": publication_local_time}
    # Native hourly prices are constant within their traded product; never interpolate prices.
    duration = (records.product_end - records.product_start).dt.total_seconds().to_numpy() / 60
    counts = duration / interval_minutes
    if not np.isfinite(counts).all() or np.any(counts < 1) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Native price products must contain a whole number of requested intervals")
    counts = counts.astype(int)
    parents = np.repeat(np.arange(len(records)), counts)
    offsets = np.arange(len(parents)) - np.repeat(np.cumsum(counts) - counts, counts)
    first = pd.DatetimeIndex(records.product_start.iloc[parents]) + pd.to_timedelta(offsets * interval_minutes, unit="min")
    history = pd.DataFrame({"product_start": first, "product_end": first + pd.Timedelta(minutes=interval_minutes),
        "available_at": records.available_at.iloc[parents].to_numpy(),
        "price_eur_per_mwh": records[price_column].iloc[parents].to_numpy()}).drop_duplicates()
    history = validate_price_history(history)
    provenance = {"database": str(Path(database).resolve()), "provider": provider, "dataset": dataset,
                  "zone": zone, "snapshot_sources": sources, "availability": availability,
                  "source_semantics": source_semantics,
                  "native_source_products": len(records),
                  "requested_native_resolution_minutes": native_resolution_minutes,
                  "native_product_counts_by_minutes": {str(int(minutes)): int(np.count_nonzero(duration == minutes))
                                                        for minutes in np.unique(duration)},
                  "expanded_intervals_are_not_independent_native_observations": True,
                  "snapshot_revision_semantics": "pinned_current_snapshots_not_historical_asof_reconstruction",
                  "native_product_conversion": "constant_price_within_each_native_product"}
    return history, provenance
