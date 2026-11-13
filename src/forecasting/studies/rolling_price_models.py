"""Chronological whole-curve price models for the rolling auction study.

Data examples are immutable dictionaries supplied by ``rolling_price_data``.
Their features must already represent information available at the example's
gate.  This module controls label publication, windowing and model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class InsufficientHistory(ValueError):
    """The published history does not support a valid chronological fit."""


@dataclass
class FittedCurve:
    """A selected curve estimator and the provenance of its training set."""

    estimator: Any
    market: str
    gate: str
    mode: str
    resolution: int
    feature_blocks: tuple[str, ...]
    parameters: dict[str, Any]
    train_days: tuple[str, ...]
    validation_days: tuple[str, ...]
    validation_mae: float
    metadata: dict[str, Any]


@dataclass
class DailyModels:
    """All selected heads fitted once at the delivery day's DA cutoff."""

    delivery_day: str
    cutoff: str
    models: dict[tuple[str, str], FittedCurve]
    unavailable: dict[str, str]
    variant: str


def _utc(value: Any) -> pd.Timestamp:
    """Reject ambiguous timestamps rather than guessing their timezone."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Timestamp requires timezone: {value!r}")
    return stamp.tz_convert("UTC")


def _local_slots(starts: list[Any], market: str, resolution: int, timezone: str) -> np.ndarray:
    """Map actual UTC products to their ordinary local-clock curve positions."""
    local = pd.DatetimeIndex(pd.to_datetime(starts, utc=True)).tz_convert(timezone)
    first_hour = 12 if market == "IDA3" else 0
    if resolution == 60:
        if market != "DA":
            raise ValueError("Only DA supports an hourly bootstrap")
        return np.asarray(local.hour, dtype=int)
    if resolution != 15:
        raise ValueError(f"Unsupported model resolution: {resolution}")
    return np.asarray((local.hour - first_hour) * 4 + local.minute // 15, dtype=int)


def _full_curve(example: dict, resolution: int, timezone: str, name: str) -> np.ndarray | None:
    """Return a complete ordinary-day target; exclude incomplete and DST labels."""
    values = np.asarray([np.nan if value is None else value for value in example[name]], dtype=float)
    starts = example["product_starts_utc"]
    if len(values) != len(starts):
        raise ValueError(f"{name} and product starts have different lengths")
    market = example["market"]
    width = 24 if resolution == 60 else 48 if market == "IDA3" else 96
    # A bootstrap training example must contain native hourly DA observations.
    if int(example["metadata"]["native_minutes"]) != resolution:
        return None
    slots = _local_slots(starts, market, resolution, timezone)
    if len(slots) != width or not np.array_equal(np.sort(slots), np.arange(width)):
        return None
    if not np.isfinite(values).all():
        return None
    result = np.empty(width, dtype=float)
    result[slots] = values
    return result


def _project(curve: np.ndarray, starts: list[Any], market: str, resolution: int, timezone: str) -> np.ndarray:
    """Project the ordinary curve to actual products, repeating autumn folds."""
    slots = _local_slots(starts, market, resolution, timezone)
    if (slots < 0).any() or (slots >= len(curve)).any():
        raise ValueError("Requested product is outside the modeled delivery window")
    return np.asarray(curve)[slots]


def _features(example: dict, blocks: tuple[str, ...], da_prediction: np.ndarray | None = None) -> np.ndarray:
    """Flatten approved blocks, with explicit out-of-sample DA conditioning."""
    arrays = []
    for name in blocks:
        if name not in example["feature_blocks"]:
            raise ValueError(f"Missing feature block {name!r}")
        values = example["feature_blocks"][name]
        arrays.append(np.asarray([np.nan if value is None else value for value in values], dtype=float))
    if da_prediction is not None:
        arrays.append(np.asarray(da_prediction, dtype=float))
    return np.concatenate(arrays)


def _da_context(example: dict, prediction: np.ndarray, timezone: str) -> np.ndarray:
    """Represent DA forecasts on fixed local-clock slots without using labels."""
    width = 48 if example["market"] == "IDA3" else 96
    slots = _local_slots(example["product_starts_utc"], example["market"], 15, timezone)
    if len(slots) != len(prediction) or (slots < 0).any() or (slots >= width).any():
        raise ValueError("DA prediction and IDA product grid do not match")
    result = np.full(width, np.nan)
    for slot in np.unique(slots):
        result[slot] = np.mean(prediction[slots == slot])
    return result


def _estimator(family: str, parameter: float, sample_count: int) -> Any:
    """Create one deterministic estimator with training-only preprocessing."""
    if family == "ridge":
        model = Ridge(alpha=parameter)
    elif family == "elastic_net":
        model = MultiTaskElasticNet(alpha=parameter, l1_ratio=0.5, max_iter=10000,
                                    tol=1e-4, random_state=0)
    elif family == "reduced_rank":
        components = int(parameter)
        if components >= sample_count:
            raise ValueError("PLS rank must be smaller than training sample count")
        model = PLSRegression(n_components=components, scale=False, max_iter=500)
    else:
        raise ValueError(f"Unknown rolling model family: {family}")
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), model)


class RollingModelEngine:
    """Fit gate-specific daily curve heads from strictly published history.

    ``settings`` accepts ``variant``, ``feature_blocks`` (variant to block list),
    ``family``, ``window_days``, ``ida_target``, ``validation_days`` and
    ``minimum_fit_days``. DA-gate IDA samples acquire their DA feature from
    historical forecasts generated strictly before that historical DA gate.
    """

    def __init__(self, examples: list[dict], settings: dict):
        self.examples = list(examples)
        self.settings = dict(settings)
        self.variant = self.settings.get("variant", "core")
        configured = self.settings.get("feature_blocks", {"core": ["core"]})
        self.blocks = tuple(configured[self.variant])
        self.family = self.settings.get("family", "ridge")
        self.window = int(self.settings.get("window_days", 90))
        if self.window not in (60, 90):
            raise ValueError("Rolling window must be 60 or 90 calendar days")
        self.mode = self.settings.get("ida_target", "spread")
        if self.mode not in ("price", "spread"):
            raise ValueError("IDA target must be price or spread")
        self.timezone = self.settings.get("timezone", "Europe/Rome")
        self.validation_days = int(self.settings.get("validation_days", 14))
        self.minimum_fit = int(self.settings.get("minimum_fit_days", 14))
        self.minimum_native = int(self.settings.get("minimum_native_da_days", 30))
        self._da_cache: dict[tuple[str, str], dict[str, float] | None] = {}

    def _resolution(self, day: date, cutoff: pd.Timestamp) -> int:
        """Switch from hourly bootstrap after 30 published normal native days."""
        first = day - timedelta(days=self.window)
        eligible = {str(e["delivery_day"]) for e in self.examples
            if e["market"] == "DA" and e["gate"] == "DA"
            and first <= date.fromisoformat(str(e["delivery_day"])) < day
            and _utc(e["target_publication_at"]) <= cutoff
            and int(e["metadata"]["native_minutes"]) == 15
            and _full_curve(e, 15, self.timezone, "target_values") is not None}
        return 15 if len(eligible) >= self.minimum_native else 60

    def _historical_da(self, day: str) -> dict[str, float] | None:
        """Generate one DA-gate forecast without accessing that day's DA label."""
        key = (day, self.variant)
        if key in self._da_cache:
            return self._da_cache[key]
        examples = [e for e in self.examples if e["market"] == "DA" and e["gate"] == "DA"
                    and str(e["delivery_day"]) == day]
        if len(examples) != 1:
            self._da_cache[key] = None
            return None
        example = examples[0]
        cutoff = _utc(example["decision_at"])
        try:
            fitted = self._fit_head(day, cutoff, "DA", "DA", None)
            forecast = self._forecast_curve(fitted, example, None)
        except InsufficientHistory:
            self._da_cache[key] = None
            return None
        self._da_cache[key] = dict(zip(example["product_starts_utc"], forecast.tolist()))
        return self._da_cache[key]

    def _aligned_historical_da(self, example: dict) -> np.ndarray | None:
        """Use only previously generated DA forecasts, expanding native hourly prices."""
        generated = self._historical_da(str(example["delivery_day"]))
        if not generated:
            return None
        source = pd.Series(generated, dtype=float)
        source.index = pd.to_datetime(source.index, utc=True)
        target = pd.DatetimeIndex(pd.to_datetime(example["product_starts_utc"], utc=True))
        da_example = next(e for e in self.examples if e["market"] == "DA" and e["gate"] == "DA"
                          and str(e["delivery_day"]) == str(example["delivery_day"]))
        # Only an hourly DA product is repeated to its constituent quarters.
        desired = target.floor("h") if int(da_example["metadata"]["native_minutes"]) == 60 else target
        positions = source.index.get_indexer(desired)
        if (positions < 0).any():
            return None
        return source.to_numpy()[positions]

    def _candidates(self) -> list[float]:
        """Use a small predetermined set, independent of the evaluation period."""
        defaults = {"ridge": [100.0, 10000.0], "elastic_net": [0.01, 0.1],
                    "reduced_rank": [2, 4]}
        return list(self.settings.get("parameters", {}).get(self.family, defaults[self.family]))

    def _fit_head(self, day: str, cutoff: pd.Timestamp, market: str, gate: str,
                  resolution: int | None) -> FittedCurve:
        """Select with earlier days and refit on the published rolling window."""
        delivery = date.fromisoformat(day)
        resolution = self._resolution(delivery, cutoff) if market == "DA" else 15
        first = delivery - timedelta(days=self.window)
        records = []
        missing_weather_days = []
        for example in self.examples:
            example_day = date.fromisoformat(str(example["delivery_day"]))
            if example["market"] != market or example["gate"] != gate:
                continue
            if not first <= example_day < delivery or _utc(example["target_publication_at"]) > cutoff:
                continue
            # A missing national weather archive must never become a fabricated
            # weather signal through the general input imputer.
            if "weather" in self.blocks and not example.get("metadata", {}).get("weather_usable", False):
                missing_weather_days.append(str(example_day))
                continue
            target = _full_curve(example, resolution, self.timezone, "target_values")
            if target is None:
                continue
            da = None
            if gate == "DA" and market != "DA":
                da = self._aligned_historical_da(example)
                if da is None:
                    continue
            features = _features(example, self.blocks,
                                 None if da is None else _da_context(example, da, self.timezone))
            mode = "price" if market == "DA" else self.mode
            if mode == "spread":
                if da is not None:
                    base = da
                else:
                    base = _full_curve(example, resolution, self.timezone, "price_reference")
                if base is None or len(base) != len(target) or not np.isfinite(base).all():
                    continue
                target = target - base
            records.append((str(example_day), _utc(example["decision_at"]),
                            _utc(example["target_publication_at"]), features, target))
        records.sort(key=lambda item: (item[0], item[1]))
        if len({r[0] for r in records}) != len(records):
            raise ValueError(f"Duplicate training day for {market}/{gate}")
        if len(records) < self.minimum_fit + self.validation_days:
            raise InsufficientHistory(f"{market}/{gate}: {len(records)} complete eligible days; "
                                      f"need {self.minimum_fit + self.validation_days}")
        validation = records[-self.validation_days:]
        first_validation_cutoff = validation[0][1]
        training = [r for r in records[:-self.validation_days] if r[2] <= first_validation_cutoff]
        if len(training) < self.minimum_fit:
            raise InsufficientHistory(f"{market}/{gate}: insufficient published inner training labels")
        lengths = {len(r[3]) for r in records}
        if len(lengths) != 1:
            raise ValueError(f"{market}/{gate}: inconsistent feature widths {sorted(lengths)}")
        train_x, train_y = np.stack([r[3] for r in training]), np.stack([r[4] for r in training])
        valid_x, valid_y = np.stack([r[3] for r in validation]), np.stack([r[4] for r in validation])
        scores = []
        with threadpool_limits(limits=1):
            for parameter in self._candidates():
                try:
                    estimator = _estimator(self.family, parameter, len(training))
                    estimator.fit(train_x, train_y)
                    score = float(np.abs(estimator.predict(valid_x) - valid_y).mean())
                    if np.isfinite(score):
                        scores.append((score, float(parameter)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if not scores:
                raise InsufficientHistory(f"{market}/{gate}: no configuration could be fitted")
            score, parameter = min(scores)
            estimator = _estimator(self.family, parameter, len(records))
            estimator.fit(np.stack([r[3] for r in records]), np.stack([r[4] for r in records]))
        return FittedCurve(estimator, market, gate, "price" if market == "DA" else self.mode,
            resolution, self.blocks, {"family": self.family, "value": parameter},
            tuple(r[0] for r in records), tuple(r[0] for r in validation), score,
            {"window_days": self.window, "window_start": first.isoformat(),
             "eligible_days": len(records), "hourly_da_bootstrap": market == "DA" and resolution == 60,
             "da_conditioning": "rolling_out_of_sample" if gate == "DA" and market != "DA" else None,
             "dst_training_policy": "exclude nonstandard or incomplete target days",
             "excluded_missing_weather_days": missing_weather_days})

    def fit_delivery_day(self, delivery_day: str, cutoff: Any) -> DailyModels:
        """Fit every head once; unavailable heads are recorded explicitly."""
        cutoff = _utc(cutoff)
        current_gates = [_utc(e["decision_at"]) for e in self.examples
                         if str(e["delivery_day"]) == delivery_day and e["gate"] == "DA"]
        if current_gates and cutoff > min(current_gates):
            raise ValueError("Daily fit cutoff is after the day's DA decision")
        models, unavailable = {}, {}
        heads = [("DA", "DA")]
        heads.extend((market, gate) for market in ("IDA1", "IDA2", "IDA3") for gate in ("DA", market))
        for market, gate in heads:
            try:
                models[(market, gate)] = self._fit_head(delivery_day, cutoff, market, gate, None)
            except InsufficientHistory as error:
                unavailable[f"{market}:{gate}"] = str(error)
        return DailyModels(delivery_day, cutoff.isoformat(), models, unavailable, self.variant)

    def _forecast_curve(self, fitted: FittedCurve, example: dict, da_prediction: np.ndarray | None) -> np.ndarray:
        """Forecast actual product starts without reading the example's label."""
        starts = example["product_starts_utc"]
        if "weather" in fitted.feature_blocks and not example.get("metadata", {}).get("weather_usable", False):
            raise ValueError("Weather feature coverage is incomplete for this gate and day")
        if fitted.market != "DA" and fitted.gate == "DA":
            if da_prediction is None or len(da_prediction) != len(starts) or not np.isfinite(da_prediction).all():
                raise ValueError("DA-gate IDA forecast requires target-aligned predicted DA prices")
        features = _features(example, fitted.feature_blocks,
                             None if da_prediction is None else _da_context(example, da_prediction, self.timezone))
        with threadpool_limits(limits=1):
            ordinary = np.asarray(fitted.estimator.predict(features[None, :])).reshape(-1)
        result = _project(ordinary, starts, fitted.market, fitted.resolution, self.timezone)
        if fitted.mode == "spread":
            base = da_prediction if fitted.gate == "DA" else np.asarray(example["price_reference"], dtype=float)
            if len(base) != len(result) or not np.isfinite(base).all():
                raise ValueError("A finite published price reference is required for spread prediction")
            result = result + base
        return result

    def predict(self, fitted: DailyModels, example: dict, da_prediction: Any = None) -> dict:
        """Return a gate curve and audit metadata without reading target prices."""
        if str(example["delivery_day"]) != fitted.delivery_day:
            raise ValueError("Daily model and requested delivery day differ")
        market, gate = example["market"], example["gate"]
        model = fitted.models.get((market, gate))
        if model is None:
            raise InsufficientHistory(fitted.unavailable.get(f"{market}:{gate}", "Model head was not fitted"))
        da = None if da_prediction is None else np.asarray(da_prediction, dtype=float)
        result = self._forecast_curve(model, example, da)
        reference = da.tolist() if da is not None and gate == "DA" and market != "DA" else example.get("price_reference")
        return {"delivery_day": fitted.delivery_day, "auction": market, "gate": gate,
            "variant": fitted.variant, "product_starts_utc": list(example["product_starts_utc"]),
            "prediction": result.tolist(), "reference": reference,
            "forecast_issue_at": str(example["decision_at"]),
            "model": {"family": model.parameters["family"], "parameter": model.parameters["value"],
                      "mode": model.mode, "resolution_minutes": model.resolution},
            "metadata": {**model.metadata, "fit_cutoff": fitted.cutoff,
                "training_days": list(model.train_days), "validation_days": list(model.validation_days),
                "validation_mae": model.validation_mae, "feature_blocks": list(model.feature_blocks),
                "weather": example.get("metadata", {}).get("weather_vintages"),
                "feature_block_coverage": example.get("metadata", {}).get("feature_block_coverage"),
                "source_manifest_sha256": example.get("metadata", {}).get("source_manifest_sha256"),
                "dst_prediction_policy": "reuse ordinary local-clock output for repeated products"}}
