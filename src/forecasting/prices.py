"""Causal calendar/lag regressions and out-of-sample conditional innovations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.market_prices import MarketDataError, available_records, import_market_csv
from ..markets import AuctionCalendar

PRECEDING = {"IDA1": "DA", "IDA2": "IDA1", "IDA3": "IDA2"}
FEATURES = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "lag_1d", "lag_7d")


@dataclass(frozen=True)
class RidgePriceModel:
    """A scaler and ridge coefficients fitted solely on eligible training rows."""
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    fit_cutoff: str
    training_rows: int
    target: str = "DA_level"

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = features.loc[:, list(self.feature_names)].to_numpy(float) if isinstance(features, pd.DataFrame) else np.asarray(features, float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or not np.isfinite(values).all():
            raise ValueError("Price predictors must be finite and match fitted feature columns")
        return (values - self.center) / self.scale @ self.coefficients + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {"feature_names": list(self.feature_names), "center": self.center.tolist(), "scale": self.scale.tolist(),
                "coefficients": self.coefficients.tolist(), "intercept": self.intercept, "alpha": self.alpha,
                "fit_cutoff": self.fit_cutoff, "training_rows": self.training_rows, "target": self.target}


def fit_ridge(features: pd.DataFrame, targets: Sequence[float], *, feature_available_at: Sequence[Any],
              target_available_at: Sequence[Any], fit_cutoff: Any, alpha: float = 1.0,
              target: str = "DA_level") -> RidgePriceModel:
    """Fit a numerically regularized linear model after publication-cutoff checks."""
    cutoff = pd.Timestamp(fit_cutoff)
    if cutoff.tzinfo is None or alpha < 0 or not np.isfinite(alpha):
        raise ValueError("Ridge fit requires an aware cutoff and nonnegative finite alpha")
    feature_times = pd.DatetimeIndex(feature_available_at)
    target_times = pd.DatetimeIndex(target_available_at)
    if feature_times.tz is None or target_times.tz is None:
        raise ValueError("Feature and target availability must be timezone-aware")
    if len(features) != len(targets) or len(features) != len(feature_times) or len(features) != len(target_times):
        raise ValueError("Training arrays and availability timestamps must align")
    if feature_times.isna().any() or target_times.isna().any() or (feature_times > cutoff).any() or (target_times > cutoff).any():
        raise ValueError("Unavailable future features/labels cannot enter price training")
    values = features.to_numpy(float)
    y = np.asarray(targets, float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all() or not np.isfinite(y).all():
        raise ValueError("Training prices/features must be nonempty and finite")
    # Normalize using the training slice only; ridge never penalizes the intercept.
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (values - center) / scale
    intercept = float(y.mean())
    design = np.vstack([normalized, np.sqrt(alpha) * np.eye(values.shape[1])])
    response = np.concatenate([y - intercept, np.zeros(values.shape[1])])
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    return RidgePriceModel(tuple(features.columns), center, scale, coefficients, intercept,
                           alpha, cutoff.isoformat(), len(features), target)


def _clock_key(value: Any, timezone: str) -> tuple[Any, int, int, int]:
    """Join by local calendar date/product and repeated-hour fold, not a 96-row lag."""
    local = pd.Timestamp(value).to_pydatetime().astimezone(ZoneInfo(timezone))
    return local.date(), local.hour, local.minute, local.fold


def build_price_features(records: pd.DataFrame, auction: str, calendar: AuctionCalendar) -> pd.DataFrame:
    """Build causal minimal features; fundamentals are explicitly absent.

    One pooled regression predicts a level for DA and spreads relative to the
    immediately preceding auction for IDAs. Same-clock day/week lags join actual
    calendar products. Unmatched repeated DST hours stay missing and are omitted
    from training rather than borrowed from another product.
    """
    if auction not in ("DA", "IDA1", "IDA2", "IDA3"):
        raise ValueError("Unknown auction target")
    frame = records.copy()
    keys = list(zip(frame.auction, frame.product_start, frame.product_end))
    by_product: dict[Any, list[int]] = {}
    by_clock: dict[Any, list[int]] = {}
    for index, key in enumerate(keys):
        by_product.setdefault(key, []).append(index)
        by_clock.setdefault((key[0], *_clock_key(key[1], calendar.timezone)), []).append(index)
    def latest(indices: list[int], cutoff: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
        eligible = frame.iloc[indices]
        eligible = eligible.loc[eligible.published_at <= cutoff]
        if eligible.empty:
            return None
        maximum = eligible.published_at.max()
        chosen = eligible.loc[eligible.published_at == maximum]
        if len(chosen) > 1:
            raise MarketDataError("Ambiguous product revision publication; normalize revision ordering first")
        item = chosen.iloc[0]
        return float(item.price_eur_mwh), item.published_at
    rows = []
    for row in frame.loc[frame.auction == auction].itertuples():
        day, hour, minute, fold = _clock_key(row.product_start, calendar.timezone)
        if not pd.Timestamp(calendar.valid_from).date() <= day <= pd.Timestamp(calendar.valid_to).date():
            continue
        event = next(item for item in calendar.schedule(day) if item.auction == auction)
        cutoff = event.submission_at
        clock = hour + minute / 60
        feature = {"hour_sin": np.sin(2 * np.pi * clock / 24), "hour_cos": np.cos(2 * np.pi * clock / 24),
            "weekday_sin": np.sin(2 * np.pi * day.weekday() / 7), "weekday_cos": np.cos(2 * np.pi * day.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * day.month / 12), "month_cos": np.cos(2 * np.pi * day.month / 12)}
        available = [cutoff]
        for lag in (1, 7):
            candidates = by_clock.get((auction, day - timedelta(days=lag), hour, minute, fold), [])
            value = latest(candidates, cutoff) if candidates else None
            feature[f"lag_{lag}d"] = np.nan if value is None else value[0]
            if value is not None:
                available.append(value[1])
        previous = None
        if auction in PRECEDING:
            candidates = by_product.get((PRECEDING[auction], row.product_start, row.product_end), [])
            previous = latest(candidates, cutoff) if candidates else None
            feature["previous_auction_price"] = np.nan if previous is None else previous[0]
            if previous is not None:
                available.append(previous[1])
        target = float(row.price_eur_mwh)
        if auction in PRECEDING:
            target = np.nan if previous is None else target - previous[0]
        rows.append({**feature, "target": target, "product_start": row.product_start,
                     "product_end": row.product_end, "delivery_day": day.isoformat(), "decision_cutoff": cutoff,
                     "feature_available_at": max(available), "target_available_at": row.published_at,
                     "auction": auction, "revision": row.revision})
    return pd.DataFrame(rows)


def rolling_origin_residuals(rows: pd.DataFrame, *, feature_names: Sequence[str],
                              min_training_rows: int = 96, alpha: float = 1.0) -> pd.DataFrame:
    """Obtain genuine out-of-sample errors from expanding chronological daily fits."""
    outputs = []
    for day in sorted(rows.delivery_day.unique()):
        validation = rows.loc[rows.delivery_day == day]
        cutoff = validation.decision_cutoff.min()
        train = rows.loc[(rows.delivery_day < day) & (rows.target_available_at <= cutoff)]
        train = train.dropna(subset=[*feature_names, "target"])
        validation = validation.dropna(subset=[*feature_names, "target"])
        if len(train) < min_training_rows or validation.empty:
            continue
        model = fit_ridge(train.loc[:, list(feature_names)], train.target,
            feature_available_at=train.feature_available_at, target_available_at=train.target_available_at,
            fit_cutoff=cutoff, alpha=alpha, target=str(validation.auction.iloc[0]))
        prediction = model.predict(validation.loc[:, list(feature_names)])
        output = validation.loc[:, ["auction", "delivery_day", "product_start", "product_end", "target_available_at"]].copy()
        output["prediction"] = prediction
        output["residual"] = validation.target.to_numpy() - prediction
        output["fit_cutoff"] = cutoff
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


@dataclass(frozen=True)
class ResidualDistribution:
    """Whole-vector residual calibration from out-of-sample errors only."""
    vectors: np.ndarray
    calibration_cutoff: str
    covariance_shrinkage: float = 0.2
    provenance: str = "rolling_origin_out_of_sample"

    def __post_init__(self) -> None:
        values = np.asarray(self.vectors, float)
        if values.ndim != 2 or values.shape[0] < 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Need at least two complete finite out-of-sample residual vectors")
        if not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("Covariance shrinkage must lie in [0,1]")
        if self.provenance != "rolling_origin_out_of_sample":
            raise ValueError("In-sample regression errors are not scenario calibration data")
        if pd.Timestamp(self.calibration_cutoff).tzinfo is None:
            raise ValueError("Residual calibration cutoff must be timezone-aware")

    def sample(self, count: int, *, method: str = "gaussian", seed: int | None = None,
               random: np.random.Generator | None = None) -> np.ndarray:
        """Draw entire temporal vectors; bootstrap never resamples individual hours."""
        if count < 1:
            raise ValueError("Residual sample count must be positive")
        rng = random if random is not None else np.random.default_rng(seed)
        values = np.asarray(self.vectors, float)
        if method == "bootstrap":
            return values[rng.integers(0, len(values), size=count)].copy()
        if method not in ("gaussian", "independent_gaussian"):
            raise ValueError("Residual method must be gaussian/bootstrap/independent_gaussian")
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        diagonal = np.diag(np.diag(covariance))
        shrink = 1.0 if method == "independent_gaussian" else self.covariance_shrinkage
        covariance = (1 - shrink) * covariance + shrink * diagonal
        return rng.multivariate_normal(values.mean(axis=0), covariance, size=count)


def calibrate_residual_vectors(residuals: pd.DataFrame, *, calibration_cutoff: Any,
                               expected_products: int, covariance_shrinkage: float = 0.2) -> ResidualDistribution:
    """Keep complete equal-length product days known by the calibration cutoff.

    DST days must be calibrated separately by product count. This deliberately
    avoids interpolating residuals across different product regimes.
    """
    cutoff = pd.Timestamp(calibration_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Calibration cutoff requires a timezone")
    eligible = residuals.loc[residuals.target_available_at <= cutoff]
    vectors = []
    for _, day in eligible.groupby("delivery_day", sort=True):
        if len(day) == expected_products and not day.product_start.duplicated().any():
            vectors.append(day.sort_values("product_start").residual.to_numpy(float))
    if len(vectors) < 2:
        raise MarketDataError("Insufficient complete out-of-sample residual days for requested product grid")
    return ResidualDistribution(np.stack(vectors), cutoff.isoformat(), covariance_shrinkage)


def train_price_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Train manual-input price models and save actual OOS calibration evidence."""
    settings = config.get("price_forecasting", {})
    path = settings.get("input_path", config.get("market_data", {}).get("output_path", "data/market/canonical_prices.csv"))
    if not Path(path).is_file():
        raise MarketDataError(f"Price training blocked: {path} missing; import actual zonal MGP/MI-A price labels first")
    if not settings.get("fit_cutoff"):
        raise MarketDataError("price_forecasting.fit_cutoff must be explicit and timezone-aware")
    cutoff = pd.Timestamp(settings["fit_cutoff"])
    records = available_records(import_market_csv(path), cutoff)
    records = records.loc[records.zone == config.get("market", {}).get("zone", "NORD")].reset_index(drop=True)
    for field in ("provider", "dataset"):
        if settings.get(field):
            records = records.loc[records[field] == settings[field]].reset_index(drop=True)
    if records.empty:
        raise MarketDataError("No eligible zonal price labels before fit_cutoff for the selected provider/dataset")
    if len(records[["provider", "dataset"]].drop_duplicates()) > 1:
        raise MarketDataError("Multiple market provider/dataset sources: explicitly select price_forecasting.provider and dataset")
    calendar = AuctionCalendar.from_config(config)
    output = Path(settings.get("output_directory", "outputs/price_models"))
    output.mkdir(parents=True, exist_ok=True)
    input_provenance = {"providers": sorted(records.provider.unique().tolist()),
                        "datasets": sorted(records.dataset.unique().tolist()),
                        "source_status": settings.get("source_status", "imported_price_records_user_supplied_provenance")}
    result = {"status": "trained", "fit_cutoff": cutoff.isoformat(), "models": {},
              "input_provenance": input_provenance,
              "omitted_features": ["load", "regional_wind", "regional_solar", "imports", "forecast_revisions"],
              "model": "pooled regularized linear DA level / successive IDA spread models"}
    for auction in settings.get("auctions", ["DA", "IDA1", "IDA2", "IDA3"]):
        rows = build_price_features(records, auction, calendar)
        if rows.empty:
            result["models"][auction] = {"status": "blocked", "reason": "no eligible auction labels"}
            continue
        features = list(FEATURES) + (["previous_auction_price"] if auction in PRECEDING else [])
        eligible = rows.dropna(subset=[*features, "target"])
        minimum = int(settings.get("minimum_training_rows", 96))
        if len(eligible) < minimum:
            result["models"][auction] = {"status": "blocked", "reason": "insufficient lag-complete training rows", "rows": len(eligible)}
            continue
        model = fit_ridge(eligible[features], eligible.target, feature_available_at=eligible.feature_available_at,
            target_available_at=eligible.target_available_at, fit_cutoff=cutoff,
            alpha=float(settings.get("ridge_alpha", 1.0)), target=f"{auction}_spread" if auction in PRECEDING else "DA_level")
        residuals = rolling_origin_residuals(rows, feature_names=features, min_training_rows=minimum,
                                           alpha=float(settings.get("ridge_alpha", 1.0)))
        model_record = {**model.to_dict(), "input_provenance": input_provenance}
        (output / f"{auction}_model.json").write_text(json.dumps(model_record, indent=2) + "\n")
        residuals.to_csv(output / f"{auction}_oos_residuals.csv", index=False)
        result["models"][auction] = {"status": "trained", "training_rows": len(eligible), "omitted_missing_lag_rows": len(rows) - len(eligible),
            "oos_residual_rows": len(residuals), "residual_calibration": "available" if len(residuals) else "blocked_insufficient_chronological_history"}
    if not any(item["status"] == "trained" for item in result["models"].values()):
        result["status"] = "blocked"
    (output / "training_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
