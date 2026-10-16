"""Audited Gaussian price paths calibrated from chronological forecast errors.

The point forecast, a separately estimated bias correction, and zero-mean
Gaussian innovations are distinct quantities. Native product grids and DST
folds remain intact; this module never expands hourly observed prices into
quarter-hour labels. Retrospective studies cannot silently become operational
forecasts when historical publication vintages are unverified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import json
from math import erf, pi, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SCHEMA = "gaussian-price-calibration-v1"
SCOPED_SCHEMA = "gaussian-price-calibration-v2"
OOS_PROVENANCE = "chronological_out_of_sample"
AUCTIONS = {"DA", "IDA1", "IDA2", "IDA3"}


def _aware_index(values: Sequence[Any], label: str) -> pd.DatetimeIndex:
    """Normalize aware timestamps without accepting an implicit naive timezone."""
    timestamps = [pd.Timestamp(value) for value in values]
    if any(pd.isna(value) or value.tzinfo is None for value in timestamps):
        raise ValueError(f"{label} must contain finite timezone-aware timestamps")
    return pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")


def _cutoff(value: Any, label: str) -> pd.Timestamp:
    """Require an explicit timezone for all information cutoffs."""
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{label} must be a finite timezone-aware timestamp")
    return result.tz_convert("UTC")


def _slot_labels(starts: pd.DatetimeIndex, timezone: str) -> tuple[str, ...]:
    """Label repeated autumn clock times by their actual timezone fold."""
    zone = ZoneInfo(timezone)
    local = [value.to_pydatetime().astimezone(zone) for value in starts]
    return tuple(f"{value.hour:02d}:{value.minute:02d}/fold{value.fold}" for value in local)


def _daily_grid(day: str, minutes: int, timezone: str, auction: str = "DA") -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Construct the native civil-day grid, including 23/25-hour DST days."""
    start = pd.Timestamp(day).tz_localize(timezone)
    following_day = (pd.Timestamp(day).date() + timedelta(days=1)).isoformat()
    end = pd.Timestamp(following_day).tz_localize(timezone)
    starts = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"),
                           freq=f"{minutes}min", inclusive="left")
    starts = starts.as_unit("ns")
    if auction == "IDA3":
        if minutes != 15 or timezone != "Europe/Rome":
            raise ValueError("IDA3 Gaussian templates require native quarter-hours in Europe/Rome")
        starts = starts[starts.tz_convert(timezone).hour >= 12]
    return starts, starts + pd.Timedelta(minutes=minutes)


def _grid_key(minutes: int, labels: Sequence[str], auction: str = "DA") -> str:
    """Use a readable, unambiguous key for each native civil-day layout."""
    repeated = any(label.endswith("fold1") for label in labels)
    base = f"{minutes}min_{len(labels)}products_{'repeated' if repeated else 'ordinary'}"
    return base if auction == "DA" else f"{auction}_{base}"


def _normal_crps(error: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute the closed Gaussian CRPS; smaller values are better.

    Formula: sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)].
    Reference: https://arxiv.org/abs/1709.04743 (scoringRules).
    """
    z = error / sigma
    cdf = 0.5 * (1 + np.fromiter((erf(value / sqrt(2)) for value in z), float, count=len(z)))
    density = np.exp(-0.5 * z ** 2) / sqrt(2 * pi)
    return sigma * (z * (2 * cdf - 1) + 2 * density - 1 / sqrt(pi))


@dataclass(frozen=True)
class GaussianGridCalibration:
    """A separate bias and temporal error covariance for one native day grid."""
    native_interval_minutes: int
    slot_labels: tuple[str, ...]
    bias_eur_mwh: np.ndarray
    covariance_eur2_mwh2: np.ndarray
    calibration_days: int
    first_delivery_day: str
    last_delivery_day: str
    auction: str = "DA"

    def __post_init__(self) -> None:
        """Reject malformed or nonpositive predictive covariance artifacts."""
        count = len(self.slot_labels)
        if self.auction not in AUCTIONS:
            raise ValueError("Gaussian calibration requires an explicit DA/IDA1/IDA2/IDA3 auction")
        bias = np.array(self.bias_eur_mwh, dtype=float, copy=True)
        covariance = np.array(self.covariance_eur2_mwh2, dtype=float, copy=True)
        if self.native_interval_minutes not in (15, 30, 60) or count < 1 or len(set(self.slot_labels)) != count:
            raise ValueError("Gaussian calibration requires unique native product slot labels")
        if bias.shape != (count,) or covariance.shape != (count, count):
            raise ValueError("Gaussian bias/covariance dimensions must match the product grid")
        if not np.isfinite(bias).all() or not np.isfinite(covariance).all():
            raise ValueError("Gaussian bias and covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Gaussian covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Gaussian covariance must be positive definite") from exc
        if self.calibration_days < 2 or self.first_delivery_day > self.last_delivery_day:
            raise ValueError("Gaussian covariance requires at least two ordered calibration days")
        if self.auction == "IDA3":
            expected, _ = _daily_grid(self.first_delivery_day, self.native_interval_minutes, "Europe/Rome", self.auction)
            if self.slot_labels != _slot_labels(expected, "Europe/Rome"):
                raise ValueError("IDA3 Gaussian artifacts require the complete 48-product noon-to-midnight template")
        bias.flags.writeable = False
        covariance.flags.writeable = False
        object.__setattr__(self, "bias_eur_mwh", bias)
        object.__setattr__(self, "covariance_eur2_mwh2", covariance)

    @property
    def key(self) -> str:
        return _grid_key(self.native_interval_minutes, self.slot_labels, self.auction)

    def to_dict(self) -> dict[str, Any]:
        """Serialize numerical parameters without hiding their physical units."""
        result = {"native_interval_minutes": self.native_interval_minutes,
                "slot_labels": list(self.slot_labels), "bias_eur_mwh": self.bias_eur_mwh.tolist(),
                "covariance_eur2_mwh2": self.covariance_eur2_mwh2.tolist(),
                "calibration_days": self.calibration_days,
                "first_delivery_day": self.first_delivery_day, "last_delivery_day": self.last_delivery_day}
        if self.auction != "DA":
            result["auction"] = self.auction
        return result


@dataclass(frozen=True)
class GaussianPriceCalibration:
    """Reusable price uncertainty parameters with explicit study limitations."""
    grids: Mapping[str, GaussianGridCalibration]
    calibration_cutoff: str
    timezone: str = "Europe/Rome"
    covariance_shrinkage: float = 0.5
    minimum_variance_eur2_mwh2: float = 1e-6
    operational_eligible: bool = False
    availability_policy: str = "retrospective_latest_revision_unverified_publication"
    excluded_days: Mapping[str, str] = field(default_factory=dict)
    provenance: str = OOS_PROVENANCE
    auction: str = "DA"

    def __post_init__(self) -> None:
        """Verify audit metadata whenever artifacts are created or reloaded."""
        cutoff = _cutoff(self.calibration_cutoff, "calibration_cutoff")
        ZoneInfo(self.timezone)
        if self.auction not in AUCTIONS or any(grid.auction != self.auction for grid in self.grids.values()):
            raise ValueError("Gaussian artifact auction must match every explicitly scoped product grid")
        if self.auction == "IDA3" and self.timezone != "Europe/Rome":
            raise ValueError("IDA3 Gaussian templates require Europe/Rome market time")
        if not self.grids or any(key != grid.key for key, grid in self.grids.items()):
            raise ValueError("Gaussian artifact needs correctly keyed calibrated product grids")
        if not np.isfinite(self.covariance_shrinkage) or not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must lie in [0,1]")
        if not np.isfinite(self.minimum_variance_eur2_mwh2) or self.minimum_variance_eur2_mwh2 <= 0:
            raise ValueError("The declared minimum predictive variance must be positive")
        if self.provenance != OOS_PROVENANCE:
            raise ValueError("Gaussian scenarios require chronological out-of-sample errors")
        if self.operational_eligible and self.availability_policy != "verified_historical_publication_vintages":
            raise ValueError("Unverified retrospective vintages are not operationally eligible")
        if any(pd.Timestamp(grid.last_delivery_day).date() > cutoff.tz_convert(self.timezone).date()
               for grid in self.grids.values()):
            raise ValueError("Gaussian calibration days exceed the calibration cutoff")

    def _select(self, products: pd.DataFrame) -> tuple[GaussianGridCalibration, np.ndarray]:
        """Select exact native slots, including a partial day for reoptimization."""
        starts = _aware_index(products.product_start, "product_start")
        ends = _aware_index(products.product_end, "product_end")
        minutes = (ends - starts).total_seconds() / 60
        if len(starts) < 1 or starts.has_duplicates or not starts.is_monotonic_increasing:
            raise ValueError("Future products must be nonempty, unique and chronologically ordered")
        if len(set(starts.tz_convert(self.timezone).date)) != 1 or len(set(minutes)) != 1:
            raise ValueError("Future price paths must use one civil day and one native product duration")
        if minutes[0] not in (15, 30, 60):
            raise ValueError("Unsupported native product duration")
        day = starts[0].tz_convert(self.timezone).date().isoformat()
        complete, complete_ends = _daily_grid(day, int(minutes[0]), self.timezone, self.auction)
        if not starts.isin(complete).all() or not ends.isin(complete_ends).all():
            raise ValueError("Future products must coincide with auction-eligible native product boundaries")
        labels = _slot_labels(complete, self.timezone)
        key = _grid_key(int(minutes[0]), labels, self.auction)
        if key not in self.grids or self.grids[key].slot_labels != labels:
            raise ValueError(f"No calibrated Gaussian grid for {key}; DST/native grids are never interpolated")
        positions = complete.get_indexer(starts)
        return self.grids[key], positions

    def parameters_for(self, products: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return bias and covariance for exact future native products."""
        model, positions = self._select(products)
        return model.bias_eur_mwh[positions].copy(), model.covariance_eur2_mwh2[np.ix_(positions, positions)].copy()

    def sample_innovations(self, products: pd.DataFrame, count: int, *, seed: int | None = None,
                           random: np.random.Generator | None = None) -> np.ndarray:
        """Draw epsilon ~ N(0, Sigma); finite Monte Carlo means need not be zero."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("Gaussian path count must be a positive integer")
        if random is not None and seed is not None:
            raise ValueError("Pass either a seed or a random generator")
        _, covariance = self.parameters_for(products)
        rng = random if random is not None else np.random.default_rng(seed)
        return rng.standard_normal((count, len(products))) @ np.linalg.cholesky(covariance).T

    def sample_future(self, mean: Sequence[float], products: pd.DataFrame, count: int, *,
                      decision_cutoff: Any, seed: int | None = None,
                      apply_bias_correction: bool = True, require_operational: bool = True) -> np.ndarray:
        """Add separate bias and zero-mean innovations to an available point mean.

        Negative prices are valid. A study must explicitly pass
        require_operational=False to sample unverified historical vintages.
        """
        cutoff = _cutoff(decision_cutoff, "decision_cutoff")
        if _cutoff(self.calibration_cutoff, "calibration_cutoff") > cutoff:
            raise ValueError("Gaussian calibration uses future information")
        if require_operational and not self.operational_eligible:
            raise ValueError("Unverified historical publication vintages prohibit operational use")
        starts = _aware_index(products.product_start, "product_start")
        if (starts < cutoff).any():
            raise ValueError("Future price products begin before the decision cutoff")
        prediction = np.asarray(mean, float)
        if prediction.shape != (len(products),) or not np.isfinite(prediction).all():
            raise ValueError("Point forecasts must be finite and match the future product grid")
        bias, _ = self.parameters_for(products)
        innovation = self.sample_innovations(products, count, seed=seed)
        return prediction + (bias if apply_bias_correction else 0) + innovation

    def to_dict(self) -> dict[str, Any]:
        """Expose all calibration parameters, cutoffs, units and limitations."""
        result = {"schema": SCHEMA if self.auction == "DA" else SCOPED_SCHEMA, "calibration_cutoff": self.calibration_cutoff,
                "timezone": self.timezone, "covariance_shrinkage": self.covariance_shrinkage,
                "minimum_variance_eur2_mwh2": self.minimum_variance_eur2_mwh2,
                "operational_eligible": self.operational_eligible, "availability_policy": self.availability_policy,
                "provenance": self.provenance, "innovation_mean_eur_mwh": 0.0,
                "variance_source": "centered chronological out-of-sample prediction errors; never raw price variance",
                "bias_policy": "calibration-only per-native-slot error mean, separate from zero-mean innovation",
                "covariance_policy": "(1-shrinkage)*sample_covariance + shrinkage*diagonal + declared variance floor",
                "grid_policy": "complete native civil days; distinct DST folds/product counts; no interpolation",
                "excluded_days": dict(self.excluded_days),
                "grids": {key: value.to_dict() for key, value in self.grids.items()}}
        if self.auction != "DA":
            result.update({"auction": self.auction,
                "product_scope": "Europe/Rome native QH noon-to-midnight" if self.auction == "IDA3" else "complete native civil day",
                "grid_policy": "complete explicitly auction-eligible native products; exact DST slots; no interpolation"})
        return result

    def save(self, path: str | Path) -> None:
        """Write a self-contained, reloadable JSON uncertainty artifact."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GaussianPriceCalibration":
        """Reload and validate numerical parameters and historical eligibility."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") not in (SCHEMA, SCOPED_SCHEMA) or data.get("innovation_mean_eur_mwh") != 0:
            raise ValueError("Unsupported or nonzero-mean Gaussian price artifact")
        grids = {key: GaussianGridCalibration(**{**value, "slot_labels": tuple(value["slot_labels"])})
                 for key, value in data["grids"].items()}
        return cls(grids=grids, calibration_cutoff=data["calibration_cutoff"], timezone=data["timezone"],
                   covariance_shrinkage=data["covariance_shrinkage"],
                   minimum_variance_eur2_mwh2=data["minimum_variance_eur2_mwh2"],
                   operational_eligible=data["operational_eligible"], availability_policy=data["availability_policy"],
                   excluded_days=data["excluded_days"], provenance=data["provenance"],
                   auction="DA" if data["schema"] == SCHEMA else data["auction"])


def _validated_rows(rows: pd.DataFrame, timezone: str, auction: str = "DA") -> pd.DataFrame:
    """Certify chronological predictions without confusing delivery and publication."""
    required = {"delivery_day", "product_start", "product_end", "actual", "prediction", "fit_cutoff",
                "forecast_cutoff", "training_max_delivery_day", "prediction_provenance"}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError(f"Gaussian calibration requires nonempty prediction rows with {sorted(required)}")
    frame = rows.copy()
    if auction not in AUCTIONS or ("auction" in frame and not frame.auction.eq(auction).all()):
        raise ValueError("Prediction rows must match the explicitly selected auction")
    for name in ("product_start", "product_end", "fit_cutoff", "forecast_cutoff"):
        frame[name] = _aware_index(frame[name], name)
    if frame.product_start.duplicated().any():
        raise ValueError("Each observed native product needs one out-of-sample forecast")
    if not np.isfinite(frame[["actual", "prediction"]].to_numpy(float)).all():
        raise ValueError("Actual prices and point forecasts must be finite")
    frame["delivery_day"] = pd.to_datetime(frame.delivery_day).dt.strftime("%Y-%m-%d")
    training_days = pd.to_datetime(frame.training_max_delivery_day).dt.strftime("%Y-%m-%d")
    actual_days = frame.product_start.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
    if not (frame.delivery_day == actual_days).all():
        raise ValueError("delivery_day disagrees with the local native product start")
    if not (training_days < frame.delivery_day).all():
        raise ValueError("In-sample or future training labels cannot calibrate Gaussian innovations")
    if not (frame.prediction_provenance == OOS_PROVENANCE).all():
        raise ValueError("Gaussian calibration requires chronological out-of-sample prediction provenance")
    if (frame.fit_cutoff > frame.forecast_cutoff).any() or (frame.forecast_cutoff > frame.product_start).any():
        raise ValueError("Model fitting must precede forecast issuance and delivery")
    duration = (frame.product_end - frame.product_start).dt.total_seconds() / 60
    if not duration.isin([15, 30, 60]).all():
        raise ValueError("Native observed products must have 15, 30 or 60 minute duration")
    if "native_interval_minutes" in frame and not np.array_equal(frame.native_interval_minutes.to_numpy(float), duration.to_numpy()):
        raise ValueError("Declared native interval disagrees with actual product boundaries")
    frame["native_interval_minutes"] = duration.astype(int)
    if auction == "IDA3" and (timezone != "Europe/Rome" or not duration.eq(15).all()
            or (frame.product_start.dt.tz_convert(timezone).dt.hour < 12).any()):
        raise ValueError("IDA3 predictions require native quarter-hour products from noon to midnight in Europe/Rome")
    return frame.sort_values("product_start").reset_index(drop=True)


def _complete_days(frame: pd.DataFrame, timezone: str, auction: str = "DA") -> tuple[dict[str, list[pd.DataFrame]], dict[str, str]]:
    """Keep entire native civil days; report gaps instead of filling prices."""
    groups: dict[str, list[pd.DataFrame]] = {}
    excluded: dict[str, str] = {}
    for day, group in frame.groupby("delivery_day", sort=True):
        if group.native_interval_minutes.nunique() != 1:
            excluded[day] = "mixed native product durations"
            continue
        minutes = int(group.native_interval_minutes.iloc[0])
        starts, ends = _daily_grid(day, minutes, timezone, auction)
        if not np.array_equal(group.product_start.array.asi8, starts.asi8) or not np.array_equal(group.product_end.array.asi8, ends.asi8):
            excluded[day] = "incomplete or noncontiguous native civil day"
            continue
        key = _grid_key(minutes, _slot_labels(starts, timezone), auction)
        groups.setdefault(key, []).append(group)
    return groups, excluded


def calibrate_gaussian_prices(rows: pd.DataFrame, *, calibration_cutoff: Any,
                              auction: str = "DA", timezone: str = "Europe/Rome", covariance_shrinkage: float = 0.5,
                              minimum_complete_days: int = 10, minimum_variance_eur2_mwh2: float = 1e-6,
                              historical_vintages_verified: bool = False,
                              availability_policy: str = "retrospective_latest_revision_unverified_publication") -> GaussianPriceCalibration:
    """Fit bias and zero-mean Gaussian covariance using calibration rows only.

    Predictions must have been issued from strictly earlier delivery-day labels.
    That condition establishes statistical out-of-sample status; it does not
    establish source publication vintages, which require additional evidence.
    """
    if not np.isfinite(covariance_shrinkage) or not 0 <= covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must lie in [0,1]")
    if isinstance(minimum_complete_days, bool) or not isinstance(minimum_complete_days, (int, np.integer)) or minimum_complete_days < 2:
        raise ValueError("minimum_complete_days must be at least two")
    if not np.isfinite(minimum_variance_eur2_mwh2) or minimum_variance_eur2_mwh2 <= 0:
        raise ValueError("minimum_variance_eur2_mwh2 must be positive")
    cutoff = _cutoff(calibration_cutoff, "calibration_cutoff")
    frame = _validated_rows(rows, timezone, auction)
    if (frame.product_end > cutoff).any():
        raise ValueError("Calibration rows extend after calibration_cutoff; keep test rows separate")
    if historical_vintages_verified:
        if availability_policy != "verified_historical_publication_vintages":
            raise ValueError("Operational eligibility requires a verified publication-vintage policy")
        for name in ("feature_available_at", "target_available_at"):
            if name not in frame:
                raise ValueError("Verified historical vintages require feature/target publication timestamps")
            frame[name] = _aware_index(frame[name], name)
        if "historical_feature_eligible" not in frame or not frame.historical_feature_eligible.eq(True).all():
            raise ValueError("Unverified source features cannot become operationally eligible")
        if (frame.feature_available_at > frame.forecast_cutoff).any() or (frame.target_available_at > cutoff).any():
            raise ValueError("Publication timestamps exceed the forecast/calibration information cutoff")
    day_groups, excluded = _complete_days(frame, timezone, auction)
    models = {}
    for key, days in day_groups.items():
        if len(days) < minimum_complete_days:
            for day in days:
                excluded[str(day.delivery_day.iloc[0])] = f"{key} has {len(days)} complete days; requires {minimum_complete_days}"
            continue
        errors = np.stack([(day.actual - day.prediction).to_numpy(float) for day in days])
        bias = errors.mean(axis=0)
        centered = errors - bias
        sample_covariance = centered.T @ centered / (len(days) - 1)
        covariance = ((1 - covariance_shrinkage) * sample_covariance
                      + covariance_shrinkage * np.diag(np.diag(sample_covariance)))
        # A declared diagonal floor makes the Gaussian density well defined even
        # when a calibration slot has identical errors or days < products.
        covariance += np.diag(np.maximum(minimum_variance_eur2_mwh2 - np.diag(covariance), 0))
        smallest = float(np.linalg.eigvalsh(covariance)[0])
        if smallest < minimum_variance_eur2_mwh2:
            covariance += (minimum_variance_eur2_mwh2 - smallest) * np.eye(len(bias))
        starts = pd.DatetimeIndex(days[0].product_start)
        models[key] = GaussianGridCalibration(int(days[0].native_interval_minutes.iloc[0]),
            _slot_labels(starts, timezone), bias, covariance, len(days),
            str(days[0].delivery_day.iloc[0]), str(days[-1].delivery_day.iloc[0]), auction=auction)
    if not models:
        raise ValueError("Insufficient complete chronological out-of-sample days for Gaussian calibration")
    return GaussianPriceCalibration(models, cutoff.isoformat(), timezone, covariance_shrinkage,
        minimum_variance_eur2_mwh2, historical_vintages_verified, availability_policy, excluded, auction=auction)


def evaluate_gaussian_prices(calibration: GaussianPriceCalibration, test_rows: pd.DataFrame, *,
                             levels: Sequence[float] = (0.5, 0.8, 0.95)) -> dict[str, Any]:
    """Score untouched later days; never refit bias or variance using test labels."""
    if not levels or any(not np.isfinite(value) or not 0 < value < 1 for value in levels):
        raise ValueError("Gaussian coverage levels must lie strictly between zero and one")
    frame = _validated_rows(test_rows, calibration.timezone, calibration.auction)
    cutoff = _cutoff(calibration.calibration_cutoff, "calibration_cutoff")
    if (frame.product_start < cutoff).any() or (frame.forecast_cutoff < cutoff).any():
        raise ValueError("Untouched test forecasts and delivery must follow the calibration cutoff")
    groups, excluded = _complete_days(frame, calibration.timezone, calibration.auction)
    output: list[pd.DataFrame] = []
    ramps: list[dict[str, float]] = []
    joint_scores: list[float] = []
    for key, days in groups.items():
        for day in days:
            if key not in calibration.grids:
                excluded[str(day.delivery_day.iloc[0])] = f"unsupported Gaussian grid {key}"
                continue
            bias, covariance = calibration.parameters_for(day)
            sigma = np.sqrt(np.diag(covariance))
            mean = day.prediction.to_numpy(float) + bias
            error = day.actual.to_numpy(float) - mean
            scored = day[["delivery_day", "product_start", "product_end", "actual", "prediction"]].copy()
            scored["bias_correction"] = bias
            scored["gaussian_mean"] = mean
            scored["sigma_eur_mwh"] = sigma
            scored["absolute_error"] = np.abs(error)
            scored["squared_error"] = error ** 2
            scored["crps_eur_mwh"] = _normal_crps(error, sigma)
            scored["marginal_nll"] = 0.5 * np.log(2 * pi * sigma ** 2) + 0.5 * (error / sigma) ** 2
            for level in levels:
                width = NormalDist().inv_cdf((1 + level) / 2) * sigma
                label = f"{100 * level:g}"
                scored[f"covered_{label}"] = np.abs(error) <= width
                scored[f"interval_width_{label}"] = 2 * width
            output.append(scored)
            cholesky = np.linalg.cholesky(covariance)
            standardized = np.linalg.solve(cholesky, error)
            joint_scores.append(float(0.5 * (len(error) * np.log(2 * pi)
                + 2 * np.log(np.diag(cholesky)).sum() + standardized @ standardized) / len(error)))
            # Evaluate uncertainty of consecutive-product price changes using
            # Var(e_t-e_t-1), which depends on the temporal covariance.
            ramp_error = np.diff(error)
            ramp_variance = np.diag(covariance)[1:] + np.diag(covariance)[:-1] - 2 * np.diag(covariance, k=1)
            ramp_sigma = np.sqrt(np.maximum(ramp_variance, calibration.minimum_variance_eur2_mwh2))
            ramps.extend({"error": float(value), "sigma": float(sd),
                          "covered_95": float(abs(value) <= NormalDist().inv_cdf(.975) * sd)}
                         for value, sd in zip(ramp_error, ramp_sigma))
    if not output:
        raise ValueError("No untouched test days match a calibrated native/DST product grid")
    scored = pd.concat(output, ignore_index=True)
    metrics = {"scored_rows": len(scored), "scored_days": int(scored.delivery_day.nunique()),
               "unscored_rows": len(frame) - len(scored),
               "mae_eur_mwh": float(scored.absolute_error.mean()),
               "rmse_eur_mwh": float(np.sqrt(scored.squared_error.mean())),
               "uncorrected_point_mae_eur_mwh": float((scored.actual - scored.prediction).abs().mean()),
               "uncorrected_point_rmse_eur_mwh": float(np.sqrt(((scored.actual - scored.prediction) ** 2).mean())),
               "mean_error_eur_mwh": float((scored.actual - scored.gaussian_mean).mean()),
               "mean_crps_eur_mwh": float(scored.crps_eur_mwh.mean()),
               "mean_marginal_nll": float(scored.marginal_nll.mean()),
               "mean_joint_nll_per_product": float(np.mean(joint_scores))}
    metrics["coverage"] = {f"{100 * level:g}": float(scored[f"covered_{100 * level:g}"].mean()) for level in levels}
    metrics["mean_interval_width_eur_mwh"] = {
        f"{100 * level:g}": float(scored[f"interval_width_{100 * level:g}"].mean()) for level in levels}
    metrics["consecutive_product_ramps"] = {"rows": len(ramps),
        "mae_eur_mwh": float(np.mean([abs(row["error"]) for row in ramps])),
        "rmse_eur_mwh": float(np.mean([row["error"] ** 2 for row in ramps]) ** .5),
        "coverage_95": float(np.mean([row["covered_95"] for row in ramps])),
        "mean_sigma_eur_mwh": float(np.mean([row["sigma"] for row in ramps]))}
    return {"metrics": metrics, "excluded_days": excluded, "scored_predictions": scored,
            "evaluation_policy": "untouched chronological test; calibration bias/covariance frozen",
            "operational_eligible": calibration.operational_eligible,
            "availability_policy": calibration.availability_policy}


def gaussian_price_tree(grid: Any, pv_ac_mw: np.ndarray, auction_schedule: Sequence[Any], *,
                        mean_prices: Mapping[str, Sequence[float]] | None = None,
                        calibrations: Mapping[str, GaussianPriceCalibration],
                        feature_available_at: Mapping[str, Any], model_fit_cutoffs: Mapping[str, Any],
                        decision_cutoff: Any, branches: int = 2,
                        conditional_means: Mapping[str, Callable[[Mapping[str, np.ndarray]], Sequence[float]]] | None = None,
                        known_prices: Mapping[str, np.ndarray] | None = None,
                        imbalance_price: float | np.ndarray = 60.0, seed: int = 0,
                        allow_retrospective: bool = False) -> tuple[Any, Any]:
    """Build the existing MILP's information tree from calibrated native paths.

    Each supplied regression predicts the absolute price level for its auction.
    A conditional_means callback overrides the corresponding fixed mean and sees
    read-only ancestor clearing curves, including sampled predecessor prices.
    Innovations are independent across equations and temporally correlated within
    each auction; conditional means propagate dependence between auctions.
    Auctions without price models must be explicitly excluded from the supplied
    schedule; this adapter never invents intraday auction training observations.
    """
    from ..scenarios import conditional_price_tree

    cutoff = _cutoff(decision_cutoff, "decision_cutoff")
    if isinstance(branches, bool) or not isinstance(branches, (int, np.integer)) or branches < 1:
        raise ValueError("Gaussian conditional branch count must be a positive integer")
    products = pd.DataFrame({"product_start": grid.starts, "product_end": grid.ends})
    known = {}
    supplied_known = known_prices or {}
    fixed_means = mean_prices or {}
    callbacks = conditional_means or {}
    means = {}
    audit = {}
    allowed_auctions = {"DA", "IDA1", "IDA2", "IDA3"}
    if (set(supplied_known) - allowed_auctions or set(callbacks) - allowed_auctions
            or any(event.auction not in allowed_auctions for event in auction_schedule)):
        raise ValueError("Gaussian price trees support only explicitly named DA/IDA1/IDA2/IDA3 auctions")
    if any(not callable(callback) for callback in callbacks.values()):
        raise ValueError("Conditional Gaussian means must be callable absolute-level predictors")
    eligible_by_auction = {}
    products_by_auction = {}
    bias_by_auction = {}
    for event in auction_schedule:
        indices = tuple(event.eligible_intervals)
        if (len(set(indices)) != len(indices) or any(isinstance(index, bool)
                or not isinstance(index, (int, np.integer)) or not 0 <= index < len(products) for index in indices)):
            raise ValueError("Auction eligible intervals must be unique valid residual-grid indices")
        selected = np.array(sorted(indices), dtype=int)
        eligible_by_auction[event.auction] = selected
        products_by_auction[event.auction] = products.iloc[selected].copy()
    for auction, value in supplied_known.items():
        array = np.asarray(value, float)
        selected = eligible_by_auction.get(auction, np.arange(len(products)))
        if array.shape == (len(selected),) and len(selected) != len(products):
            expanded = np.zeros(len(products))
            expanded[selected] = array
            array = expanded
        elif array.ndim == 0:
            array = np.full(len(products), float(array))
        if array.shape != (len(products),) or not np.isfinite(array).all():
            raise ValueError(f"Known {auction} prices must match eligible native products or the full residual grid")
        outside = np.setdiff1d(np.arange(len(products)), selected)
        if len(outside) and np.any(array[outside] != 0):
            raise ValueError(f"Known {auction} has prices outside eligible products; supply only actual native labels")
        known[auction] = array.copy()
    for event in auction_schedule:
        auction = event.auction
        if not event.available or event.results_at is None or auction in known:
            continue
        if (auction not in fixed_means and auction not in callbacks) or auction not in calibrations:
            raise ValueError(f"Missing actual fitted price mean or Gaussian calibration for {auction}")
        if auction not in feature_available_at or auction not in model_fit_cutoffs:
            raise ValueError(f"Missing price model/feature information cutoffs for {auction}")
        if (_cutoff(feature_available_at[auction], f"{auction} feature_available_at") > cutoff
                or _cutoff(model_fit_cutoffs[auction], f"{auction} model_fit_cutoff") > cutoff):
            raise ValueError(f"{auction} point forecast uses future model/features")
        calibration = calibrations[auction]
        if calibration.auction != auction:
            raise ValueError(f"{auction} needs calibration explicitly scoped to the same auction")
        if _cutoff(calibration.calibration_cutoff, "calibration_cutoff") > cutoff:
            raise ValueError(f"{auction} Gaussian calibration uses future information")
        if not allow_retrospective and not calibration.operational_eligible:
            raise ValueError("Unverified publication vintages require explicit allow_retrospective=True for studies")
        selected = eligible_by_auction[auction]
        if not len(selected):
            raise ValueError(f"{auction} has no eligible residual products; omit it from the scenario schedule")
        bias, _ = calibration.parameters_for(products_by_auction[auction])
        bias_by_auction[auction] = bias
        if auction not in callbacks:
            mean = np.asarray(fixed_means[auction], float)
            if mean.shape == (len(products),):
                mean = mean[selected]
            if mean.shape != (len(selected),) or not np.isfinite(mean).all():
                raise ValueError(f"{auction} point forecast must match the finite auction-eligible native delivery grid")
            means[auction] = mean.copy()
        audit[auction] = {"calibration_cutoff": calibration.calibration_cutoff,
                          "auction": calibration.auction,
                          "availability_policy": calibration.availability_policy,
                          "operational_eligible": calibration.operational_eligible,
                          "feature_available_at": str(feature_available_at[auction]),
                          "model_fit_cutoff": str(model_fit_cutoffs[auction]),
                          "conditional_mean": "history-aware absolute regression" if auction in callbacks else "fixed absolute regression"}

    def sampler(auction: str, count: int, random: np.random.Generator) -> np.ndarray:
        """Draw complete zero-mean native vectors at each conditional result node."""
        if count != len(products):
            raise ValueError("Scenario tree product grid changed after Gaussian calibration selection")
        selected = eligible_by_auction[auction]
        result = np.zeros((branches, len(products)))
        result[:, selected] = calibrations[auction].sample_innovations(products_by_auction[auction], branches, random=random)
        return result

    def conditional_absolute_mean(auction: str, history: Mapping[str, np.ndarray]) -> np.ndarray:
        """Rebuild branch-conditional means only on the auction's true products."""
        selected = eligible_by_auction[auction]
        prediction = np.asarray(callbacks[auction](history) if auction in callbacks else means[auction], float)
        if prediction.shape == (len(products),):
            prediction = prediction[selected]
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise ValueError(f"{auction} conditional absolute mean must match its finite eligible native products")
        expanded = np.zeros(len(products))
        expanded[selected] = prediction + bias_by_auction[auction]
        return expanded

    return conditional_price_tree(grid, pv_ac_mw, auction_schedule,
        innovation_sampler=sampler, history_mean=conditional_absolute_mean,
        branch_probabilities=np.full(branches, 1 / branches), known_prices=known,
        decision_cutoff=cutoff, imbalance_price=imbalance_price, seed=seed,
        origin="fitted linear prices with zero-mean Gaussian innovations / deterministic-PV",
        provenance={"residual_method": "zero-mean multivariate Gaussian from chronological OOS errors",
                    "bias_policy": "separate calibration-only slot bias added to linear price mean",
                    "conditional_dependence": ("history-aware absolute means propagate sampled/known ancestor prices; equation innovations independent"
                                               if callbacks else "auction innovations independent; each fixed mean predicts an absolute price level"),
                    "operational_eligible": bool(audit) and all(value["operational_eligible"] for value in audit.values()),
                    "retrospective_study_explicitly_enabled": allow_retrospective,
                    "calibration_audit": audit,
                    "auction_eligible_product_indices": {auction: indices.tolist() for auction, indices in eligible_by_auction.items()},
                    "untradable_price_placeholders": {auction: {"value_eur_mwh": 0.0,
                        "product_indices": np.setdiff1d(np.arange(len(products)), indices).tolist(),
                        "policy": "untradable rectangular coefficients only; no morning prices, residuals or covariance invented"}
                        for auction, indices in eligible_by_auction.items() if len(indices) != len(products)}})
