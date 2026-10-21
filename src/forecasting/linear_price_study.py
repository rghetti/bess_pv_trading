"""Chronological linear price studies with separate selection and calibration.

API snapshots contain retrieved revisions, not reconstructed historical vintages.
The study preserves that limitation while making model selection, rolling fits,
out-of-sample errors and native product regimes reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.api_catalog import load_api_cache_records
from .fundamental_experiment import align_interval_forecast
from .price_experiment import (_configured_bound, _daily_quality,
    _fit_retrospective_ridge, _metrics, _write_frame, prepare_native_prices)
from .price_features import build_linear_price_features
from .prices import RidgePriceModel


SCHEMA = "linear-price-study-v1"
BASELINES = ("previous_day", "previous_week", "day_week_average")
LIMITATION = ("Retrospective latest-retrieved price and forecast revisions; historical "
    "publication/vintage availability is unverified. Chronological out-of-sample "
    "evaluation does not establish an executable historical trading backtest.")


def _integer(settings: Mapping[str, Any], name: str, default: int, minimum: int = 1) -> int:
    """Reject rounded, boolean or nonpositive study calendar settings."""
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _stamp(day: str, timezone_name: str, *, previous_noon: bool = False) -> pd.Timestamp:
    """Use a declared DA study cutoff, never a claimed provider publication."""
    local_day = date.fromisoformat(day) - timedelta(days=int(previous_noon))
    # Local noon is a wall-clock choice, not twelve elapsed hours after midnight.
    result = pd.Timestamp(f"{local_day.isoformat()} {'12:00:00' if previous_noon else '00:00:00'}", tz=timezone_name)
    return result.tz_convert("UTC")


def _group_keys(features: pd.DataFrame, structure: str, timezone_name: str) -> np.ndarray:
    """Identify native local-hour/product regressions without joining DST folds.

    Quarter-hour products can share coefficients within one local hour. The
    repeated autumn hour remains a distinct, potentially unsupported group;
    neither hourly products nor the other fold can supply its coefficients.
    """
    if structure not in ("by_hour", "by_product"):
        raise ValueError("Grouped price structure must be by_hour or by_product")
    if not {"product_start", "product_end", "native_interval_minutes"}.issubset(features):
        raise ValueError("Grouped price predictions need explicit native product boundaries")
    starts = pd.DatetimeIndex(features.product_start)
    ends = pd.DatetimeIndex(features.product_end)
    if starts.tz is None or ends.tz is None or starts.isna().any() or ends.isna().any():
        raise ValueError("Grouped price products must use nonmissing aware timestamps")
    durations = features.native_interval_minutes.to_numpy(float)
    observed = (ends - starts).total_seconds() / 60
    if not np.isin(durations, [15, 30, 60]).all() or not np.equal(durations, observed).all():
        raise ValueError("Grouped price native duration differs from its product boundaries")
    local = starts.tz_convert(ZoneInfo(timezone_name))
    if any(stamp.minute % int(minutes) or stamp.second or stamp.microsecond or stamp.nanosecond
           for minutes, stamp in zip(durations, local)):
        raise ValueError("Grouped prices must start on native local-clock product boundaries")
    if structure == "by_hour":
        return np.asarray([f"{int(minutes)}min/hour{stamp.hour:02d}/fold{stamp.fold}"
                           for minutes, stamp in zip(durations, local)])
    return np.asarray([f"{int(minutes)}min/{stamp.hour:02d}:{stamp.minute:02d}/fold{stamp.fold}"
                       for minutes, stamp in zip(durations, local)])


def _restore_ridge(values: Mapping[str, Any]) -> RidgePriceModel:
    """Restore one trained scaler and coefficient vector from JSON numbers."""
    return RidgePriceModel(tuple(values["feature_names"]), np.asarray(values["center"], float),
        np.asarray(values["scale"], float), np.asarray(values["coefficients"], float),
        float(values["intercept"]), float(values["alpha"]), values["fit_cutoff"],
        int(values["training_rows"]), values["target"])


@dataclass(frozen=True)
class LinearPriceForecast:
    """Save and reuse a selected point model without certifying API vintages."""
    candidate: Mapping[str, Any]
    fit_cutoff: str
    training_max_delivery_day: str
    model: RidgePriceModel | None = None
    operational_eligible: bool = False
    group_models: Mapping[str, RidgePriceModel] = field(default_factory=dict)
    timezone: str = "Europe/Rome"

    def __post_init__(self) -> None:
        """Check persisted model boundaries before coefficients become reusable."""
        cutoff = pd.Timestamp(self.fit_cutoff)
        if pd.isna(cutoff) or cutoff.tzinfo is None or self.operational_eligible:
            raise ValueError("Retrospective point artifacts need an aware fit cutoff and are not operationally eligible")
        date.fromisoformat(self.training_max_delivery_day)
        ZoneInfo(self.timezone)
        kind = self.candidate.get("kind")
        if kind == "baseline":
            if self.candidate.get("baseline") not in BASELINES or self.model is not None or self.group_models:
                raise ValueError("Invalid baseline point artifact")
        elif kind in ("ridge", "linear"):
            structure = self.candidate.get("model_structure", "pooled")
            if self.candidate.get("target") not in ("level", "delta_day"):
                raise ValueError("Invalid linear point artifact")
            if kind == "linear":
                from .linear_estimators import validate_estimator
                validate_estimator(self.candidate.get("estimator", {}))
            if structure == "pooled":
                if self.model is None or self.group_models:
                    raise ValueError("Pooled ridge artifact requires one fitted model")
                fitted = [self.model]
            elif structure in ("by_hour", "by_product"):
                if self.model is not None or not self.group_models:
                    raise ValueError("Grouped ridge artifact requires separate fitted group models")
                fitted = list(self.group_models.values())
            else:
                raise ValueError("Unknown persisted ridge model structure")
            for model in fitted:
                if tuple(self.candidate.get("feature_names", ())) != model.feature_names:
                    raise ValueError("Persisted ridge feature specification differs from its coefficients")
                if pd.Timestamp(model.fit_cutoff) != cutoff:
                    raise ValueError("Persisted ridge fit cutoffs disagree")
        else:
            raise ValueError("Unknown point price model kind")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Evaluate a baseline or linear level/delta model on explicit features."""
        if self.candidate["kind"] == "baseline":
            name = self.candidate["baseline"]
            if name == "previous_day":
                result = features.lag_1d.to_numpy(float)
            elif name == "previous_week":
                result = features.lag_7d.to_numpy(float)
            elif name == "day_week_average":
                result = (features.lag_1d.to_numpy(float) + features.lag_7d.to_numpy(float)) / 2
            else:
                raise ValueError("Unknown persisted price baseline")
        else:
            structure = self.candidate.get("model_structure", "pooled")
            if structure == "pooled":
                result = self.model.predict(features)
            else:
                keys = _group_keys(features, structure, self.timezone)
                unknown = set(keys) - set(self.group_models)
                if unknown:
                    raise ValueError("No trained price coefficients for native clock groups: " + ", ".join(sorted(unknown)))
                result = np.empty(len(features), dtype=float)
                for key in np.unique(keys):
                    mask = keys == key
                    result[mask] = self.group_models[key].predict(features.loc[mask])
            if self.candidate["target"] == "delta_day":
                result = result + features.lag_1d.to_numpy(float)
        if not np.isfinite(result).all():
            raise ValueError("Point price predictions require finite complete features")
        return result

    def to_dict(self) -> dict:
        """Retain target offsets, training boundaries and retrospective status."""
        grouped = self.candidate.get("model_structure", "pooled") != "pooled"
        schema = ("linear-price-point-model-v3" if self.candidate.get("kind") == "linear" else
                  "linear-price-point-model-v2" if grouped else "linear-price-point-model-v1")
        return {"schema_version": schema, "candidate": dict(self.candidate),
            "fit_cutoff": self.fit_cutoff, "training_max_delivery_day": self.training_max_delivery_day,
            "model": None if self.model is None else self.model.to_dict(),
            "group_models": {key: model.to_dict() for key, model in self.group_models.items()},
            "grouping_timezone": self.timezone,
            "unsupported_group_policy": "Reject unseen/insufficiently trained native clock groups; no pooled or DST-fold borrowing",
            "target_offset": "lag_1d" if self.candidate.get("target") == "delta_day" else None,
            "operational_eligible": self.operational_eligible,
            "availability_policy": "retrospective_latest_revision_unverified_publication"}

    def save(self, path: str | Path) -> None:
        """Write a portable artifact that contains no source credentials."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LinearPriceForecast":
        """Restore coefficients while rejecting unsupported audit claims."""
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") not in ("linear-price-point-model-v1", "linear-price-point-model-v2", "linear-price-point-model-v3") or payload.get("operational_eligible"):
            raise ValueError("Unsupported or operationally certified retrospective price artifact")
        values = payload.get("model")
        model = None if values is None else _restore_ridge(values)
        groups = {key: _restore_ridge(value) for key, value in payload.get("group_models", {}).items()}
        return cls(payload["candidate"], payload["fit_cutoff"], payload["training_max_delivery_day"],
            model, group_models=groups, timezone=payload.get("grouping_timezone", "Europe/Rome"))


def _candidates(groups: Mapping[str, Sequence[str]], settings: Mapping[str, Any]) -> list[dict]:
    """Expand an explicit, deterministic candidate grid including useful baselines."""
    candidates = [{"id": f"baseline_{name}", "kind": "baseline", "baseline": name,
                   "feature_names": ["lag_1d", "lag_7d"]} for name in BASELINES]
    specifications = settings.get("feature_sets", {"lags": ["calendar", "autoregressive"]})
    alphas = settings.get("alphas", [1.0, 100.0])
    windows = settings.get("training_windows_days", [90, 180])
    targets = settings.get("targets", ["level", "delta_day"])
    structures = settings.get("model_structures", ["pooled"])
    if not isinstance(specifications, dict) or not specifications or not alphas or not windows or not targets:
        raise ValueError("Declare nonempty feature_sets, alphas, training_windows_days and targets")
    if any(isinstance(alpha, bool) or not np.isfinite(float(alpha)) or float(alpha) < 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be finite and nonnegative")
    if any(window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1) for window in windows):
        raise ValueError("Training windows must be positive integer days or null for expanding history")
    if set(targets) - {"level", "delta_day"}:
        raise ValueError("Price targets must be level or delta_day")
    if (not isinstance(structures, list) or not structures or len(set(structures)) != len(structures)
            or set(structures) - {"pooled", "by_hour", "by_product"}):
        raise ValueError("model_structures must uniquely select pooled, by_hour or by_product")
    for name, selected_groups in specifications.items():
        if not isinstance(selected_groups, list) or not selected_groups or set(selected_groups) - set(groups):
            raise ValueError(f"Feature set {name} must select known nonempty feature groups")
        names = list(dict.fromkeys(column for group in selected_groups for column in groups[group]))
        if not names:
            raise ValueError(f"Feature set {name} has no columns")
        for alpha in alphas:
            for window in windows:
                for target in targets:
                    for structure in structures:
                        suffix = "" if structure == "pooled" else f"_{structure}"
                        candidates.append({"id": f"ridge_{name}_a{float(alpha):g}_w{window or 'expanding'}_{target}{suffix}",
                            "kind": "ridge", "feature_set": name, "feature_groups": selected_groups,
                            "feature_names": names, "alpha": float(alpha), "training_window_days": window,
                            "target": target, "model_structure": structure})
    additional = settings.get("additional_estimators", [])
    if not isinstance(additional, list):
        raise ValueError("additional_estimators must be a list of named fitting specifications")
    if additional:
        from .linear_estimators import validate_estimator
        declared = set()
        for specification in additional:
            if not isinstance(specification, dict):
                raise ValueError("Each additional estimator must be a named mapping")
            label = specification.get("name")
            if (not isinstance(label, str) or not label or not label.replace("_", "").isalnum()
                    or label in declared):
                raise ValueError("Additional estimator names must be unique nonempty alphanumeric labels")
            declared.add(label)
            estimator = validate_estimator({key: value for key, value in specification.items() if key != "name"})
            for name, selected_groups in specifications.items():
                names = list(dict.fromkeys(column for group in selected_groups for column in groups[group]))
                for window in windows:
                    for target in targets:
                        for structure in structures:
                            suffix = "" if structure == "pooled" else f"_{structure}"
                            candidates.append({"id": f"{label}_{name}_w{window or 'expanding'}_{target}{suffix}",
                                "kind": "linear", "estimator": estimator,
                                "feature_set": name, "feature_groups": selected_groups,
                                "feature_names": names, "alpha": float(estimator.get("alpha", 0.0)),
                                "training_window_days": window, "target": target,
                                "model_structure": structure})
    if len({candidate["id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate grid contains duplicate specifications")
    return candidates


def _split(last_day: str, settings: Mapping[str, Any]) -> dict:
    """Reserve ordered validation, calibration, maturity-gap and untouched test days."""
    test_days = _integer(settings, "test_days", 42)
    calibration_days = _integer(settings, "calibration_days", 42)
    validation_days = _integer(settings, "validation_days", 28)
    gap_days = _integer(settings, "maturity_gap_days", 1)
    last = date.fromisoformat(last_day)
    stop = last + timedelta(days=1)
    test = stop - timedelta(days=test_days)
    gap = test - timedelta(days=gap_days)
    calibration = gap - timedelta(days=calibration_days)
    validation = calibration - timedelta(days=validation_days)
    return {"validation_start_day": validation.isoformat(), "validation_end_day": (calibration - timedelta(days=1)).isoformat(),
        "calibration_start_day": calibration.isoformat(), "calibration_end_day": (gap - timedelta(days=1)).isoformat(),
        "maturity_gap_start_day": gap.isoformat(), "maturity_gap_end_day": (test - timedelta(days=1)).isoformat(),
        "test_start_day": test.isoformat(), "test_end_day": last_day,
        "calibration_cutoff": _stamp(gap.isoformat(), str(settings.get("timezone", "Europe/Rome"))).isoformat()}


def _predict_candidate(frame: pd.DataFrame, candidate: Mapping[str, Any], days: Sequence[str],
                       settings: Mapping[str, Any], common_eligible: pd.Series,
                       phase_for_day: Callable[[str], str]) -> tuple[pd.DataFrame, list[dict]]:
    """Refit using strictly earlier delivery days, then keep coefficients for a week."""
    timezone_name = str(settings.get("timezone", "Europe/Rome"))
    minimum = _integer(settings, "min_training_days", 56)
    refit_every = _integer(settings, "refit_every_days", 7)
    minimum_group_days = _integer(settings, "min_group_training_days", minimum)
    names = list(candidate["feature_names"])
    structure = candidate.get("model_structure", "pooled")
    grouped_keys = (None if structure == "pooled" else
        pd.Series(_group_keys(frame, structure, timezone_name), index=frame.index))
    finite = np.isfinite(frame[names + ["price_eur_mwh", "lag_1d"]].to_numpy(float)).all(axis=1)
    artifacts, outputs = [], []
    artifact = None
    last_fit_day = None
    for day in days:
        selected = frame.loc[frame.delivery_day.eq(day) & common_eligible]
        if selected.empty:
            continue
        forecast_cutoff = _stamp(day, timezone_name, previous_noon=True)
        if candidate["kind"] == "baseline":
            source_lag = 7 if candidate["baseline"] == "previous_week" else 1
            maximum_day = (date.fromisoformat(day) - timedelta(days=source_lag)).isoformat()
            artifact = LinearPriceForecast(dict(candidate), forecast_cutoff.isoformat(), maximum_day, timezone=timezone_name)
        elif last_fit_day is None or (date.fromisoformat(day) - date.fromisoformat(last_fit_day)).days >= refit_every:
            mask = frame.delivery_day.lt(day) & finite
            window = candidate["training_window_days"]
            if window is not None:
                beginning = (date.fromisoformat(day) - timedelta(days=int(window))).isoformat()
                mask &= frame.delivery_day.ge(beginning)
            train = frame.loc[mask].copy()
            if train.delivery_day.nunique() < minimum:
                continue
            if candidate["target"] == "delta_day":
                train["price_eur_mwh"] = train.price_eur_mwh - train.lag_1d
            def fit(portion: pd.DataFrame) -> RidgePriceModel:
                """Scale and fit this training subset without other clock groups."""
                if candidate["kind"] == "linear":
                    from .linear_estimators import fit_linear_estimator
                    model = fit_linear_estimator(portion, feature_names=names,
                        fit_cutoff=forecast_cutoff.isoformat(), method=candidate["estimator"]["method"],
                        settings=candidate["estimator"])
                else:
                    model = _fit_retrospective_ridge(portion, float(candidate["alpha"]), forecast_cutoff.isoformat(), names)
                return replace(model, target=f"retrospective_DA_{candidate['target']}_unverified_vintages")
            if structure == "pooled":
                artifact = LinearPriceForecast(dict(candidate), forecast_cutoff.isoformat(), train.delivery_day.max(), fit(train), timezone=timezone_name)
            else:
                group_models = {}
                # Only 24 hour groups (or 96 product groups) are fitted. Feature
                # filtering and weekly training-window selection happen once.
                train_keys = grouped_keys.loc[train.index]
                for key, indexes in train_keys.groupby(train_keys, sort=True).groups.items():
                    portion = train.loc[indexes]
                    if portion.delivery_day.nunique() >= minimum_group_days:
                        group_models[key] = fit(portion)
                if not group_models:
                    continue
                artifact = LinearPriceForecast(dict(candidate), forecast_cutoff.isoformat(), train.delivery_day.max(),
                    group_models=group_models, timezone=timezone_name)
            artifacts.append({"refit_delivery_day": day, **artifact.to_dict()})
            last_fit_day = day
        if artifact is None:
            continue
        if artifact.group_models:
            # An unsupported autumn fold or missing-hour history cannot borrow
            # coefficients. Partial validation coverage rejects this candidate.
            selected = selected.loc[grouped_keys.loc[selected.index].isin(artifact.group_models)]
            if selected.empty:
                continue
        columns = ["product_start", "product_end", "delivery_day", "native_interval_minutes"]
        output = selected[columns].copy()
        output["actual"] = selected.price_eur_mwh.to_numpy(float)
        output["prediction"] = artifact.predict(selected)
        output["residual"] = output.actual - output.prediction
        output["forecast_cutoff"] = forecast_cutoff
        output["fit_cutoff"] = pd.Timestamp(artifact.fit_cutoff)
        output["training_max_delivery_day"] = artifact.training_max_delivery_day
        output["prediction_provenance"] = "chronological_out_of_sample"
        output["candidate_id"] = candidate["id"]
        output["model_structure"] = "baseline" if candidate["kind"] == "baseline" else structure
        output["phase"] = phase_for_day(day)
        output["operational_eligible"] = False
        outputs.append(output)
    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), artifacts


def _scores(frame: pd.DataFrame) -> dict:
    """Report average and largest absolute held-out price errors in EUR/MWh."""
    return {**_metrics(frame.actual.to_numpy(float), frame.prediction.to_numpy(float)),
            "max_absolute_error_eur_mwh": float((frame.actual - frame.prediction).abs().max())}


def evaluate_linear_price_study(features: pd.DataFrame, groups: Mapping[str, Sequence[str]],
                                settings: Mapping[str, Any], progress: Callable[[str], None] | None = None) -> dict:
    """Select specifications only on validation, then evaluate independent later days."""
    candidates = _candidates(groups, settings)
    _integer(settings, "min_training_days", 56)
    _integer(settings, "refit_every_days", 7)
    outputs, baseline_outputs, validation_outputs, validation_results, regimes, models = [], [], [], [], [], {}
    required = list(dict.fromkeys(["price_eur_mwh", "lag_1d", "lag_7d",
        *(column for candidate in candidates for column in candidate["feature_names"])]))
    for minutes, portion in features.groupby("native_interval_minutes", sort=True):
        portion = portion.sort_values("product_start").reset_index(drop=True)
        common = pd.Series(np.isfinite(portion[required].to_numpy(float)).all(axis=1), index=portion.index)
        splits = _split(str(portion.delivery_day.max()), settings)
        validation_days = sorted(portion.loc[portion.delivery_day.between(splits["validation_start_day"], splits["validation_end_day"]), "delivery_day"].unique())
        expected_rows = int((common & portion.delivery_day.isin(validation_days)).sum())
        info = {"native_interval_minutes": int(minutes), "split": splits,
            "source_rows": len(portion), "common_feature_complete_rows": int(common.sum()),
            "incomplete_feature_rows": int((~common).sum()), "validation_rows": expected_rows,
            "feature_missing_rows": {name: int(portion[name].isna().sum()) for name in required if portion[name].isna().any()},
            "days_with_excluded_products": sorted(portion.loc[~common, "delivery_day"].unique().tolist())}
        ranked = []
        for index, candidate in enumerate(candidates):
            if progress is not None:
                progress(f"{minutes}min validation candidate {index + 1}/{len(candidates)}: {candidate['id']}")
            try:
                prediction, _ = _predict_candidate(portion, candidate, validation_days, settings, common, lambda day: "validation")
            except RuntimeError as error:
                from .linear_estimators import LinearEstimatorConvergenceError
                if not isinstance(error, LinearEstimatorConvergenceError):
                    raise
                validation_results.append({"native_interval_minutes": int(minutes), "candidate": candidate,
                    "status": "estimator_did_not_converge", "reason": str(error),
                    "predicted_rows": 0, "expected_common_rows": expected_rows})
                continue
            record = {"native_interval_minutes": int(minutes), "candidate": candidate,
                "status": "evaluated" if expected_rows and len(prediction) == expected_rows else "insufficient_training_or_validation",
                "predicted_rows": len(prediction), "expected_common_rows": expected_rows}
            if record["status"] == "evaluated":
                record["metrics"] = _scores(prediction)
                ranked.append(record)
                validation_outputs.append(prediction)
            validation_results.append(record)
        # Baselines cannot make a short regime look sufficiently trained.
        fitted_ranked = [record for record in ranked if record["candidate"]["kind"] != "baseline"]
        if not fitted_ranked:
            regimes.append({**info, "status": "insufficient_data", "reason": "No linear candidate has the configured earlier training history and full common validation coverage"})
            continue
        winner = min(ranked, key=lambda record: (record["metrics"]["mae_eur_mwh"], record["candidate"]["id"]))
        candidate = winner["candidate"]
        later_days = sorted(portion.loc[portion.delivery_day.ge(splits["calibration_start_day"]), "delivery_day"].unique())
        def phase(day: str) -> str:
            return "calibration" if day < splits["maturity_gap_start_day"] else "maturity_gap" if day < splits["test_start_day"] else "test"
        selected, artifacts = _predict_candidate(portion, candidate, later_days, settings, common, phase)
        if selected.empty or not {"calibration", "test"}.issubset(set(selected.phase)):
            regimes.append({**info, "status": "insufficient_data", "reason": "No complete-feature forecasts in a reserved calibration or final holdout period"})
            continue
        outputs.append(selected)
        for name in BASELINES:
            baseline = next(value for value in candidates if value.get("baseline") == name)
            baseline_rows, _ = _predict_candidate(portion, baseline, later_days, settings, common, phase)
            baseline_outputs.append(baseline_rows)
        tests = selected.loc[selected.phase.eq("test")]
        calibration = selected.loc[selected.phase.eq("calibration")]
        models[f"{int(minutes)}min"] = artifacts
        regimes.append({**info, "status": "evaluated", "selected_candidate": candidate,
            "selection_metric": "validation_mae_eur_mwh", "selection_metrics": winner["metrics"],
            "calibration_rows": len(calibration), "test_rows": len(tests), "test_metrics": _scores(tests),
            "coefficient_update_policy": "Selected specification fixed after validation; coefficients refitted on strictly earlier delivery days only"})
    concatenate = lambda values: pd.concat(values, ignore_index=True) if values else pd.DataFrame()
    return {"predictions": concatenate(outputs), "baseline_predictions": concatenate(baseline_outputs),
        "validation_predictions": concatenate(validation_outputs), "validation_results": validation_results,
        "regimes": regimes, "models": models, "feature_groups": dict(groups)}


def _common_end(native: pd.DataFrame, columns: Sequence[str], timezone_name: str) -> str:
    """Find the last full local day with every required forecast, excluding partial downloads."""
    complete = []
    for day, portion in native.groupby("delivery_day", sort=True):
        minutes = int(portion.native_interval_minutes.iloc[0])
        start = _stamp(str(day), timezone_name)
        stop = _stamp((date.fromisoformat(str(day)) + timedelta(days=1)).isoformat(), timezone_name)
        expected = pd.date_range(start, stop, freq=f"{minutes}min", inclusive="left")
        if pd.DatetimeIndex(portion.product_start).as_unit("ns").equals(expected.as_unit("ns")) and np.isfinite(portion[list(columns)].to_numpy(float)).all():
            complete.append(str(day))
    if not complete:
        raise ValueError("No complete local delivery day has all configured regional forecast features")
    return complete[-1]


def _plots(predictions: pd.DataFrame, baselines: pd.DataFrame, directory: Path) -> list[str]:
    """Save native-regime test traces and daily errors for direct baseline comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    test = predictions.loc[predictions.phase.eq("test")]
    for minutes, portion in test.groupby("native_interval_minutes", sort=True):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
        recent_day = (date.fromisoformat(portion.delivery_day.max()) - timedelta(days=6)).isoformat()
        recent = portion.loc[portion.delivery_day.ge(recent_day)]
        axes[0].plot(recent.product_start, recent.actual, color="black", linewidth=.8, label="Retrieved price")
        axes[0].plot(recent.product_start, recent.prediction, linewidth=.8, label="Validation-selected forecast")
        axes[0].set(title=f"NORD {minutes}-minute native prices: final chronological holdout", ylabel="EUR/MWh", xlabel="UTC product start")
        axes[0].legend()
        comparison = pd.concat([portion.assign(candidate_id="selected"), baselines.loc[baselines.phase.eq("test") & baselines.native_interval_minutes.eq(minutes)]])
        for name, rows in comparison.groupby("candidate_id", sort=False):
            errors = (rows.actual - rows.prediction).abs().groupby(rows.delivery_day).mean()
            axes[1].plot(pd.to_datetime(errors.index), errors.values, label=name)
        axes[1].set(ylabel="Daily MAE (EUR/MWh)", xlabel="Local delivery day")
        axes[1].legend(ncol=2)
        filename = f"test_{int(minutes)}min.png"
        fig.savefig(directory / filename, dpi=130)
        plt.close(fig)
        paths.append(filename)
    return paths


def run_linear_price_study(config: Mapping[str, Any], progress: Callable[[str], None] | None = None) -> dict:
    """Freeze cache inputs and save a reproducible validation/calibration/test study."""
    began = time.monotonic()
    settings = {**config.get("price_experiment", {}), **config.get("linear_price_study", {})}
    timezone_name = str(settings.get("timezone", "Europe/Rome"))
    zone = str(settings.get("zone", "NORD"))
    cache = settings.get("cache_directory", "data/api")
    bounds = {"start": _configured_bound(settings.get("start"), timezone_name),
              "end": _configured_bound(settings.get("end"), timezone_name)}
    fundamentals = settings.get("fundamental_features", {})
    frozen_source = None
    common_end = None
    if settings.get("frozen_feature_manifest"):
        # Estimator comparisons reuse immutable old features instead of silently
        # appending newly downloaded labels or rebuilding a different dataset.
        source_path = Path(settings["frozen_feature_manifest"]).resolve()
        source_bytes = source_path.read_bytes()
        source_manifest = json.loads(source_bytes)
        if source_manifest.get("status") != "completed":
            raise ValueError("Frozen feature inputs require a completed linear price study")
        feature_bytes = (source_path.parent / "features.parquet").read_bytes()
        feature_sha = hashlib.sha256(feature_bytes).hexdigest()
        if feature_sha != source_manifest.get("frozen_features_sha256"):
            raise ValueError("Frozen feature checksum differs from the source study")
        ancillary = {name: (source_path.parent / name).read_bytes() for name in
                     ("settings.json", "feature_groups.json", "source_caches.json", "feature_coverage.json")}
        source_settings = json.loads(ancillary["settings.json"])
        for key, default in (("timezone", "Europe/Rome"), ("zone", "NORD"),
                             ("feature_settings", {}), ("fundamental_features", {})):
            if settings.get(key, default) != source_settings.get(key, default):
                raise ValueError(f"Frozen feature inputs cannot change {key}")
        features = pd.read_parquet(BytesIO(feature_bytes))
        keep = pd.Series(True, index=features.index)
        if bounds["start"] is not None:
            keep &= features.product_start.ge(bounds["start"])
        if bounds["end"] is not None:
            keep &= features.product_end.le(bounds["end"])
        features = features.loc[keep].reset_index(drop=True)
        native = features
        groups = json.loads(ancillary["feature_groups.json"])
        sources = json.loads(ancillary["source_caches.json"])
        coverage = json.loads(ancillary["feature_coverage.json"])
        frozen_source = {"manifest": str(source_path),
            "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "features_sha256": feature_sha,
            "ancillary_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in ancillary.items()},
            "policy": "Reuse saved features and pinned source snapshots; no API or current database reads"}
    else:
        records, provenance = load_api_cache_records(cache, provider=settings.get("provider", "entsoe"),
            dataset=settings.get("dataset", "day_ahead_prices"), zone=zone, **bounds)
        native = prepare_native_prices(records, settings)
        sources, coverage = {"prices": provenance}, {}
        for name, specification in fundamentals.items():
            if name in native:
                raise ValueError("Fundamental feature names cannot overwrite native price columns")
            values, source = load_api_cache_records(cache, provider=specification["provider"],
                dataset=specification["dataset"], zone=specification.get("zone", zone), **bounds)
            native[name], coverage[name] = align_interval_forecast(native, values)
            sources[name] = source
        if fundamentals and settings.get("auto_common_end", True):
            common_end = _common_end(native, list(fundamentals), timezone_name)
            native = native.loc[native.delivery_day.le(common_end)].reset_index(drop=True)
        feature_settings = dict(settings.get("feature_settings", {}))
        feature_settings.setdefault("fundamental_columns", list(fundamentals))
        features, groups = build_linear_price_features(native, timezone_name, settings=feature_settings)
    evaluated = evaluate_linear_price_study(features, groups, settings, progress)
    if evaluated["predictions"].empty:
        return {"schema_version": SCHEMA, "status": "insufficient_data", "regimes": evaluated["regimes"],
            "operational_eligible": False, "limitation": LIMITATION}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = Path(settings.get("output_directory", "outputs/linear_price_studies")) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    predictions, baselines = evaluated["predictions"], evaluated["baseline_predictions"]
    _write_frame(features, directory, "features")
    quality = _daily_quality(native, timezone_name, settings)
    _write_frame(quality, directory, "daily_quality")
    _write_frame(predictions, directory, "selected_predictions")
    _write_frame(evaluated["validation_predictions"], directory, "validation_predictions")
    for phase in ("calibration", "test"):
        _write_frame(predictions.loc[predictions.phase.eq(phase)], directory, f"{phase}_predictions")
    _write_frame(baselines.loc[baselines.phase.eq("test")], directory, "baseline_test_predictions")
    all_test = pd.concat([predictions.loc[predictions.phase.eq("test")].assign(comparison="selected"),
        baselines.loc[baselines.phase.eq("test")].assign(comparison=lambda frame: frame.candidate_id)], ignore_index=True)
    daily = []
    comparisons = []
    for (minutes, name), portion in all_test.groupby(["native_interval_minutes", "comparison"], sort=True):
        comparisons.append({"native_interval_minutes": int(minutes), "model": name, **_scores(portion)})
        for day, values in portion.groupby("delivery_day", sort=True):
            daily.append({"native_interval_minutes": int(minutes), "model": name, "delivery_day": day, **_scores(values)})
    _write_frame(pd.DataFrame(daily), directory, "daily_metrics")
    for regime in evaluated["regimes"]:
        if regime["status"] != "evaluated":
            continue
        minutes = regime["native_interval_minutes"]
        destination = directory / f"{minutes}min"
        destination.mkdir()
        rows = predictions.loc[predictions.native_interval_minutes.eq(minutes)]
        for phase in ("calibration", "test"):
            _write_frame(rows.loc[rows.phase.eq(phase)], destination, f"{phase}_predictions")
            regime[f"{phase}_predictions"] = str((destination / f"{phase}_predictions.parquet").resolve())
        refits = evaluated["models"][f"{minutes}min"]
        if refits:
            final = dict(refits[-1])
            final.pop("refit_delivery_day")
        else:
            last = rows.iloc[-1]
            final = LinearPriceForecast(regime["selected_candidate"], last.fit_cutoff.isoformat(), last.training_max_delivery_day, timezone=timezone_name).to_dict()
        (destination / "point_model.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
        regime["point_model"] = str((destination / "point_model.json").resolve())
    values = {"settings.json": settings, "feature_groups.json": groups,
        "validation_candidate_scores.json": evaluated["validation_results"],
        "selected_specs.json": evaluated["regimes"], "models.json": evaluated["models"],
        "source_caches.json": sources, "feature_coverage.json": coverage, "test_metrics.json": comparisons}
    for filename, value in values.items():
        (directory / filename).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    monthly_directories = []
    for month, portion in features.groupby(features.delivery_day.str[:7], sort=True):
        year, month_number = month.split("-")
        monthly = Path(settings.get("analysis_directory", "data/analysis/linear_prices")) / zone / f"year={year}" / f"month={month_number}"
        _write_frame(portion, monthly, "features")
        _write_frame(quality.loc[quality.delivery_day.str.startswith(month)], monthly, "daily_quality")
        (monthly / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA, "month": month,
            "rows": len(portion), "source_run": str(directory.resolve()),
            "source_caches": str((directory / "source_caches.json").resolve()),
            "operational_eligible": False, "limitation": LIMITATION}, indent=2) + "\n")
        monthly_directories.append(str(monthly.resolve()))
    plots = _plots(predictions, baselines, directory)
    result = {"schema_version": SCHEMA, "status": "completed", "run_id": run_id,
        "output_dir": str(directory.resolve()), "run_directory": str(directory.resolve()),
        "manifest": str((directory / "manifest.json").resolve()),
        "calibration_predictions": str((directory / "calibration_predictions.parquet").resolve()),
        "test_predictions": str((directory / "test_predictions.parquet").resolve()),
        "regimes": evaluated["regimes"], "test_comparisons": comparisons,
        "operational_eligible": False, "historical_publication_verified": False, "limitation": LIMITATION,
        "forecast_cutoff_policy": "Assumed D-1 12:00 Europe/Rome study decision; not a provider publication assertion",
        "selection_policy": "Lowest validation MAE on identical complete-feature products; final test never selects specification",
        "prior_evaluation_overlap": settings.get("prior_evaluation_overlap", "Earlier exploratory studies inspected September 2025 hourly and August 2026 quarter-hour outcomes. The final holdout is reserved from this study's selection; it is not an unseen prospective sample."),
        "training_policy": "Strictly earlier delivery days; train-only scaling; weekly coefficient refits; frozen selected specification",
        "first_delivery_day": features.delivery_day.min(), "last_delivery_day": features.delivery_day.max(),
        "auto_common_end_day": common_end, "rows": len(features), "source_cache_count": sum(map(len, sources.values())),
        "derived_monthly_directories": monthly_directories,
        "frozen_features_sha256": hashlib.sha256((directory / "features.parquet").read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - began, "plots": plots}
    if frozen_source is not None:
        result["frozen_feature_source"] = frozen_source
    (directory / "manifest.json").write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    return result
