"""Retrospective price experiments from verified cached API responses.

These experiments use the latest retrieved revisions. They deliberately do not
manufacture historical publication timestamps or claim an executable backtest.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .prices import FEATURES, RidgePriceModel


STUDY_LIMITATION = (
    "Retrospective study using latest retrieved price revisions. Historical "
    "publication and revision availability are unverified; these results are "
    "not a validated historical trading backtest or operational scenario calibration."
)


def _native_regimes(settings: Mapping[str, Any]) -> list[tuple[date, date, int]]:
    """Validate explicit market durations independently of compressed API blocks."""
    regimes = []
    for entry in settings.get("native_product_regimes", []):
        start = date.fromisoformat(str(entry["start"]))
        end = date.max if entry.get("end") is None else date.fromisoformat(str(entry["end"]))
        minutes = int(entry["minutes"])
        if end <= start or minutes not in (15, 30, 60) or minutes != float(entry["minutes"]):
            raise ValueError("Price product regimes require increasing dates and 15, 30, or 60 minutes")
        regimes.append((start, end, minutes))
    regimes.sort()
    if not regimes or any(first[1] > second[0] for first, second in zip(regimes, regimes[1:])):
        raise ValueError("Declare non-overlapping native_product_regimes; API block lengths are not market durations")
    return regimes


def prepare_native_prices(records: pd.DataFrame, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Recover declared native products without interpolation or price averaging.

    ENTSO-E can encode historical hourly prices on a PT15M A03 grid. Four
    complete, equal-valued quarters recover that hourly observation. Repeated
    prices in the quarter-hour market remain distinct quarter-hour products.
    """
    required = {"product_start", "product_end", "price_eur_mwh", "retrieved_at"}
    if records.empty or not required.issubset(records.columns):
        raise ValueError("Completed cached day-ahead prices are required for an experiment")
    frame = records.copy()
    for column in ("product_start", "product_end", "retrieved_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"Cached prices contain missing {column}")
    if not np.isfinite(frame.price_eur_mwh.to_numpy(float)).all():
        raise ValueError("Cached prices must be finite")
    # Latest retrieval is an explicit retrospective choice, never a publication assertion.
    keys = ["product_start", "product_end"]
    same_revision = frame.groupby([*keys, "retrieved_at"]).price_eur_mwh.nunique()
    if (same_revision > 1).any():
        raise ValueError("Ambiguous prices for an identical product and retrieval time")
    frame = frame.sort_values([*keys, "retrieved_at"]).drop_duplicates(keys, keep="last")
    frame = frame.sort_values("product_start").reset_index(drop=True)
    starts = frame.product_start
    ends = frame.product_end
    if (ends <= starts).any() or (starts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any():
        raise ValueError("Cached price products overlap or have invalid durations")
    local = starts.dt.tz_convert(str(settings.get("timezone", "Europe/Rome")))
    local_days = local.dt.date
    declared = np.zeros(len(frame), dtype=int)
    for start, end, minutes in _native_regimes(settings):
        declared[(local_days >= start) & (local_days < end)] = minutes
    if (declared == 0).any():
        raise ValueError("Every cached delivery day must be covered by a native product regime")
    frame["native_interval_minutes"] = declared
    frame["delivery_day"] = local_days.astype(str)
    # UTC binning preserves both repeated local hours and their independent products.
    native_rows = []
    for minutes, portion in frame.groupby("native_interval_minutes", sort=True):
        bins = portion.product_start.dt.floor(f"{minutes}min")
        durations = (portion.product_end - portion.product_start).dt.total_seconds() / 60
        if durations.eq(minutes).all() and portion.product_start.eq(bins).all():
            # Preserve independent QH products even when A03 repeats their price.
            direct = portion.copy()
            direct["provider_grid_rows"] = 1
            direct["provider_grid_min_minutes"] = float(minutes)
            native_rows.extend(direct.to_dict(orient="records"))
            continue
        for start, group in portion.groupby(bins, sort=True):
            end = start + pd.Timedelta(minutes=int(minutes))
            ordered = group.sort_values("product_start")
            if (ordered.product_start.iloc[0] != start or ordered.product_end.iloc[-1] != end
                    or (ordered.product_start.iloc[1:].to_numpy() != ordered.product_end.iloc[:-1].to_numpy()).any()
                    or (ordered.product_end > end).any()):
                raise ValueError("Cannot recover a native price product from incomplete provider intervals")
            prices = ordered.price_eur_mwh.to_numpy(float)
            if not np.equal(prices, prices[0]).all():
                raise ValueError("Provider prices differ within a declared native product; do not average them")
            item = ordered.iloc[0].to_dict()
            item.update(product_start=start, product_end=end, price_eur_mwh=float(prices[0]),
                        retrieved_at=ordered.retrieved_at.max(), provider_grid_rows=len(ordered),
                        provider_grid_min_minutes=float((ordered.product_end - ordered.product_start).dt.total_seconds().min() / 60))
            native_rows.append(item)
    result = pd.DataFrame(native_rows).sort_values("product_start").reset_index(drop=True)
    # Keep native duration separate from the original API resolution metadata.
    result["native_interval_minutes"] = result.native_interval_minutes.astype(int)
    return result


def build_retrospective_features(records: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Join observed same-clock lags by local day, fold, and native duration."""
    zone = ZoneInfo(timezone_name)
    frame = records.copy().reset_index(drop=True)
    local = [stamp.to_pydatetime().astimezone(zone) for stamp in frame.product_start]
    days = [stamp.date() for stamp in local]
    hours = np.array([stamp.hour + stamp.minute / 60 for stamp in local])
    weekdays = np.array([day.weekday() for day in days])
    months = np.array([day.month for day in days])
    for label, values, period in (("hour", hours, 24), ("weekday", weekdays, 7), ("month", months, 12)):
        frame[f"{label}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"{label}_cos"] = np.cos(2 * np.pi * values / period)
    clock_keys = [(day, stamp.hour, stamp.minute, stamp.fold, int(minutes))
                  for day, stamp, minutes in zip(days, local, frame.native_interval_minutes)]
    lookup = dict(zip(clock_keys, frame.price_eur_mwh.astype(float)))
    if len(lookup) != len(frame):
        raise ValueError("Duplicate native local-clock price products")
    for lag in (1, 7):
        frame[f"lag_{lag}d"] = [lookup.get((day - timedelta(days=lag), hour, minute, fold, minutes), np.nan)
                               for day, hour, minute, fold, minutes in clock_keys]
    frame["local_hour"] = hours
    frame["local_fold"] = [stamp.fold for stamp in local]
    frame["delivery_day"] = [day.isoformat() for day in days]
    return frame


def _fit_retrospective_ridge(train: pd.DataFrame, alpha: float, boundary: str,
                             feature_names: Sequence[str] = FEATURES) -> RidgePriceModel:
    """Fit train-only scaling while avoiding fabricated publication assertions."""
    values = train.loc[:, list(feature_names)].to_numpy(float)
    target = train.price_eur_mwh.to_numpy(float)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (values - center) / scale
    intercept = float(target.mean())
    design = np.vstack([normalized, np.sqrt(alpha) * np.eye(len(feature_names))])
    response = np.concatenate([target - intercept, np.zeros(len(feature_names))])
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    return RidgePriceModel(tuple(feature_names), center, scale, coefficients, intercept, alpha,
                           boundary, len(train), "retrospective_DA_level_unverified_vintages")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Measure comparable held-out products without dropping negative prices."""
    error = actual - predicted
    denominator = float(np.square(actual - actual.mean()).sum())
    return {"rows": len(actual), "mae_eur_mwh": float(np.abs(error).mean()),
            "rmse_eur_mwh": float(np.sqrt(np.square(error).mean())),
            "bias_actual_minus_prediction_eur_mwh": float(error.mean()),
            "r2": None if denominator == 0 else float(1 - np.square(error).sum() / denominator)}


def evaluate_price_holdout(features: pd.DataFrame, settings: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict], dict]:
    """Fit once before each duration's final holdout, then use observed past lags.

    Holdout labels never fit the model. A later holdout day's lag may use an
    earlier holdout day's price, as in a sequential forecasting experiment.
    """
    holdout_days = int(settings.get("holdout_days", 30))
    minimum_days = int(settings.get("min_training_days", 30))
    alpha = float(settings.get("alpha", 1.0))
    if holdout_days < 1 or minimum_days < 1 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Positive holdout/training days and nonnegative finite alpha are required")
    outputs, evaluations, models = [], [], {}
    for minutes, portion in features.groupby("native_interval_minutes", sort=True):
        all_days = sorted(portion.delivery_day.unique())
        last_day = date.fromisoformat(all_days[-1])
        first_holdout_day = (last_day - timedelta(days=holdout_days - 1)).isoformat()
        finite = portion.dropna(subset=[*FEATURES, "price_eur_mwh"])
        train = finite.loc[finite.delivery_day < first_holdout_day]
        validation = finite.loc[finite.delivery_day >= first_holdout_day]
        label = f"{int(minutes)}min"
        details = {"native_interval_minutes": int(minutes), "holdout_start_day": first_holdout_day,
                   "holdout_end_day": all_days[-1], "training_rows": len(train),
                   "training_start_day": None if train.empty else train.delivery_day.min(),
                   "training_end_day": None if train.empty else train.delivery_day.max(),
                   "training_days": int(train.delivery_day.nunique()), "holdout_rows": len(validation),
                   "holdout_days_observed": int(validation.delivery_day.nunique()),
                   "feature_incomplete_rows": int(len(portion) - len(finite)),
                   "feature_incomplete_holdout_rows": int(((portion.delivery_day >= first_holdout_day)
                                                           & portion[list(FEATURES)].isna().any(axis=1)).sum())}
        if train.delivery_day.nunique() < minimum_days or validation.empty:
            evaluations.append({**details, "status": "insufficient_data"})
            continue
        boundary = pd.Timestamp(first_holdout_day, tz=str(settings.get("timezone", "Europe/Rome"))).tz_convert("UTC").isoformat()
        model = _fit_retrospective_ridge(train, alpha, boundary)
        selected = validation.loc[:, ["product_start", "product_end", "delivery_day", "native_interval_minutes",
                                      "local_hour", "local_fold", "price_eur_mwh", "lag_1d", "lag_7d"]].copy()
        selected["ridge_prediction"] = model.predict(validation.loc[:, list(FEATURES)])
        selected["ridge_residual"] = selected.price_eur_mwh - selected.ridge_prediction
        selected["training_delivery_boundary"] = boundary
        outputs.append(selected)
        scores = {name: _metrics(selected.price_eur_mwh.to_numpy(float), selected[column].to_numpy(float))
                  for name, column in (("ridge", "ridge_prediction"), ("previous_day", "lag_1d"), ("previous_week", "lag_7d"))}
        evaluations.append({**details, "status": "evaluated", "models": scores})
        models[label] = {**model.to_dict(), "cutoff_semantics": "Exclusive delivery-day training boundary; not verified historical publication",
                         "training_start_day": train.delivery_day.min(), "training_end_day": train.delivery_day.max()}
    predictions = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return predictions, evaluations, models


def _daily_quality(records: pd.DataFrame, timezone_name: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Check each observed native day against its real 23/24/25-hour grid."""
    rows = []
    regimes = _native_regimes(settings)
    for day_stamp in pd.date_range(records.delivery_day.min(), records.delivery_day.max(), freq="D"):
        day = day_stamp.date().isoformat()
        portion = records.loc[records.delivery_day == day]
        midnight = pd.Timestamp(day, tz=timezone_name)
        following = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=timezone_name)
        minutes = next(minutes for start, end, minutes in regimes if start <= day_stamp.date() < end)
        expected = int((following - midnight).total_seconds() / (minutes * 60))
        observed = len(portion)
        complete = (observed == expected and portion.product_start.min() == midnight.tz_convert("UTC")
                    and portion.product_end.max() == following.tz_convert("UTC"))
        rows.append({"delivery_day": day, "native_interval_minutes": minutes,
                     "expected_products": expected, "observed_products": observed,
                     "missing_products": max(0, expected - observed), "complete": bool(complete),
                     "mean_eur_mwh": float(portion.price_eur_mwh.mean()),
                     "min_eur_mwh": float(portion.price_eur_mwh.min()),
                     "max_eur_mwh": float(portion.price_eur_mwh.max()),
                     "negative_products": int((portion.price_eur_mwh < 0).sum())})
    return pd.DataFrame(rows)


def _configured_bound(value: Any, timezone_name: str) -> str | None:
    """Interpret calendar-date filters locally and require explicit clock zones."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
            raise ValueError("Price experiment timestamp bounds need a timezone; calendar dates use the configured timezone")
        timestamp = timestamp.tz_localize(timezone_name)
    return timestamp.tz_convert("UTC").isoformat()


def _write_frame(frame: pd.DataFrame, directory: Path, name: str) -> None:
    """Publish reusable typed and human-readable derived tables atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("parquet", "csv"):
        target = directory / f"{name}.{extension}"
        pending = directory / f".{name}.{extension}.pending"
        if extension == "parquet":
            frame.to_parquet(pending, index=False)
        else:
            frame.to_csv(pending, index=False)
        pending.replace(target)


def _plot_results(quality: pd.DataFrame, predictions: pd.DataFrame, evaluations: Sequence[dict], directory: Path) -> list[str]:
    """Draw daily coverage/price context and comparable holdout diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    dates = pd.to_datetime(quality.delivery_day)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].fill_between(dates, quality.min_eur_mwh, quality.max_eur_mwh, alpha=.2, label="Daily range")
    axes[0].plot(dates, quality.mean_eur_mwh, linewidth=1, label="Daily mean")
    axes[0].set(ylabel="EUR/MWh", title="Italy NORD day-ahead prices: retrospective retrieved revisions")
    axes[0].legend(loc="upper right")
    axes[1].plot(dates, quality.observed_products, label="Observed native products")
    axes[1].plot(dates, quality.expected_products, linestyle="--", label="Expected products (DST-aware)")
    axes[1].set(ylabel="Products/day", xlabel="Local delivery day")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(directory / "price_history.png", dpi=130)
    plt.close(fig)
    files.append("price_history.png")
    for evaluation in evaluations:
        if evaluation["status"] != "evaluated":
            continue
        minutes = evaluation["native_interval_minutes"]
        portion = predictions.loc[predictions.native_interval_minutes == minutes].copy()
        recent = portion.loc[portion.delivery_day >= (date.fromisoformat(portion.delivery_day.max()) - timedelta(days=6)).isoformat()]
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for column, label in (("price_eur_mwh", "Retrieved price"), ("ridge_prediction", "Ridge forecast")):
            axes[0].plot(recent.product_start, recent[column], label=label, linewidth=.8)
        axes[0].set(title=f"{minutes}-minute native products: final seven holdout days", ylabel="EUR/MWh", xlabel="UTC product start")
        axes[0].legend()
        for column, label in (("ridge_prediction", "Ridge"), ("lag_1d", "Previous day"), ("lag_7d", "Previous week")):
            errors = portion.assign(absolute_error=(portion.price_eur_mwh - portion[column]).abs()).groupby("delivery_day").absolute_error.mean()
            axes[1].plot(pd.to_datetime(errors.index), errors.values, label=label)
        axes[1].set(ylabel="Daily MAE (EUR/MWh)", xlabel="Local delivery day")
        axes[1].legend()
        axes[2].hist(portion.ridge_residual, bins=40, alpha=.7)
        axes[2].axvline(0, color="black", linewidth=.8)
        axes[2].set(xlabel="Held-out actual minus ridge forecast (EUR/MWh)", ylabel="Products", title="Observed residual distribution; no Gaussian fit assumed")
        fig.tight_layout()
        name = f"holdout_{minutes}min.png"
        fig.savefig(directory / name, dpi=130)
        plt.close(fig)
        files.append(name)
    return files


def run_price_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read completed caches only and save an honest reproducible first experiment."""
    from ..data.api_catalog import load_api_cache_records

    settings = config.get("price_experiment", {})
    timezone_name = str(settings.get("timezone", "Europe/Rome"))
    provider, dataset, zone = (str(settings.get(name, default)) for name, default in
                               (("provider", "entsoe"), ("dataset", "day_ahead_prices"), ("zone", "NORD")))
    records, provenance = load_api_cache_records(settings.get("cache_directory", "data/api"), provider=provider,
                                                dataset=dataset, zone=zone,
                                                start=_configured_bound(settings.get("start"), timezone_name),
                                                end=_configured_bound(settings.get("end"), timezone_name), verify_checksums=True)
    native = prepare_native_prices(records, settings)
    features = build_retrospective_features(native, timezone_name)
    quality = _daily_quality(native, timezone_name, settings)
    predictions, evaluations, models = evaluate_price_holdout(features, settings)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = Path(settings.get("output_directory", "outputs/price_experiments")) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    derived_base = Path(settings.get("analysis_directory", "data/analysis/api_prices")) / provider / dataset / zone
    monthly_directories = []
    # Derived monthly views can be reused without revisiting credentials or XML.
    for month, portion in features.groupby(features.delivery_day.str[:7], sort=True):
        year, month_number = month.split("-")
        destination = derived_base / f"year={year}" / f"month={month_number}"
        _write_frame(portion, destination, "prices_with_features")
        monthly_quality = quality.loc[quality.delivery_day.str.startswith(month)]
        _write_frame(monthly_quality, destination, "daily_quality")
        month_manifest = {"schema_version": "retrospective-prices-v1", "month": month,
                          "provider": provider, "dataset": dataset, "zone": zone, "rows": len(portion),
                          "timezone": timezone_name, "run_directory": str(run_directory.resolve()),
                          "source_caches": str((run_directory / "source_caches.json").resolve()),
                          "revision_selection_policy": "Latest retrieved price for each provider-grid product; no claim of historical vintage availability",
                          "limitation": STUDY_LIMITATION, "native_product_regimes": settings["native_product_regimes"]}
        (destination / "manifest.json").write_text(json.dumps(month_manifest, indent=2, default=str) + "\n", encoding="utf-8")
        monthly_directories.append(str(destination.resolve()))
    _write_frame(quality, run_directory, "daily_quality")
    # Freeze the exact fit inputs; the convenient monthly views can grow on reruns.
    features.to_parquet(run_directory / "prices_with_features.parquet", index=False)
    if not predictions.empty:
        _write_frame(predictions, run_directory, "holdout_predictions")
    plot_paths = _plot_results(quality, predictions, evaluations, run_directory)
    (run_directory / "models.json").write_text(json.dumps(models, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps(evaluations, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_directory / "source_caches.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    summary = {"schema_version": "retrospective-prices-v1", "status": "completed",
               "study_type": "fixed chronological holdout by native product duration",
               "limitation": STUDY_LIMITATION, "historical_publication_verified": False,
               "training_method": "Ridge with train-only centering/scaling; no noise added",
               "features": list(FEATURES), "lag_semantics": "Same local clock, fold and native duration; observed past holdout lags allowed",
               "evaluation_comparison": "All models evaluated on identical complete-feature holdout products",
               "revision_selection_policy": "Latest retrieved price for each provider-grid product",
               "frozen_feature_inputs": "prices_with_features.parquet",
               "settings": dict(settings), "source_cache_count": len(provenance), "provider_grid_rows": len(records),
               "native_products": len(native), "first_delivery_day": native.delivery_day.min(),
               "last_delivery_day": native.delivery_day.max(), "complete_days": int(quality.complete.sum()),
               "incomplete_days": int((~quality.complete).sum()), "negative_native_prices": int((native.price_eur_mwh < 0).sum()),
               "metrics": evaluations, "run_directory": str(run_directory.resolve()),
               "derived_monthly_directories": monthly_directories, "plots": plot_paths}
    (run_directory / "manifest.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return summary
