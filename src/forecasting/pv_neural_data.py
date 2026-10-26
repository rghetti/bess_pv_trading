"""Leakage-safe weather/calendar features and day/auction sequence tensors."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .pv_study_inputs import prepare_corpus


def validate_neural_config(config: dict):
    """Require an ordered chronological split and bounded CPU resource settings."""
    section = config["pv_neural"]
    dates = [pd.Timestamp(section[name]) for name in (
        "start_day", "train_end_day_exclusive", "validation_end_day_exclusive", "end_day_exclusive")]
    if any(value.tzinfo is not None or value != value.normalize() for value in dates):
        raise ValueError("Neural split dates must be timezone-free local calendar dates")
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError("Neural train/validation/test boundaries must be strictly increasing")
    if section["device"] != "cpu":
        raise ValueError("This benchmark is explicitly CPU-only")
    if section["split_embargo_days"] < 0 or int(section["split_embargo_days"]) != section["split_embargo_days"]:
        raise ValueError("Neural split_embargo_days must be a nonnegative integer")
    for name in ("workers", "threads_per_worker", "interop_threads_per_worker", "max_epochs", "patience",
                 "batch_size", "prediction_batch_size", "minimum_training_days", "minimum_validation_days", "minimum_test_days"):
        if isinstance(section[name], bool) or int(section[name]) != section[name] or section[name] < 1:
            raise ValueError(f"pv_neural.{name} must be a positive integer")
    if section["loss"] not in {"mse", "huber"}:
        raise ValueError("Neural loss must be mse or huber")
    if not section["farm_ids"] or len(set(section["farm_ids"])) != len(section["farm_ids"]):
        raise ValueError("Neural farm_ids must be nonempty and unique")
    names = []
    for candidate in section["candidates"]:
        name = candidate["name"]
        if not name or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for x in name):
            raise ValueError("Neural candidate names must be safe lowercase file identifiers")
        if candidate["architecture"] not in {"mlp", "residual", "tcn"} or candidate["features"] not in {"core", "weather"}:
            raise ValueError("Unsupported neural architecture or feature set")
        if candidate["width"] < 1 or candidate["depth"] < 1:
            raise ValueError("Neural width/depth must be positive")
        if candidate["architecture"] == "tcn" and (candidate["kernel_size"] < 1 or candidate["kernel_size"] % 2 == 0):
            raise ValueError("TCN kernel_size must be positive and odd")
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise ValueError("Neural candidates must be nonempty with distinct names")


def prepare_neural_corpus(config, farm_id, destination, progress=None):
    """Reuse operational alignment and calibrated pvlib with raw weather retained."""
    local = deepcopy(config)
    section = config["pv_neural"]
    local["pv_calibration"].update(start_day=section["start_day"],
        end_day_exclusive=section["end_day_exclusive"], delivery_timezone=section["delivery_timezone"])
    frame, farm, settings, provenance = prepare_corpus(local, farm_id, destination, progress, retain_weather=True)
    frame = frame.copy()
    days = frame.delivery_day
    frame["split"] = np.select([days < section["train_end_day_exclusive"],
        days < section["validation_end_day_exclusive"]], ["train", "validation"], default="test")
    embargo = pd.DateOffset(days=int(section["split_embargo_days"]))
    cutoff = (pd.Timestamp(section["train_end_day_exclusive"])-embargo).tz_localize(section["delivery_timezone"]).tz_convert("UTC")
    frame.loc[(frame.split == "train") & (frame.target_available_at_utc > cutoff), "eligible"] = False
    validation_cutoff = (pd.Timestamp(section["validation_end_day_exclusive"])-embargo).tz_localize(section["delivery_timezone"]).tz_convert("UTC")
    frame.loc[(frame.split == "validation") & (frame.target_available_at_utc > validation_cutoff), "eligible"] = False
    for label in ("train", "validation", "test"):
        observed = frame.loc[frame.eligible & frame.daylight & (frame.split == label)].delivery_day.nunique()
        minimum = section["minimum_training_days" if label == "train" else f"minimum_{label}_days"]
        if observed < minimum:
            raise ValueError(f"Farm {farm_id}: insufficient {label} coverage ({observed} days)")
    return frame, farm, settings, provenance


def feature_frame(frame: pd.DataFrame, feature_set: str, timezone: str, *,
                  include_solar_geometry: bool = True,
                  include_national_holiday: bool = False) -> pd.DataFrame:
    """Use only forecast weather and deterministic geometry/time, never fitted PV.

    Constant farm attributes are recorded in the model contract rather than
    repeated as unidentifiable features inside a separate model for each farm.
    Cloud layers and other weather are optional additions to the pvlib inputs.
    No POA, fitted temperature, output coefficient, GCR or observed power enters.
    """
    core = ["forecast_ghi", "forecast_bhi", "temperature_c", "wind_u_10m", "wind_v_10m"]
    extras = ["dew_point_c", "cloud_total_pct", "cloud_low_pct", "cloud_mid_pct", "cloud_high_pct",
              "precipitation_mm", "snow_water_equivalent_m", "visibility_km",
              "wind_u_100m", "wind_v_100m"]
    columns = core + (extras if feature_set == "weather" else [])
    result = frame[columns].astype(float).copy()
    starts = pd.DatetimeIndex(frame.interval_start_utc)
    local = (starts + pd.Timedelta(minutes=7.5)).tz_convert(timezone)
    hour = local.hour + local.minute / 60 + local.second / 3600
    year_fraction = (local.dayofyear - 1 + hour / 24) / np.where(local.is_leap_year, 366, 365)
    for name, angle in (("hour", 2*np.pi*hour/24), ("annual", 2*np.pi*year_fraction),
                        ("month", 2*np.pi*(local.month-1)/12)):
        result[f"{name}_sin"] = np.sin(angle)
        result[f"{name}_cos"] = np.cos(angle)
    if include_national_holiday:
        from .pv_calendar import italian_national_holiday
        result["italian_national_holiday"] = italian_national_holiday(starts, timezone)
    if include_solar_geometry:
        zenith = np.radians(frame.solar_zenith_deg.to_numpy())
        azimuth = np.radians(frame.solar_azimuth_deg.to_numpy())
        result["cos_solar_zenith"] = np.cos(zenith)
        result["solar_azimuth_sin"] = np.sin(azimuth)
        result["solar_azimuth_cos"] = np.cos(azimuth)
    result["lead_hours"] = frame.lead_hours.to_numpy()
    cycles = pd.DatetimeIndex(frame.forecast_cycle_utc)
    result["cycle_hour_sin"] = np.sin(2*np.pi*cycles.hour/24)
    result["cycle_hour_cos"] = np.cos(2*np.pi*cycles.hour/24)
    result["wind_speed_10m"] = np.hypot(frame.wind_u_10m, frame.wind_v_10m)
    return result.replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(features: pd.DataFrame, mask: np.ndarray, section: dict, *,
                    include_missing_indicators: bool = True) -> dict:
    """Fit imputation/scaling only on finite training targets and daylight inputs."""
    selected = features.loc[mask].to_numpy(dtype=float)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0
                       for col in selected.T])
    filled = np.where(np.isfinite(selected), selected, median)
    mean = filled.mean(axis=0)
    scale = np.maximum(filled.std(axis=0), float(section["standardization_floor"]))
    output_columns = list(features)
    if include_missing_indicators:
        output_columns += [f"{name}__missing" for name in features]
    return {"columns": list(features), "median": median.tolist(), "mean": mean.tolist(),
            "scale": scale.tolist(), "clip_standard_deviations": section["input_clip_standard_deviations"],
            "fit_rows": len(selected), "target_columns_used": [],
            "include_missing_indicators": bool(include_missing_indicators),
            "output_columns": output_columns}


def transform_features(features: pd.DataFrame, fitted: dict) -> np.ndarray:
    """Apply persisted train-only statistics and retain missingness indicators."""
    if list(features) != fitted["columns"]:
        raise ValueError("Neural feature order disagrees with the fitted artifact")
    values = features.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    filled = np.where(missing, fitted["median"], values)
    scaled = (filled-fitted["mean"])/fitted["scale"]
    clip = fitted["clip_standard_deviations"]
    output = np.clip(scaled, -clip, clip)
    # Older saved preprocessors omit this field and retain their indicator channels.
    if fitted.get("include_missing_indicators", True):
        output = np.concatenate([output, missing.astype(float)], axis=1)
    return output.astype(np.float32)


def pack_sequences(frame: pd.DataFrame, features: np.ndarray, rating: float) -> dict:
    """Pad each date/auction separately, retaining row indices for exact reconstruction."""
    groups = list(frame.groupby(["delivery_day", "auction"], sort=True).indices.values())
    maximum = max(map(len, groups))
    shape = (len(groups), maximum)
    x = np.zeros((*shape, features.shape[1]), dtype=np.float32)
    y = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    daylight = np.zeros(shape, dtype=bool)
    months = np.zeros(shape, dtype=np.int64)
    auctions = np.zeros(shape, dtype=np.int64)
    indices = np.full(shape, -1, dtype=np.int64)
    split, keys = [], []
    auction_ids = {name: i for i, name in enumerate(("DA", "IDA1", "IDA2", "IDA3"))}
    for i, positions in enumerate(groups):
        sample = frame.iloc[positions]
        if sample.split.nunique() != 1 or sample.forecast_cycle_utc.nunique() != 1:
            raise ValueError("A neural sequence must not cross split or weather-vintage boundaries")
        starts = pd.DatetimeIndex(sample.interval_start_utc)
        if len(starts) > 1 and not np.all(np.diff(starts.as_unit("ns").asi8) == pd.Timedelta(minutes=15).value):
            raise ValueError("Neural sequence must preserve native quarter-hour adjacency")
        size = len(positions)
        x[i, :size] = features[positions]
        raw = sample.measured_ac_mw.to_numpy(dtype=float)/rating
        y[i, :size] = np.where(np.isfinite(raw), raw, 0)
        mask[i, :size] = (sample.eligible & sample.daylight).to_numpy()
        daylight[i, :size] = sample.daylight.to_numpy()
        months[i, :size] = pd.Timestamp(sample.delivery_day.iloc[0]).month-1
        auctions[i, :size] = auction_ids[sample.auction.iloc[0]]
        indices[i, :size] = positions
        split.append(sample.split.iloc[0])
        keys.append((sample.delivery_day.iloc[0], sample.auction.iloc[0]))
    return {"x": x, "y": y, "mask": mask, "daylight": daylight,
            "month": months, "auction": auctions, "indices": indices,
            "split": np.asarray(split), "keys": keys}


def unpack_predictions(packed: dict, prediction: np.ndarray, rows: int, rating: float) -> np.ndarray:
    """Restore original corpus order, rejecting gaps, duplicates or nonfinite power."""
    result = np.full(rows, np.nan)
    valid = packed["indices"] >= 0
    indices = packed["indices"][valid]
    if len(indices) != rows or len(np.unique(indices)) != rows:
        raise ValueError("Neural sequence reconstruction changed the row population")
    result[indices] = prediction[valid]*rating
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > rating+1e-7):
        raise ValueError("Neural predictions violate the finite AC output contract")
    return result
