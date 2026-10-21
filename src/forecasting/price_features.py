"""Configurable linear price features with explicit local-day history.

Only lagged observed prices enter a delivery day's price features. Forecast
fundamentals must already have been aligned to the native products by the
caller; this module never substitutes realized demand or renewable generation.
Retrieved API revisions remain retrospective inputs, not verified vintages.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .price_experiment import build_retrospective_features
from .prices import FEATURES


def _positive_integers(settings: Mapping[str, Any], key: str, default: Sequence[int]) -> list[int]:
    """Reject ambiguous lag settings rather than silently rounding them."""
    values = settings.get(key, default)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a list of positive integer calendar-day offsets")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} must contain positive integer calendar-day offsets")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate offsets")
    return result


def _complete_daily_statistics(frame: pd.DataFrame, timezone_name: str,
                               profile_block_hours: int = 6) -> dict[int, pd.DataFrame]:
    """Summarize complete local days, leaving every missing calendar day empty.

    Exact UTC product grids determine completeness, so both daylight-saving
    days and repeated local hours retain their actual native products. A day
    with a missing product contributes no daily statistic, even if its mean
    could otherwise be calculated from the remaining observations.
    """
    result = {}
    block_names = [f"block_{hour:02d}_{hour + profile_block_hours:02d}_mean"
                   for hour in range(0, 24, profile_block_hours)]
    statistics = ["mean", "min", "max", "std", "spread", "last", *block_names]
    for minutes, portion in frame.groupby("native_interval_minutes", sort=True):
        daily_rows = []
        for day_name, day_rows in portion.groupby("delivery_day", sort=True):
            local_day = date.fromisoformat(str(day_name))
            midnight = pd.Timestamp(local_day, tz=timezone_name)
            following = pd.Timestamp(local_day + timedelta(days=1), tz=timezone_name)
            expected = pd.date_range(midnight.tz_convert("UTC"), following.tz_convert("UTC"),
                                     freq=f"{int(minutes)}min", inclusive="left")
            ordered = day_rows.sort_values("product_start")
            observed = pd.DatetimeIndex(ordered.product_start)
            complete = (observed.as_unit("ns").equals(expected.as_unit("ns"))
                        and ordered.product_end.eq(ordered.product_start + pd.Timedelta(minutes=int(minutes))).all())
            values = ordered.price_eur_mwh.to_numpy(float)
            summary = {"day": pd.Timestamp(local_day), "mean": values.mean() if complete else np.nan,
                       "min": values.min() if complete else np.nan,
                       "max": values.max() if complete else np.nan,
                       "std": values.std(ddof=0) if complete else np.nan,
                       "spread": np.ptp(values) if complete else np.nan,
                       "last": values[-1] if complete else np.nan}
            local_hour = ordered.product_start.dt.tz_convert(timezone_name).dt.hour
            for hour, name in zip(range(0, 24, profile_block_hours), block_names):
                in_block = (local_hour >= hour) & (local_hour < hour + profile_block_hours)
                summary[name] = values[in_block].mean() if complete and in_block.any() else np.nan
            daily_rows.append(summary)
        daily = pd.DataFrame(daily_rows).set_index("day")
        # A regular calendar index prevents rolling windows from jumping gaps.
        all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        result[int(minutes)] = daily.reindex(all_days).loc[:, statistics]
    return result


def _historical_daily_features(frame: pd.DataFrame, timezone_name: str,
                               summary_lags: Sequence[int], windows: Sequence[int],
                               profile_lags: Sequence[int], profile_block_hours: int) -> tuple[list[str], list[str], list[str]]:
    """Map strictly earlier complete-day summaries back to native products."""
    daily_sources = _complete_daily_statistics(frame, timezone_name, profile_block_hours)
    summaries, rolling, profiles = [], [], []
    for lag in summary_lags:
        for statistic in ("mean", "min", "max", "spread"):
            name = f"daily_lag_{lag}d_{statistic}"
            frame[name] = np.nan
            summaries.append(name)
    for window in windows:
        for statistic in ("mean", "mean_std", "min", "max", "spread_mean"):
            name = f"rolling_{window}d_{statistic}"
            frame[name] = np.nan
            rolling.append(name)
    profile_statistics = ["last", *[f"block_{hour:02d}_{hour + profile_block_hours:02d}_mean"
                                   for hour in range(0, 24, profile_block_hours)]]
    for lag in profile_lags:
        for statistic in profile_statistics:
            name = f"profile_lag_{lag}d_{statistic}"
            frame[name] = np.nan
            profiles.append(name)
    for minutes, indexes in frame.groupby("native_interval_minutes", sort=True).groups.items():
        daily = daily_sources[int(minutes)]
        target_days = pd.to_datetime(frame.loc[indexes, "delivery_day"])
        for lag in summary_lags:
            source_days = target_days - pd.Timedelta(days=int(lag))
            for statistic in ("mean", "min", "max", "spread"):
                frame.loc[indexes, f"daily_lag_{lag}d_{statistic}"] = daily[statistic].reindex(source_days).to_numpy()
        for lag in profile_lags:
            source_days = target_days - pd.Timedelta(days=int(lag))
            for statistic in profile_statistics:
                frame.loc[indexes, f"profile_lag_{lag}d_{statistic}"] = daily[statistic].reindex(source_days).to_numpy()
        for window in windows:
            # Every statistic first excludes the target day, then requires a
            # full consecutive window of complete prior local delivery days.
            previous = daily.shift(1)
            aggregates = {
                "mean": previous["mean"].rolling(window, min_periods=window).mean(),
                "mean_std": previous["mean"].rolling(window, min_periods=window).std(ddof=0),
                "min": previous["min"].rolling(window, min_periods=window).min(),
                "max": previous["max"].rolling(window, min_periods=window).max(),
                "spread_mean": previous["spread"].rolling(window, min_periods=window).mean(),
            }
            for statistic, values in aggregates.items():
                frame.loc[indexes, f"rolling_{window}d_{statistic}"] = values.reindex(target_days).to_numpy()
    return summaries, rolling, profiles


def _forecast_features(frame: pd.DataFrame, settings: Mapping[str, Any]) -> list[str]:
    """Derive regional net demand and shares only from explicit forecast fields."""
    configured = settings.get("fundamental_columns", ["load_forecast_mw", "solar_forecast_mw", "wind_forecast_mw"])
    if not isinstance(configured, (list, tuple)) or any(not isinstance(name, str) for name in configured):
        raise ValueError("fundamental_columns must list aligned forecast column names")
    if len(set(configured)) != len(configured):
        raise ValueError("fundamental_columns must not contain duplicate names")
    names = [name for name in configured if name in frame.columns]
    for name in names:
        # Preserve missing forecasts; infinite values must never enter a fit.
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(float)
        if np.isinf(values).any():
            raise ValueError(f"Aligned forecast column {name} contains infinite values")
        frame[name] = values
    canonical = {"load_forecast_mw", "solar_forecast_mw", "wind_forecast_mw"}
    if canonical.issubset(names):
        frame["residual_load_forecast_mw"] = frame.load_forecast_mw - frame.solar_forecast_mw - frame.wind_forecast_mw
        names.append("residual_load_forecast_mw")
    if "load_forecast_mw" in names:
        positive_load = frame.load_forecast_mw.where(frame.load_forecast_mw > 0)
        for renewable in ("solar", "wind"):
            source = f"{renewable}_forecast_mw"
            if source in names:
                name = f"{renewable}_forecast_share"
                frame[name] = frame[source] / positive_load
                names.append(name)
    return names


def build_linear_price_features(native_prices: pd.DataFrame, timezone_name: str = "Europe/Rome", *,
                                settings: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construct a small set of reusable feature groups for linear candidates.

    ``native_prices`` comes from ``prepare_native_prices`` and may include
    forecast fundamentals already aligned to each product. All lags use local
    calendar dates, clock minutes, DST fold, and native product duration. Daily
    summaries require every native product; rolling histories require every
    complete calendar day. No missing observation is filled or interpolated.

    Configurable keys are ``lag_days`` (default 1/2/3/7/14),
    ``daily_summary_lags`` (1/7), ``rolling_windows`` (7/28),
    ``daily_profile_lags`` (1/2/3/7), ``profile_block_hours`` (6),
    ``holiday_dates`` (explicit ISO dates), and ``fundamental_columns``.
    Holidays are included only when an explicit calendar has been supplied.
    """
    settings = {} if settings is None else settings
    lag_days = _positive_integers(settings, "lag_days", [1, 2, 3, 7, 14])
    summary_lags = _positive_integers(settings, "daily_summary_lags", [1, 7])
    windows = _positive_integers(settings, "rolling_windows", [7, 28])
    profile_lags = _positive_integers(settings, "daily_profile_lags", [1, 2, 3, 7])
    profile_block_hours = settings.get("profile_block_hours", 6)
    if (isinstance(profile_block_hours, bool) or not isinstance(profile_block_hours, int)
            or profile_block_hours <= 0 or 24 % profile_block_hours):
        raise ValueError("profile_block_hours must be a positive integer divisor of 24")
    required = {"product_start", "product_end", "native_interval_minutes", "price_eur_mwh"}
    if native_prices.empty or not required.issubset(native_prices.columns):
        raise ValueError("Nonempty prepared native prices are required")
    frame = native_prices.copy().reset_index(drop=True)
    for name in ("product_start", "product_end"):
        stamps = pd.to_datetime(frame[name], errors="raise")
        if stamps.dt.tz is None or stamps.isna().any():
            raise ValueError("Native product timestamps must be timezone-aware and nonmissing")
        frame[name] = stamps.dt.tz_convert("UTC")
    values = frame.price_eur_mwh.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Native prices must be finite")
    durations = frame.native_interval_minutes.to_numpy(float)
    if not np.isin(durations, [15, 30, 60]).all():
        raise ValueError("Native product durations must be 15, 30, or 60 minutes")
    observed_durations = (frame.product_end - frame.product_start).dt.total_seconds().to_numpy() / 60
    if not np.equal(durations, observed_durations).all():
        raise ValueError("Prepared product duration differs from its declared native duration")
    frame = frame.sort_values("product_start").reset_index(drop=True)
    if (frame.product_start.iloc[1:].to_numpy() < frame.product_end.iloc[:-1].to_numpy()).any():
        raise ValueError("Native price products must not overlap")
    frame = build_retrospective_features(frame, timezone_name)
    zone = ZoneInfo(timezone_name)
    clocks = [stamp.to_pydatetime().astimezone(zone) for stamp in frame.product_start]
    dates = [stamp.date() for stamp in clocks]
    clock_keys = [(day, stamp.hour, stamp.minute, stamp.fold, int(minutes))
                  for day, stamp, minutes in zip(dates, clocks, frame.native_interval_minutes)]
    lookup = dict(zip(clock_keys, frame.price_eur_mwh.to_numpy(float)))
    for lag in lag_days:
        frame[f"lag_{lag}d"] = [lookup.get((day - timedelta(days=lag), hour, minute, fold, minutes), np.nan)
                               for day, hour, minute, fold, minutes in clock_keys]
    groups = {"calendar": list(FEATURES[:6]), "autoregressive": ["lag_1d", "lag_7d"],
              "autoregressive_extended": [f"lag_{lag}d" for lag in lag_days]}
    # Deterministic delivery-calendar features never inspect prices or demand.
    weekdays = np.array([day.weekday() for day in dates])
    frame["is_weekend"] = (weekdays >= 5).astype(float)
    annual_phase = np.array([(day.timetuple().tm_yday - 1) / (366 if date(day.year, 12, 31).timetuple().tm_yday == 366 else 365)
                             for day in dates])
    frame["year_sin"] = np.sin(2 * np.pi * annual_phase)
    frame["year_cos"] = np.cos(2 * np.pi * annual_phase)
    groups["calendar_extended"] = [*groups["calendar"], "is_weekend", "year_sin", "year_cos"]
    configured_holidays = settings.get("holiday_dates", [])
    if not isinstance(configured_holidays, (list, tuple)):
        raise ValueError("holiday_dates must list explicit ISO calendar dates")
    try:
        holidays = {date.fromisoformat(str(value)) for value in configured_holidays}
    except (TypeError, ValueError) as exc:
        raise ValueError("holiday_dates must list explicit ISO calendar dates") from exc
    if holidays:
        for label, offset in (("is_configured_holiday", 0), ("day_before_configured_holiday", 1),
                              ("day_after_configured_holiday", -1)):
            frame[label] = [float(day + timedelta(days=offset) in holidays) for day in dates]
            groups["calendar_extended"].append(label)
    groups["hour_categories"] = []
    for hour in range(1, 24):
        name = f"hour_{hour:02d}"
        frame[name] = [float(stamp.hour == hour) for stamp in clocks]
        groups["hour_categories"].append(name)
    groups["weekday_categories"] = []
    for weekday in range(1, 7):
        name = f"weekday_{weekday}"
        frame[name] = (weekdays == weekday).astype(float)
        groups["weekday_categories"].append(name)
    groups["daily_summary"], groups["rolling_history"], groups["daily_profile_lags"] = _historical_daily_features(
        frame, timezone_name, summary_lags, windows, profile_lags, profile_block_hours)
    for window in windows:
        groups[f"rolling_history_{window}d"] = [name for name in groups["rolling_history"] if name.startswith(f"rolling_{window}d_")]
    # Interactions let linear coefficients vary with delivery-hour and daytype.
    groups["interactions"] = []
    for left, right in (("is_weekend", "hour_sin"), ("is_weekend", "hour_cos"),
                        ("lag_1d", "is_weekend"), ("lag_7d", "is_weekend"),
                        ("lag_1d", "hour_sin"), ("lag_1d", "hour_cos")):
        name = f"{left}_x_{right}"
        frame[name] = frame[left] * frame[right]
        groups["interactions"].append(name)
    groups["fundamentals"] = _forecast_features(frame, settings)
    # Keep interpretable ablations separate: residual demand is an exact linear
    # combination of its components and should not be added alongside all of
    # them in the same candidate merely to duplicate the design span.
    groups["fundamental_components"] = [name for name in groups["fundamentals"]
                                        if name not in {"residual_load_forecast_mw", "solar_forecast_share", "wind_forecast_share"}]
    groups["residual_load"] = [name for name in groups["fundamentals"] if name == "residual_load_forecast_mw"]
    groups["renewable_shares"] = [name for name in groups["fundamentals"] if name.endswith("_forecast_share")]
    return frame, groups


def build_future_linear_price_features(history_native: pd.DataFrame, future_products: pd.DataFrame,
                                       timezone_name: str = "Europe/Rome", *,
                                       settings: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build an unknown delivery day's features without supplying target prices.

    ``history_native`` contains finite observed native prices from strictly
    earlier local delivery days. ``future_products`` contains one complete
    native civil-day grid and may contain aligned forecast fundamentals. Its
    native duration is inferred from product boundaries when not supplied.
    Nonmissing realized price or actual target columns are rejected; blank
    target columns are discarded. Returned future rows have no target column.

    All price histories use the same implementation as retrospective studies.
    Internally a temporary zero target lets that implementation validate the
    future grid; positive calendar-day lags and shifted daily histories never
    read it. The temporary target never leaves this function. Missing history
    remains NaN and no hourly/quarter-hour regime conversion is introduced.

    This function does not establish source publication or revision
    availability. Supplied source metadata is preserved without certifying
    operational eligibility; callers must enforce their information cutoff.
    """
    required = {"product_start", "product_end"}
    if future_products.empty or not required.issubset(future_products.columns):
        raise ValueError("Future products require one complete native local delivery day")
    if history_native.empty or not {*required, "native_interval_minutes", "price_eur_mwh"}.issubset(history_native.columns):
        raise ValueError("Finite native price history from earlier delivery days is required")
    future = future_products.copy().reset_index(drop=True)
    # Explicitly reject target labels even when the caller names them using a
    # conventional actual/realized alias instead of price_eur_mwh.
    target_aliases = {"price_eur_mwh", "price", "actual", "target", "y", "value", "value_eur_mwh",
                      "target_price_eur_mwh", "observed_price_eur_mwh"}
    target_columns = [name for name in future if str(name).lower() in target_aliases
                      or str(name).lower().startswith(("actual_", "realized_", "realised_"))]
    if any(future[name].notna().any() for name in target_columns):
        raise ValueError("Future realized prices or actual targets must not be supplied")
    future = future.drop(columns=target_columns)
    history = history_native.copy().reset_index(drop=True)
    for frame in (history, future):
        for name in ("product_start", "product_end"):
            timestamps = pd.to_datetime(frame[name], errors="raise")
            if timestamps.isna().any() or timestamps.dt.tz is None:
                raise ValueError("Native product timestamps must be timezone-aware and nonmissing")
            frame[name] = timestamps.dt.tz_convert("UTC")
    future = future.sort_values("product_start").reset_index(drop=True)
    local_days = future.product_start.dt.tz_convert(timezone_name).dt.date
    if local_days.nunique() != 1:
        raise ValueError("Future products must contain exactly one complete local delivery day")
    future_day = local_days.iloc[0]
    if not (history.product_start.dt.tz_convert(timezone_name).dt.date < future_day).all():
        raise ValueError("Every historical delivery day must be strictly before the future delivery day")
    minutes = (future.product_end - future.product_start).dt.total_seconds() / 60
    if minutes.nunique() != 1 or not minutes.isin([15, 30, 60]).all():
        raise ValueError("Future products must share one supported native product duration")
    native_minutes = int(minutes.iloc[0])
    if "native_interval_minutes" in future and not future.native_interval_minutes.eq(native_minutes).all():
        raise ValueError("Future declared native duration disagrees with product boundaries")
    future["native_interval_minutes"] = native_minutes
    midnight = pd.Timestamp(future_day, tz=timezone_name)
    following = pd.Timestamp(future_day + timedelta(days=1), tz=timezone_name)
    expected = pd.date_range(midnight.tz_convert("UTC"), following.tz_convert("UTC"),
                             freq=f"{native_minutes}min", inclusive="left")
    if (not pd.DatetimeIndex(future.product_start).as_unit("ns").equals(expected.as_unit("ns"))
            or not future.product_end.eq(future.product_start + pd.Timedelta(minutes=native_minutes)).all()):
        raise ValueError("Future products must form an exact complete native local delivery-day grid")
    if (history.product_end > future.product_start.iloc[0]).any():
        raise ValueError("Historical price products overlap the future delivery day")
    # The shared builder's histories exclusively read earlier calendar days.
    # Keeping this sentinel local avoids a second, diverging feature formula.
    future["price_eur_mwh"] = 0.
    combined = pd.concat([history, future], ignore_index=True)
    features, groups = build_linear_price_features(combined, timezone_name, settings=settings)
    # Historical source tables can carry additional aliases of their labels;
    # none belongs in the target-free future result, even as an empty column.
    result_target_columns = [name for name in features if str(name).lower() in target_aliases
                             or str(name).lower().startswith(("actual_", "realized_", "realised_"))]
    result = features.loc[features.delivery_day.eq(future_day.isoformat())].drop(columns=result_target_columns)
    return result.reset_index(drop=True), groups
