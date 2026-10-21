"""Read PV observations and forecast vintages without altering source values.

Historical timestamps mark the start of 15-minute intervals. Forecast radiation
timestamps mark the end of hourly intervals; a forecast at 11:00 UTC therefore
describes [10:00, 11:00) UTC. All timestamps returned here are UTC aware.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
import xarray as xr


_HISTORICAL_VARIABLES = ("power", "ghi", "fraction_diffuse")
_INTERVAL_FORECAST_VARIABLES = frozenset(
    ("shortwave_radiation", "direct_radiation", "precipitation")
)
_FORECAST_DIMS = ("farm_id", "cycle_time", "lead_time")


# Interpret date-only window endpoints as complete UTC calendar days.
def _window_bounds(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive UTC bounds for historical interval start times."""
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    start_time = start_time.tz_localize("UTC") if start_time.tzinfo is None else start_time.tz_convert("UTC")
    end_time = end_time.tz_localize("UTC") if end_time.tzinfo is None else end_time.tz_convert("UTC")
    if isinstance(end, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end.strip()):
        end_time += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if start_time > end_time:
        raise ValueError("Window start must be no later than window end")
    return start_time, end_time


# Keep raw stores read-only and join static metadata later by the farm key.
def load_inputs(zarr_path: str | Path, parquet_path: str | Path) -> tuple[xr.Dataset, pd.DataFrame]:
    """Open the Zarr dataset and read the independent farm metadata table."""
    dataset = xr.open_zarr(Path(zarr_path))
    metadata = pd.read_parquet(Path(parquet_path))
    if "farm_id" not in metadata or "total_power" not in metadata:
        raise ValueError("Parquet metadata requires farm_id and total_power columns")
    if metadata["farm_id"].isna().any() or metadata["farm_id"].duplicated().any():
        raise ValueError("Parquet farm_id values must be unique and nonmissing")
    return dataset, metadata


# Report shape, timing, and expected missingness without changing any values.
def validate_inputs(
    dataset: xr.Dataset,
    metadata: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Validate grid conventions and describe raw missingness and ranges.

    Forecast missingness is evaluated over the whole store. Missing values with
    ``valid_time`` outside the historical date range are expected. Lead zero is
    additionally expected missing for interval means and accumulations, since
    no preceding hourly interval is represented at that lead. Range statistics
    are descriptive: negative temperature and wind components can be normal.
    """
    start_time, end_time = _window_bounds(start, end)
    required_coords = ("farm_id", "time", "cycle_time", "lead_time", "valid_time")
    absent = [name for name in required_coords if name not in dataset.coords]
    if absent:
        raise ValueError(f"Zarr is missing required coordinates: {absent}")

    farm_ids = np.asarray(dataset.farm_id.values)
    times = np.asarray(dataset.time.values, dtype="datetime64[ns]")
    cycles = np.asarray(dataset.cycle_time.values, dtype="datetime64[ns]")
    leads = np.asarray(dataset.lead_time.values, dtype="timedelta64[ns]")
    valid_times = np.asarray(dataset.valid_time.values, dtype="datetime64[ns]")
    expected_valid_times = cycles[:, None] + leads[None, :]

    metadata_ids = set(metadata["farm_id"].tolist())
    dataset_ids = set(farm_ids.tolist())
    grid = {
        "historical_15_minute": bool(len(times) > 0 and np.all(np.diff(times) == np.timedelta64(15, "m"))),
        "forecast_cycles_6_hour": bool(len(cycles) > 0 and np.all(np.diff(cycles) == np.timedelta64(6, "h"))),
        "forecast_leads_hourly": bool(
            len(leads) > 0
            and leads[0] == np.timedelta64(0, "h")
            and np.all(np.diff(leads) == np.timedelta64(1, "h"))
        ),
        "valid_time_equals_cycle_plus_lead": bool(
            valid_times.shape == expected_valid_times.shape
            and np.array_equal(valid_times, expected_valid_times)
        ),
        "farm_metadata_match": dataset_ids == metadata_ids,
    }

    # The original file's date range determines whether forecast boundary NaNs are expected.
    outside = (valid_times < times[0]) | (valid_times > times[-1])
    zero_lead = leads == np.timedelta64(0, "h")
    forecast_missing: list[dict[str, Any]] = []
    for name, variable in dataset.data_vars.items():
        if set(variable.dims) != set(_FORECAST_DIMS):
            continue
        values = np.asarray(variable.transpose(*_FORECAST_DIMS).values)
        is_missing = np.asarray(pd.isna(values), dtype=bool)
        finite = np.isfinite(values)
        boundary = np.broadcast_to(outside, is_missing.shape)
        interval_kind = variable.attrs.get("temporal_semantics") in (
            "interval_average",
            "interval_accumulation",
        ) or name in _INTERVAL_FORECAST_VARIABLES
        zero = np.broadcast_to((~outside) & zero_lead[None, :], is_missing.shape) if interval_kind else np.zeros_like(is_missing)
        unexpected = is_missing & ~(boundary | zero)
        unexpected_present = ~is_missing & (boundary | zero)
        forecast_missing.append(
            {
                "variable": name,
                "temporal_semantics": variable.attrs.get("temporal_semantics", "unspecified"),
                "missing_total": int(is_missing.sum()),
                "finite_min": float(values[finite].min()) if finite.any() else None,
                "finite_max": float(values[finite].max()) if finite.any() else None,
                "negative_count": int((finite & (values < 0)).sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "expected_boundary_missing": int((is_missing & boundary).sum()),
                "expected_lead_zero_missing": int((is_missing & zero).sum()),
                "unexpected_missing": int(unexpected.sum()),
                "present_in_expected_missing_region": int(unexpected_present.sum()),
            }
        )

    # Keep a separate descriptive check for the two horizontal irradiance components.
    radiation_excess: dict[str, Any] | None = None
    if "direct_radiation" in dataset and "shortwave_radiation" in dataset:
        direct = np.asarray(dataset["direct_radiation"].transpose(*_FORECAST_DIMS).values)
        shortwave = np.asarray(dataset["shortwave_radiation"].transpose(*_FORECAST_DIMS).values)
        finite_pair = np.isfinite(direct) & np.isfinite(shortwave)
        excess = direct - shortwave
        exceeds = finite_pair & (excess > 0)
        radiation_excess = {
            "count": int(exceeds.sum()),
            "max_excess_w_m2": float(excess[exceeds].max()) if exceeds.any() else None,
            "finite_pair_count": int(finite_pair.sum()),
        }

    # Count raw observed nulls in the requested historical window, by farm.
    historical = dataset.sel(time=slice(start_time.tz_localize(None), end_time.tz_localize(None)))
    historical_missing: list[dict[str, Any]] = []
    for farm_id in farm_ids:
        row: dict[str, Any] = {"farm_id": int(farm_id), "intervals": int(historical.sizes.get("time", 0))}
        for name in _HISTORICAL_VARIABLES:
            row[f"{name}_missing"] = (
                int(np.asarray(pd.isna(historical[name].sel(farm_id=farm_id).values)).sum())
                if name in historical
                else None
            )
        historical_missing.append(row)

    return {
        "timezone": dataset.attrs.get("timezone", "unspecified"),
        "window_start_utc": start_time.isoformat(),
        "window_end_utc": end_time.isoformat(),
        "historical_start_utc": pd.Timestamp(times[0], tz="UTC").isoformat(),
        "historical_end_utc": pd.Timestamp(times[-1], tz="UTC").isoformat(),
        "historical_intervals": int(len(times)),
        "forecast_cycles": int(len(cycles)),
        "forecast_leads": int(len(leads)),
        "farm_ids": [int(farm_id) for farm_id in farm_ids],
        "metadata_only_farm_ids": sorted(int(value) for value in metadata_ids - dataset_ids),
        "zarr_only_farm_ids": sorted(int(value) for value in dataset_ids - metadata_ids),
        "grid": grid,
        "historical_missing_by_farm": historical_missing,
        "forecast_missing_by_variable": forecast_missing,
        "direct_radiation_exceeds_shortwave": radiation_excess,
        "unexpected_forecast_missing_total": sum(row["unexpected_missing"] for row in forecast_missing),
        "expected_forecast_boundary_missing_total": sum(row["expected_boundary_missing"] for row in forecast_missing),
        "expected_forecast_lead_zero_missing_total": sum(row["expected_lead_zero_missing"] for row in forecast_missing),
    }


# Flatten forecast vintages while preserving their cycle and lead identities.
def _forecast_frame(dataset: xr.Dataset, variables: tuple[str, ...]) -> pd.DataFrame:
    """Represent selected forecast arrays in tidy farm/cycle/lead rows."""
    farm_ids = np.asarray(dataset.farm_id.values)
    cycles = np.asarray(dataset.cycle_time.values, dtype="datetime64[ns]")
    leads = np.asarray(dataset.lead_time.values, dtype="timedelta64[ns]")
    valid_times = np.asarray(dataset.valid_time.values, dtype="datetime64[ns]")
    n_farms, n_cycles, n_leads = len(farm_ids), len(cycles), len(leads)
    rows_per_farm = n_cycles * n_leads
    frame = pd.DataFrame(
        {
            "farm_id": np.repeat(farm_ids, rows_per_farm),
            "cycle_time_utc": pd.to_datetime(np.tile(np.repeat(cycles, n_leads), n_farms), utc=True),
            "valid_time_utc": pd.to_datetime(np.tile(valid_times.ravel(), n_farms), utc=True),
            "lead_hours": np.tile(np.tile(leads / np.timedelta64(1, "h"), n_cycles), n_farms).astype(int),
        }
    )
    for name in variables:
        if name in dataset:
            frame[name] = np.asarray(dataset[name].transpose(*_FORECAST_DIMS).values).ravel()
        else:
            frame[name] = np.nan
    return frame


# Keep each historical quarter-hour and attach the latest usable prior vintage.
def build_historical_context(
    dataset: xr.Dataset,
    metadata: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    latitude: float,
    longitude: float,
    availability_lag_hours: float,
) -> pd.DataFrame:
    """Return raw PV/GHI observations, solar elevation, and forecast context.

    At each historical interval, forecast irradiance refers to the enclosing
    hour ending at ``forecast_hour_end_utc``. The chosen cycle is the newest
    with ``cycle_time + availability_lag <= hour_start``. No irradiance is
    interpolated, shifted, clipped, or filled.
    """
    start_time, end_time = _window_bounds(start, end)
    historical = dataset.sel(time=slice(start_time.tz_localize(None), end_time.tz_localize(None)))
    times = pd.DatetimeIndex(pd.to_datetime(historical.time.values, utc=True))
    farm_ids = np.asarray(dataset.farm_id.values)
    if len(times) == 0:
        raise ValueError("The requested historical window contains no observations")
    if "power" not in historical or "ghi" not in historical:
        raise ValueError("Historical Zarr data require power and ghi")
    if set(farm_ids.tolist()) != set(metadata.farm_id.tolist()):
        raise ValueError("Zarr and Parquet farm_id sets differ")

    context = pd.DataFrame(
        {
            "farm_id": np.repeat(farm_ids, len(times)),
            "time_utc": pd.DatetimeIndex(np.tile(times.values, len(farm_ids)), tz="UTC"),
        }
    )
    for source, target in (
        ("power", "power_ac_kw"),
        ("ghi", "ghi_w_m2"),
        ("fraction_diffuse", "fraction_diffuse"),
    ):
        context[target] = (
            np.asarray(historical[source].transpose("farm_id", "time").values).ravel()
            if source in historical
            else np.nan
        )
    metadata_columns = metadata.rename(columns={"total_power": "total_power_ac_kw"})
    context = context.merge(metadata_columns, on="farm_id", how="left", validate="many_to_one", sort=False)

    # Evaluate the proxy sun position at the midpoint of each source interval.
    midpoint_times = times + pd.Timedelta(minutes=7, seconds=30)
    solar_elevation = pvlib.solarposition.get_solarposition(
        midpoint_times, latitude=latitude, longitude=longitude
    )["elevation"].to_numpy(dtype=float)
    context["solar_elevation_deg"] = np.tile(solar_elevation, len(farm_ids))

    # A forecast hour is [hour_end - 1 hour, hour_end). Retain the cycle key.
    context["forecast_hour_end_utc"] = context.time_utc.dt.floor("h") + pd.Timedelta(hours=1)
    forecast = _forecast_frame(dataset, ("shortwave_radiation", "direct_radiation"))
    forecast = forecast.rename(
        columns={
            "cycle_time_utc": "forecast_cycle_time_utc",
            "valid_time_utc": "forecast_hour_end_utc",
            "lead_hours": "forecast_lead_hours",
            "shortwave_radiation": "forecast_shortwave_w_m2",
            "direct_radiation": "forecast_direct_w_m2",
        }
    )
    forecast = forecast.loc[
        forecast.forecast_hour_end_utc.between(
            context.forecast_hour_end_utc.min(), context.forecast_hour_end_utc.max()
        )
    ].copy()
    forecast["forecast_available_time_utc"] = forecast.forecast_cycle_time_utc + pd.to_timedelta(
        availability_lag_hours, unit="h"
    )
    eligible = forecast.forecast_available_time_utc <= forecast.forecast_hour_end_utc - pd.Timedelta(hours=1)
    latest = (
        forecast.loc[eligible]
        .sort_values(["farm_id", "forecast_hour_end_utc", "forecast_cycle_time_utc"])
        .drop_duplicates(["farm_id", "forecast_hour_end_utc"], keep="last")
    )
    context = context.merge(
        latest[
            [
                "farm_id",
                "forecast_hour_end_utc",
                "forecast_cycle_time_utc",
                "forecast_available_time_utc",
                "forecast_lead_hours",
                "forecast_shortwave_w_m2",
                "forecast_direct_w_m2",
            ]
        ],
        on=["farm_id", "forecast_hour_end_utc"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return context


# Align an hourly forecast only with its exact four source quarter-hours.
def _complete_hourly_ghi(context: pd.DataFrame) -> pd.DataFrame:
    """Average four available GHI intervals into each forecast target hour."""
    observations = context[["farm_id", "time_utc", "ghi_w_m2"]].copy()
    observations["valid_time_utc"] = observations.time_utc.dt.floor("h") + pd.Timedelta(hours=1)
    observations["canonical_quarter"] = (
        observations.time_utc.dt.minute.isin((0, 15, 30, 45))
        & observations.time_utc.dt.second.eq(0)
        & observations.time_utc.dt.microsecond.eq(0)
        & observations.time_utc.dt.nanosecond.eq(0)
    )
    aggregated = observations.groupby(["farm_id", "valid_time_utc"], sort=False).agg(
        historical_ghi_w_m2=("ghi_w_m2", "mean"),
        quarter_hours=("time_utc", "size"),
        nonmissing_quarter_hours=("ghi_w_m2", "count"),
        unique_quarter_hours=("time_utc", "nunique"),
        canonical_quarter_hours=("canonical_quarter", "sum"),
    )
    complete = aggregated.loc[
        (aggregated.quarter_hours == 4)
        & (aggregated.nonmissing_quarter_hours == 4)
        & (aggregated.unique_quarter_hours == 4)
        & (aggregated.canonical_quarter_hours == 4)
    ].reset_index()
    return complete[["farm_id", "valid_time_utc", "historical_ghi_w_m2"]]


# Summarize all forecast vintages separately by farm and lead time.
def build_forecast_checks(
    dataset: xr.Dataset,
    context: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    availability_lag_hours: float = 4,
) -> pd.DataFrame:
    """Compare hourly shortwave forecasts with complete satellite GHI hours.

    Error equals forecast minus historical GHI. Rows at every lead are retained
    for diagnostic skill assessment, including leads unavailable before their
    target hour starts; ``eligible_before_hour_start`` identifies operational
    lead times. A complete GHI hour requires all four 15-minute source values.
    """
    start_time, end_time = _window_bounds(start, end)
    complete = _complete_hourly_ghi(context)
    forecast = _forecast_frame(dataset, ("shortwave_radiation",))
    forecast = forecast.loc[
        forecast.valid_time_utc.between(start_time, end_time + pd.Timedelta(hours=1))
    ]
    joined = forecast.merge(complete, on=["farm_id", "valid_time_utc"], how="inner", validate="many_to_one")
    joined = joined.loc[joined.shortwave_radiation.notna()].copy()
    joined["error_w_m2"] = joined.shortwave_radiation - joined.historical_ghi_w_m2
    joined["absolute_error_w_m2"] = joined.error_w_m2.abs()
    joined["squared_error_w_m4"] = joined.error_w_m2.pow(2)
    joined["daylight"] = joined.historical_ghi_w_m2 >= 50.0
    joined["daylight_absolute_error_w_m2"] = joined.absolute_error_w_m2.where(joined.daylight)

    result = joined.groupby(["farm_id", "lead_hours"], sort=True).agg(
        compared_hours=("error_w_m2", "size"),
        mean_error_w_m2=("error_w_m2", "mean"),
        mae_w_m2=("absolute_error_w_m2", "mean"),
        mean_squared_error_w_m4=("squared_error_w_m4", "mean"),
        observed_ghi_mean_w_m2=("historical_ghi_w_m2", "mean"),
        forecast_shortwave_mean_w_m2=("shortwave_radiation", "mean"),
        daylight_compared_hours=("daylight", "sum"),
        daylight_mae_w_m2=("daylight_absolute_error_w_m2", "mean"),
    ).reset_index()
    full_index = pd.MultiIndex.from_product(
        [np.asarray(dataset.farm_id.values), (dataset.lead_time.values / np.timedelta64(1, "h")).astype(int)],
        names=["farm_id", "lead_hours"],
    )
    result = result.set_index(["farm_id", "lead_hours"]).reindex(full_index).reset_index()
    result["compared_hours"] = result.compared_hours.fillna(0).astype(int)
    result["daylight_compared_hours"] = result.daylight_compared_hours.fillna(0).astype(int)
    result["rmse_w_m2"] = np.sqrt(result.pop("mean_squared_error_w_m4"))
    result["eligible_before_hour_start"] = result.lead_hours >= availability_lag_hours + 1
    return result


# Expose both raw-level and change correlations without shifting observations.
def build_lag_checks(context: pd.DataFrame, max_lag_quarters: int = 4) -> pd.DataFrame:
    """Calculate power/GHI correlations over symmetric 15-minute offsets.

    A positive ``lag_minutes`` compares current power with earlier GHI, i.e.
    power trails irradiance. No lag is applied to the audit's raw context.
    """
    if max_lag_quarters < 0:
        raise ValueError("max_lag_quarters must be nonnegative")
    rows: list[dict[str, Any]] = []
    for farm_id, farm in context.groupby("farm_id", sort=True):
        farm = farm.sort_values("time_utc")
        times = farm.time_utc.reset_index(drop=True)
        power = pd.Series(farm.power_ac_kw.to_numpy(dtype=float))
        ghi = pd.Series(farm.ghi_w_m2.to_numpy(dtype=float))
        consecutive = times.diff().eq(pd.Timedelta(minutes=15))
        power_change = power.diff().where(consecutive)
        ghi_change = ghi.diff().where(consecutive)
        for lag in range(-max_lag_quarters, max_lag_quarters + 1):
            aligned = (times - times.shift(lag)).eq(pd.Timedelta(minutes=15 * lag))
            shifted_ghi = ghi.shift(lag).where(aligned)
            shifted_change = ghi_change.shift(lag).where(aligned)
            rows.append(
                {
                    "farm_id": int(farm_id),
                    "lag_quarters": lag,
                    "lag_minutes": lag * 15,
                    "paired_levels": int((power.notna() & shifted_ghi.notna()).sum()),
                    "power_ghi_correlation": power.corr(shifted_ghi),
                    "paired_changes": int((power_change.notna() & shifted_change.notna()).sum()),
                    "change_correlation": power_change.corr(shifted_change),
                }
            )
    return pd.DataFrame(rows)
