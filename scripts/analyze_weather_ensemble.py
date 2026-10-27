"""Create plots, tables, and descriptive spread statistics for a farm weather ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = {
    "shortwave_radiation": {
        "meaning": "Downwelling shortwave irradiance; used here as forecast GHI.",
        "units": "W/m²",
        "semantics": "Hourly interval average, labelled at interval end.",
        "current": "The analogous deterministic field feeds the PV conversion as GHI; this ensemble store is not integrated.",
        "future": "Run the PV conversion for each member; retain per-member PV output and derive quantiles/scenarios after conversion.",
    },
    "direct_radiation": {
        "meaning": "Direct shortwave irradiance on the horizontal plane; working interpretation as BHI.",
        "units": "W/m²",
        "semantics": "Hourly interval average, labelled at interval end.",
        "current": "The analogous deterministic field is used as horizontal direct irradiance under a documented working assumption.",
        "future": "Keep the same meaning and unit; convert each member with its matching shortwave member. Confirm source definition before physical calibration.",
    },
    "temperature_2m": {
        "meaning": "Near-surface air temperature at 2 metres.",
        "units": "°C",
        "semantics": "Instantaneous forecast value.",
        "current": "The analogous deterministic value drives the configured module-temperature estimate.",
        "future": "Use member-wise with irradiance and wind in the PV model; do not average weather first if preserving nonlinear PV uncertainty.",
    },
    "dew_point_2m": {
        "meaning": "Near-surface dew-point temperature at 2 metres.",
        "units": "°C",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned but not an additional PV power correction.",
        "future": "Possible humidity or condensation proxy; only use after a defined physical or predictive feature study.",
    },
    "cloud_cover": {
        "meaning": "Total cloud cover.",
        "units": "%",
        "semantics": "Instantaneous forecast value.",
        "current": "Available for diagnostics; not a separate correction in the PV conversion.",
        "future": "Useful ensemble weather feature or diagnostic; avoid double-counting when irradiance already carries cloud effects.",
    },
    "cloud_cover_low": {
        "meaning": "Low-level cloud cover.",
        "units": "%",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned; not used directly in PV power conversion.",
        "future": "Potential cloud-regime or irradiance-uncertainty feature, subject to ablation.",
    },
    "cloud_cover_mid": {
        "meaning": "Mid-level cloud cover.",
        "units": "%",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned; not used directly in PV power conversion.",
        "future": "Potential cloud-regime or irradiance-uncertainty feature, subject to ablation.",
    },
    "cloud_cover_high": {
        "meaning": "High-level cloud cover.",
        "units": "%",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned; not used directly in PV power conversion.",
        "future": "Potential cloud-regime or irradiance-uncertainty feature, subject to ablation.",
    },
    "precipitation": {
        "meaning": "Precipitation amount accumulated over the source interval.",
        "units": "mm",
        "semantics": "Hourly interval accumulation, labelled at interval end.",
        "current": "Stored/aligned; not a direct term in the PV power conversion.",
        "future": "Possible wetness/soiling or weather-regime signal; do not interpret as panel soiling without a separate model.",
    },
    "snow_water_equivalent": {
        "meaning": "Liquid-water equivalent of snow on/in the atmospheric column or surface product, as defined by the source.",
        "units": "m",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned; not used as a PV correction.",
        "future": "Potential snow regime indicator; does not directly measure snow coverage on panels.",
    },
    "visibility": {
        "meaning": "Meteorological visibility distance.",
        "units": "km",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored/aligned; not used in the PV power conversion.",
        "future": "Possible aerosol/fog proxy after validating source definition and predictive value.",
    },
    "wind_u_component_10m": {
        "meaning": "East-west (u) wind component at 10 metres; sign follows the source convention.",
        "units": "m/s",
        "semantics": "Instantaneous forecast value.",
        "current": "The analogous deterministic 10 m u component contributes to wind-speed magnitude for module cooling.",
        "future": "Preserve signed component per member; combine with v component before calculating speed/direction.",
    },
    "wind_v_component_10m": {
        "meaning": "North-south (v) wind component at 10 metres; sign follows the source convention.",
        "units": "m/s",
        "semantics": "Instantaneous forecast value.",
        "current": "The analogous deterministic 10 m v component contributes to wind-speed magnitude for module cooling.",
        "future": "Preserve signed component per member; combine with u component before calculating speed/direction.",
    },
    "wind_u_component_100m": {
        "meaning": "East-west (u) wind component at 100 metres; sign follows the source convention.",
        "units": "m/s",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored but the current PV temperature model uses 10 m wind only.",
        "future": "Potential wind-resource or alternative cooling-height feature; validate height before use.",
    },
    "wind_v_component_100m": {
        "meaning": "North-south (v) wind component at 100 metres; sign follows the source convention.",
        "units": "m/s",
        "semantics": "Instantaneous forecast value.",
        "current": "Stored but the current PV temperature model uses 10 m wind only.",
        "future": "Potential wind-resource or alternative cooling-height feature; validate height before use.",
    },
}

COORDINATES = {
    "farm_id": ("Farm identity; joins the four anonymous plants to metadata.parquet.", "Index over plant dimension; use as a key, never average farms implicitly."),
    "cycle_time": ("Forecast reference/initialization time in UTC.", "Determines vintage availability. Apply the chosen publication lag to the cycle, not to member IDs."),
    "lead_time": ("Elapsed forecast horizon from cycle_time; hourly from 0 to 48 h.", "Aligns forecast values to target delivery intervals; lead time is not a member."),
    "valid_time": ("Valid timestamp for each cycle/lead pair; cycle_time + lead_time.", "Use for delivery alignment. Interval averages and accumulations are labelled at their ending valid_time."),
    "member": ("Ensemble realization identifier, labelled 1 through 50.", "Keep as an explicit scenario/member axis. Equal weights are an assumption unless source documentation supplies weights."),
}

PLOT_VARIABLES = [
    "shortwave_radiation", "direct_radiation", "temperature_2m", "dew_point_2m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_speed_100m", "wind_u_component_10m", "wind_v_component_10m",
    "wind_u_component_100m", "wind_v_component_100m", "precipitation",
    "snow_water_equivalent", "visibility",
]
PLOT_GROUPS = {
    "radiation": ["shortwave_radiation", "direct_radiation"],
    "temperature_clouds": ["temperature_2m", "dew_point_2m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"],
    "wind": ["wind_speed_10m", "wind_speed_100m", "wind_u_component_10m", "wind_v_component_10m", "wind_u_component_100m", "wind_v_component_100m"],
    "precipitation_snow_visibility": ["precipitation", "snow_water_equivalent", "visibility"],
}

ARRAY_CACHE: dict[str, np.ndarray] = {}
VALID_TIME_CACHE: np.ndarray | None = None
CYCLE_INDEX: dict[int, int] = {}
FARM_INDEX: dict[int, int] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=Path("data/farms/nwp_ens_20260101_20260131.zarr"))
    parser.add_argument("--metadata", type=Path, default=Path("data/farms/metadata.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/weather_ensemble_jan2026_7days"))
    parser.add_argument("--plot-start", type=date.fromisoformat, default=date(2026, 1, 2))
    parser.add_argument("--plot-days", type=int, default=7)
    parser.add_argument("--analysis-start", type=date.fromisoformat, default=date(2026, 1, 2))
    parser.add_argument("--analysis-end", type=date.fromisoformat, default=date(2026, 1, 31))
    parser.add_argument("--timezone", default="Europe/Rome")
    parser.add_argument("--availability-lag-hours", type=float, default=6.0)
    parser.add_argument("--decision-buffer-minutes", type=float, default=5.0)
    parser.add_argument("--da-gate-local", default="12:00")
    parser.add_argument("--ida1-gate-local", default="15:00")
    parser.add_argument("--overlay-members", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="replace files in the output directory")
    parser.add_argument("--reuse-existing-plots", action="store_true", help="keep non-empty plot PNGs when resuming an interrupted report")
    args = parser.parse_args()
    if args.plot_days < 1 or args.overlay_members < 1:
        parser.error("plot-days and overlay-members must be positive")
    return args


def cutoff_for(day: date, gate: str, args: argparse.Namespace) -> pd.Timestamp:
    hour, minute = map(int, gate.split(":"))
    local = pd.Timestamp(datetime.combine(day - timedelta(days=1), time(hour, minute)), tz=args.timezone)
    return local.tz_convert("UTC") - pd.Timedelta(minutes=args.decision_buffer_minutes)


def select_cycle(ds: xr.Dataset, delivery_day: date, gate: str, args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    cycles = pd.DatetimeIndex(pd.to_datetime(ds.cycle_time.values, utc=True))
    cutoff = cutoff_for(delivery_day, gate, args)
    available = cycles + pd.Timedelta(hours=args.availability_lag_hours)
    eligible = cycles[(cycles <= cutoff) & (available <= cutoff)]
    if not len(eligible):
        raise ValueError(f"No cycle eligible for {delivery_day} at {gate} cutoff {cutoff}")
    return eligible[-1], cutoff


def local_day_values(ds: xr.Dataset, variable: str, farm_id: int, cycle: pd.Timestamp,
                     delivery_day: date, timezone_name: str) -> np.ndarray:
    """Return a 24-by-member forecast matrix aligned to local delivery-hour starts."""
    if VALID_TIME_CACHE is None or variable not in ARRAY_CACHE:
        raise RuntimeError("Forecast arrays must be loaded before extraction")
    cycle_index = CYCLE_INDEX[cycle.value]
    valid = pd.DatetimeIndex(pd.to_datetime(VALID_TIME_CACHE[cycle_index], utc=True))
    semantics = str(ds[variable].attrs.get("temporal_semantics", "instantaneous"))
    if semantics in {"interval_average", "interval_accumulation"}:
        hour_starts = (valid - pd.Timedelta(hours=1)).tz_convert(timezone_name)
    else:
        hour_starts = valid.tz_convert(timezone_name)
    mask = np.asarray([stamp.date() == delivery_day for stamp in hour_starts])
    hours = np.asarray([stamp.hour for stamp in hour_starts[mask]], dtype=int)
    values = ARRAY_CACHE[variable][FARM_INDEX[farm_id], cycle_index, mask, :]
    result = np.full((24, ds.sizes["member"]), np.nan, dtype=float)
    if len(hours):
        if len(set(hours)) != len(hours):
            raise ValueError(f"Duplicate local delivery-hour values for {variable}, {farm_id}, {delivery_day}")
        result[hours, :] = values
    return result


def all_member_matrix(ds: xr.Dataset, variable: str, farm_id: int, cycle: pd.Timestamp,
                      delivery_day: date, timezone_name: str) -> np.ndarray:
    if variable == "wind_speed_10m":
        u = local_day_values(ds, "wind_u_component_10m", farm_id, cycle, delivery_day, timezone_name)
        v = local_day_values(ds, "wind_v_component_10m", farm_id, cycle, delivery_day, timezone_name)
        return np.hypot(u, v)
    if variable == "wind_speed_100m":
        u = local_day_values(ds, "wind_u_component_100m", farm_id, cycle, delivery_day, timezone_name)
        v = local_day_values(ds, "wind_v_component_100m", farm_id, cycle, delivery_day, timezone_name)
        return np.hypot(u, v)
    return local_day_values(ds, variable, farm_id, cycle, delivery_day, timezone_name)


def point_statistics(matrix: np.ndarray) -> dict[str, float | int]:
    finite = matrix[np.isfinite(matrix)]
    if not len(finite):
        return {"member_count": 0, **{key: np.nan for key in ("mean", "median", "std", "min", "p05", "p10", "p25", "p75", "p90", "p95", "max", "range", "p90_p10", "iqr")}}
    qs = np.nanquantile(finite, [0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
    return {
        "member_count": int(len(finite)), "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(np.min(finite)), "p05": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]),
        "p75": float(qs[3]), "p90": float(qs[4]), "p95": float(qs[5]), "max": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)), "p90_p10": float(qs[4] - qs[1]),
        "iqr": float(qs[3] - qs[2]),
    }


def make_dictionary(ds: xr.Dataset, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, info in VARIABLES.items():
        attrs = ds[name].attrs
        rows.append({
            "zarr_group": name, "group_type": "forecast variable", "meaning": info["meaning"],
            "units": attrs.get("units", info["units"]), "temporal_semantics": attrs.get("temporal_semantics", info["semantics"]),
            "array_dimensions": ", ".join(ds[name].dims), "current_pipeline_use": info["current"], "future_ensemble_use": info["future"],
        })
    for name, (meaning, future) in COORDINATES.items():
        rows.append({
            "zarr_group": name, "group_type": "coordinate", "meaning": meaning,
            "units": str(ds[name].attrs.get("units", "UTC" if name in {"cycle_time", "valid_time"} else "hours" if name == "lead_time" else "identifier")),
            "temporal_semantics": str(ds[name].attrs.get("standard_name", "coordinate")),
            "array_dimensions": ", ".join(ds[name].dims),
            "current_pipeline_use": "New ensemble store is not wired into the current deterministic PV workflow.",
            "future_ensemble_use": future,
        })
    rows.extend([
        {"zarr_group": "wind_speed_10m", "group_type": "derived feature", "meaning": "Horizontal wind speed derived per member as hypot(u10, v10).", "units": "m/s", "temporal_semantics": "Instantaneous; member-wise nonlinear transform.", "array_dimensions": "farm_id, cycle_time, lead_time, member", "current_pipeline_use": "The deterministic PV model uses the analogous 10 m component magnitude for module cooling.", "future_ensemble_use": "Derive separately for every member before computing ensemble spread or PV output."},
        {"zarr_group": "wind_speed_100m", "group_type": "derived feature", "meaning": "Horizontal wind speed derived per member as hypot(u100, v100).", "units": "m/s", "temporal_semantics": "Instantaneous; member-wise nonlinear transform.", "array_dimensions": "farm_id, cycle_time, lead_time, member", "current_pipeline_use": "Not used in current PV conversion.", "future_ensemble_use": "Potential wind-resource or cooling-height sensitivity; do not substitute for 10 m wind without justification."},
    ])
    # Static plant attributes live in metadata.parquet, outside the Zarr hierarchy.
    rows.append({
        "zarr_group": "metadata.parquet", "group_type": "external static metadata", "meaning": "One row per farm: zone, azimuth, tilt, PV type/tracking IDs, AC nameplate and elevation.",
        "units": "Mixed; total_power is kW and geometry is degrees/metres.", "temporal_semantics": "Static row keyed by farm_id.",
        "array_dimensions": f"{len(metadata)} farms", "current_pipeline_use": "Supplies plant nameplate and geometry for the current PV conversion.",
        "future_ensemble_use": "Join by farm_id, then apply each member's forecast through the same plant-specific PV model.",
    })
    return pd.DataFrame(rows)


def make_day_group_page(ds: xr.Dataset, day: date, farm_id: int, cycle: pd.Timestamp, da_cutoff: pd.Timestamp,
                        ida_cutoff: pd.Timestamp, member_ids: np.ndarray, output_png: Path,
                        timezone_name: str, group_name: str, variables: list[str], dpi: int = 170) -> None:
    columns = 2 if len(variables) == 2 else 3
    rows = int(np.ceil(len(variables) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows), sharex=True, squeeze=False)
    axes = axes.ravel()
    member_coords = ds.member.values.astype(int)
    selected_indices = np.linspace(0, len(member_coords) - 1, len(member_ids)).round().astype(int)
    selected_ids = member_coords[selected_indices]
    cmap = plt.get_cmap("tab10")
    x = np.arange(24)
    for index, variable in enumerate(variables):
        ax = axes[index]
        values = all_member_matrix(ds, variable, farm_id, cycle, day, timezone_name)
        p10, median, p90 = np.nanquantile(values, [0.10, 0.50, 0.90], axis=1)
        ax.fill_between(x, p10, p90, color="#aab7c4", alpha=0.38, linewidth=0, label="All 50: p10–p90")
        for color_index, member_id in enumerate(selected_ids):
            idx = int(np.flatnonzero(member_coords == member_id)[0])
            ax.plot(x, values[:, idx], color=cmap(color_index), linewidth=0.9, alpha=0.78)
        ax.plot(x, median, color="#20252b", linewidth=1.8, label="All 50: median")
        info = VARIABLES.get(variable, {})
        unit = info.get("units", "m/s")
        ax.set_title(variable.replace("_", " "), fontsize=9, loc="left")
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(True, color="#d7dce0", linewidth=0.45, alpha=0.8)
        ax.tick_params(labelsize=7)
        if variable in {"shortwave_radiation", "direct_radiation", "precipitation"}:
            ax.set_xlim(-0.5, 23.5)
    for ax in axes[len(variables):]:
        ax.axis("off")
    for ax in axes:
        if ax.axison:
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_xlabel("Local delivery hour (hour-start aligned)", fontsize=8)
    fig.suptitle(
        f"{group_name.replace('_', ' ').title()} ensemble weather — {day.isoformat()} — farm {farm_id}\n"
        f"Vintage {cycle.isoformat()} (available { (cycle + pd.Timedelta(hours=6)).isoformat() }, 6 h assumed); "
        f"DA cutoff {da_cutoff.isoformat()}, IDA1 cutoff {ida_cutoff.isoformat()}",
        fontsize=12, y=0.985,
    )
    legend_handles = [plt.Line2D([0], [0], color=cmap(i), lw=1.2, label=f"member {m}") for i, m in enumerate(selected_ids)]
    legend_handles += [
        plt.Line2D([0], [0], color="#20252b", lw=1.8, label="all 50: median"),
        plt.Line2D([0], [0], color="#aab7c4", lw=6, alpha=0.55, label="all 50: p10–p90"),
    ]
    fig.legend(handles=legend_handles, ncol=6, loc="lower center", fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.065))
    fig.text(0.5, 0.018, "10 sampled members; gray = all-member p10–p90; dark = all-member median. "
             "Radiation and precipitation are interval-start aligned; point variables use valid time.",
             ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=[0.015, 0.19, 0.99, 0.91], h_pad=1.2, w_pad=1.0)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a Markdown table without relying on pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(column).replace("|", "\\|") for column in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.zarr.exists():
        raise FileNotFoundError(args.zarr)
    if not args.metadata.exists():
        raise FileNotFoundError(args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {args.output}; choose another --output or pass --force")
    if args.force:
        # Remove only the exact root-level artifacts written by the earlier, mixed-farm layout.
        for name in (
            "ensemble_variability_summary.csv", "ensemble_variability_by_day_farm.csv",
            "ensemble_variability_by_delivery_hour.csv", "pointwise_ensemble_statistics.csv",
            "source_value_coverage_and_negative_values.csv", "ensemble_seven_day_all_farms.pdf",
        ):
            candidate = args.output / name
            if candidate.is_file():
                candidate.unlink()
        for day_index in range(args.plot_days):
            candidate = args.output / f"ensemble_{(args.plot_start + timedelta(days=day_index)).isoformat()}.pdf"
            if candidate.is_file():
                candidate.unlink()

    ds = xr.open_zarr(args.zarr, consolidated=True)
    required_dims = {"farm_id", "cycle_time", "lead_time", "member"}
    if not required_dims.issubset(ds.sizes):
        raise ValueError(f"Expected explicit ensemble dimensions {sorted(required_dims)}, found {dict(ds.sizes)}")
    if "valid_time" not in ds.coords:
        raise ValueError("Zarr has no valid_time coordinate")
    expected_valid = ds.cycle_time.values[:, None] + ds.lead_time.values[None, :]
    if not np.array_equal(ds.valid_time.values, expected_valid):
        raise ValueError("valid_time does not equal cycle_time + lead_time")
    if not set(VARIABLES).issubset(ds.data_vars):
        raise ValueError("One or more documented forecast variables are absent from the Zarr")

    global VALID_TIME_CACHE, CYCLE_INDEX, FARM_INDEX
    VALID_TIME_CACHE = ds.valid_time.values
    CYCLE_INDEX = {pd.Timestamp(cycle).tz_localize("UTC").value: index for index, cycle in enumerate(ds.cycle_time.values)}
    FARM_INDEX = {int(farm_id): index for index, farm_id in enumerate(ds.farm_id.values)}
    ARRAY_CACHE.clear()
    ARRAY_CACHE.update({name: ds[name].values for name in VARIABLES})

    metadata = pd.read_parquet(args.metadata)
    farm_ids = [int(item) for item in ds.farm_id.values]
    if "farm_id" not in metadata or not set(farm_ids).issubset(set(metadata.farm_id.astype(int))):
        raise ValueError("metadata.parquet does not contain matching farm_id keys")
    metadata = metadata.loc[metadata.farm_id.astype(int).isin(farm_ids)].sort_values("farm_id")
    dictionary = make_dictionary(ds, metadata)
    dictionary.to_csv(args.output / "zarr_group_dictionary.csv", index=False)

    analysis_days = [args.analysis_start + timedelta(days=i) for i in range((args.analysis_end - args.analysis_start).days + 1)]
    plot_days = [args.plot_start + timedelta(days=i) for i in range(args.plot_days)]
    farm_ids = sorted(farm_ids)
    farm_dirs = {farm_id: args.output / f"farm_{farm_id}" for farm_id in farm_ids}
    for directory in farm_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    member_coords = ds.member.values.astype(int)
    if args.overlay_members > len(member_coords):
        raise ValueError(f"Requested {args.overlay_members} overlay members but only {len(member_coords)} exist")
    overlay_indices = np.linspace(0, len(member_coords) - 1, args.overlay_members).round().astype(int)
    overlay_ids = member_coords[overlay_indices]

    gate_records = []
    selected_cycles: dict[date, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = {}
    for day in analysis_days:
        da_cycle, da_cutoff = select_cycle(ds, day, args.da_gate_local, args)
        ida_cycle, ida_cutoff = select_cycle(ds, day, args.ida1_gate_local, args)
        selected_cycles[day] = (da_cycle, da_cutoff, ida_cycle, ida_cutoff)
        gate_records.append({
            "delivery_day": day.isoformat(), "DA_cutoff_utc": da_cutoff.isoformat(), "IDA1_cutoff_utc": ida_cutoff.isoformat(),
            "DA_cycle_utc": da_cycle.isoformat(), "IDA1_cycle_utc": ida_cycle.isoformat(),
            "DA_available_at_assumed_utc": (da_cycle + pd.Timedelta(hours=args.availability_lag_hours)).isoformat(),
            "IDA1_available_at_assumed_utc": (ida_cycle + pd.Timedelta(hours=args.availability_lag_hours)).isoformat(),
            "same_vintage": bool(da_cycle == ida_cycle),
        })
    pd.DataFrame(gate_records).to_csv(args.output / "auction_vintage_selection.csv", index=False)

    # Materialize one member matrix per day/farm/feature and retain complete pointwise quantiles.
    matrices: dict[tuple[date, int, str], np.ndarray] = {}
    point_rows = []
    for day in analysis_days:
        cycle, _, _, _ = selected_cycles[day]
        for farm_id in farm_ids:
            for variable in PLOT_VARIABLES:
                matrix = all_member_matrix(ds, variable, farm_id, cycle, day, args.timezone)
                matrices[(day, farm_id, variable)] = matrix
                unit = VARIABLES.get(variable, {}).get("units", "m/s")
                for hour in range(24):
                    stats = point_statistics(matrix[hour, :])
                    point_rows.append({
                        "delivery_day": day.isoformat(), "farm_id": farm_id, "delivery_hour_local": hour,
                        "forecast_cycle_utc": cycle.isoformat(), "variable": variable, "unit": unit, **stats,
                    })
    point_frame = pd.DataFrame(point_rows)
    for farm_id, directory in farm_dirs.items():
        point_frame.loc[point_frame.farm_id.eq(farm_id)].to_csv(directory / "pointwise_ensemble_statistics.csv", index=False)

    summary_rows = []
    for (farm_id, variable), subset in point_frame.groupby(["farm_id", "variable"], sort=True):
        expected_points = len(analysis_days) * 24
        valid = subset.loc[subset.member_count.ge(2)]
        summary_rows.append({
            "farm_id": int(farm_id),
            "variable": variable,
            "unit": VARIABLES.get(variable, {}).get("units", "m/s"),
            "temporal_semantics": VARIABLES.get(variable, {}).get("semantics", "Derived instantaneous magnitude from u/v"),
            "expected_delivery_hours": expected_points,
            "valid_delivery_hours": int(len(valid)),
            "coverage_pct": 100 * len(valid) / expected_points if expected_points else np.nan,
            "mean_finite_members_per_point": float(valid.member_count.mean()) if len(valid) else np.nan,
            "full_50_member_points_pct": 100 * (valid.member_count.eq(50).mean()) if len(valid) else np.nan,
            "points_with_nonzero_p90_p10_pct": 100 * (valid.p90_p10.gt(1e-9).mean()) if len(valid) else np.nan,
            "mean_ensemble_sd": float(valid["std"].mean()) if len(valid) else np.nan,
            "median_ensemble_sd": float(valid["std"].median()) if len(valid) else np.nan,
            "p90_ensemble_sd": float(valid["std"].quantile(0.90)) if len(valid) else np.nan,
            "mean_p90_p10_width": float(valid.p90_p10.mean()) if len(valid) else np.nan,
            "median_p90_p10_width": float(valid.p90_p10.median()) if len(valid) else np.nan,
            "p90_p90_p10_width": float(valid.p90_p10.quantile(0.90)) if len(valid) else np.nan,
            "max_p90_p10_width": float(valid.p90_p10.max()) if len(valid) else np.nan,
            "mean_full_member_range": float(valid["range"].mean()) if len(valid) else np.nan,
            "max_full_member_range": float(valid["range"].max()) if len(valid) else np.nan,
        })
        by_day_rows = []
        for day_text, group in subset.groupby("delivery_day", sort=True):
            finite_group = group.loc[group.member_count.ge(2)]
            by_day_rows.append({
                "delivery_day": day_text, "farm_id": int(farm_id), "variable": variable,
                "valid_hours": int(len(finite_group)), "coverage_pct": 100 * len(finite_group) / 24,
                "mean_ensemble_sd": float(finite_group["std"].mean()) if len(finite_group) else np.nan,
                "median_ensemble_sd": float(finite_group["std"].median()) if len(finite_group) else np.nan,
                "mean_p90_p10_width": float(finite_group.p90_p10.mean()) if len(finite_group) else np.nan,
                "median_p90_p10_width": float(finite_group.p90_p10.median()) if len(finite_group) else np.nan,
                "max_p90_p10_width": float(finite_group.p90_p10.max()) if len(finite_group) else np.nan,
            })
        pd.DataFrame(by_day_rows).to_csv(farm_dirs[int(farm_id)] / "ensemble_variability_by_day.csv", mode="a", index=False,
                                         header=not (farm_dirs[int(farm_id)] / "ensemble_variability_by_day.csv").exists())
        lead_rows = []
        for hour, group in subset.groupby("delivery_hour_local", sort=True):
            finite_group = group.loc[group.member_count.ge(2)]
            lead_rows.append({
                "variable": variable, "unit": VARIABLES.get(variable, {}).get("units", "m/s"),
                "delivery_hour_local": int(hour), "valid_farm_days": int(len(finite_group)),
                "mean_ensemble_sd": float(finite_group["std"].mean()) if len(finite_group) else np.nan,
                "median_ensemble_sd": float(finite_group["std"].median()) if len(finite_group) else np.nan,
                "mean_p90_p10_width": float(finite_group.p90_p10.mean()) if len(finite_group) else np.nan,
                "median_p90_p10_width": float(finite_group.p90_p10.median()) if len(finite_group) else np.nan,
                "p90_p90_p10_width": float(finite_group.p90_p10.quantile(0.90)) if len(finite_group) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    for farm_id, directory in farm_dirs.items():
        farm_summary = summary.loc[summary.farm_id.eq(farm_id)]
        farm_summary.to_csv(directory / "ensemble_variability_summary.csv", index=False)
        farm_lead = []
        for variable in PLOT_VARIABLES:
            subset = point_frame.loc[point_frame.farm_id.eq(farm_id) & point_frame.variable.eq(variable)]
            for hour, group in subset.groupby("delivery_hour_local", sort=True):
                finite_group = group.loc[group.member_count.ge(2)]
                farm_lead.append({
                    "variable": variable, "unit": VARIABLES.get(variable, {}).get("units", "m/s"),
                    "delivery_hour_local": int(hour), "valid_days": int(len(finite_group)),
                    "mean_ensemble_sd": float(finite_group["std"].mean()) if len(finite_group) else np.nan,
                    "median_ensemble_sd": float(finite_group["std"].median()) if len(finite_group) else np.nan,
                    "mean_p90_p10_width": float(finite_group.p90_p10.mean()) if len(finite_group) else np.nan,
                    "median_p90_p10_width": float(finite_group.p90_p10.median()) if len(finite_group) else np.nan,
                    "p90_p90_p10_width": float(finite_group.p90_p10.quantile(0.90)) if len(finite_group) else np.nan,
                })
        pd.DataFrame(farm_lead).to_csv(directory / "ensemble_variability_by_delivery_hour.csv", index=False)
        for day in analysis_days:
            day_dir = directory / day.isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            daily = point_frame.loc[point_frame.farm_id.eq(farm_id) & point_frame.delivery_day.eq(day.isoformat())].copy()
            daily.to_csv(day_dir / "pointwise_ensemble_statistics.csv", index=False)
            quantity_rows = []
            for variable, group in daily.groupby("variable", sort=True):
                valid = group.loc[group.member_count.ge(2)]
                quantity_rows.append({
                    "delivery_day": day.isoformat(), "farm_id": farm_id, "variable": variable,
                    "unit": group.unit.iloc[0], "expected_hours": 24, "valid_hours": int(len(valid)),
                    "coverage_pct": 100 * len(valid) / 24,
                    "mean_ensemble_sd": float(valid["std"].mean()) if len(valid) else np.nan,
                    "median_ensemble_sd": float(valid["std"].median()) if len(valid) else np.nan,
                    "mean_p90_p10_width": float(valid.p90_p10.mean()) if len(valid) else np.nan,
                    "median_p90_p10_width": float(valid.p90_p10.median()) if len(valid) else np.nan,
                    "p90_p90_p10_width": float(valid.p90_p10.quantile(0.90)) if len(valid) else np.nan,
                    "max_p90_p10_width": float(valid.p90_p10.max()) if len(valid) else np.nan,
                    "mean_full_member_range": float(valid["range"].mean()) if len(valid) else np.nan,
                })
            pd.DataFrame(quantity_rows).to_csv(day_dir / "ensemble_statistics_by_quantity.csv", index=False)
    summary.to_csv(args.output / "farm_comparison_summary.csv", index=False)
    overall_summary = point_frame.groupby("variable", sort=True).apply(
        lambda group: pd.Series({
            "unit": group.unit.iloc[0],
            "valid_farm_day_hours": int(group.member_count.ge(2).sum()),
            "expected_farm_day_hours": int(len(analysis_days) * len(farm_ids) * 24),
            "coverage_pct": 100 * group.member_count.ge(2).mean(),
            "mean_ensemble_sd": group.loc[group.member_count.ge(2), "std"].mean(),
            "median_ensemble_sd": group.loc[group.member_count.ge(2), "std"].median(),
            "median_p90_p10_width": group.loc[group.member_count.ge(2), "p90_p10"].median(),
            "mean_p90_p10_width": group.loc[group.member_count.ge(2), "p90_p10"].mean(),
            "p90_p90_p10_width": group.loc[group.member_count.ge(2), "p90_p10"].quantile(0.90),
            "max_p90_p10_width": group.loc[group.member_count.ge(2), "p90_p10"].max(),
        }), include_groups=False,
    ).reset_index()
    overall_summary.to_csv(args.output / "ensemble_variability_summary_all_farms.csv", index=False)

    # Count negative forecasts and explain mask coverage against the declared January valid-time window.
    quality_rows = []
    valid_utc = (
        pd.DatetimeIndex(pd.to_datetime(ds.valid_time.values.ravel(), utc=True))
        .tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
        .reshape(ds.sizes["cycle_time"], ds.sizes["lead_time"])
    )
    in_month = (valid_utc >= np.datetime64("2026-01-01T00:00:00")) & (valid_utc < np.datetime64("2026-02-01T00:00:00"))
    for farm_id, directory in farm_dirs.items():
        quality_rows = []
        for variable in VARIABLES:
            arr = ARRAY_CACHE[variable][FARM_INDEX[farm_id]]
            finite = np.isfinite(arr)
            negative = finite & (arr < 0)
            finite_total = int(finite.sum())
            quality_rows.append({
                "farm_id": farm_id, "variable": variable, "unit": ds[variable].attrs.get("units", ""),
                "total_array_cells": int(arr.size), "finite_cells": finite_total,
                "finite_pct": 100 * finite_total / arr.size,
                "negative_finite_cells": int(negative.sum()),
                "negative_pct_of_finite": 100 * negative.sum() / finite_total if finite_total else np.nan,
                "minimum_finite_value": float(np.nanmin(arr)) if finite_total else np.nan,
                "maximum_finite_value": float(np.nanmax(arr)) if finite_total else np.nan,
                "missing_in_declared_january_valid_window": int((~finite & in_month[:, :, None]).sum()),
            })
        pd.DataFrame(quality_rows).to_csv(directory / "source_value_coverage_and_negative_values.csv", index=False)

    # Keep each delivery date under its farm folder and separate plots by weather quantity group.
    plot_outputs: dict[int, list[str]] = {farm_id: [] for farm_id in farm_ids}
    for day in plot_days:
        if day not in selected_cycles:
            raise ValueError(f"Plot day {day} must be inside the analysis window")
        da_cycle, da_cutoff, ida_cycle, ida_cutoff = selected_cycles[day]
        if da_cycle != ida_cycle:
            raise ValueError(f"DA and IDA1 select different cycles on {day}; current day plot expects one shared vintage")
        for farm_id in farm_ids:
            day_dir = farm_dirs[farm_id] / day.isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            for group_name, variables in PLOT_GROUPS.items():
                path = day_dir / f"{group_name}.png"
                legacy_pdf = day_dir / f"{group_name}.pdf"
                if args.reuse_existing_plots and path.is_file() and path.stat().st_size > 0:
                    if legacy_pdf.is_file():
                        legacy_pdf.unlink()
                    plot_outputs[farm_id].append(path.relative_to(args.output).as_posix())
                    continue
                make_day_group_page(ds, day, farm_id, da_cycle, da_cutoff, ida_cutoff,
                                    overlay_ids, path, args.timezone, group_name, variables)
                if legacy_pdf.is_file():
                    legacy_pdf.unlink()
                plot_outputs[farm_id].append(path.relative_to(args.output).as_posix())

    dictionary_markdown = markdown_table(dictionary)
    best = overall_summary.sort_values("median_p90_p10_width", ascending=False).head(8)
    best_lines = "\n".join(
        f"- `{row.variable}`: median p90–p10 width {fmt(row.median_p90_p10_width)} {row.unit}; "
        f"mean member SD {fmt(row.mean_ensemble_sd)} {row.unit}; "
        f"{fmt(row.valid_farm_day_hours, 0)}/{fmt(row.expected_farm_day_hours, 0)} farm-hour points have at least two finite members."
        for row in best.itertuples()
    )
    same_vintage_pct = 100 * pd.DataFrame(gate_records).same_vintage.mean()
    farm_folder_lines = "\n".join(
        f"- `farm_{farm_id}/`: seven daily plots, pointwise member distributions, per-feature spread, spread by hour/day, and source coverage checks."
        for farm_id in farm_ids
    )
    readme = f"""# January 2026 farm weather ensemble review

## Scope and files

Input: `{args.zarr}` with static farm metadata in `{args.metadata}`. The Zarr was read without changes. All plot and detailed statistics are separated into four farm folders:

{farm_folder_lines}

- `zarr_group_dictionary.csv`: physical meaning and current/future use for every array group, coordinate and derived wind speed.
- `auction_vintage_selection.csv`: selected DA/IDA1 forecast cycle and assumed availability timestamps.
- `farm_comparison_summary.csv`: per-farm and per-feature variability table; each row also has its own detailed files in that farm folder.
- `ensemble_variability_summary_all_farms.csv`: cross-farm aggregate for comparison.

## Date and availability conventions

The plots cover **Jan 2–8, 2026**. The statistical summaries cover **Jan 2–31**, the complete local delivery days whose required values fall inside the declared UTC valid-time range. Jan 1 is excluded because its first Europe/Rome delivery hour begins at 2025-12-31 23:00 UTC, outside the store's declared `2026-01-01/2026-01-31` valid-time range. No values were filled.

For delivery day D, the latest available cycle is selected separately at the configured DA and IDA1 cutoffs on D−1, subtracting the decision buffer and requiring `cycle_time + {args.availability_lag_hours:g}h <= cutoff`. Forecast release time is an **assumption from cycle time**, not a recorded publication timestamp. The new store has one 00 UTC cycle per day, so DA and IDA1 select the same vintage on all sampled days ({same_vintage_pct:.0f}% of analyzed days). January local time is UTC+1. Each selected run is the 00 UTC run from the preceding UTC date.

For interval averages and accumulations, source `valid_time` is an interval-end label; plots shift it back one hour to align values to delivery-hour starts. Instantaneous fields are plotted at their valid hour. The x-axis is local delivery hour 0–23. For a delivery day, instantaneous variables use leads 23–46 and interval fields use leads 24–47 from the preceding 00 UTC run.

## How to read the graphs

Each farm has one subfolder per delivery date. Each day folder has four quantity-group PNGs: radiation, temperature/clouds, wind, and precipitation/snow/visibility. Each file contains same-date plots for related quantities for that farm. Thin colored lines show 10 ensemble members sampled evenly from the 50 member IDs: `{', '.join(map(str, overlay_ids))}`. The shaded gray band is the p10–p90 range using all finite members; the dark line is the all-member median. Panels use native units; wind speeds are derived per member as `sqrt(u² + v²)` before plotting. The selected 10 traces illustrate member paths; the statistics use all 50.

Each delivery-date folder also contains `pointwise_ensemble_statistics.csv` and `ensemble_statistics_by_quantity.csv`. Monthly farm summaries and source coverage are saved at the farm-folder level.

## Statistical findings

Spread values below summarize **all 30 analyzed delivery days × four farms × 24 hours**. They describe disagreement among forecasts, not forecast error or calibrated probabilities.

{best_lines}

### Source coverage and raw-value notes

The Zarr provides 50 explicit member values at each farm/cycle/lead cell where the source value is finite. Its variables are masked outside the declared January UTC valid-time window; coverage is not uniform across the extra lead horizon. Read each farm folder's `source_value_coverage_and_negative_values.csv` for exact counts.

Small negative radiation or precipitation values are preserved in these charts and tables. The existing PV path has a configurable nonnegative forecast treatment after temporal alignment; this descriptive review does not clip the source.

## Interpretation and limits

- The new arrays have `(farm_id, cycle_time, lead_time, member)` dimensions. The current deterministic farm workflow expects one forecast value per `(farm_id, cycle_time, lead_time)` and does not yet ingest this member axis.
- Existing PV inputs use shortwave as GHI, direct radiation as a documented horizontal-direct/BHI working assumption, 2 m temperature, and derived 10 m wind speed. The new store is not yet wired into that model. Other variables are retained for analysis but are not current PV corrections.
- The member IDs are listed, but the Zarr metadata does not identify the NWP system, explain member construction, or publish member weights. Do not infer equal scenario probabilities from IDs alone.
- Member spread is not automatically calibrated uncertainty. Assess against observed irradiance/power by farm, lead, variable and regime before using it as scenario probabilities.
- The current deterministic farm archive runs every six hours and assumes cycle+4 h availability. This review applies the requested six-hour lag to auction gates. The distinct timing conventions must be resolved explicitly before operational ensemble integration.

## Zarr group dictionary

{dictionary_markdown}

Root files `.zgroup`, `.zattrs` and `.zmetadata` describe the Zarr group, dataset attributes and consolidated metadata; they are not physical forecast fields. `metadata.parquet` is an external static farm table joined by `farm_id`.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "input_zarr": str(args.zarr.resolve()), "metadata_parquet": str(args.metadata.resolve()),
        "analysis_start": args.analysis_start.isoformat(), "analysis_end_inclusive": args.analysis_end.isoformat(),
        "plot_start": args.plot_start.isoformat(), "plot_days": args.plot_days,
        "timezone": args.timezone, "availability_lag_hours_assumed": args.availability_lag_hours,
        "decision_buffer_minutes": args.decision_buffer_minutes, "overlay_member_ids": overlay_ids.tolist(),
        "dimensions": {key: int(value) for key, value in ds.sizes.items()},
        "farms": farm_ids, "plot_files_by_farm": plot_outputs, "plot_groups": PLOT_GROUPS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_modified": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
