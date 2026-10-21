"""Find reviewable PV data episodes without changing any source observation.

The functions in this module consume the long, UTC historical context table. A
flag indicates a pattern worth examining, not a verdict that a measurement is
wrong. Thresholds are deliberately exposed through ``config['diagnostics']``.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd


INTERVAL = pd.Timedelta(minutes=15)
EVENT_COLUMNS = [
    "event_id",
    "category",
    "farm_id",
    "start_utc",
    "end_utc",
    "duration_quarters",
    "duration_minutes",
    "score",
    "related_farms",
    "min_power_ac_kw",
    "max_power_ac_kw",
    "min_ghi_w_m2",
    "max_ghi_w_m2",
    "min_solar_elevation_deg",
    "max_solar_elevation_deg",
    "power_before_kw",
    "power_after_kw",
    "ghi_before_w_m2",
    "ghi_after_w_m2",
    "forecast_shortwave_w_m2",
    "forecast_shortwave_min_w_m2",
    "forecast_shortwave_max_w_m2",
    "forecast_cycle_time_utc",
    "forecast_lead_hours",
    "simultaneous_farm_count_min",
    "simultaneous_farm_count_max",
]


def _diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge documented defaults with the audit YAML diagnostic settings."""

    supplied = config.get("diagnostics", {})
    defaults = {
        "night_solar_elevation_deg": -18.0,
        "positive_power_fraction": 0.001,
        "daylight_solar_elevation_deg": 10.0,
        "substantial_ghi_w_m2": 200.0,
        "low_power_fraction": 0.01,
        "abrupt_power_fraction": 0.25,
        "abrupt_ghi_w_m2": 200.0,
        "plateau_min_intervals": 8,
        "physical_power_fraction_max": 1.1,
        "physical_ghi_max": 1400.0,
        "min_shared_farms": 2,
        "stable_ghi_tolerance_w_m2": 100.0,
        "plateau_atol_kw": 1e-6,
        "zero_ghi_tolerance_w_m2": 0.0,
        "range_tolerance": 1e-6,
    }
    return {**supplied, "thresholds": {**defaults, **supplied.get("thresholds", {})}}


def _prepare_context(context: pd.DataFrame) -> pd.DataFrame:
    """Check the minimum schema and put all farms on ordered UTC intervals."""

    required = {
        "farm_id",
        "time_utc",
        "power_ac_kw",
        "ghi_w_m2",
        "fraction_diffuse",
        "total_power_ac_kw",
        "solar_elevation_deg",
    }
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"Historical context is missing columns: {sorted(missing)}")

    ordered = context.copy()
    ordered["farm_id"] = ordered["farm_id"].astype(str)
    ordered["time_utc"] = pd.to_datetime(ordered["time_utc"], utc=True)
    for column in (
        "power_ac_kw",
        "ghi_w_m2",
        "fraction_diffuse",
        "total_power_ac_kw",
        "solar_elevation_deg",
        "forecast_shortwave_w_m2",
        "forecast_lead_hours",
    ):
        if column not in ordered:
            ordered[column] = np.nan
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    if "forecast_cycle_time_utc" not in ordered:
        ordered["forecast_cycle_time_utc"] = pd.NaT
    ordered["forecast_cycle_time_utc"] = pd.to_datetime(
        ordered["forecast_cycle_time_utc"], utc=True
    )
    return ordered.sort_values(["farm_id", "time_utc"], kind="stable").reset_index(drop=True)


def _event_id(category: str, farm_id: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Make a stable identifier from the episode's category and UTC bounds."""

    identity = f"{category}|{farm_id}|{start.isoformat()}|{end.isoformat()}"
    return "evt_" + sha256(identity.encode("utf-8")).hexdigest()[:16]


def _finite_min(values: pd.Series) -> float:
    """Return a raw minimum while retaining NaN for an entirely missing range."""

    return float(values.min()) if values.notna().any() else np.nan


def _finite_max(values: pd.Series) -> float:
    """Return a raw maximum while retaining NaN for an entirely missing range."""

    return float(values.max()) if values.notna().any() else np.nan


def _episodes_for_flag(
    ordered: pd.DataFrame,
    flag: np.ndarray | pd.Series,
    category: str,
    point_score: np.ndarray | pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Group adjoining flagged quarters per farm and retain raw neighboring values."""

    selected = np.flatnonzero(np.asarray(pd.Series(flag).fillna(False), dtype=bool))
    if len(selected) == 0:
        return []

    flagged = ordered.iloc[selected].copy()
    flagged["_position"] = selected
    if point_score is None:
        flagged["_score"] = 1.0
    else:
        scores = np.asarray(point_score, dtype=float)
        flagged["_score"] = scores[selected]
    boundary = flagged["farm_id"].ne(flagged["farm_id"].shift()) | flagged[
        "time_utc"
    ].diff().ne(INTERVAL)
    flagged["_episode"] = boundary.cumsum()
    rows: list[dict[str, Any]] = []

    # Episode aggregation is inexpensive relative to the quarter-hour flag pass.
    for _, episode in flagged.groupby("_episode", sort=False):
        first = episode.iloc[0]
        last = episode.iloc[-1]
        start = first["time_utc"]
        end = last["time_utc"] + INTERVAL
        farm_id = str(first["farm_id"])
        before_position = int(first["_position"]) - 1
        after_position = int(last["_position"]) + 1
        before = None
        after = None
        if before_position >= 0:
            candidate = ordered.iloc[before_position]
            if candidate["farm_id"] == farm_id and start - candidate["time_utc"] == INTERVAL:
                before = candidate
        if after_position < len(ordered):
            candidate = ordered.iloc[after_position]
            if candidate["farm_id"] == farm_id and candidate["time_utc"] - end == pd.Timedelta(0):
                after = candidate

        rows.append(
            {
                "event_id": _event_id(category, farm_id, start, end),
                "category": category,
                "farm_id": farm_id,
                "start_utc": start,
                "end_utc": end,
                "duration_quarters": len(episode),
                "duration_minutes": len(episode) * 15,
                "score": _finite_max(episode["_score"]),
                "related_farms": farm_id,
                "min_power_ac_kw": _finite_min(episode["power_ac_kw"]),
                "max_power_ac_kw": _finite_max(episode["power_ac_kw"]),
                "min_ghi_w_m2": _finite_min(episode["ghi_w_m2"]),
                "max_ghi_w_m2": _finite_max(episode["ghi_w_m2"]),
                "min_solar_elevation_deg": _finite_min(episode["solar_elevation_deg"]),
                "max_solar_elevation_deg": _finite_max(episode["solar_elevation_deg"]),
                "power_before_kw": before["power_ac_kw"] if before is not None else np.nan,
                "power_after_kw": after["power_ac_kw"] if after is not None else np.nan,
                "ghi_before_w_m2": before["ghi_w_m2"] if before is not None else np.nan,
                "ghi_after_w_m2": after["ghi_w_m2"] if after is not None else np.nan,
                "forecast_shortwave_w_m2": first["forecast_shortwave_w_m2"],
                "forecast_shortwave_min_w_m2": _finite_min(episode["forecast_shortwave_w_m2"]),
                "forecast_shortwave_max_w_m2": _finite_max(episode["forecast_shortwave_w_m2"]),
                "forecast_cycle_time_utc": first["forecast_cycle_time_utc"],
                "forecast_lead_hours": first["forecast_lead_hours"],
                "simultaneous_farm_count_min": np.nan,
                "simultaneous_farm_count_max": np.nan,
            }
        )
    return rows


def _shared_ghi_episodes(
    ordered: pd.DataFrame,
    eligible: pd.Series,
    min_shared_farms: int,
) -> list[dict[str, Any]]:
    """Identify daylight zero-GHI intervals seen at multiple producing farms."""

    shared_source = ordered.loc[eligible].copy()
    if shared_source.empty:
        return []
    farm_count = shared_source.groupby("time_utc", sort=True)["farm_id"].nunique()
    shared_times = farm_count.index[farm_count >= min_shared_farms]
    if len(shared_times) == 0:
        return []
    shared_source = shared_source[shared_source["time_utc"].isin(shared_times)]
    boundary = pd.Series(shared_times, index=shared_times).diff().ne(INTERVAL)
    episode_key = pd.Series(boundary.cumsum().to_numpy(), index=shared_times)
    shared_source["_episode"] = shared_source["time_utc"].map(episode_key)
    rows: list[dict[str, Any]] = []

    # Every row remains a measured farm value; aggregation only builds review context.
    for _, episode in shared_source.groupby("_episode", sort=False):
        start = episode["time_utc"].min()
        end = episode["time_utc"].max() + INTERVAL
        farms = sorted(episode["farm_id"].unique(), key=lambda value: int(value) if value.isdigit() else value)
        counts = farm_count.loc[episode["time_utc"].unique()]
        start_context = episode.loc[episode["time_utc"].eq(start)].iloc[0]
        duration = int((end - start) / INTERVAL)
        rows.append(
            {
                "event_id": _event_id("shared_daylight_zero_ghi", "portfolio", start, end),
                "category": "shared_daylight_zero_ghi",
                "farm_id": "portfolio",
                "start_utc": start,
                "end_utc": end,
                "duration_quarters": duration,
                "duration_minutes": duration * 15,
                "score": float(counts.max()) * duration / 4.0,
                "related_farms": ";".join(farms),
                "min_power_ac_kw": _finite_min(episode["power_ac_kw"]),
                "max_power_ac_kw": _finite_max(episode["power_ac_kw"]),
                "min_ghi_w_m2": _finite_min(episode["ghi_w_m2"]),
                "max_ghi_w_m2": _finite_max(episode["ghi_w_m2"]),
                "min_solar_elevation_deg": _finite_min(episode["solar_elevation_deg"]),
                "max_solar_elevation_deg": _finite_max(episode["solar_elevation_deg"]),
                "power_before_kw": np.nan,
                "power_after_kw": np.nan,
                "ghi_before_w_m2": np.nan,
                "ghi_after_w_m2": np.nan,
                "forecast_shortwave_w_m2": start_context["forecast_shortwave_w_m2"],
                "forecast_shortwave_min_w_m2": _finite_min(episode["forecast_shortwave_w_m2"]),
                "forecast_shortwave_max_w_m2": _finite_max(episode["forecast_shortwave_w_m2"]),
                "forecast_cycle_time_utc": start_context["forecast_cycle_time_utc"],
                "forecast_lead_hours": start_context["forecast_lead_hours"],
                "simultaneous_farm_count_min": int(counts.min()),
                "simultaneous_farm_count_max": int(counts.max()),
            }
        )
    return rows


def detect_events(context: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return grouped candidate episodes, retaining raw power and irradiance ranges.

    ``end_utc`` is exclusive: it marks the end of the final 15-minute interval.
    All timestamp fields are UTC. Event categories deliberately overlap when a
    single record has several notable properties.
    """

    ordered = _prepare_context(context)
    settings = _diagnostic_config(config)["thresholds"]
    power = ordered["power_ac_kw"]
    ghi = ordered["ghi_w_m2"]
    fraction = ordered["fraction_diffuse"]
    capacity = ordered["total_power_ac_kw"]
    elevation = ordered["solar_elevation_deg"]
    prev_power = ordered.groupby("farm_id", sort=False)["power_ac_kw"].shift()
    next_power = ordered.groupby("farm_id", sort=False)["power_ac_kw"].shift(-1)
    prev_ghi = ordered.groupby("farm_id", sort=False)["ghi_w_m2"].shift()
    next_ghi = ordered.groupby("farm_id", sort=False)["ghi_w_m2"].shift(-1)
    prev_time = ordered.groupby("farm_id", sort=False)["time_utc"].shift()
    next_time = ordered.groupby("farm_id", sort=False)["time_utc"].shift(-1)
    contiguous_before = (ordered["time_utc"] - prev_time).eq(INTERVAL)
    contiguous_after = (next_time - ordered["time_utc"]).eq(INTERVAL)
    power_change = (power - prev_power).where(contiguous_before)
    ghi_change = (ghi - prev_ghi).where(contiguous_before)
    positive_power = power > capacity * float(settings["positive_power_fraction"])
    daylight_zero_ghi = (
        elevation >= float(settings["daylight_solar_elevation_deg"])
    ) & ghi.between(
        -float(settings["zero_ghi_tolerance_w_m2"]),
        float(settings["zero_ghi_tolerance_w_m2"]),
    )
    power_ramp = capacity * float(settings["abrupt_power_fraction"])
    ghi_ramp = float(settings["abrupt_ghi_w_m2"])

    # Exact, positive plateaus are candidates even when the level resembles clipping.
    same_power = pd.Series(
        np.isclose(
            power.to_numpy(dtype=float),
            prev_power.to_numpy(dtype=float),
            rtol=0,
            atol=float(settings["plateau_atol_kw"]),
            equal_nan=False,
        ),
        index=ordered.index,
    ) & contiguous_before
    plateau_key = (~same_power).cumsum()
    plateau_length = plateau_key.map(plateau_key.value_counts(sort=False))
    plateau = positive_power & (plateau_length >= int(settings["plateau_min_intervals"]))

    # One-quarter power valleys with stable GHI make sensor or curtailment events easy to find.
    stable_ghi_tolerance = float(settings["stable_ghi_tolerance_w_m2"])
    power_dip = (
        contiguous_before
        & contiguous_after
        & (prev_power - power >= power_ramp)
        & (next_power - power >= power_ramp)
        & (prev_ghi - ghi).abs().le(stable_ghi_tolerance)
        & (next_ghi - ghi).abs().le(stable_ghi_tolerance)
    )
    ghi_dip = (
        contiguous_before
        & contiguous_after
        & (prev_ghi - ghi >= ghi_ramp)
        & (next_ghi - ghi >= ghi_ramp)
    )

    # Each mask has a severity scale for ranking examples, not for making a verdict.
    mask_score = [
        (
            "night_positive_power",
            (elevation <= float(settings["night_solar_elevation_deg"])) & positive_power,
            power / capacity,
        ),
        (
            "daylight_zero_ghi",
            daylight_zero_ghi,
            elevation.clip(lower=0) / 90 + (power / capacity).clip(lower=0),
        ),
        (
            "low_power_high_ghi",
            (ghi >= float(settings["substantial_ghi_w_m2"]))
            & (power <= capacity * float(settings["low_power_fraction"])),
            ghi / max(float(settings["substantial_ghi_w_m2"]), 1),
        ),
        ("power_drop", power_change <= -power_ramp, (-power_change / capacity)),
        ("power_recovery", power_change >= power_ramp, power_change / capacity),
        ("ghi_drop", ghi_change <= -ghi_ramp, -ghi_change / max(ghi_ramp, 1)),
        ("ghi_recovery", ghi_change >= ghi_ramp, ghi_change / max(ghi_ramp, 1)),
        ("power_dip", power_dip, (prev_power - power) / capacity),
        ("ghi_dip", ghi_dip, (prev_ghi - ghi) / max(ghi_ramp, 1)),
        ("constant_positive_power", plateau, plateau_length / int(settings["plateau_min_intervals"])),
        ("missing_power", power.isna(), pd.Series(1.0, index=ordered.index)),
        ("missing_ghi", ghi.isna(), pd.Series(1.0, index=ordered.index)),
        ("missing_fraction_diffuse", fraction.isna(), pd.Series(1.0, index=ordered.index)),
        (
            "negative_power",
            power < -float(settings["range_tolerance"]),
            -power / capacity,
        ),
        (
            "power_above_capacity",
            power > capacity * float(settings["physical_power_fraction_max"]),
            power / capacity,
        ),
        (
            "negative_ghi",
            ghi < -float(settings["range_tolerance"]),
            -ghi / max(float(settings["physical_ghi_max"]), 1),
        ),
        (
            "ghi_above_physical_max",
            ghi > float(settings["physical_ghi_max"]),
            ghi / max(float(settings["physical_ghi_max"]), 1),
        ),
        (
            "diffuse_fraction_out_of_range",
            (fraction < -float(settings["range_tolerance"]))
            | (fraction > 1 + float(settings["range_tolerance"])),
            (fraction - 0.5).abs() * 2,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for category, mask, score in mask_score:
        rows.extend(_episodes_for_flag(ordered, mask, category, score))
    rows.extend(
        _shared_ghi_episodes(
            ordered,
            daylight_zero_ghi & positive_power,
            int(settings["min_shared_farms"]),
        )
    )
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = pd.DataFrame(rows)
    events["score"] = pd.to_numeric(events["score"], errors="coerce")
    return events.sort_values(
        ["start_utc", "farm_id", "category", "event_id"], kind="stable"
    ).reset_index(drop=True)[EVENT_COLUMNS]


def sensitivity_table(context: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Count observations across reporting bins without assigning data validity.

    The table is long so each threshold combination can be graphed or pivoted.
    Cutoff columns irrelevant to a metric are blank; ``count_intervals`` counts
    quarter-hours, not episodes.
    """

    ordered = _prepare_context(context)
    settings = _diagnostic_config(config)
    solar_bins = settings.get("solar_elevation_bins_deg", [-18, -6, 0, 10])
    power_bins = settings.get("power_fraction_bins", [0, 0.001, 0.01, 0.05, 0.2])
    ghi_bins = settings.get("ghi_bins_w_m2", [0, 50, 100, 200, 400])
    ramp_bins = settings.get("ramp_fraction_bins", [0.1, 0.25, 0.5])
    rows: list[dict[str, Any]] = []

    # Sensitivity counts expose how strongly a conclusion depends on a cutoff.
    for farm_id, farm in ordered.groupby("farm_id", sort=False):
        power = farm["power_ac_kw"].to_numpy(dtype=float)
        capacity = farm["total_power_ac_kw"].to_numpy(dtype=float)
        ghi = farm["ghi_w_m2"].to_numpy(dtype=float)
        elevation = farm["solar_elevation_deg"].to_numpy(dtype=float)
        time = farm["time_utc"]
        power_change = np.diff(power, prepend=np.nan)
        contiguous = time.diff().eq(INTERVAL).to_numpy()
        power_change[~contiguous] = np.nan
        for solar_cutoff in solar_bins:
            for power_cutoff in power_bins:
                rows.append(
                    {
                        "farm_id": farm_id,
                        "metric": "positive_power_below_solar_elevation",
                        "solar_elevation_cutoff_deg": solar_cutoff,
                        "power_fraction_cutoff": power_cutoff,
                        "ghi_cutoff_w_m2": np.nan,
                        "ramp_fraction_cutoff": np.nan,
                        "count_intervals": int(
                            np.count_nonzero((elevation <= solar_cutoff) & (power > capacity * power_cutoff))
                        ),
                    }
                )
            rows.append(
                {
                    "farm_id": farm_id,
                    "metric": "zero_ghi_above_solar_elevation",
                    "solar_elevation_cutoff_deg": solar_cutoff,
                    "power_fraction_cutoff": np.nan,
                    "ghi_cutoff_w_m2": 0,
                    "ramp_fraction_cutoff": np.nan,
                    "count_intervals": int(np.count_nonzero((elevation >= solar_cutoff) & (ghi == 0))),
                }
            )
        for power_cutoff in power_bins:
            for ghi_cutoff in ghi_bins:
                rows.append(
                    {
                        "farm_id": farm_id,
                        "metric": "low_power_above_ghi",
                        "solar_elevation_cutoff_deg": np.nan,
                        "power_fraction_cutoff": power_cutoff,
                        "ghi_cutoff_w_m2": ghi_cutoff,
                        "ramp_fraction_cutoff": np.nan,
                        "count_intervals": int(
                            np.count_nonzero((power <= capacity * power_cutoff) & (ghi >= ghi_cutoff))
                        ),
                    }
                )
        for ramp_cutoff in ramp_bins:
            rows.append(
                {
                    "farm_id": farm_id,
                    "metric": "absolute_power_change_above_capacity_fraction",
                    "solar_elevation_cutoff_deg": np.nan,
                    "power_fraction_cutoff": np.nan,
                    "ghi_cutoff_w_m2": np.nan,
                    "ramp_fraction_cutoff": ramp_cutoff,
                    "count_intervals": int(
                        np.count_nonzero(np.abs(power_change) >= capacity * ramp_cutoff)
                    ),
                }
            )
    return pd.DataFrame(rows)
