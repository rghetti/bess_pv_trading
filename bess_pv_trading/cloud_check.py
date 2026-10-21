"""Compare forecast cloud cover with shared GHI zeros and abrupt PV changes.

Example::

    python -m bess_pv_trading.cloud_check --config configs/audit.yaml \
        --audit-dir outputs/qa/20261015T140957Z

Clouds are forecast context, not measured cloud observations. Every result is
descriptive: serial dependence and repeated forecast values preclude treating
the quarter-hours as independent statistical trials.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .audit import _project_relative, _read_config
from .cloud_plots import write_cloud_plots
from .report import _markdown_table
from .diagnostics import _prepare_context, _shared_ghi_episodes, _event_id


CLOUD_VARIABLES = ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")
QUARTER = pd.Timedelta(minutes=15)


def _sample_cloud(dataset: xr.Dataset, context: pd.DataFrame, target: pd.DatetimeIndex, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate instantaneous hourly clouds within each row's fixed vintage."""
    cycles = pd.DatetimeIndex(pd.to_datetime(context["forecast_cycle_time_utc"], utc=True))
    starts = target.floor("h")
    lead_start = starts - cycles
    farm_index = dataset.farm_id.to_index().get_indexer(context["farm_id"])
    cycle_index = pd.DatetimeIndex(pd.to_datetime(dataset.cycle_time.values, utc=True)).get_indexer(cycles)
    start_index = dataset.lead_time.to_index().get_indexer(lead_start)
    end_index = dataset.lead_time.to_index().get_indexer(lead_start + pd.Timedelta(hours=1))
    valid = (farm_index >= 0) & (cycle_index >= 0) & (start_index >= 0) & (end_index >= 0)
    first = np.full(len(context), np.nan)
    last = np.full(len(context), np.nan)
    values = np.asarray(dataset[variable].transpose("farm_id", "cycle_time", "lead_time").values)
    first[valid] = values[farm_index[valid], cycle_index[valid], start_index[valid]]
    last[valid] = values[farm_index[valid], cycle_index[valid], end_index[valid]]
    weight = np.asarray((target - starts) / pd.Timedelta(hours=1), dtype=float)
    return first + weight * (last - first), last


def attach_cloud_context(dataset: xr.Dataset, context: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add forecast clouds at measurement midpoints, retaining raw measurements."""
    result = context.copy().sort_values(["farm_id", "time_utc"]).reset_index(drop=True)
    midpoint = pd.DatetimeIndex(result["time_utc"]) + QUARTER / 2
    for variable in CLOUD_VARIABLES:
        result[f"{variable}_midpoint_pct"], result[f"{variable}_end_pct"] = _sample_cloud(dataset, result, midpoint, variable)

    # Use the SAME current forecast vintage for both ends of a cloud tendency.
    # This prevents a forecast revision from masquerading as a passing cloud.
    for minutes in (15, 60):
        prior, _ = _sample_cloud(dataset, result, midpoint - pd.Timedelta(minutes=minutes), "cloud_cover")
        result[f"cloud_tendency_{minutes}min_pct"] = result["cloud_cover_midpoint_pct"] - prior
    result["cloud_change_pct"] = result["cloud_tendency_15min_pct"]
    grouped = result.groupby("farm_id", sort=False)
    contiguous = grouped["time_utc"].diff().eq(QUARTER)
    result["cycle_changed"] = result["forecast_cycle_time_utc"].ne(grouped["forecast_cycle_time_utc"].shift())
    result["rolling_cloud_change_pct"] = grouped["cloud_cover_midpoint_pct"].diff().where(contiguous)
    result["power_fraction"] = result["power_ac_kw"] / result["total_power_ac_kw"]
    result["power_change_fraction"] = grouped["power_ac_kw"].diff().where(contiguous) / result["total_power_ac_kw"]
    result["ghi_change_w_m2"] = grouped["ghi_w_m2"].diff().where(contiguous)
    result["forecast_shortwave_change_w_m2"] = grouped["forecast_shortwave_w_m2"].diff().where(contiguous)
    settings = config["cloud_analysis"]
    threshold = config["diagnostics"]["thresholds"]["daylight_solar_elevation_deg"]
    result["proxy_solar_above_threshold"] = result["solar_elevation_deg"].ge(threshold)
    result["ghi_signal"] = result["ghi_w_m2"].ge(settings["activity_ghi_w_m2"])
    result["forecast_shortwave_signal"] = result["forecast_shortwave_w_m2"].ge(settings["activity_forecast_shortwave_w_m2"])
    result["power_signal"] = result["power_fraction"].ge(settings["activity_power_fraction"])
    result["radiation_evidence"] = result["ghi_signal"] | result["forecast_shortwave_signal"]
    result["activity_evidence"] = result["radiation_evidence"] | result["power_signal"]
    result["valid_power_pair"] = contiguous & result["power_change_fraction"].notna()
    result["previous_solar_elevation_deg"] = grouped["solar_elevation_deg"].shift().where(contiguous)
    previous_activity = grouped["activity_evidence"].shift().eq(True)
    result["drop_eligible"] = result["valid_power_pair"] & (result["activity_evidence"] | previous_activity)
    result["previous_power_fraction"] = grouped["power_fraction"].shift().where(contiguous)
    result["power_drop"] = result["drop_eligible"] & result["power_change_fraction"].le(-config["diagnostics"]["thresholds"]["abrupt_power_fraction"])
    next_change = grouped["power_ac_kw"].shift(-1) - result["power_ac_kw"]
    next_contiguous = (grouped["time_utc"].shift(-1) - result["time_utc"]).eq(QUARTER)
    recovery = config["cloud_analysis"]["recovery_fraction_of_drop"]
    result["quick_recovery"] = next_contiguous & next_change.ge(-result["power_change_fraction"] * result["total_power_ac_kw"] * recovery)
    result["next_ghi_change_w_m2"] = (grouped["ghi_w_m2"].shift(-1) - result["ghi_w_m2"]).where(next_contiguous)
    return result


def portfolio_context(context: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count common zero GHI at unique timestamps rather than duplicating farms."""
    frame = context.copy()
    positive = config["diagnostics"]["thresholds"]["positive_power_fraction"]
    frame["_zero"] = frame["ghi_w_m2"].eq(0)
    frame["_zero_producing"] = frame["_zero"] & frame["power_fraction"].gt(positive)
    aggregations = {
        "solar_elevation_deg": ("solar_elevation_deg", "first"),
        "ghi_observed_farms": ("ghi_w_m2", "count"),
        "zero_ghi_farms": ("_zero", "sum"),
        "zero_producing_farms": ("_zero_producing", "sum"),
        "historical_ghi_mean_w_m2": ("ghi_w_m2", "mean"),
        "forecast_shortwave_mean_w_m2": ("forecast_shortwave_w_m2", "mean"),
        "radiation_evidence": ("radiation_evidence", "any"),
        "activity_evidence": ("activity_evidence", "any"),
    }
    for name in CLOUD_VARIABLES:
        aggregations[f"{name}_mean_pct"] = (f"{name}_midpoint_pct", "mean")
    portfolio = frame.groupby("time_utc", sort=True).agg(**aggregations).reset_index()
    farm_count = context["farm_id"].nunique()
    minimum = config["diagnostics"]["thresholds"]["min_shared_farms"]
    portfolio["shared_zero_pattern"] = portfolio["zero_producing_farms"].ge(minimum)
    portfolio["all_zero_pattern"] = portfolio["zero_ghi_farms"].eq(farm_count) & portfolio["ghi_observed_farms"].eq(farm_count)
    portfolio["shared_eligible"] = portfolio["activity_evidence"] & portfolio["ghi_observed_farms"].ge(minimum)
    portfolio["all_zero_eligible"] = portfolio["activity_evidence"] & portfolio["ghi_observed_farms"].eq(farm_count)
    portfolio["shared_zero"] = portfolio["shared_eligible"] & portfolio["shared_zero_pattern"]
    portfolio["all_zero"] = portfolio["all_zero_eligible"] & portfolio["all_zero_pattern"]
    frame = frame.drop(columns=["_zero", "_zero_producing"]).merge(portfolio[["time_utc", "shared_zero", "all_zero"]], on="time_utc", validate="many_to_one")
    return frame, portfolio


def cloud_bin_summary(portfolio: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Report event frequencies with explicit observed-activity denominators."""
    eligible = portfolio.loc[portfolio["shared_eligible"]].copy()
    eligible["cloud_bin"] = pd.cut(eligible["cloud_cover_mean_pct"], config["cloud_analysis"]["cloud_bins_pct"], include_lowest=True)
    rows = []
    for name, group in eligible.groupby("cloud_bin", observed=False):
        strict = group.loc[group["all_zero_eligible"]]
        rows.append({"cloud_bin": str(name), "eligible_timestamps": len(group),
                     "shared_zero_timestamps": int(group["shared_zero"].sum()),
                     "shared_zero_rate_pct": 100 * group["shared_zero"].mean() if len(group) else np.nan,
                     "all_observed_timestamps": len(strict), "all_zero_timestamps": int(strict["all_zero"].sum()),
                     "all_zero_rate_pct": 100 * strict["all_zero"].mean() if len(strict) else np.nan})
    return pd.DataFrame(rows)


def correlations(context: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive Pearson coefficients, including binary event flags."""
    rows = []
    for flag, eligibility in (("shared_zero", "shared_eligible"), ("all_zero", "all_zero_eligible")):
        frame = portfolio.loc[portfolio[eligibility]]
        for cloud in CLOUD_VARIABLES:
            column = f"{cloud}_mean_pct"
            paired = frame[[flag, column]].dropna()
            rows.append({"scope": flag, "farm_id": "portfolio", "x": column, "y": flag, "pairs": len(paired), "pearson_r": paired[column].corr(paired[flag].astype(float))})
    for farm, frame in [("all", context), *list(context.groupby("farm_id", sort=True))]:
        frame = frame.loc[frame["drop_eligible"]]
        for x, y in (("cloud_cover_midpoint_pct", "power_fraction"), ("ghi_w_m2", "power_fraction"),
                     ("cloud_cover_midpoint_pct", "power_change_fraction"),
                     ("cloud_tendency_15min_pct", "power_change_fraction"),
                     ("cloud_tendency_60min_pct", "power_change_fraction"),
                     ("ghi_change_w_m2", "power_change_fraction"),
                     ("forecast_shortwave_change_w_m2", "power_change_fraction")):
            paired = frame[[x, y]].dropna()
            rows.append({"scope": "active_power", "farm_id": farm, "x": x, "y": y, "pairs": len(paired), "pearson_r": paired[x].corr(paired[y])})
    return pd.DataFrame(rows)


def drop_summary(context: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count ramps and distinguish GHI co-movement from near-stable GHI."""
    rows = []
    settings = config["cloud_analysis"]
    for farm, group in [("all", context), *list(context.groupby("farm_id", sort=True))]:
        active = group.loc[group["drop_eligible"]]
        for threshold in settings["drop_capacity_fractions"]:
            drops = active.loc[active["power_change_fraction"].le(-threshold)]
            feasible = active.loc[active["previous_power_fraction"].ge(threshold)]
            stable = drops["ghi_change_w_m2"].abs().le(settings["stable_ghi_change_w_m2"])
            rows.append({"farm_id": farm, "drop_capacity_fraction": threshold,
                         "active_pairs": len(active), "pre_power_at_least_threshold_pairs": len(feasible),
                         "drop_intervals": len(drops), "drop_rate_pct": 100 * len(drops) / len(active) if len(active) else np.nan,
                         "opportunity_adjusted_drop_rate_pct": 100 * len(drops.loc[drops["previous_power_fraction"].ge(threshold)]) / len(feasible) if len(feasible) else np.nan,
                         "mean_cloud_cover_pct": drops["cloud_cover_midpoint_pct"].mean(),
                         "mean_cloud_change_15min_pp": drops["cloud_tendency_15min_pct"].mean(),
                         "mean_cloud_change_60min_pp": drops["cloud_tendency_60min_pct"].mean(),
                         "stable_ghi_drop_intervals": int(stable.sum()),
                         "ghi_fall_cutoff_w_m2": settings["substantial_ghi_drop_w_m2"],
                         "ghi_fall_at_least_cutoff_intervals": int(drops["ghi_change_w_m2"].le(-settings["substantial_ghi_drop_w_m2"]).sum()),
                         "missing_ghi_change_intervals": int(drops["ghi_change_w_m2"].isna().sum()),
                         "stable_ghi_quick_recovery_intervals": int((stable & drops["quick_recovery"] & drops["next_ghi_change_w_m2"].abs().le(settings["stable_ghi_change_w_m2"])).sum()),
                         "forecast_cycle_changed_intervals": int(drops["cycle_changed"].sum())})
    return pd.DataFrame(rows)


def power_drop_cloud_bins(context: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare drop frequencies after excluding impossible low-output drops."""
    rows = []
    for farm, group in [("all", context), *list(context.groupby("farm_id", sort=True))]:
        for threshold in config["cloud_analysis"]["drop_capacity_fractions"]:
            eligible = group.loc[group["drop_eligible"] & group["previous_power_fraction"].ge(threshold)].copy()
            eligible["cloud_bin"] = pd.cut(eligible["cloud_cover_midpoint_pct"], config["cloud_analysis"]["cloud_bins_pct"], include_lowest=True)
            for cloud_bin, values in eligible.groupby("cloud_bin", observed=False):
                count = int(values["power_change_fraction"].le(-threshold).sum())
                rows.append({"farm_id": farm, "drop_capacity_fraction": threshold, "cloud_bin": str(cloud_bin),
                             "eligible_pairs": len(values), "drop_intervals": count,
                             "drop_rate_pct": 100 * count / len(values) if len(values) else np.nan})
    return pd.DataFrame(rows)


def hourly_shared_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Check whether the cloud association survives reducing repeated quarters."""
    rows = []
    for flag, eligibility in (("shared_zero", "shared_eligible"), ("all_zero", "all_zero_eligible")):
        frame = portfolio.loc[portfolio[eligibility]].copy()
        frame["hour_utc"] = frame["time_utc"].dt.floor("h")
        hours = frame.groupby("hour_utc").agg(cloud_pct=("cloud_cover_mean_pct", "mean"), **{flag: (flag, "any")})
        paired = hours.dropna(subset=["cloud_pct"])
        event = paired.loc[paired[flag], "cloud_pct"]
        control = paired.loc[~paired[flag], "cloud_pct"]
        rows.append({"scope": flag, "eligible_hours": len(paired), "affected_hours": len(event),
                     "event_cloud_pct": event.mean(), "control_cloud_pct": control.mean(),
                     "point_biserial_r": paired[flag].astype(float).corr(paired["cloud_pct"])})
    return pd.DataFrame(rows)


def matched_cloud_comparisons(portfolio: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare clouds within the same month and UTC hour, without coordinates."""
    rows = []
    for flag, eligibility in (("shared_zero", "shared_eligible"), ("all_zero", "all_zero_eligible")):
        frame = portfolio.loc[portfolio[eligibility]].copy()
        frame["month"] = frame["time_utc"].dt.strftime("%Y-%m")
        frame["hour_utc"] = frame["time_utc"].dt.hour
        for variable in CLOUD_VARIABLES:
            cloud = f"{variable}_mean_pct"
            strata = frame.groupby(["month", "hour_utc"], observed=True)
            cases = strata.apply(lambda x: x.loc[x[flag], cloud].agg(["count", "mean"]), include_groups=False)
            controls = strata.apply(lambda x: x.loc[~x[flag], cloud].mean(), include_groups=False).rename("control_mean")
            joined = cases.join(controls).dropna()
            joined = joined.loc[joined["count"] > 0]
            count = int(joined["count"].sum())
            rows.append({"scope": flag, "cloud_variable": variable, "matched_event_timestamps": count,
                         "event_cloud_pct": np.average(joined["mean"], weights=joined["count"]) if count else np.nan,
                         "matched_control_cloud_pct": np.average(joined["control_mean"], weights=joined["count"]) if count else np.nan})
    return pd.DataFrame(rows)


def selection_sensitivity(context: pd.DataFrame, portfolio: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Show how activity evidence and proxy solar selections change the result."""
    settings = config["cloud_analysis"]
    minimum = config["diagnostics"]["thresholds"]["min_shared_farms"]
    farm_count = context["farm_id"].nunique()
    previous_radiation = context.groupby("farm_id")["radiation_evidence"].shift().eq(True)
    radiation_pairs = context["valid_power_pair"] & (context["radiation_evidence"] | previous_radiation)
    selections = [("irradiance_or_production_evidence", portfolio["activity_evidence"], context["drop_eligible"]),
                  ("irradiance_evidence_only", portfolio["radiation_evidence"], radiation_pairs),
                  ("production_evidence_only_without_irradiance", portfolio["activity_evidence"] & ~portfolio["radiation_evidence"], context["drop_eligible"] & ~radiation_pairs)]
    for cutoff in settings["proxy_solar_cutoffs_deg"]:
        solar_pairs = context["valid_power_pair"] & context["solar_elevation_deg"].ge(cutoff) & context["previous_solar_elevation_deg"].ge(cutoff)
        selections.append((f"Milan_proxy_ge_{cutoff:g}_deg_REFERENCE_ONLY", portfolio["solar_elevation_deg"].ge(cutoff), solar_pairs))
    rows = []
    for name, mask, pairs in selections:
        for pattern, observed in (("shared_zero", portfolio["ghi_observed_farms"].ge(minimum)), ("all_zero", portfolio["ghi_observed_farms"].eq(farm_count))):
            sample = portfolio.loc[mask & observed]
            cases = sample.loc[sample[f"{pattern}_pattern"]]
            controls = sample.loc[~sample[f"{pattern}_pattern"]]
            rows.append({"selection": name, "pattern": pattern, "eligible_timestamps": len(sample),
                         "event_timestamps": len(cases), "event_cloud_pct": cases["cloud_cover_mean_pct"].mean(),
                         "control_cloud_pct": controls["cloud_cover_mean_pct"].mean(),
                         "power_pairs": int(pairs.sum()),
                         "large_power_drop_intervals": int((pairs & context["power_change_fraction"].le(-config["diagnostics"]["thresholds"]["abrupt_power_fraction"])).sum())})
    return pd.DataFrame(rows)


def study_event_catalog(context: pd.DataFrame, portfolio: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build shared candidate episodes without a solar-position condition."""
    ordered = _prepare_context(context)
    eligible_times = portfolio.loc[portfolio["shared_eligible"], "time_utc"]
    threshold = config["diagnostics"]["thresholds"]["positive_power_fraction"]
    eligible = ordered["time_utc"].isin(eligible_times) & ordered["ghi_w_m2"].eq(0) & ordered["power_ac_kw"].gt(ordered["total_power_ac_kw"] * threshold)
    rows = _shared_ghi_episodes(ordered, eligible, config["diagnostics"]["thresholds"]["min_shared_farms"])
    for row in rows:
        row["category"] = "shared_zero_ghi_candidate"
        row["event_id"] = _event_id(row["category"], "portfolio", row["start_utc"], row["end_utc"])
    power = events.loc[events["category"].isin(("power_drop", "power_dip"))]
    return pd.concat([pd.DataFrame(rows), power], ignore_index=True)


def event_weather_summary(context: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach weather evidence to the existing stable event IDs."""
    groups = {str(farm): group.set_index("time_utc").sort_index() for farm, group in context.groupby("farm_id")}
    all_rows = context.set_index("time_utc").sort_index()
    rows = []
    chosen = events.loc[events["category"].isin(("shared_zero_ghi_candidate", "power_drop", "power_dip"))]
    for event in chosen.to_dict("records"):
        source = all_rows if str(event["farm_id"]) == "portfolio" else groups[str(event["farm_id"])]
        subset = source.loc[(source.index >= event["start_utc"]) & (source.index < event["end_utc"])]
        if event["category"] != "shared_zero_ghi_candidate" and not subset["drop_eligible"].any():
            continue
        rows.append({**event, "mean_cloud_cover_pct": subset["cloud_cover_midpoint_pct"].mean(),
                     "min_cloud_cover_pct": subset["cloud_cover_midpoint_pct"].min(),
                     "max_cloud_cover_pct": subset["cloud_cover_midpoint_pct"].max(),
                     "mean_cloud_change_15min_pp": subset["cloud_tendency_15min_pct"].mean(),
                     "mean_cloud_change_60min_pp": subset["cloud_tendency_60min_pct"].mean(),
                     "min_power_change_fraction": subset["power_change_fraction"].min(),
                     "mean_ghi_change_w_m2": subset["ghi_change_w_m2"].mean(),
                     "mean_power_fraction": subset["power_fraction"].mean()})
    return pd.DataFrame(rows)


def _report(context: pd.DataFrame, portfolio: pd.DataFrame, bins: pd.DataFrame, corr: pd.DataFrame, drops: pd.DataFrame, matched: pd.DataFrame, event_summary: pd.DataFrame, hourly: pd.DataFrame, drop_bins: pd.DataFrame, sensitivity: pd.DataFrame, config: dict) -> str:
    """Explain both associations and the limits of forecast cloud evidence."""
    eligible = portfolio.loc[portfolio["shared_eligible"]]
    shared = eligible.loc[eligible["shared_zero"]]
    control = eligible.loc[~eligible["shared_zero"]]
    strict = portfolio.loc[portfolio["all_zero"]]
    threshold = config["diagnostics"]["thresholds"]["abrupt_power_fraction"]
    selected_drop = drops.loc[drops["farm_id"].astype(str).eq("all") & drops["drop_capacity_fraction"].eq(threshold)]
    key_drop = selected_drop.iloc[0] if len(selected_drop) else None
    key_text = (f"There are **{int(key_drop.drop_intervals):,}** power drops of at least **{100 * threshold:g}% of AC nameplate capacity** per quarter-hour in the activity-evidence cohort. **{int(key_drop.stable_ghi_drop_intervals):,}** have |ΔGHI| ≤ {config['cloud_analysis']['stable_ghi_change_w_m2']:g} W/m²; **{int(key_drop.stable_ghi_quick_recovery_intervals):,}** also recover at least {100 * config['cloud_analysis']['recovery_fraction_of_drop']:g}% of the lost power in the next interval while GHI remains within that tolerance on both sides. These counts justify review without identifying a cause." if key_drop is not None else "Power-drop thresholds and their sensitivity are reported below.")
    zero_context = context.loc[context["all_zero"]]
    largest_output = zero_context.sort_values("power_fraction", ascending=False).groupby("farm_id", sort=True).head(1)
    example_columns = ["farm_id", "time_utc", "power_ac_kw", "power_fraction", "ghi_w_m2", "cloud_cover_midpoint_pct", "forecast_shortwave_w_m2"]
    lines = ["# Cloud cover, shared GHI zeros and sudden PV drops", "",
             f"Shared zero-GHI patterns occur at **{len(shared):,} unique quarter-hours** out of **{len(eligible):,}** with irradiance/production evidence and at least two observed GHI values. Mean forecast cloud cover is **{shared.cloud_cover_mean_pct.mean():.2f}%** during these periods, versus **{control.cloud_cover_mean_pct.mean():.2f}%** at other eligible times.", "",
             f"All farms report exactly zero GHI at **{len(strict)}** eligible timestamps. Mean cloud cover is **{strict.cloud_cover_mean_pct.mean():.2f}%**, with a minimum of **{strict.cloud_cover_mean_pct.min():.2f}%**. These are counts of timestamps, not independent weather events.", "",
             key_text, "",
             "## Alignment and interpretation", "",
             "Cloud variables in the Zarr are instantaneous forecasts in percent. Each quarter-hour uses its existing latest operationally available cycle (publication lag from YAML). Cloud values are interpolated between hourly targets **within that same cycle** to the historical interval midpoint. This interpolates forecasts, not measurements. The endpoint-cloud columns are retained for sensitivity checks. Both ends of each 15/60-minute cloud tendency also use the same cycle, so a new forecast issue cannot create an artificial cloud movement.", "",
             f"Historical GHI/power remain the original interval means. The primary comparison includes timestamps with recorded GHI ≥{config['cloud_analysis']['activity_ghi_w_m2']:g} W/m², forecast shortwave ≥{config['cloud_analysis']['activity_forecast_shortwave_w_m2']:g} W/m², or power ≥{100 * config['cloud_analysis']['activity_power_fraction']:g}% of AC nameplate capacity in at least one farm. This is an evidence-based reporting cohort, **not a declaration of actual daylight or data validity**. Production-only activation can itself arise from a power defect. Shared-zero patterns require at least two farms with GHI exactly zero and power above the configured positive-power fraction. All-farm patterns require all GHI values observed.", "",
             "**Milan solar elevation is reference information only.** It does not determine the primary cohort or a cleaning verdict. The sensitivity table shows the effect of using that uncertain proxy as a comparison. Actual farm coordinates are needed before making plant-specific solar geometry claims.", "",
             "Cloud cover measures spatial coverage, and attenuation also depends on cloud water/ice and other properties. See [ECMWF cloud cover](https://codes.ecmwf.int/grib/param-db/164) and [atmospheric physics](https://www.ecmwf.int/en/research/modelling-and-prediction/atmospheric-physics). [Total shortwave](https://codes.ecmwf.int/grib/param-db/169) includes direct and diffuse components. These model forecasts support a weather hypothesis but do not observe the actual cloud over an anonymous farm.", "",
             "## Shared GHI zeros by cloud cover", "", _markdown_table(bins), "",
             "### Positive production despite all farms reporting zero GHI", "",
             "The largest normalized output per farm during these timestamps is shown below. Some power series also have plateaus or other defects, so positive power is evidence to inspect, not an independent guarantee of irradiance quality.", "",
             _markdown_table(largest_output, example_columns), "",
             "## Selection sensitivity: evidence and solar proxy", "", _markdown_table(sensitivity), "",
             "The irradiance-only cohort avoids activation through a potentially faulty power reading. Solar-proxy rows are comparisons, not final day/night classifications. All these choices can change which episodes are worth examining; none decides which values to clean.", "",
             "## Comparison within the same month and UTC hour", "",
             "Controls are weighted by the number of event timestamps in each matching stratum. This reduces calendar/time-of-day confounding without geographic coordinates; it does not establish causation.", "", _markdown_table(matched), "",
             "A clock hour is counted once if any of its eligible quarters has the event:", "", _markdown_table(hourly), "",
             "## Sudden power drops", "",
             "Drops are measured in fractions of installed AC nameplate capacity per 15 minutes. A contiguous pair enters the primary cohort if either interval has recorded/forecast irradiance or production evidence. The opportunity-adjusted rate restricts the denominator to intervals whose preceding output was large enough for that drop. No solar-position exclusion is applied. Stable GHI and falling-GHI counts describe evidence; they assign no fault verdict.", "",
             _markdown_table(drops), "", "### Cloud bins with feasible preceding output", "",
             _markdown_table(drop_bins.loc[drop_bins["farm_id"].astype(str).eq("all")]), "", "## Correlations", "",
             "A binary event versus cloud correlation is point-biserial (Pearson with a 0/1 event indicator). Cloud level versus production and cloud **changes** versus power **changes** answer different questions. Serial dependence, hourly forecast resolution, forecast error and proxy coordinates limit interpretation; no significance tests or causal claims are made.", "",
             _markdown_table(corr), "",
             "## Review examples", "",
             _markdown_table(event_summary.sort_values("min_power_change_fraction").head(12), ["event_id", "category", "farm_id", "start_utc", "mean_cloud_cover_pct", "min_power_change_fraction", "mean_ghi_change_w_m2"]), "",
             "Every event and its weather context is in events_with_cloud.csv; plots are grouped by farm and event date. Shared episodes are rebuilt from the primary evidence cohort without a solar filter; their IDs are study-specific. Exact zero GHI under heavy forecast cloud can reflect very low irradiance, satellite retrieval/threshold behavior, or another data process. Power drops under stable GHI merit further review, while coarse cloud forecasts cannot rule out a local transient. No raw values or cleaning decisions were changed.", "",
             "## How to read the plots", "",
             "Event windows have four panels: recorded AC power (with installed capacity dotted); recorded GHI plus available hourly shortwave forecast; forecast total/low/middle/high cloud cover; and Milan solar elevation **for reference only**. The pink span marks the candidate episode from its first interval start to its exclusive end. It does not mark a confirmed fault. Vertical dotted lines mark changes of forecast vintage. Power/GHI are plotted at historical interval starts; the cloud estimate represents the interval midpoint.", "",
             "The association figure compares cloud-cover distributions (each group normalized to 100%) and the frequency of zero-GHI patterns within each cloud bin. The scatter figure uses one point per large power-drop interval: horizontal is same-vintage forecast-cloud change; vertical is power change as percent of AC nameplate. Blue denotes |ΔGHI| within the configured tolerance, orange falling GHI beyond it, purple rising/missing GHI, and hollow black circles indicate a change of available forecast cycle. These colors describe observed co-movement, not causes or cleaning decisions.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Read an existing audit and write a reproducible supplementary study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    config = _read_config(args.config.resolve())
    audit_dir = args.audit_dir.resolve()
    with (audit_dir / "config_used.yaml").open(encoding="utf-8") as stream:
        audit_config = yaml.safe_load(stream)
    # Existing event IDs and selected vintages belong to the original audit choices.
    for section, keys in (("solar", ("latitude", "longitude")), ("forecast", ("availability_lag_hours",))):
        if any(config[section][key] != audit_config[section][key] for key in keys):
            raise ValueError(f"{section} alignment differs from the supplied audit; rerun the main audit first")
    if any(pd.Timestamp(config["window"][key]) != pd.Timestamp(audit_config["window"][key]) for key in ("start", "end")):
        raise ValueError("The date window differs from the supplied audit; rerun the main audit first")
    for key in ("daylight_solar_elevation_deg", "positive_power_fraction", "min_shared_farms", "abrupt_power_fraction"):
        if config["diagnostics"]["thresholds"][key] != audit_config["diagnostics"]["thresholds"][key]:
            raise ValueError(f"Diagnostic {key} differs from the supplied audit; rerun the main audit first")
    project = Path(__file__).resolve().parents[1]
    dataset = xr.open_zarr(_project_relative(config["inputs"]["zarr"], project))
    context = pd.read_csv(audit_dir / "historical_context.csv", float_precision="round_trip")
    if set(context["farm_id"].unique()) != set(dataset.farm_id.values):
        raise ValueError("The source Zarr and historical audit have different farm IDs")
    for name in ("time_utc", "forecast_cycle_time_utc", "forecast_hour_end_utc"):
        context[name] = pd.to_datetime(context[name], utc=True)
    events = pd.read_csv(audit_dir / "events.csv", dtype={"farm_id": str})
    for name in ("start_utc", "end_utc"):
        events[name] = pd.to_datetime(events[name], utc=True)
    print("Aligning forecast cloud coverage and within-vintage tendencies", flush=True)
    context = attach_cloud_context(dataset, context, config)
    context, portfolio = portfolio_context(context, config)
    bins = cloud_bin_summary(portfolio, config)
    corr = correlations(context, portfolio)
    drops = drop_summary(context, config)
    drop_bins = power_drop_cloud_bins(context, config)
    hourly = hourly_shared_summary(portfolio)
    matched = matched_cloud_comparisons(portfolio, config)
    sensitivity = selection_sensitivity(context, portfolio, config)
    catalog = study_event_catalog(context, portfolio, events, config)
    weather_events = event_weather_summary(context, catalog)
    output = audit_dir / "cloud_cover_check" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    for filename, frame in (("historical_cloud_context.csv", context), ("portfolio_cloud_context.csv", portfolio),
                            ("cloud_bins.csv", bins), ("correlations.csv", corr), ("power_drop_summary.csv", drops),
                            ("power_drop_cloud_bins.csv", drop_bins), ("hourly_shared_checks.csv", hourly),
                            ("selection_sensitivity.csv", sensitivity),
                            ("matched_cloud_comparisons.csv", matched), ("events_with_cloud.csv", weather_events)):
        frame.to_csv(output / filename, index=False)
    # Keep YAML timestamps serializable, matching the main audit's provenance.
    config["window"] = {key: value.isoformat() for key, value in config["window"].items()}
    (output / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Writing cloud association and power-drop plots", flush=True)
    write_cloud_plots(output, context, portfolio, bins, weather_events, config)
    (output / "summary.md").write_text(_report(context, portfolio, bins, corr, drops, matched, weather_events, hourly, drop_bins, sensitivity, config), encoding="utf-8")
    print(f"Cloud study complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
