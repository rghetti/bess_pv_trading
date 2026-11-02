"""Evaluate pinned ensemble forecasts at each auction and without double counting."""
from pathlib import Path

import numpy as np
import pandas as pd


def save_ensemble_diagnostics(destination, forecast_inputs, prepared, *, plots=True):
    """Compare 50 weather members with actual PV at each available decision cutoff."""
    source = pd.DataFrame(forecast_inputs)
    required = {"delivery_day", "auction", "cutoff_utc", "interval_start_utc", "pv_ac_forecast_mw", "scenario_id", "forecast_role"}
    if source.empty or not required.issubset(source):
        return
    weather = source.loc[source.scenario_id.fillna("").str.startswith("weather_")].copy()
    deterministic = source.loc[source.forecast_role.eq("deterministic_diagnostic")].copy()
    # Older checkpoints stored deterministic targets without a scenario label.
    legacy = source.loc[source.forecast_role.eq("target_delivery") & source.scenario_id.isna()]
    deterministic = pd.concat([deterministic, legacy], ignore_index=True).drop_duplicates(
        ["delivery_day", "auction", "cutoff_utc", "interval_start_utc"], keep="first")
    if weather.empty or deterministic.empty:
        return
    keys = ["delivery_day", "auction", "cutoff_utc", "interval_start_utc"]
    if weather.duplicated(keys + ["scenario_id"]).any() or deterministic.duplicated(keys).any():
        raise ValueError("Duplicate ensemble forecast diagnostics")
    grouped = weather.groupby(keys, sort=False).pv_ac_forecast_mw.agg(["mean", "min", "max", "count"]).reset_index()
    metadata = [column for column in ("pv_forecast_vintage_utc", "pv_forecast_available_at_utc") if column in deterministic]
    grouped = grouped.merge(deterministic[keys + ["pv_ac_forecast_mw", *metadata]], on=keys,
                            validate="one_to_one", how="left")
    if grouped.pv_ac_forecast_mw.isna().any():
        raise ValueError("Missing deterministic forecast for ensemble diagnostics")
    expected_members = {day: len(prepared["days_data"][day]["ensemble_members"]) for day in prepared["days"]}
    if any(int(row['count']) != expected_members[row['delivery_day']] for row in grouped.to_dict('records')):
        raise ValueError("Incomplete weather-member diagnostics")
    actual = []
    for day in prepared["days"]:
        item = prepared["days_data"][day]
        for start, duration, value in zip(item["grid"].starts, item["grid"].delta_hours, item["realized_pv"]):
            actual.append({"delivery_day": day, "interval_start_utc": pd.Timestamp(start),
                           "duration_hours": float(duration), "realized_pv_mw": float(value)})
    grouped["interval_start_utc"] = pd.to_datetime(grouped.interval_start_utc, utc=True)
    actual = pd.DataFrame(actual)
    grouped = grouped.merge(actual, on=["delivery_day", "interval_start_utc"], validate="many_to_one", how="left")
    if grouped.realized_pv_mw.isna().any():
        raise ValueError("Ensemble diagnostics cannot align realized PV")
    grouped["mean_minus_min_mwh"] = (grouped['mean'] - grouped['min']) * grouped.duration_hours
    grouped["weather_spread_mw"] = grouped['max'] - grouped['min']
    grouped["realized_below_min_mwh"] = (grouped['min'] - grouped.realized_pv_mw).clip(lower=0) * grouped.duration_hours
    grouped["realized_below_min"] = grouped.realized_pv_mw < grouped['min']
    grouped["deterministic_below_weather_min"] = grouped.pv_ac_forecast_mw < grouped['min']
    grouped["deterministic_at_envelope"] = grouped.pv_ac_forecast_mw <= grouped['min']
    grouped["deterministic_envelope_reduction_mwh"] = (grouped['min'] - grouped.pv_ac_forecast_mw).clip(lower=0) * grouped.duration_hours
    provenance = prepared["provenance"]["pv_ensemble"]
    grouped["ensemble_vintage_utc"] = grouped.delivery_day.map(lambda day: provenance[day]["forecast_vintage_utc"])
    grouped["ensemble_available_at_utc"] = grouped.delivery_day.map(lambda day: provenance[day]["available_at_utc"])
    grouped["realized_pv_fallback_intervals_on_day"] = grouped.delivery_day.map(
        lambda day: prepared["days_data"][day].get("pv_provenance", {}).get("realized_power_missing_fallback_intervals", 0))
    grouped = grouped.rename(columns={"mean": "weather_mean_mw", "min": "weather_min_mw", "max": "weather_max_mw", "pv_ac_forecast_mw": "deterministic_mw", "count": "weather_member_count"})
    directory = Path(destination) / "csv"
    grouped.to_csv(directory / "ensemble_diagnostics_by_interval_cutoff.csv", index=False)

    def aggregate(frame, keys):
        """Aggregate durations and energy gaps without averaging overlapping cutoffs."""
        result = frame.groupby(keys, sort=True).agg(intervals=("realized_pv_mw", "size"),
            hours=("duration_hours", "sum"), mean_min_gap_mwh=("mean_minus_min_mwh", "sum"),
            realized_below_min_intervals=("realized_below_min", "sum"),
            realized_below_min_mwh=("realized_below_min_mwh", "sum"),
            deterministic_lower_intervals=("deterministic_below_weather_min", "sum"),
            deterministic_at_envelope_intervals=("deterministic_at_envelope", "sum"),
            deterministic_envelope_reduction_mwh=("deterministic_envelope_reduction_mwh", "sum"),
            mean_weather_mean_mw=("weather_mean_mw", "mean"),
            mean_weather_min_mw=("weather_min_mw", "mean"),
            mean_weather_spread_mw=("weather_spread_mw", "mean")).reset_index()
        result["realized_below_min_frequency"] = result.realized_below_min_intervals / result.intervals
        result["deterministic_lower_frequency"] = result.deterministic_lower_intervals / result.intervals
        result["deterministic_at_envelope_frequency"] = result.deterministic_at_envelope_intervals / result.intervals
        return result

    decisions = aggregate(grouped, ["delivery_day", "auction", "cutoff_utc"])
    decisions.to_csv(directory / "ensemble_diagnostics_by_decision.csv", index=False)
    # Select the latest available modeled forecast for each delivery interval.
    grouped["cutoff_utc"] = pd.to_datetime(grouped.cutoff_utc, utc=True)
    latest = grouped.sort_values("cutoff_utc").drop_duplicates(["delivery_day", "interval_start_utc"], keep="last")
    daily = aggregate(latest, ["delivery_day"])
    daily.to_csv(directory / "ensemble_diagnostics_daily.csv", index=False)
    latest["delivery_month"] = latest.delivery_day.str[:7]
    aggregate(latest, ["delivery_month"]).to_csv(directory / "ensemble_diagnostics_monthly.csv", index=False)
    if plots:
        import matplotlib.pyplot as plt
        output = Path(destination) / "plots"
        output.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")
        axis.bar(pd.to_datetime(daily.delivery_day), daily.mean_min_gap_mwh)
        axis.set(title="Latest planned weather ensemble mean minus minimum", ylabel="Energy gap (MWh)", xlabel="Delivery day")
        fig.savefig(output / "ensemble_mean_min_gap.png", dpi=140)
        plt.close(fig)
