"""Plot cloud associations with shared GHI zeros and sudden DC power drops.

The figures display observations alongside available weather forecasts. They
provide review context without interpreting a cloud association as a fault verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep names safe for the existing farm/date plot directory convention.
def _safe_name(value: object) -> str:
    """Convert an event or farm identifier to a bounded filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120] or "unknown"


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    """Write a figure and release its memory after the bounded plot selection."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(destination, dpi=140)
    finally:
        plt.close(figure)
    return destination


# Compare distributions on a common percentage scale despite unequal sample sizes.
def _plot_shared_association(
    output_dir: Path,
    portfolio: pd.DataFrame,
    cloud_bins: pd.DataFrame,
) -> Path:
    """Show forecast cloud distributions and the frequency of shared GHI zeros."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    active = portfolio.loc[portfolio["shared_eligible"].fillna(False)]
    subsets = (
        ("All active intervals", active, "#64748b"),
        ("Shared GHI zero", active.loc[active["shared_zero"].fillna(False)], "#b45309"),
        ("All farms GHI zero", active.loc[active["all_zero"].fillna(False)], "#be123c"),
    )
    histogram_edges = np.linspace(0, 100, 11)
    for label, frame, color in subsets:
        values = pd.to_numeric(frame["cloud_cover_mean_pct"], errors="coerce").dropna()
        if values.empty:
            continue
        axes[0].hist(
            values,
            bins=histogram_edges,
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            lw=1.7,
            label=f"{label} (n={len(values):,})",
            color=color,
        )
    axes[0].set(xlabel="Mean forecast cloud cover across farms (%)", ylabel="Share within each group (%)")
    axes[0].set_title("Cloud distribution with irradiance/production evidence")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)

    if not cloud_bins.empty:
        positions = np.arange(len(cloud_bins))
        width = 0.38
        axes[1].bar(
            positions - width / 2,
            cloud_bins["shared_zero_rate_pct"],
            width,
            color="#b45309",
            label="Shared GHI zero",
        )
        axes[1].bar(
            positions + width / 2,
            cloud_bins["all_zero_rate_pct"],
            width,
            color="#be123c",
            label="All farms GHI zero",
        )
        axes[1].set_xticks(positions, cloud_bins["cloud_bin"].astype(str), rotation=25, ha="right")
        for position, count in zip(positions, cloud_bins["eligible_timestamps"]):
            axes[1].annotate(
                f"n={int(count):,}", (position, 0), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=7,
                rotation=90,
            )
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Forecast cloud cover bin (%)", ylabel="Active intervals with event (%)")
    axes[1].set_title("Event rate within each cloud bin")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Forecast cloud context for shared GHI zeros in active intervals")
    return _save_figure(figure, output_dir / "plots" / "association.png")


# Retain the cloud forecast changes and GHI changes around every power drop.
def _plot_power_association(
    output_dir: Path,
    context: pd.DataFrame,
    stable_ghi_change: float,
) -> Path:
    """Compare 15-minute power and forecast-cloud changes for each farm's drops."""
    farms = sorted(context["farm_id"].drop_duplicates(), key=str)
    columns = min(2, max(1, len(farms)))
    rows = max(1, (len(farms) + columns - 1) // columns)
    figure, grid = plt.subplots(rows, columns, figsize=(7 * columns, 4 * rows), squeeze=False, constrained_layout=True)
    for axis, farm in zip(grid.flat, farms):
        frame = context.loc[
            context["farm_id"].eq(farm)
            & context["drop_eligible"].fillna(False)
            & context["power_drop"].fillna(False)
        ].copy()
        stable = frame["ghi_change_w_m2"].abs().le(stable_ghi_change)
        falling = frame["ghi_change_w_m2"].lt(-stable_ghi_change)
        groups = (
            ("Stable GHI", stable, "#1d4ed8", "o"),
            ("Falling GHI", falling, "#b45309", "v"),
            ("Rising / missing GHI", ~(stable | falling), "#7c3aed", "s"),
        )
        for label, mask, color, marker in groups:
            points = frame.loc[mask].dropna(subset=["cloud_change_pct", "power_change_fraction"])
            if points.empty:
                continue
            axis.scatter(
                points["cloud_change_pct"], 100 * points["power_change_fraction"],
                s=20, alpha=0.65, color=color, marker=marker,
                label=f"{label} (n={len(points):,})",
            )
        cycle_switch = frame.loc[frame["cycle_changed"].fillna(False)].dropna(
            subset=["cloud_change_pct", "power_change_fraction"]
        )
        if not cycle_switch.empty:
            axis.scatter(
                cycle_switch["cloud_change_pct"], 100 * cycle_switch["power_change_fraction"],
                s=58, facecolors="none", edgecolors="#111827", linewidths=0.8,
                label="Forecast cycle changed",
            )
        axis.axvline(0, color="#475569", ls="--", lw=0.8)
        axis.axhline(0, color="#475569", ls="--", lw=0.8)
        axis.set(
            title=f"Farm {farm}: sudden power drops in active intervals",
            xlabel="15-minute forecast cloud change (percentage points)",
            ylabel="15-minute power change (% of DC capacity)",
        )
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No eligible power drops", transform=axis.transAxes, ha="center")
    for axis in list(grid.flat)[len(farms):]:
        axis.set_visible(False)
    figure.suptitle(
        f"Cloud changes at sudden power drops; stable GHI: |ΔGHI| ≤ {stable_ghi_change:g} W/m²\n"
        "Cloud tendencies use one forecast vintage; circles mark changes of the available cycle"
    )
    return _save_figure(figure, output_dir / "plots" / "power_drop_association.png")


# Select a few informative episodes while honoring configured known examples.
def _pinned_rows(events: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Find configured event times that fall within this cloud-study event set."""
    plots_config = config.get("plots", {})
    pins = plots_config.get("pinned_events", []) if isinstance(plots_config, Mapping) else []
    found = []
    for pin in pins:
        if not isinstance(pin, Mapping) or "time_utc" not in pin:
            continue
        moment = pd.to_datetime(pin["time_utc"], utc=True)
        matching = events.loc[
            events["farm_id"].astype(str).eq(str(pin.get("farm_id", "")))
            & events["start_utc"].le(moment)
            & events["end_utc"].gt(moment)
        ]
        if "category" in pin:
            # Reuse the known shared-event time after activity-based regrouping.
            category = pin["category"]
            if category == "shared_daylight_zero_ghi":
                category = "shared_zero_ghi_candidate"
            matching = matching.loc[matching["category"].eq(category)]
        if not matching.empty:
            found.append(matching.sort_values(["start_utc", "event_id"]).head(1))
    return pd.concat(found, ignore_index=True) if found else events.head(0)


def _selected_events(
    context: pd.DataFrame,
    event_summary: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Choose longest and clearest shared events plus the strongest farm drops."""
    events = event_summary.copy()
    if events.empty:
        return events
    options = config.get("cloud_analysis", {})
    options = options if isinstance(options, Mapping) else {}
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True)
    shared = events.loc[events["farm_id"].astype(str).eq("portfolio")]
    selected = []
    if not shared.empty:
        longest = shared.sort_values(["duration_quarters", "start_utc"], ascending=[False, True])
        clearest = shared.sort_values(["mean_cloud_cover_pct", "start_utc"], na_position="last")
        selected.append(
            pd.concat([_pinned_rows(shared, config), longest.head(1), clearest.head(1), longest])
            .drop_duplicates("event_id")
            .head(max(0, int(options.get("max_shared_event_plots", 3))))
        )
    drops = events.loc[~events["farm_id"].astype(str).eq("portfolio")].copy()
    if not drops.empty:
        if "min_power_change_fraction" in drops:
            drops["_strength"] = -pd.to_numeric(drops["min_power_change_fraction"], errors="coerce").fillna(np.inf)
        else:
            # Restrict each search to one indexed farm instead of rescanning all farms.
            farm_frames = {str(farm): frame.set_index("time_utc").sort_index() for farm, frame in context.groupby("farm_id")}
            strengths = []
            for event in drops.itertuples(index=False):
                frame = farm_frames[str(event.farm_id)]
                values = frame.loc[(frame.index >= event.start_utc) & (frame.index < event.end_utc), "power_change_fraction"]
                strengths.append(-values.min() if values.notna().any() else -np.inf)
            drops["_strength"] = strengths
        per_farm = max(0, int(options.get("max_drop_event_plots_per_farm", 2)))
        for _, farm_events in drops.groupby("farm_id", sort=False):
            ranked = farm_events.sort_values(["_strength", "start_utc"], ascending=[False, True])
            selected.append(
                pd.concat([_pinned_rows(farm_events, config), ranked])
                .drop_duplicates("event_id").head(per_farm)
            )
    return pd.concat(selected, ignore_index=True).drop(columns="_strength", errors="ignore") if selected else events.head(0)


# Put raw interval means and the available forecast on separate, readable panels.
def _plot_event(
    output_dir: Path,
    farm: object,
    farm_context: pd.DataFrame,
    event: pd.Series,
    context_hours: float,
) -> Path | None:
    """Plot power, irradiance, cloud layers and solar elevation around one event."""
    start = event["start_utc"]
    end = event["end_utc"]
    margin = pd.Timedelta(hours=max(0, context_hours))
    window = farm_context.loc[farm_context["time_utc"].ge(start - margin) & farm_context["time_utc"].lt(end + margin)]
    if window.empty:
        return None
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    time = window["time_utc"]
    axes[0].plot(time, window["power_dc_kw"], color="#1d4ed8", lw=1.4, label="Historical DC power")
    capacity = window["total_power_dc_kw"].dropna()
    if not capacity.empty:
        axes[0].axhline(capacity.iloc[0], color="#475569", ls=":", lw=0.9, label="Installed DC capacity")
    axes[0].set_ylabel("DC power (kW)")
    axes[1].plot(time, window["ghi_w_m2"], color="#d97706", lw=1.4, label="Historical GHI")
    axes[1].plot(
        time, window["forecast_shortwave_w_m2"], color="#047857", lw=1.2,
        drawstyle="steps-post", label="Available forecast shortwave (hour mean)",
    )
    axes[1].set_ylabel("Irradiance (W/m²)")
    cloud_series = (
        ("cloud_cover_midpoint_pct", "Total cloud", "#111827", 1.7),
        ("cloud_cover_low_midpoint_pct", "Low cloud", "#0284c7", 1.0),
        ("cloud_cover_mid_midpoint_pct", "Middle cloud", "#7c3aed", 1.0),
        ("cloud_cover_high_midpoint_pct", "High cloud", "#db2777", 1.0),
    )
    for column, label, color, width in cloud_series:
        if column in window:
            axes[2].plot(time, window[column], color=color, lw=width, label=label)
    axes[2].set(ylabel="Forecast cloud (%)", ylim=(-3, 103))
    axes[3].plot(time, window["solar_elevation_deg"], color="#7c3aed", lw=1.2, label="Interval midpoint")
    axes[3].axhline(0, color="#475569", ls="--", lw=0.8)
    axes[3].set(ylabel="Proxy solar elevation (°)", xlabel="UTC historical interval start")
    axes[3].set_title("Milan proxy (reference only)", loc="left", fontsize=9)
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=timezone.utc))
    switches = window.loc[window["cycle_changed"].fillna(False), "time_utc"]
    for axis in axes:
        axis.axvspan(start, end, color="#f43f5e", alpha=0.11, lw=0)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8, ncols=2 if axis is axes[2] else 1)
    for moment in switches:
        for axis in axes[1:3]:
            axis.axvline(moment, color="#64748b", ls=":", lw=0.8, alpha=0.7)
    # Retain original event IDs while distinguishing their earlier audit labels.
    category = str(event["category"])
    caption = category.replace("daylight_", "").replace("_", " ")
    if "daylight" in category:
        caption += " (original audit candidate)"
    figure.suptitle(
        f"Farm {farm} · {caption} · {start.strftime('%Y-%m-%d %H:%M')} UTC\n"
        "Shading marks episode; cloud curves use the available cycle; dotted lines mark cycle changes"
    )
    destination = output_dir / "plots" / _safe_name(farm) / start.strftime("%Y-%m-%d") / f"{_safe_name(event['event_id'])}.png"
    return _save_figure(figure, destination)


# Expose one reporting entry point for the cloud study CLI.
def write_cloud_plots(
    output_dir: Path,
    context: pd.DataFrame,
    portfolio: pd.DataFrame,
    cloud_bins: pd.DataFrame,
    event_summary: pd.DataFrame,
    config: Mapping[str, object],
) -> list[Path]:
    """Write association figures and a bounded set of farm/date event windows."""
    output_dir = Path(output_dir)
    if context.empty or portfolio.empty:
        return []
    options = config.get("cloud_analysis", {})
    options = options if isinstance(options, Mapping) else {}
    stable_ghi = float(options.get("stable_ghi_change_w_m2", 50))
    context = context.copy()
    context["time_utc"] = pd.to_datetime(context["time_utc"], utc=True)
    written = [
        _plot_shared_association(output_dir, portfolio, cloud_bins),
        _plot_power_association(output_dir, context, stable_ghi),
    ]
    frames = {farm: frame.sort_values("time_utc") for farm, frame in context.groupby("farm_id", sort=False)}
    for _, event in _selected_events(context, event_summary, config).iterrows():
        event_farm = str(event["farm_id"])
        for farm, frame in frames.items():
            if event_farm not in ("portfolio", str(farm)):
                continue
            destination = _plot_event(output_dir, farm, frame, event, float(options.get("context_hours", 2)))
            if destination is not None:
                written.append(destination)
    return written
