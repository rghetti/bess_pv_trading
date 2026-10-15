"""Write reproducible, graph-ready review material for the PV data audit.

Every value in the CSVs comes from the input data or a clearly named diagnostic.
The reporting layer does not edit the source Zarr or Parquet files, and it never
assigns a cleaning verdict to a flagged episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_CONTEXT_COLUMNS = (
    "farm_id",
    "time_utc",
    "power_dc_kw",
    "ghi_w_m2",
    "fraction_diffuse",
    "total_power_dc_kw",
    "solar_elevation_deg",
    "forecast_shortwave_w_m2",
    "forecast_cycle_time_utc",
    "forecast_lead_hours",
)
_EVENT_COLUMNS = (
    "event_id",
    "category",
    "farm_id",
    "start_utc",
    "end_utc",
    "duration_quarters",
    "score",
    "related_farms",
)


# Keep the report helpers independent of optional Markdown/table packages.
def _display_value(value: object) -> str:
    """Format one value for a compact Markdown table without changing CSV data."""
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    if frame.empty:
        return "No rows."
    selected = frame if columns is None else frame.loc[:, [c for c in columns if c in frame]]
    header = "| " + " | ".join(str(c) for c in selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(_display_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _safe_name(value: object) -> str:
    """Keep event IDs and farm IDs usable as local file and folder names."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned[:120] or "unknown"


def _serialize_json(value: object) -> object:
    """Turn diagnostic counts and dates into readable validation JSON."""
    if isinstance(value, Mapping):
        return {str(key): _serialize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, np.datetime64, np.timedelta64)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Keep CSVs complete: plot limits never trim the records available for review.
def _write_tables(
    output_dir: Path,
    context: pd.DataFrame,
    events: pd.DataFrame,
    forecast_checks: pd.DataFrame,
    lag_checks: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation: Mapping[str, object],
) -> pd.DataFrame:
    """Write complete CSVs, a blank review worksheet, and validation counts."""
    for label, required in (("context", _CONTEXT_COLUMNS), ("events", _EVENT_COLUMNS)):
        frame = context if label == "context" else events
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    context.to_csv(output_dir / "historical_context.csv", index=False)
    events.to_csv(output_dir / "events.csv", index=False)
    forecast_checks.to_csv(output_dir / "forecast_checks.csv", index=False)
    lag_checks.to_csv(output_dir / "lag_checks.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)

    review_columns = ["event_id", "farm_id", "category", "start_utc", "end_utc"]
    review = events.loc[:, review_columns].copy()
    review["decision"] = ""
    review["notes"] = ""
    review.to_csv(output_dir / "review_template.csv", index=False)

    (output_dir / "validation.json").write_text(
        json.dumps(_serialize_json(validation), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return review


# Describe the evidence and thresholds in a short, navigable text report.
def _coverage_section(context: pd.DataFrame) -> list[str]:
    """Summarize the full historical grid and observed missingness by farm."""
    times = pd.to_datetime(context["time_utc"], utc=True)
    start = times.min()
    end = times.max()
    lines = [
        "## Coverage",
        "",
        f"Historical interval starts: **{start.isoformat()}** through **{end.isoformat()}** "
        f"({len(context):,} farm intervals; {context['farm_id'].nunique()} farms).",
        "",
    ]
    counts = context.groupby("farm_id", dropna=False).agg(
        intervals=("time_utc", "size"),
        missing_power=("power_dc_kw", lambda s: int(s.isna().sum())),
        missing_ghi=("ghi_w_m2", lambda s: int(s.isna().sum())),
        missing_fraction_diffuse=("fraction_diffuse", lambda s: int(s.isna().sum())),
        missing_available_forecast=("forecast_shortwave_w_m2", lambda s: int(s.isna().sum())),
    )
    lines.extend([_markdown_table(counts.reset_index()), ""])
    return lines


def _event_sections(events: pd.DataFrame, output_dir: Path) -> list[str]:
    """Summarize complete event counts and strong review examples by farm."""
    lines = ["## Review episodes", ""]
    if events.empty:
        lines.extend(["No episodes were flagged by the configured diagnostics.", ""])
        return lines

    counts = (
        events.groupby(["farm_id", "category"], dropna=False)
        .size()
        .rename("episodes")
        .reset_index()
        .sort_values(["farm_id", "category"], key=lambda column: column.astype(str))
    )
    lines.extend([f"Total episodes: **{len(events):,}**.", "", _markdown_table(counts), ""])

    dated = events.copy()
    dated["month_utc"] = pd.to_datetime(dated["start_utc"], utc=True).dt.strftime("%Y-%m")
    monthly = (
        dated.groupby(["month_utc", "farm_id", "category"], dropna=False)
        .size()
        .rename("episodes")
        .reset_index()
    )
    monthly.to_csv(output_dir / "monthly_event_counts.csv", index=False)
    month_totals = monthly.groupby(["month_utc", "farm_id"], dropna=False)["episodes"].sum().unstack(fill_value=0)
    month_totals = month_totals.reset_index()
    lines.extend(
        [
            "### Monthly counts",
            "",
            "The complete farm × month × category counts are in "
            "[monthly_event_counts.csv](monthly_event_counts.csv).",
            "",
            _markdown_table(month_totals),
            "",
            "### Strong examples by farm",
            "",
        ]
    )

    ranked = events.copy()
    ranked["_score"] = pd.to_numeric(ranked["score"], errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(
        ["farm_id", "category", "_score", "start_utc"],
        ascending=[True, True, False, True],
    )
    examples = (
        ranked.groupby(["farm_id", "category"], dropna=False, sort=False)
        .head(1)
        .drop(columns="_score")
    )
    example_columns = [
        "farm_id",
        "category",
        "start_utc",
        "end_utc",
        "duration_quarters",
        "score",
        "event_id",
    ]
    lines.extend([_markdown_table(examples, example_columns), ""])
    return lines


def _sensitivity_section(sensitivity: pd.DataFrame) -> list[str]:
    """Show reporting-bin counts without presenting them as cleaning rules."""
    lines = [
        "## Threshold sensitivity",
        "",
        "These thresholds count observations for review. They are **not validity rules**, "
        "replacement rules, or a calibration mask. The complete table is in "
        "[sensitivity.csv](sensitivity.csv).",
        "",
    ]
    if sensitivity.empty:
        lines.extend(["No sensitivity rows were produced.", ""])
        return lines
    if "metric" in sensitivity:
        for metric, frame in sensitivity.groupby("metric", sort=False, dropna=False):
            active_columns = [
                column for column in frame.columns
                if column in ("farm_id", "metric", "count_intervals") or frame[column].notna().any()
            ]
            lines.extend([f"### {metric}", "", _markdown_table(frame, active_columns), ""])
    else:
        lines.extend([_markdown_table(sensitivity.head(100)), ""])
    return lines


def _forecast_section(forecast_checks: pd.DataFrame, lag_checks: pd.DataFrame) -> list[str]:
    """Present forecast and temporal-alignment diagnostics compactly."""
    lines = [
        "## Forecast comparison and timing",
        "",
        "[forecast_checks.csv](forecast_checks.csv) contains all farm and lead-time "
        "comparisons. Each forecast hour is compared with the mean of its four "
        "matching historical 15-minute GHI intervals when all four are present.",
        "",
    ]
    if not forecast_checks.empty:
        lead_column = next(
            (column for column in ("lead_hours", "lead_time_hours", "lead_time") if column in forecast_checks),
            None,
        )
        if lead_column is not None:
            selected_leads = {1, 6, 12, 24, 36, 48}
            sample = forecast_checks.loc[
                pd.to_numeric(forecast_checks[lead_column], errors="coerce").isin(selected_leads)
            ]
            if sample.empty:
                sample = forecast_checks.head(32)
        else:
            sample = forecast_checks.head(32)
        lines.extend(["### Selected lead times", "", _markdown_table(sample), ""])
    else:
        lines.extend(["No valid hourly forecast comparisons were available.", ""])

    lines.extend(
        [
            "### Power–GHI lag correlations",
            "",
            "Positive lag means power is later than GHI. Lags are diagnostic only; "
            "the audit applies no automatic time shift. The complete curves are in "
            "[lag_checks.csv](lag_checks.csv).",
            "",
        ]
    )
    lines.extend([_markdown_table(lag_checks.head(100)), ""])
    return lines


def _summary_text(
    context: pd.DataFrame,
    events: pd.DataFrame,
    forecast_checks: pd.DataFrame,
    lag_checks: pd.DataFrame,
    validation: Mapping[str, object],
    sensitivity: pd.DataFrame,
    config: Mapping[str, object],
    plotted_events: int,
    output_dir: Path,
) -> str:
    """Build the report with explicit interval and source interpretations."""
    forecast_config = config.get("forecast", {}) if isinstance(config.get("forecast"), Mapping) else {}
    solar_config = config.get("solar", {}) if isinstance(config.get("solar"), Mapping) else {}
    lag_hours = forecast_config.get("availability_lag_hours", 4)
    latitude = solar_config.get("latitude", solar_config.get("latitude_deg", "configured"))
    longitude = solar_config.get("longitude", solar_config.get("longitude_deg", "configured"))

    lines = [
        "# PV data audit",
        "",
        "This run preserves raw measurements and forecasts. An episode is a review "
        "candidate; its category is **not a fault verdict**. No values were replaced "
        "and no calibration mask was assigned.",
        "",
        "## Time and radiation conventions",
        "",
        "- All timestamps are UTC. A historical `time_utc` is the **start** of "
        "a 15-minute interval; `power_dc_kw` and historical `ghi_w_m2` are "
        "interval means over `[time_utc, time_utc + 15 min)`. Solar elevation "
        "is computed at that interval's midpoint.",
        "- A forecast `cycle_time` identifies a forecast vintage. `lead_time` "
        "advances from that vintage, so `valid_time = cycle_time + lead_time`. "
        "Radiation forecasts represent the preceding one-hour interval "
        "`[valid_time - 1 h, valid_time)`; their values are hourly means "
        "in W/m². Lead-zero radiation and values beyond the historical "
        "window have documented missingness.",
        f"- Event context uses the latest cycle published by the forecast hour's "
        f"start, assuming a {lag_hours}-hour delay after `cycle_time`. "
        "It does not use later forecast vintages.",
        f"- Solar position uses the Milan proxy ({latitude}° N, {longitude}° E), "
        "because farm coordinates are anonymous. Twilight counts therefore "
        "have approximate geographic context.",
        "- Historical `ghi_w_m2` is treated as satellite horizontal irradiance. "
        "The interpretation of forecast `shortwave_radiation` as horizontal "
        "total shortwave (GHI) and `direct_radiation` as horizontal direct beam "
        "is inferred from the earlier store and "
        "[ECMWF shortwave radiation](https://codes.ecmwf.int/grib/param-db/169) "
        "and [ECMWF direct radiation](https://codes.ecmwf.int/grib/param-db/228021) "
        "definitions. The new Zarr does not identify those source parameters "
        "explicitly.",
        "",
    ]
    lines.extend(_coverage_section(context))
    lines.extend(
        [
            "## Structural checks",
            "",
            "Full validation counters and examples are in "
            "[validation.json](validation.json). Expected boundary and lead-zero "
            "missing values are counted separately from unexpected missingness.",
            "",
            "```json",
            json.dumps(_serialize_json(validation), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    lines.extend(_event_sections(events, output_dir))
    lines.extend(_sensitivity_section(sensitivity))
    lines.extend(_forecast_section(forecast_checks, lag_checks))
    lines.extend(
        [
            "## Review files and plots",
            "",
            "- [events.csv](events.csv): every grouped episode, raw ranges, "
            "neighboring values, forecast context, and stable event ID.",
            "- [historical_context.csv](historical_context.csv): graph-ready "
            "15-minute farm series; the forecast column is the latest eligible "
            "forecast for each observation's hour.",
            "- [review_template.csv](review_template.csv): blank `decision` and "
            "`notes` columns for human review.",
            "- [config_used.yaml](config_used.yaml): exact configuration for this run.",
            f"- Plots: {plotted_events} selected event windows plus their "
            "event-centered UTC overviews, grouped under `plots/<farm_id>/<date>/`. Plot limits "
            "do not limit any CSV.",
            "",
        ]
    )
    return "\n".join(lines)


# Rank a bounded set of plots while retaining the full episode catalog in CSV.
def _selected_plot_events(events: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    """Choose configured examples and high-scoring events for bounded plotting."""
    if events.empty:
        return events.copy()
    plot_config = config.get("plots", {}) if isinstance(config.get("plots"), Mapping) else {}
    per_category = max(0, int(plot_config.get("max_events_per_farm_category", 4)))
    max_total = max(0, int(plot_config.get("max_total_events", 80)))
    pinned_rows: list[pd.DataFrame] = []
    for pin in plot_config.get("pinned_events", []):
        if not isinstance(pin, Mapping):
            raise ValueError("Each plots.pinned_events entry must be a mapping")
        at_time = pd.to_datetime(pin["time_utc"], utc=True)
        matches = events.loc[
            events["farm_id"].astype(str).eq(str(pin["farm_id"]))
            & events["category"].eq(str(pin["category"]))
            & pd.to_datetime(events["start_utc"], utc=True).le(at_time)
            & pd.to_datetime(events["end_utc"], utc=True).gt(at_time)
        ]
        if not matches.empty:
            pinned_rows.append(matches.sort_values(["start_utc", "event_id"]).head(1))
    pinned = pd.concat(pinned_rows, ignore_index=True) if pinned_rows else events.head(0).copy()
    ranked = events.copy()
    ranked["_score"] = pd.to_numeric(ranked["score"], errors="coerce").fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_score", "start_utc", "event_id"], ascending=[False, True, True], kind="mergesort"
    )
    ranked["_category_rank"] = ranked.groupby(
        ["farm_id", "category"], dropna=False, sort=False
    ).cumcount()
    ranked = ranked.loc[ranked["_category_rank"] < per_category]
    first_per_category = ranked.loc[ranked["_category_rank"] == 0].sort_values(
        ["farm_id", "category"], key=lambda column: column.astype(str)
    )
    remaining = ranked.loc[ranked["_category_rank"] > 0]
    selected = pd.concat([pinned, first_per_category, remaining], ignore_index=True)
    selected = selected.drop_duplicates("event_id", keep="first").head(max_total)
    return selected.drop(columns=["_score", "_category_rank"], errors="ignore")


def _related_farms(event: pd.Series, available: list[object]) -> list[object]:
    """Resolve a shared event into farm plots while preserving farm ID types."""
    farm = str(event["farm_id"])
    matched = [candidate for candidate in available if str(candidate) == farm]
    if matched:
        return matched
    related = str(event.get("related_farms", ""))
    listed = {part.strip() for part in re.split(r"[,;\s]+", related) if part.strip()}
    selected = [candidate for candidate in available if str(candidate) in listed]
    return selected or available


# Use one plotting function for a full UTC day and a focused event window.
def _plot_farm_window(
    farm_context: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    destination: Path,
    title: str,
    highlight: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> None:
    """Plot raw DC power, GHI, eligible forecast, and proxy solar elevation."""
    window = farm_context.loc[
        (farm_context["time_utc"] >= start) & (farm_context["time_utc"] < end)
    ]
    if window.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    time = window["time_utc"]
    axes[0].plot(time, window["power_dc_kw"], color="#1d4ed8", lw=1.35, label="DC power")
    axes[0].set_ylabel("DC power (kW)")
    axes[0].legend(loc="upper right")

    axes[1].plot(time, window["ghi_w_m2"], color="#d97706", lw=1.35, label="Historical GHI")
    if window["forecast_shortwave_w_m2"].notna().any():
        axes[1].plot(
            time,
            window["forecast_shortwave_w_m2"],
            color="#047857",
            lw=1.15,
            alpha=0.9,
            drawstyle="steps-post",
            label="Latest available forecast shortwave",
        )
    axes[1].set_ylabel("Irradiance (W/m²)")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, window["solar_elevation_deg"], color="#7c3aed", lw=1.3)
    axes[2].axhline(0, color="#475569", lw=0.8, ls="--")
    axes[2].set_ylabel("Solar elevation (°)")
    axes[2].set_xlabel("UTC")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=timezone.utc))

    if highlight is not None:
        shaded_end = highlight[1]
        for axis in axes:
            axis.axvspan(highlight[0], shaded_end, color="#f43f5e", alpha=0.12, lw=0)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=125)
    finally:
        plt.close(fig)


def _write_plots(
    output_dir: Path,
    context: pd.DataFrame,
    events: pd.DataFrame,
    config: Mapping[str, object],
) -> int:
    """Write bounded event windows and one day overview per selected farm/date."""
    chosen = _selected_plot_events(events, config)
    if chosen.empty:
        return 0
    plot_config = config.get("plots", {}) if isinstance(config.get("plots"), Mapping) else {}
    context_hours = float(plot_config.get("context_hours", 2))
    overview_days = max(1, int(plot_config.get("overview_days", 7)))
    margin = pd.Timedelta(hours=max(0.0, context_hours))
    farm_frames = {
        farm: frame.sort_values("time_utc")
        for farm, frame in context.groupby("farm_id", sort=False, dropna=False)
    }
    available_farms = list(farm_frames)
    overview_done: set[tuple[str, str]] = set()
    written = 0

    for _, event in chosen.iterrows():
        start = pd.to_datetime(event["start_utc"], utc=True)
        end = pd.to_datetime(event["end_utc"], utc=True)
        date_start = start.floor("D")
        date_label = date_start.strftime("%Y-%m-%d")
        for farm in _related_farms(event, available_farms):
            farm_label = _safe_name(farm)
            folder = output_dir / "plots" / farm_label / date_label
            farm_frame = farm_frames[farm]
            key = (farm_label, date_label)
            if key not in overview_done:
                overview_start = date_start - pd.Timedelta(days=(overview_days - 1) // 2)
                overview_end = overview_start + pd.Timedelta(days=overview_days)
                _plot_farm_window(
                    farm_frame,
                    overview_start,
                    overview_end,
                    folder / "overview.png",
                    f"Farm {farm}: {overview_days}-day UTC overview around {date_label}",
                    None,
                )
                overview_done.add(key)
            filename = f"{_safe_name(event['event_id'])}.png"
            _plot_farm_window(
                farm_frame,
                start - margin,
                end + margin,
                folder / filename,
                f"Farm {farm}: {event['category']} ({event['event_id']})",
                (start, end),
            )
            written += 1
    return written


def write_outputs(
    output_dir: str | Path,
    context: pd.DataFrame,
    events: pd.DataFrame,
    forecast_checks: pd.DataFrame,
    lag_checks: pd.DataFrame,
    validation: Mapping[str, object],
    sensitivity: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    """Write all audit tables, plots, and the human-readable summary.

    ``context`` and ``events`` are complete inputs; only figure selection is
    limited by ``plots.max_events_per_farm_category`` and
    ``plots.max_total_events``. Source data and these input frames are untouched.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_tables(
        destination,
        context,
        events,
        forecast_checks,
        lag_checks,
        sensitivity,
        validation,
    )
    plot_count = _write_plots(destination, context, events, config)
    summary = _summary_text(
        context,
        events,
        forecast_checks,
        lag_checks,
        validation,
        sensitivity,
        config,
        plot_count,
        destination,
    )
    (destination / "summary.md").write_text(summary, encoding="utf-8")
