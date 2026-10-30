"""Inspect final static farm parameters through the operational forecast path."""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ...config import load_config
from ...data import config_path, load_metadata, open_store
from ...markets import AuctionCalendar
from ...simulation.gme_oracle import _forecast_pv_at_cutoff


def _common_example_days(root: Path, farm_ids: list[int], quantiles) -> list[str]:
    """Select common days by production level, independently of forecast errors."""
    daily = []
    # Select common days by production level, independently of forecast errors for each `farm_id` in `farm_ids`.
    for farm_id in farm_ids:
        frame = pd.read_parquet(root / f"farm_{farm_id}" / "parquet" / "corpus.parquet")
        view = frame.loc[frame.auction == "DA"]
        complete = view.groupby("delivery_day").measured_ac_mw.apply(
            lambda values: bool(np.isfinite(values).all() and (values >= 0).all()))
        energy = view.groupby("delivery_day").measured_ac_mw.sum() * .25
        rating = json.loads((root / f"farm_{farm_id}" / "json" / "summary.json").read_text())["ac_rating_mw"]
        daily.append((energy.loc[complete] / rating).rename(str(farm_id)))
    common = pd.concat(daily, axis=1).dropna().mean(axis=1).sort_index()
    # Enforce the input contract: Too few common complete measurement days for trajectory examples.
    if len(common) < len(quantiles):
        raise ValueError("Too few common complete measurement days for trajectory examples")
    chosen = []
    # Select common days by production level, independently of forecast errors for each `quantile` in `quantiles`.
    for quantile in quantiles:
        # Enforce the input contract: Production quantiles must be between zero and one.
        if not 0 <= float(quantile) <= 1:
            raise ValueError("Production quantiles must be between zero and one")
        distance = (common - common.quantile(quantile)).abs().drop(index=chosen)
        chosen.append(str(distance.sort_values(kind="stable").index[0]))
    return chosen


def plot_global_trajectories(study_root: str | Path, config_file: str | Path,
                             *, quantiles=(.2, .5, .8), dates=None) -> dict:
    """Plot three representative measured/forecast days per farm, with AC limits.

    Forecasts use the resolved standard parameters and actual submission cutoffs.
    Measurements are raw AC, including the following midnight when available;
    measured values are never clipped to disguise nameplate exceedances.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    root = Path(study_root)
    original = json.loads((root / "json" / "study_config.json").read_text())
    farm_ids = list(map(int, original["pv_calibration"]["farm_ids"]))
    chosen = list(dates) if dates is not None else _common_example_days(root, farm_ids, quantiles)
    # Plot three representative measured/forecast days per farm, with AC limits for each `folder` in `('plots', 'csv', 'json')`.
    for folder in ("plots", "csv", "json"):
        (root / folder / "trajectory_examples").mkdir(parents=True, exist_ok=True)
    result = {"dates": chosen, "selection": "common_complete_days_nearest_portfolio_normalized_energy_quantiles"
              if dates is None else "explicit_dates", "quantiles": list(quantiles) if dates is None else None,
              "configuration_source": str(config_file), "farms": {}}
    colors = {"DA": "tab:blue", "IDA1": "tab:orange", "IDA2": "tab:green", "IDA3": "tab:purple"}
    # Write the corresponding artifacts for `farm_id` from `farm_ids`.
    for farm_id in farm_ids:
        config = load_config(config_file, overrides={"farm_id": farm_id, "market": {
            "valid_from": original["pv_calibration"]["start_day"],
            "valid_to": (pd.Timestamp(original["pv_calibration"]["end_day_exclusive"])
                         - pd.Timedelta(days=1)).date().isoformat()}})
        calendar = AuctionCalendar.from_config(config)
        farm = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
        rating = float(config["farms"][str(farm_id)]["ac_capacity_kw"]) / 1000
        fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 3.8 * len(chosen)),
                                 sharey=True, squeeze=False, layout="constrained")
        rows, provenance, checks = [], [], []
        with open_store(config) as store:
            dataset = store.sel(farm_id=[farm_id]).load()
        try:
            power = pd.Series(dataset.power.sel(farm_id=farm_id).values,
                              index=pd.to_datetime(dataset.time.values, utc=True))
            power *= float(config["forecast"]["realized_power_scale_to_mw"])
            highest = rating
            # Plot three representative measured/forecast days per farm, with AC limits for each `(axis, day)` in `zip(axes.flat, chosen)`.
            for axis, day in zip(axes.flat, chosen):
                grid = calendar.delivery_grid(day)
                # Include the true following-midnight sample, rather than
                # copying the last quarter-hour measurement as a fictitious point.
                starts = grid.starts.append(pd.DatetimeIndex([grid.ends[-1]]))
                actual = power.reindex(starts).to_numpy(dtype=float)
                frame = pd.DataFrame({"interval_start_utc": starts,
                    "interval_start_local": starts.tz_convert(calendar.timezone),
                    "delivery_day": day, "measured_ac_mw": actual,
                    "ac_rating_mw": rating, "next_midnight_boundary": starts == grid.ends[-1]})
                local = starts.tz_convert(calendar.timezone)
                axis.step(local, actual, where="post", color="black", linewidth=2, label="Measured AC", zorder=5)
                day_checks = {"delivery_day": day, "measured_peak_mw": float(np.nanmax(actual)),
                    "measured_above_rating_intervals": int(np.sum(actual[:-1] > rating)),
                    "midnight_measurement_available": bool(np.isfinite(actual[-1]))}
                highest = max(highest, float(np.nanmax(actual)))
                # Plot three representative measured/forecast days per farm, with AC limits for each `event` in `calendar.schedule(day)`.
                for event in calendar.schedule(day):
                    eligible = grid.starts[list(event.eligible_intervals)]
                    # The boundary follows the same weather vintage and model.
                    forecast_starts = eligible.append(pd.DatetimeIndex([grid.ends[-1]]))
                    forecast = _forecast_pv_at_cutoff(config, calendar, dataset, farm,
                                                      event.submission_at, forecast_starts)
                    values = forecast.values.pv_ac_mw.to_numpy(dtype=float)
                    axis.step(forecast_starts.tz_convert(calendar.timezone), values, where="post",
                              color=colors[event.auction], linewidth=1.2, alpha=.85, label=event.auction)
                    frame[f"{event.auction}_forecast_mw"] = pd.Series(values, index=forecast_starts).reindex(starts).to_numpy()
                    frame[f"{event.auction}_forecast_cycle_utc"] = forecast.forecast_vintage
                    frame[f"{event.auction}_forecast_available_at_utc"] = forecast.available_at
                    frame[f"{event.auction}_submission_cutoff_utc"] = event.submission_at
                    day_checks[f"{event.auction}_forecast_peak_mw"] = float(np.max(values))
                    day_checks[f"{event.auction}_forecast_above_rating_intervals"] = int(np.sum(values > rating + 1e-10))
                    # Enforce the input contract: Farm `farm_id` `event.auction` forecast exceeds its AC rating.
                    if np.any(values > rating + 1e-10):
                        raise ValueError(f"Farm {farm_id} {event.auction} forecast exceeds its AC rating")
                    provenance.append({"delivery_day": day, "auction": event.auction,
                        "decision_cutoff_utc": event.submission_at, "provenance": forecast.provenance})
                axis.axhline(rating, color="crimson", linestyle="--", linewidth=1.5,
                             label=f"Installed AC rating: {rating:g} MW")
                axis.set(title=day, ylabel="AC power (MW)", xlim=(local[0], local[-1]))
                axis.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 3), tz=ZoneInfo(calendar.timezone)))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(calendar.timezone)))
                axis.grid(alpha=.18)
                rows.append(frame)
                checks.append(day_checks)
            # Plot three representative measured/forecast days per farm, with AC limits for each `axis` in `axes.flat`.
            for axis in axes.flat:
                axis.set_ylim(0, highest * 1.09)
            axes.flat[-1].set_xlabel(f"Delivery time ({calendar.timezone}); final 00:00 is the following day")
            axes.flat[0].legend(loc="upper left", ncol=3, fontsize=9, framealpha=.95)
            fig.suptitle(f"Farm {farm_id} — globally calibrated PV forecasts and measured production", fontsize=14)
            plot = root / "plots" / "trajectory_examples" / f"farm_{farm_id}_three_days.png"
            fig.savefig(plot, dpi=160)
            plt.close(fig)
            csv = root / "csv" / "trajectory_examples" / f"farm_{farm_id}_three_days.csv"
            pd.concat(rows, ignore_index=True).to_csv(csv, index=False)
            detail = {"farm_id": farm_id, "ac_rating_mw": rating, "checks": checks,
                      "resolved_configuration": config, "forecast_provenance": provenance}
            (root / "json" / "trajectory_examples" / f"farm_{farm_id}.json").write_text(
                json.dumps(detail, indent=2, default=str) + "\n")
            result["farms"][str(farm_id)] = {"plot": str(plot), "csv": str(csv), "checks": checks}
        finally:
            dataset.close()
    (root / "json" / "trajectory_examples" / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
