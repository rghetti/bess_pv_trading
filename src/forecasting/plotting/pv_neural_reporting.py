"""Paired neural/pvlib benchmark tables, operating diagnostics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..training.pv_fitting import balanced_weights
from ..plotting.pv_study_reporting import error_metrics, paired_bootstrap


def report_neural_farm(frame, farm, methods, section, destination, diagnostics):
    """Report untouched test data separately from training and validation scores."""
    data = frame.copy()
    data["month"] = pd.to_datetime(data.delivery_day).dt.strftime("%Y-%m")
    data["delivery_hour"] = pd.DatetimeIndex(data.interval_start_utc).tz_convert(section["delivery_timezone"]).hour
    rows = []
    # Report untouched test data separately from training and validation scores for each `split` in `('train', 'validation', 'test')`.
    for split in ("train", "validation", "test"):
        # Report untouched test data separately from training and validation scores for each `scope` in `('all_intervals', 'daylight')`.
        for scope in ("all_intervals", "daylight"):
            view = data.loc[data.eligible & (data.split == split)]
            # When `scope == 'daylight'`, set `view` using `view.loc[view.daylight]`.
            if scope == "daylight":
                view = view.loc[view.daylight]
            # Report untouched test data separately from training and validation scores for each `auction` in `('ALL_BALANCED', 'DA', 'IDA1', 'IDA2', 'IDA3')`.
            for auction in ("ALL_BALANCED", "DA", "IDA1", "IDA2", "IDA3"):
                group = view if auction == "ALL_BALANCED" else view.loc[view.auction == auction]
                # Skip the current item when `group.empty`.
                if group.empty:
                    continue
                weights = balanced_weights(group)
                # Report untouched test data separately from training and validation scores for each `method` in `methods`.
                for method in methods:
                    prediction = group[f"prediction_{method}_mw"].to_numpy()
                    actual = group.measured_ac_mw.to_numpy()
                    error = prediction-actual
                    scores = error_metrics(actual, prediction, farm.ac_capacity_mw)
                    # Override pooled moments with the explicit day/auction weighting.
                    scores.update(mae_mw=float(np.dot(weights, abs(error))),
                        rmse_mw=float(np.sqrt(np.dot(weights, error**2))),
                        bias_mw=float(np.dot(weights, error)),
                        min_absolute_error_mw=float(np.min(abs(error))),
                        max_absolute_error_mw=float(np.max(abs(error))))
                    scores["nrmse_percent_ac_rating"] = 100*scores["rmse_mw"]/farm.ac_capacity_mw
                    # Energy across overlapping views is deliberately not aggregated.
                    if auction == "ALL_BALANCED":
                        # Process each `name` in this collection.
                        for name in ("paired_energy_bias_mwh", "measured_energy_mwh", "forecast_energy_mwh"):
                            scores.pop(name, None)
                    rows.append({"farm_id": farm.farm_id, "split": split, "scope": scope,
                                 "auction": auction, "method": method, **scores})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(destination/"csv"/"metrics.csv", index=False)
    daily = []
    # Report untouched test data separately from training and validation scores for each `((split, day, auction), group)` in `data.loc[data.eligible].groupby(['split', 'delivery_day', 'auction'])`.
    for (split, day, auction), group in data.loc[data.eligible].groupby(["split", "delivery_day", "auction"]):
        # Report untouched test data separately from training and validation scores for each `method` in `methods`.
        for method in methods:
            error = group[f"prediction_{method}_mw"].to_numpy()-group.measured_ac_mw.to_numpy()
            daily.append({"split": split, "delivery_day": day, "auction": auction, "method": method,
                "forecast_energy_mwh": float(group[f"prediction_{method}_mw"].sum()*.25),
                "measured_energy_mwh": float(group.measured_ac_mw.sum()*.25),
                "energy_bias_mwh": float(error.sum()*.25), "mae_mw": float(abs(error).mean()),
                "max_absolute_error_mw": float(abs(error).max()), "scored_intervals": len(group)})
    pd.DataFrame(daily).to_csv(destination/"csv"/"daily_errors.csv", index=False)
    coverage = data.groupby(["split", "delivery_day", "auction"]).agg(
        intervals=("eligible", "size"), eligible=("eligible", "sum"),
        daylight=("daylight", "sum"), suspected_outages=("suspected_outage", "sum"),
        plateau_intervals=("plateau_flag", "sum")).reset_index()
    coverage.to_csv(destination/"csv"/"coverage.csv", index=False)
    grouped_rows = []
    test = data.loc[data.eligible & (data.split == "test")]
    # Report untouched test data separately from training and validation scores for each `dimension` in `('month', 'delivery_hour')`.
    for dimension in ("month", "delivery_hour"):
        # Report untouched test data separately from training and validation scores for each `((auction, value), group)` in `test.groupby(['auction', dimension])`.
        for (auction, value), group in test.groupby(["auction", dimension]):
            # Report untouched test data separately from training and validation scores for each `method` in `methods`.
            for method in methods:
                grouped_rows.append({"dimension": dimension, "value": str(value), "auction": auction,
                    "method": method, **error_metrics(group.measured_ac_mw, group[f"prediction_{method}_mw"], farm.ac_capacity_mw)})
    pd.DataFrame(grouped_rows).to_csv(destination/"csv"/"test_error_groups.csv", index=False)
    data["phase"] = np.where(data.split == "test", "held_out", "warmup")
    paired = paired_bootstrap(data, methods, farm.ac_capacity_mw, section)
    pd.DataFrame(paired).to_csv(destination/"csv"/"paired_test_comparison.csv", index=False)
    selected = metrics.loc[(metrics.split == "test") & (metrics.scope == "daylight")
                           & (metrics.auction == "ALL_BALANCED")]
    summary = {"farm_id": farm.farm_id, "ac_rating_mw": farm.ac_capacity_mw,
        "test_daylight_metrics": selected.to_dict("records"), "paired_test_comparison": paired,
        "models": diagnostics, "mean_weighting": "equal_day_auction_daylight_view",
        "target": "raw_quarter_hour_AC_MW_no_conversion_or_imputation",
        "selection": "minimum_prior_validation_NRMSE_over_neural_candidates_only",
        "test_used_for_selection": False,
        "pvlib": "fixed_global_2025_calibrated_parameters_from_standard_config",
        "limitations": "Proxy coordinates, weather errors and operating anomalies remain. Auction forecasts overlap; their energy totals must not be summed."}
    (destination/"json"/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    _farm_plots(data, metrics, diagnostics, farm, section, destination)
    return summary


def _farm_plots(data, metrics, diagnostics, farm, section, destination):
    """Show test skill, training histories and complete daily forecast trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    test = metrics.loc[(metrics.split == "test") & (metrics.scope == "daylight")]
    # Show test skill, training histories and complete daily forecast trajectories for each `(axis, auction)` in `zip(axes.flat, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
    for axis, auction in zip(axes.flat, ("DA", "IDA1", "IDA2", "IDA3")):
        group = test.loc[test.auction == auction].sort_values("rmse_mw")
        axis.barh(group.method, group.nrmse_percent_ac_rating,
                  color=["tab:orange" if name == "pvlib" else "tab:blue" for name in group.method])
        axis.set(title=auction, xlabel="Test daylight RMSE / AC rating (%)")
    fig.suptitle(f"Farm {farm.farm_id} — untouched test forecast accuracy")
    fig.savefig(destination/"plots"/"test_accuracy.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    # Show test skill, training histories and complete daily forecast trajectories for each `item` in `diagnostics`.
    for item in diagnostics:
        # When `'history' in item`, set `history` using `pd.DataFrame(item['history'])`.
        if "history" in item:
            history = pd.DataFrame(item["history"])
            axis.plot(history.epoch, 100*history.validation_nrmse, label=item["candidate"]["name"])
    axis.set(xlabel="Epoch", ylabel="Validation RMSE / AC rating (%)", title=f"Farm {farm.farm_id}: prior validation learning curves")
    axis.legend(); axis.grid(alpha=.2)
    fig.savefig(destination/"plots"/"learning_curves.png", dpi=150); plt.close(fig)
    # Select representative test days by production level, not prediction residual.
    view = data.loc[(data.split == "test") & (data.auction == "DA") & data.eligible]
    energy = view.groupby("delivery_day").measured_ac_mw.sum()
    examples = []
    # Show test skill, training histories and complete daily forecast trajectories for each `quantile` in `np.linspace(0.2, 0.8, int(section['representative_days']))`.
    for quantile in np.linspace(.2, .8, int(section["representative_days"])):
        distances = (energy-energy.quantile(quantile)).abs().drop(index=examples)
        # Show test skill, training histories and complete daily forecast trajectories when `not distances.empty`.
        if not distances.empty:
            examples.append(distances.sort_values(kind="stable").index[0])
    # Show test skill, training histories and complete daily forecast trajectories for each `day` in `examples`.
    for day in examples:
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, sharey=True, layout="constrained")
        actual = data.loc[(data.delivery_day == day) & (data.auction == "DA")]
        midnight = pd.Timestamp(day).tz_localize(section["delivery_timezone"])+pd.DateOffset(days=1)
        next_actual = data.loc[(data.auction == "DA") & (data.interval_start_utc == midnight.tz_convert("UTC"))]
        # Show test skill, training histories and complete daily forecast trajectories for each `(axis, auction)` in `zip(axes, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
        for axis, auction in zip(axes, ("DA", "IDA1", "IDA2", "IDA3")):
            group = data.loc[(data.delivery_day == day) & (data.auction == auction)]
            times = pd.DatetimeIndex(group.interval_start_utc).tz_convert(section["delivery_timezone"])
            measured_times = pd.DatetimeIndex(actual.interval_start_utc).tz_convert(section["delivery_timezone"])
            measured = actual.measured_ac_mw.to_numpy()
            # Plot the measured next-midnight point when it exists; forecast
            # interval means extend to their true final delivery boundary.
            if not next_actual.empty:
                measured_times = measured_times.append(pd.DatetimeIndex([midnight]))
                measured = np.r_[measured, next_actual.measured_ac_mw.iloc[0]]
            axis.step(measured_times, measured, where="post", label="Measured AC", color="black", linewidth=1.6)
            extended = times.append(pd.DatetimeIndex([midnight]))
            # Process each `(name, label, color)` in this collection.
            for name, label, color in (("pvlib", "Calibrated pvlib", "tab:orange"),
                                       ("selected", "Validation-selected neural", "tab:blue")):
                axis.step(extended, np.r_[group[f"prediction_{name}_mw"], group[f"prediction_{name}_mw"].iloc[-1]],
                          where="post", label=label, color=color)
            axis.axhline(farm.ac_capacity_mw, color="crimson", linestyle="--", label="Installed AC rating")
            axis.set(title=auction, ylabel="AC MW", ylim=(0, farm.ac_capacity_mw*1.08))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(section["delivery_timezone"])))
            axis.grid(alpha=.2)
        axes[0].legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel(f"Delivery time ({section['delivery_timezone']}); last midnight is the following day")
        fig.suptitle(f"Farm {farm.farm_id}: {day}, held-out test")
        fig.savefig(destination/"plots"/f"forecast_{day}.png", dpi=150); plt.close(fig)
