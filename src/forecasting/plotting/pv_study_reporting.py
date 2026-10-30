"""Paired chronological PV accuracy, energy diagnostics and compact study figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def error_metrics(actual, prediction, rating: float) -> dict:
    """Describe signed power errors and paired energy; never use MAPE at zero PV."""
    error = np.asarray(prediction) - np.asarray(actual)
    # Return `{'pairs': 0}` when `not len(error)`.
    if not len(error):
        return {"pairs": 0}
    over, under = error[error > 0], error[error < 0]
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"pairs": len(error), "mae_mw": float(np.mean(abs(error))), "rmse_mw": rmse,
        "nrmse_percent_ac_rating": 100*rmse/rating, "bias_mw": float(np.mean(error)),
        "overprediction_fraction": float(np.mean(error > 0)),
        "underprediction_fraction": float(np.mean(error < 0)),
        "mean_overprediction_mw_when_over": float(np.mean(over)) if len(over) else 0.0,
        "mean_underprediction_mw_when_under": float(-np.mean(under)) if len(under) else 0.0,
        "paired_energy_bias_mwh": float(error.sum()*0.25),
        "measured_energy_mwh": float(np.sum(actual)*0.25),
        "forecast_energy_mwh": float(np.sum(prediction)*0.25),
        "forecast_at_ac_rating_fraction": float(np.mean(np.asarray(prediction) >= rating-1e-8))}


def paired_bootstrap(data: pd.DataFrame, methods: list[str], rating: float, section: dict):
    """Resample contiguous seven-day blocks, retaining paired auction/day errors."""
    chosen = data.loc[data.eligible & data.daylight & (data.phase == "held_out")].copy()
    # Return `[]` when `chosen.empty`.
    if chosen.empty:
        return []
    group = ["delivery_day", "auction"]
    baseline = (chosen.baseline_mw-chosen.measured_ac_mw)**2
    base = baseline.groupby([chosen[k] for k in group]).mean().groupby(level=0).mean()
    count = len(base)
    rng = np.random.default_rng(int(section["seed"]))
    block = int(section["bootstrap_block_days"])
    repetitions = int(section["bootstrap_repetitions"])
    starts = rng.integers(0, count, size=(repetitions, int(np.ceil(count/block))))
    indexes = ((starts[:, :, None]+np.arange(block)) % count).reshape(repetitions, -1)[:, :count]
    sampled_base = np.sqrt(base.to_numpy()[indexes].mean(axis=1))
    rows = []
    # Resample contiguous seven-day blocks, retaining paired auction/day errors for each `method` in `methods`.
    for method in methods:
        error = (chosen[f"prediction_{method}_mw"]-chosen.measured_ac_mw)**2
        daily = error.groupby([chosen[k] for k in group]).mean().groupby(level=0).mean().reindex(base.index)
        sampled = np.sqrt(daily.to_numpy()[indexes].mean(axis=1))
        delta = sampled_base-sampled
        rows.append({"method": method, "daylight_balanced_rmse_mw": float(np.sqrt(daily.mean())),
            "baseline_rmse_mw": float(np.sqrt(base.mean())),
            "rmse_improvement_mw": float(np.sqrt(base.mean())-np.sqrt(daily.mean())),
            "improvement_ci_2_5_mw": float(np.quantile(delta, .025)),
            "improvement_ci_97_5_mw": float(np.quantile(delta, .975)),
            "improvement_ci_2_5_percent_rating": float(np.quantile(delta, .025)/rating*100),
            "improvement_ci_97_5_percent_rating": float(np.quantile(delta, .975)/rating*100),
            "block_days": block, "resamples": repetitions, "delivery_days": count})
    return rows


def report_farm(data: pd.DataFrame, methods: list[str], farm, section: dict, destination: Path,
                parameter_history: list[dict], *, retrospective=False) -> dict:
    """Write full-year coverage separately from the common held-out score population."""
    data = data.copy()
    dates = pd.to_datetime(data.delivery_day)
    data["month"] = dates.dt.strftime("%Y-%m")
    data["season"] = dates.dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
    local = pd.DatetimeIndex(data.interval_start_utc).tz_convert(section["delivery_timezone"])
    data["delivery_hour"] = local.hour
    data["lead_bin"] = pd.cut(data.lead_hours, [0,12,24,36,48,np.inf], right=False).astype(str)
    data["irradiance_bin"] = pd.cut(data.ghi_w_m2, [-np.inf,100,300,600,900,np.inf]).astype(str)
    data["temperature_bin"] = pd.cut(data.temperature_c, [-np.inf,0,10,20,30,np.inf]).astype(str)
    data["wind_bin"] = pd.cut(data.wind_speed_m_s, [-np.inf,2,5,10,np.inf]).astype(str)
    prefix = "retrospective_" if retrospective else ""
    scoring = data.loc[data.eligible & ((data.phase == "held_out") if not retrospective else True)]
    tables = {}
    # Process each `(label, keys)` in this collection.
    for label, keys in {"overall": [], "by_auction": ["auction"],
        "by_month": ["auction","month"], "by_season": ["auction","season"],
        "by_hour": ["auction","delivery_hour"], "by_lead": ["auction","lead_bin"],
        "by_irradiance": ["auction","irradiance_bin"],
        "by_temperature": ["auction","temperature_bin"], "by_wind": ["auction","wind_bin"]}.items():
        rows = []
        groups = scoring.groupby(keys, observed=True) if keys else [((), scoring)]
        # Write full-year coverage separately from the common held-out score population for each `(key, group)` in `groups`.
        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            # Write full-year coverage separately from the common held-out score population for each `scope` in `('all_intervals', 'daylight')`.
            for scope in ("all_intervals", "daylight"):
                view = group if scope == "all_intervals" else group.loc[group.daylight]
                # Skip the current item when `view.empty`.
                if view.empty:
                    continue
                # Write full-year coverage separately from the common held-out score population for each `method` in `methods`.
                for method in methods:
                    scores = error_metrics(view.measured_ac_mw, view[f"prediction_{method}_mw"], farm.ac_capacity_mw)
                    base = error_metrics(view.measured_ac_mw, view.baseline_mw, farm.ac_capacity_mw)
                    scores["rmse_improvement_percent"] = 100*(1-scores["rmse_mw"]/base["rmse_mw"]) if base["rmse_mw"] else 0.0
                    rows.append({**dict(zip(keys, key)), "method": method, "scope": scope, **scores})
        table = pd.DataFrame(rows)
        table.to_csv(destination/"csv"/f"{prefix}metrics_{label}.csv", index=False)
        tables[label] = table
    daily = []
    # Write full-year coverage separately from the common held-out score population for each `((day, auction), group)` in `scoring.groupby(['delivery_day', 'auction'])`.
    for (day, auction), group in scoring.groupby(["delivery_day", "auction"]):
        # Write full-year coverage separately from the common held-out score population for each `method` in `methods`.
        for method in methods:
            errors = group[f"prediction_{method}_mw"]-group.measured_ac_mw
            daily.append({"delivery_day": day, "auction": auction, "method": method,
                "energy_error_mwh": float(errors.sum()*.25), "mae_mw": float(abs(errors).mean()),
                "rmse_mw": float(np.sqrt((errors**2).mean())), "pairs": len(group),
                "expected_intervals": int(((data.delivery_day == day) & (data.auction == auction)).sum()),
                "measured_energy_mwh": float(group.measured_ac_mw.sum()*.25)})
    daily = pd.DataFrame(daily)
    daily.to_csv(destination/"csv"/f"{prefix}daily_energy_errors.csv", index=False)
    daily_summary = []
    # Write full-year coverage separately from the common held-out score population for each `((auction, method), group)` in `daily.groupby(['auction', 'method'])`.
    for (auction, method), group in daily.groupby(["auction", "method"]):
        errors = group.energy_error_mwh
        daily_summary.append({"auction": auction, "method": method, "days": len(group),
            "mean_daily_energy_error_mwh": float(errors.mean()),
            "mean_daily_absolute_energy_error_mwh": float(abs(errors).mean()),
            "mean_overprediction_mwh_on_overpredicted_days": float(errors[errors>0].mean()) if (errors>0).any() else 0.0,
            "mean_underprediction_mwh_on_underpredicted_days": float(-errors[errors<0].mean()) if (errors<0).any() else 0.0,
            "accumulated_energy_bias_mwh": float(errors.sum()),
            "partial_days": int((group.pairs != group.expected_intervals).sum())})
    pd.DataFrame(daily_summary).to_csv(destination/"csv"/f"{prefix}daily_energy_summary.csv", index=False)
    # Return `{'metrics': tables['overall'].to_dict('records')}` when `retrospective`.
    if retrospective:
        return {"metrics": tables["overall"].to_dict("records")}
    bootstrap = paired_bootstrap(data, methods, farm.ac_capacity_mw, section)
    pd.DataFrame(bootstrap).to_csv(destination/"csv"/"paired_block_bootstrap.csv", index=False)
    weather_rows = []
    # Process each `(auction, group)` in this collection.
    for auction, group in data.loc[np.isfinite(data.satellite_ghi_w_m2) & data.daylight].groupby("auction"):
        residual = group.ghi_w_m2-group.satellite_ghi_w_m2
        weather_rows.append({"auction": auction, "pairs": len(group),
            "ghi_mae_w_m2": float(abs(residual).mean()), "ghi_bias_w_m2": float(residual.mean()),
            "ghi_rmse_w_m2": float(np.sqrt((residual**2).mean()))})
    pd.DataFrame(weather_rows).to_csv(destination/"csv"/"weather_error_diagnostics.csv", index=False)
    coverage = data.groupby(["delivery_day", "auction"]).agg(rows=("eligible","size"),
        eligible=("eligible","sum"), daylight=("daylight","sum"),
        suspected_outages=("suspected_outage","sum"), plateaus=("plateau_flag","sum"),
        contradictory_radiation=("bhi_exceeds_ghi","sum")).reset_index()
    coverage.to_csv(destination/"csv"/"coverage.csv", index=False)
    # Write full-year coverage separately from the common held-out score population when `parameter_history`.
    if parameter_history:
        pd.DataFrame(parameter_history).to_csv(destination/"csv"/"parameter_history.csv", index=False)
    _plots(data, methods, farm, section, destination, tables, daily, parameter_history,
           retrospective=retrospective)
    summary = {"farm_id": farm.farm_id, "ac_rating_mw": farm.ac_capacity_mw,
        "methods": methods, "held_out_metrics": tables["overall"].to_dict("records"),
        "paired_daylight_comparison": bootstrap,
        "warmup_baseline": error_metrics(data.loc[data.eligible & (data.phase=="warmup"), "measured_ac_mw"],
            data.loc[data.eligible & (data.phase=="warmup"), "baseline_mw"], farm.ac_capacity_mw),
        "limitations": "Effective physical-model fits; proxy coordinates, unmeasured module temperatures, outages and weather errors prevent equipment identification. Auction energy totals overlap and must not be summed."}
    (destination/"json"/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def _plots(data, methods, farm, section, destination, tables, daily, history, *, retrospective=False):
    """Prefer per-auction panels and heatmaps to crowded overlapping method curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_root = destination/"plots"
    # When `retrospective`, set `plot_root` using `plot_root / 'retrospective'`.
    if retrospective:
        plot_root = plot_root/"retrospective"
    plot_root.mkdir(parents=True, exist_ok=True)
    evaluation = "retrospective in-sample" if retrospective else "held-out"
    monthly = tables["by_month"].loc[tables["by_month"].scope == "daylight"]
    fig, axes = plt.subplots(2,2,figsize=(14,10),layout="constrained")
    # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(axis, auction)` in `zip(axes.flat, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
    for axis, auction in zip(axes.flat, ("DA","IDA1","IDA2","IDA3")):
        pivot = monthly.loc[monthly.auction==auction].pivot(index="method",columns="month",values="nrmse_percent_ac_rating").reindex(methods)
        colored = axis.imshow(pivot.to_numpy(),aspect="auto",cmap="viridis_r")
        axis.set(yticks=np.arange(len(methods)),yticklabels=methods,xticks=np.arange(len(pivot.columns)),
                 xticklabels=[x[-2:] for x in pivot.columns],title=auction,xlabel="Delivery month")
        fig.colorbar(colored,ax=axis,label="Daylight RMSE / AC rating (%)")
    fig.suptitle(f"Farm {farm.farm_id}: {evaluation} monthly forecast accuracy")
    fig.savefig(plot_root/"monthly_method_comparison.png",dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(axis, auction)` in `zip(axes.flat, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
    for axis,auction in zip(axes.flat,("DA","IDA1","IDA2","IDA3")):
        # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `method` in `('fixed', 'selected')`.
        for method in ("fixed","selected"):
            view=daily.loc[(daily.auction==auction)&(daily.method==method)]
            axis.plot(pd.to_datetime(view.delivery_day),view.energy_error_mwh,label=method,linewidth=.9)
        axis.axhline(0,color="grey",linewidth=.5)
        axis.set(title=auction,ylabel="Forecast − measured (MWh)")
        axis.legend();axis.grid(alpha=.2)
    fig.suptitle(f"Farm {farm.farm_id}: {evaluation} daily energy forecast bias")
    fig.savefig(plot_root/"daily_energy_bias.png",dpi=140);plt.close(fig)
    plot_daily_energy_distribution(daily, farm.farm_id,
        plot_root/"daily_energy_error_distribution.png", evaluation=evaluation)
    # When `history`, set `parameters` using `pd.DataFrame(history)`.
    if history:
        parameters=pd.DataFrame(history)
        fig,axes=plt.subplots(3,2,figsize=(13,10),layout="constrained")
        # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(axis, name)` in `zip(axes.flat, ('coefficient', 'u0', 'u1', 'gcr', 'max_angle'))`.
        for axis,name in zip(axes.flat,("coefficient","u0","u1","gcr","max_angle")):
            # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(method, group)` in `parameters.groupby('method')`.
            for method,group in parameters.groupby("method"):
                axis.plot(pd.to_datetime(group.issue_utc),group[name],marker=".",label=method,linewidth=.8)
            axis.set(title=name);axis.grid(alpha=.2)
        axes.flat[-1].axis("off")
        handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles,labels,loc="center",fontsize=8)
        fig.suptitle(f"Farm {farm.farm_id}: monthly fitted parameters")
        fig.savefig(plot_root/"parameter_history.png",dpi=140);plt.close(fig)
    # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(season, group)` in `data.loc[data.phase == 'held_out'].groupby('season')`.
    for season,group in data.loc[data.phase=="held_out"].groupby("season"):
        energies=group.loc[(group.auction=="DA")&group.eligible].groupby("delivery_day").measured_ac_mw.sum()
        # Skip the current item when `energies.empty`.
        if energies.empty:
            continue
        ordered=(energies-energies.median()).abs().sort_values().index[:int(section["representative_days_per_season"])]
        # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `day` in `ordered`.
        for day in ordered:
            fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout="constrained")
            # Prefer per-auction panels and heatmaps to crowded overlapping method curves for each `(axis, auction)` in `zip(axes, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
            for axis,auction in zip(axes,("DA","IDA1","IDA2","IDA3")):
                view=group.loc[(group.delivery_day==day)&(group.auction==auction)]
                times=pd.DatetimeIndex(view.interval_start_utc).tz_convert(section["delivery_timezone"])
                # Process each `(column, label)` in this collection.
                for column,label in (("measured_ac_mw","Measured AC"),("prediction_fixed_mw","Fixed forecast"),("prediction_selected_mw","Selected physical forecast")):
                    # Extend each interval-mean curve to its last boundary.
                    axis.step(times.append(pd.DatetimeIndex([times[-1]+pd.Timedelta(minutes=15)])),
                        np.r_[view[column].to_numpy(),view[column].iloc[-1]],where="post",label=label,linewidth=1)
                axis.set(title=auction,ylabel="MW",ylim=(0,farm.ac_capacity_mw));axis.legend(fontsize=8)
            fig.suptitle(f"Farm {farm.farm_id} — {day} ({season})")
            fig.savefig(plot_root/f"forecast_{day}.png",dpi=140);plt.close(fig)


def plot_daily_energy_distribution(daily: pd.DataFrame, farm_id: int, output: Path, *, evaluation="held-out"):
    """Regenerate paired daily bias distributions from a saved CSV without fitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    # Regenerate paired daily bias distributions from a saved CSV without fitting for each `(axis, auction)` in `zip(axes.flat, ('DA', 'IDA1', 'IDA2', 'IDA3'))`.
    for axis,auction in zip(axes.flat,("DA","IDA1","IDA2","IDA3")):
        values=[daily.loc[(daily.auction==auction)&(daily.method==name),"energy_error_mwh"].to_numpy()
                for name in ("fixed","selected")]
        axis.boxplot(values,tick_labels=["Fixed","Selected"],showmeans=True)
        axis.axhline(0,color="grey",linewidth=.5)
        axis.set(title=auction,ylabel="Daily forecast − measured (MWh)")
    fig.suptitle(f"Farm {farm_id}: {evaluation} daily energy error distributions")
    fig.savefig(output,dpi=140)
    plt.close(fig)


def regenerate_saved_plots(study_root: str | Path):
    """Refresh figures from saved predictions/metrics, preserving fit artifacts."""
    from ...data.contracts import FarmMetadata
    root = Path(study_root)
    config = json.loads((root/"json"/"study_config.json").read_text())
    section = config["pv_calibration"]
    # Refresh figures from saved predictions/metrics, preserving fit artifacts for each `farm_id` in `section['farm_ids']`.
    for farm_id in section["farm_ids"]:
        destination = root/f"farm_{farm_id}"
        summary = json.loads((destination/"json"/"summary.json").read_text())
        source = json.loads((destination/"json"/"input_provenance.json").read_text())
        farm = FarmMetadata(**source["farm_metadata"])
        # Refresh figures from saved predictions/metrics, preserving fit artifacts for each `retrospective` in `(False, True)`.
        for retrospective in (False, True):
            prefix = "retrospective_" if retrospective else ""
            filename = "retrospective_predictions.parquet" if retrospective else "predictions.parquet"
            path = destination/"parquet"/filename
            # Skip the current item when `not path.exists()`.
            if not path.exists():
                continue
            data = pd.read_parquet(path)
            data["season"] = pd.to_datetime(data.delivery_day).dt.month.map({
                12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"})
            tables = {"by_month": pd.read_csv(destination/"csv"/f"{prefix}metrics_by_month.csv")}
            daily = pd.read_csv(destination/"csv"/f"{prefix}daily_energy_errors.csv")
            history = [] if retrospective else pd.read_csv(destination/"csv"/"parameter_history.csv").to_dict("records")
            _plots(data, summary["methods"], farm, section, destination, tables, daily, history,
                   retrospective=retrospective)
