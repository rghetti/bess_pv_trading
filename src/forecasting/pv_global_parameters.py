"""Export full-history physical fits as explicitly assumed static farm metadata."""
from __future__ import annotations

import json
from pathlib import Path

from .pv_artifacts import read_json


def export_global_parameters(study_root: str | Path, destination: str | Path) -> dict:
    """Select a family on held-out 2025 scores and export its final all-year refit.

    This export is intentionally distinct from issue-time artifacts: the user
    treats the inferred equipment parameters as prior farm knowledge at every
    experiment date. Its global fit scores are therefore explicitly in-sample.
    """
    root, output = Path(study_root), Path(destination)
    output.mkdir(parents=True,exist_ok=True)
    config=read_json(root/"json"/"study_config.json")
    overrides={}
    for farm_id in config["pv_calibration"]["farm_ids"]:
        farm=root/f"farm_{farm_id}"
        summary=read_json(farm/"json"/"summary.json")
        candidates=[row for row in summary["paired_daylight_comparison"] if row["method"]!="selected"]
        best=min(candidates,key=lambda row:row["daylight_balanced_rmse_mw"])
        method=best["method"]
        checkpoint=read_json(farm/"json"/"fits"/"retrospective_full_year"/f"{method}.json")
        fit=checkpoint["fit"]
        source=read_json(farm/"json"/"input_provenance.json")
        artifact_path=output/f"farm_{farm_id}.json"
        artifact={"schema_version":1,"farm_id":farm_id,"farm_metadata":source["farm_metadata"],
            "treatment":"static_farm_knowledge_inferred_from_2025_applied_to_all_experiment_dates",
            "selection":"best_family_on_February_December_2025_balanced_daylight_RMSE",
            "training_start_day":config["pv_calibration"]["start_day"],
            "training_end_day_exclusive":config["pv_calibration"]["end_day_exclusive"],
            "fit":fit,"held_out_family_metrics":best,
            "global_in_sample_metrics":[row for row in summary["retrospective_in_sample"]["metrics"] if row["method"]==method],
            "source_fingerprint":source["source_fingerprint"],
            "study_configuration_fingerprint":checkpoint["configuration_fingerprint"],
            "interpretation":"Full-year global fits are assumed plant metadata, not causally trained early-2025 forecasts or identified hardware measurements."}
        artifact_path.write_text(json.dumps(artifact,indent=2)+"\n")
        parameters=fit["parameters"]
        override={"direct_exceeds_ghi":source["settings"]["direct_exceeds_ghi"],
            "transposition_model":fit["transposition_model"],"albedo":fit["albedo"],
            "temperature":{"u0":parameters["u0"],"u1":parameters["u1"]},
            "bifacial":fit["bifacial_settings"],
            "output":{"method":"calibrated_scale","calibrated_coefficient":parameters["coefficient"],
                "assumption_status":f"global_2025_fit_treated_as_prior_farm_metadata; source={artifact_path}"}}
        if source["farm_metadata"]["tracking_type_id"] in {1,3}:
            override["tracker"]={"gcr":parameters["gcr"],"max_angle":parameters["max_angle"]}
        overrides[str(farm_id)]={"pv":override}
    (output/"global_parameters.json").write_text(json.dumps({"farms":overrides},indent=2)+"\n")
    return overrides


def compare_global_errors(study_root: str | Path, parameters_dir: str | Path) -> dict:
    """Compare old and promoted forecasts on identical raw targets, without refitting.

    Full-year global fits and chronological held-out family predictions are
    reported separately. Pooled means weight each day/auction view equally;
    per-auction rows avoid confusing overlapping delivery energy totals.
    """
    import numpy as np
    import pandas as pd
    from .pv_fitting import balanced_weights

    root, parameters = Path(study_root), Path(parameters_dir)
    config = read_json(root/"json"/"study_config.json")
    rows = []
    for farm_id in config["pv_calibration"]["farm_ids"]:
        artifact = read_json(parameters/f"farm_{farm_id}.json")
        method = artifact["fit"]["method"]
        rating = artifact["farm_metadata"]["ac_capacity_kw"] / 1000
        for phase, filename in (("all_year_static_in_sample", "retrospective_predictions.parquet"),
                                ("chronological_held_out_family", "predictions.parquet")):
            data = pd.read_parquet(root/f"farm_{farm_id}"/"parquet"/filename)
            data = data.loc[data.eligible & ((data.phase == "held_out") if phase.startswith("chronological") else True)]
            for scope in ("all_intervals", "daylight"):
                view = data.loc[data.daylight] if scope == "daylight" else data
                for auction in ("ALL_BALANCED", "DA", "IDA1", "IDA2", "IDA3"):
                    sample = view if auction == "ALL_BALANCED" else view.loc[view.auction == auction]
                    if sample.empty:
                        continue
                    weights = balanced_weights(sample)
                    row = {"farm_id": farm_id, "method": method, "comparison": phase, "scope": scope,
                           "auction": auction, "ac_rating_mw": rating, "pairs": len(sample),
                           "delivery_days": sample.delivery_day.nunique()}
                    for label, column in (("before", "prediction_fixed_mw"), ("after", f"prediction_{method}_mw")):
                        error = (sample[column] - sample.measured_ac_mw).to_numpy()
                        if not np.isfinite(error).all():
                            raise ValueError("Before/after errors must use the same finite evaluation mask")
                        metrics = {"mean_abs_error": np.dot(weights, np.abs(error)),
                                   "min_abs_error": np.min(np.abs(error)), "max_abs_error": np.max(np.abs(error)),
                                   "rmse": np.sqrt(np.dot(weights, error ** 2)), "signed_bias": np.dot(weights, error)}
                        for name, value in metrics.items():
                            row[f"{name}_{label}_mw"] = float(value)
                            row[f"{name}_{label}_percent_ac"] = float(value / rating * 100)
                    row["mae_improvement_percent"] = 100 * (1 - row["mean_abs_error_after_mw"] / row["mean_abs_error_before_mw"])
                    row["rmse_improvement_percent"] = 100 * (1 - row["rmse_after_mw"] / row["rmse_before_mw"])
                    rows.append(row)
    table = pd.DataFrame(rows)
    csv = root/"csv"/"before_after_global_parameters.csv"
    table.to_csv(csv, index=False)
    summary = {"mean": "equal aggregate weight per daylight/day or full-day auction view",
               "minimum_and_maximum": "minimum/maximum absolute quarter-hour AC power error on the shared mask",
               "before": "original fixed physical parameters and output coefficient",
               "after_static": "final all-2025 family refit installed as prior farm metadata; in-sample",
               "after_chronological": "monthly historical fits of that family; February-December held out",
               "energy_warning": "Auction forecast horizons overlap; do not sum their energy totals",
               "csv": str(csv), "comparisons": rows}
    (root/"json"/"before_after_global_parameters.json").write_text(json.dumps(summary, indent=2)+"\n")
    _plot_global_errors(table, root/"plots"/"before_after_global_parameters.png")
    return summary


def _plot_global_errors(table, output: Path):
    """Show nameplate-normalized mean/minimum/maximum errors for each farm."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = table.loc[(table.comparison == "all_year_static_in_sample")
                       & (table.scope == "daylight") & (table.auction == "ALL_BALANCED")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    names = ("mean_abs_error", "min_abs_error", "max_abs_error")
    for axis, row in zip(axes.flat, chosen.itertuples(index=False)):
        for label, offset, color in (("before", -.18, "grey"), ("after", .18, "tab:blue")):
            values = [getattr(row, f"{name}_{label}_percent_ac") for name in names]
            bars = axis.bar(np.arange(len(names))+offset, values, width=.36, color=color,
                            label="Original parameters" if label == "before" else "Final global parameters")
            axis.bar_label(bars, labels=[f"{getattr(row, f'{name}_{label}_mw'):.3f} MW" for name in names],
                           padding=3, fontsize=8)
        axis.set(title=f"Farm {row.farm_id} ({row.ac_rating_mw:g} MW AC)",
                 xticks=np.arange(len(names)), xticklabels=["Mean absolute", "Minimum absolute", "Maximum absolute"],
                 ylabel="Error / installed AC rating (%)")
        axis.margins(y=.18)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("2025 daylight errors: original vs final global PV model\n"
                 "Same measured AC targets; final global fit is in-sample", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)
