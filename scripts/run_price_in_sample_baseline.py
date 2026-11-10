"""Fit a small, explicitly in-sample 2026 auction-price baseline from pinned arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep this standalone diagnostic within one CPU thread and away from GPUs.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_models import make_model, point_metrics
from bess_pv_trading.forecasting.studies.price_holiday_features import build_holiday_features


def load_arrays(root, group):
    """Validate the exact previously prepared input arrays before reusing them."""
    arrays = {}
    for name in ("x", "y", "anchor", "reference", "starts", "decisions", "days"):
        descriptor = group["files"][name]
        path = root / descriptor["path"]
        if digest(path) != descriptor["sha256"]:
            raise ValueError(f"Input array checksum changed: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays


def fit_group(root, destination, name, group, settings, calendar, holiday):
    """Fit one point-price model and save only its observed, usable 2026 rows."""
    arrays = load_arrays(root, group)
    starts = pd.to_datetime(arrays["starts"], utc=True)
    calendar_values, calendar_names = build_holiday_features(starts, calendar, holiday)
    base = np.asarray(arrays["x"][:, :group["base_width"]])
    features = np.column_stack((base, calendar_values))
    actual = np.asarray(arrays["y"])
    anchor = np.asarray(arrays["anchor"])
    reference = np.asarray(arrays["reference"])
    days = arrays["days"]
    complete = np.isfinite(features).all(axis=1) & np.isfinite(actual) & np.isfinite(anchor) & np.isfinite(reference)
    training = complete & (days >= settings["train_start"]) & (days < settings["train_end_exclusive"])
    evaluation = complete & (days >= settings["prediction_start"]) & (days < settings["prediction_end_exclusive"])
    if not training.any() or not evaluation.any():
        raise ValueError(f"No complete training or prediction rows for {name}")
    model = make_model(settings["model_family"], settings["model_parameters"], settings["seed"])
    with threadpool_limits(limits=1):
        model.fit(features[training], actual[training] - anchor[training])
        prediction = model.predict(features[evaluation]) + anchor[evaluation]
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / f"{name}.joblib"
    joblib.dump(model, model_path, compress=3)
    frame = pd.DataFrame({"auction": group["market"], "product_start": starts[evaluation],
        "decision_cutoff": pd.to_datetime(arrays["decisions"][evaluation], utc=True),
        "actual_eur_per_mwh": actual[evaluation], "forecast_eur_per_mwh": prediction,
        "reference_eur_per_mwh": reference[evaluation], "local_day": days[evaluation],
        "provenance": "in_sample_fit_uses_future_labels_relative_to_decision"})
    path = destination / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    score = point_metrics(frame.actual_eur_per_mwh, frame.forecast_eur_per_mwh, frame.reference_eur_per_mwh)
    return {"group": name, "auction": group["market"], "training_rows": int(training.sum()),
        "training_first_day": str(sorted(days[training])[0]), "training_last_day": str(sorted(days[training])[-1]),
        "scored_first_day": str(sorted(days[evaluation])[0]), "scored_last_day": str(sorted(days[evaluation])[-1]),
        "observed_target_rows_in_requested_period": int(((days >= settings["prediction_start"]) &
            (days < settings["prediction_end_exclusive"])).sum()),
        "scores_in_sample": score, "feature_names": group["feature_names"][:group["base_width"]] + calendar_names,
        "predictions": str(path), "predictions_sha256": digest(path),
        "model": str(model_path), "model_sha256": digest(model_path)}


def plot_group(path, destination, timezone):
    """Plot monthly in-sample errors and two observed versus predicted days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(path)
    frame["error"] = (frame["forecast_eur_per_mwh"] - frame["actual_eur_per_mwh"]).abs()
    frame["month"] = frame["local_day"].str.slice(0, 7)
    monthly = frame.groupby("month")["error"].mean()
    destination.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4))
    monthly.plot.bar(ax=axis)
    axis.set(ylabel="MAE (EUR/MWh)", title="Monthly in-sample error", xlabel="Delivery month")
    fig.tight_layout()
    fig.savefig(destination / "monthly_mae.png", dpi=130)
    plt.close(fig)
    daily = frame.groupby("local_day")["error"].mean().sort_values()
    for label, day in (("best", daily.index[0]), ("worst", daily.index[-1])):
        selected = frame.loc[frame["local_day"] == day]
        fig, axis = plt.subplots(figsize=(10, 4))
        local = selected["product_start"].dt.tz_convert(timezone)
        axis.plot(local, selected["actual_eur_per_mwh"], label="Actual")
        axis.plot(local, selected["forecast_eur_per_mwh"], label="In-sample prediction")
        axis.set(title=f"{label.title()} observed day: {day}", ylabel="EUR/MWh", xlabel=timezone)
        axis.legend()
        fig.tight_layout()
        fig.savefig(destination / f"{label}_{day}.png", dpi=130)
        plt.close(fig)


def main():
    """Require --run and retain separate outputs without editing the simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_in_sample_baseline_2026.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["price_in_sample_baseline"]
    plan = {"status": "plan_only", "settings": settings, "fits": len(settings["group_names"]),
        "cpu_workers": 1, "in_sample": True, "simulator_modified": False}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    root = Path(settings["source_search"])
    output = Path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    prepared_path = root / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    calendar = prepared["scientific_settings"]
    holiday = load_config(settings["holiday_config"])["price_holiday_expansion"]
    results = []
    for name in settings["group_names"]:
        print(f"Fitting {name} on available 2025–2026 rows", flush=True)
        result = fit_group(root, output, name, prepared["groups"][name], settings, calendar, holiday)
        plot_group(result["predictions"], output / "plots" / name, calendar["timezone"])
        results.append(result)
    manifest = {**plan, "status": "complete", "source_prepared_sha256": digest(prepared_path),
        "source_selection_sha256": prepared["source_selection_sha256"],
        "interpretation": "In-sample diagnostic; IDA predictions use actual earlier cleared auction prices.",
        "results": results}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: result[key] for key in ("auction", "training_rows", "scored_first_day",
        "scored_last_day", "scores_in_sample")} for result in results], indent=2))


if __name__ == "__main__":
    main()
