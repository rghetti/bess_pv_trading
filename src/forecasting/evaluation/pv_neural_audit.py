"""Independent saved-output checks for completed neural forecast benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..training.pv_fitting import balanced_weights
from ..inference.pv_neural_predict import predict_neural_frame
from ..training.pv_neural_training import load_network


def audit_neural_study(study_root: str | Path) -> dict:
    """Check every farm/model and replay saved selected artifacts without targets."""
    root = Path(study_root)
    config = json.loads((root/"json"/"study_config.json").read_text())
    section = config["pv_neural"]
    import torch
    torch.set_num_threads(int(section["threads_per_worker"]))
    expected_names = {candidate["name"] for candidate in section["candidates"]}
    reports = []
    for farm_id in section["farm_ids"]:
        farm = root/f"farm_{farm_id}"
        frame = pd.read_parquet(farm/"parquet"/"predictions.parquet")
        corpus = pd.read_parquet(farm/"parquet"/"corpus.parquet")
        np.testing.assert_array_equal(frame.measured_ac_mw, corpus.measured_ac_mw)
        np.testing.assert_array_equal(frame.prediction_pvlib_mw, corpus.baseline_mw)
        pd.testing.assert_series_equal(frame.interval_start_utc, corpus.interval_start_utc)
        if not (frame.forecast_available_at_utc <= frame.decision_cutoff_utc).all():
            raise ValueError("Saved neural input violates forecast availability")
        embargo = pd.DateOffset(days=int(section["split_embargo_days"]))
        for split, boundary in (("train", "train_end_day_exclusive"), ("validation", "validation_end_day_exclusive")):
            cutoff = (pd.Timestamp(section[boundary])-embargo).tz_localize(section["delivery_timezone"]).tz_convert("UTC")
            if not (frame.loc[(frame.split == split) & frame.eligible, "target_available_at_utc"] <= cutoff).all():
                raise ValueError("Saved fit/selection targets violate split embargo")
        selection = json.loads((farm/"json"/"model_selection.json").read_text())
        candidates = {row["name"]: row for row in selection["all_candidates"]}
        if set(candidates) != expected_names:
            raise ValueError("Completed neural candidate set is incomplete")
        chosen = min(candidates, key=lambda name: candidates[name]["validation_nrmse"])
        if chosen != selection["selected_candidate"] or selection["test_used_for_selection"]:
            raise ValueError("Saved neural selection disagrees with prior validation")
        np.testing.assert_array_equal(frame.prediction_selected_mw, frame[f"prediction_{chosen}_mw"])
        validation = frame.loc[(frame.split == "validation") & frame.eligible & frame.daylight]
        count = 0
        for name in expected_names:
            model, artifact = load_network(farm/"models"/f"{name}.pt", farm_id=farm_id)
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise ValueError("Saved neural model is not on CPU")
            rating = artifact["farm_metadata"]["ac_capacity_kw"]/1000
            values = frame[f"prediction_{name}_mw"].to_numpy()
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > rating+1e-7):
                raise ValueError("Saved neural outputs violate the AC rating")
            if not np.isfinite(frame.loc[frame.eligible, "prediction_pvlib_mw"]).all():
                raise ValueError("Common baseline evaluation mask contains invalid predictions")
            residual = (validation[f"prediction_{name}_mw"]-validation.measured_ac_mw).to_numpy()/rating
            nrmse = float(np.sqrt(np.dot(balanced_weights(validation), residual**2)))
            np.testing.assert_allclose(nrmse, candidates[name]["validation_nrmse"], rtol=1e-5, atol=1e-7)
            if artifact["fitted_pvlib_parameters_used_as_neural_inputs"]:
                raise ValueError("Neural artifact includes fitted physical inputs")
            count += 1
        dates = sorted(frame.loc[frame.split == "test", "delivery_day"].unique())
        sample_days = [dates[0], dates[len(dates)//2], dates[-1]]
        sample = frame.loc[frame.delivery_day.isin(sample_days)].reset_index(drop=True)
        target_free = sample.drop(columns=["measured_ac_mw"])
        prediction = predict_neural_frame(target_free, farm/"models"/"selected.pt", farm_id=farm_id)
        difference = float(np.max(abs(prediction-sample.prediction_selected_mw.to_numpy())))
        if difference > 1e-5:
            raise ValueError("Target-free saved neural inference differs from benchmark predictions")
        reports.append({"farm_id": farm_id, "model_count": count, "forecast_rows": len(frame),
            "delivery_days": frame.delivery_day.nunique(), "selected_candidate": chosen,
            "target_free_max_difference_mw": difference, "raw_targets_unchanged": True,
            "pvlib_baseline_unchanged": True, "availability_and_embargo_checked": True,
            "common_finite_mask_and_capacity_checked": True, "validation_selection_reconciled": True,
            "test_delivery_days": len(dates), "artifact_replay_days": sample_days})
    report = {"status": "passed", "farms": reports, "checked_models": sum(row["model_count"] for row in reports)}
    (root/"json"/"validation_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
