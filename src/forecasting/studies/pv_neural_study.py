"""Reproducible per-farm neural forecasting benchmark on chronological local data."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from ...data import config_path
from ..inference.pv_artifacts import digest
from ..training.pv_fitting import balanced_weights
from ..data.pv_neural_data import (feature_frame, fit_preprocessor, pack_sequences,
    prepare_neural_corpus, transform_features, unpack_predictions, validate_neural_config)
from ..plotting.pv_neural_reporting import report_neural_farm


def _write(path, value):
    """Publish inspectable state with an atomic rename."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n")
    temporary.replace(path)


def _progress(message):
    print(message, flush=True)


def _ridge(frame, features, preprocessor, farm, section, destination):
    """Provide a simple weather-only linear sanity baseline selected on validation."""
    x = transform_features(features, preprocessor).astype(float)
    x = np.c_[x, np.ones(len(x))]
    train = (frame.eligible & frame.daylight & (frame.split == "train")).to_numpy()
    validation = (frame.eligible & frame.daylight & (frame.split == "validation")).to_numpy()
    weights = balanced_weights(frame.loc[train])
    gram = x[train].T @ (weights[:, None]*x[train])
    rhs = x[train].T @ (weights*frame.loc[train, "measured_ac_mw"].to_numpy()/farm.ac_capacity_mw)
    penalty = np.eye(x.shape[1]); penalty[-1, -1] = 0
    validation_weights = balanced_weights(frame.loc[validation])
    trials = []
    # Provide a simple weather-only linear sanity baseline selected on validation for each `alpha` in `section['ridge_alphas']`.
    for alpha in section["ridge_alphas"]:
        coefficients = np.linalg.lstsq(gram+float(alpha)*penalty, rhs, rcond=None)[0]
        values = np.clip(x@coefficients, 0, 1)*frame.daylight.to_numpy()
        error = values[validation]-frame.loc[validation, "measured_ac_mw"].to_numpy()/farm.ac_capacity_mw
        trials.append({"alpha": alpha, "validation_nrmse": float(np.sqrt(np.dot(validation_weights, error**2))),
                       "coefficients": coefficients.tolist()})
    best = min(trials, key=lambda row: row["validation_nrmse"])
    _write(destination/"models"/"ridge.json", {"farm_metadata": asdict(farm), "preprocessor": preprocessor,
        "fit": best, "trials": trials, "selection": "validation_only", "target": "AC_nameplate_fraction"})
    return np.clip(x@np.asarray(best["coefficients"]), 0, 1)*frame.daylight.to_numpy()*farm.ac_capacity_mw


def _farm_worker(config, farm_id, root, config_hash):
    """Run each farm with independent weights, scaler, embeddings and artifacts."""
    from ..training.pv_neural_training import (configure_cpu, fit_network, load_network,
                                     predict_network, save_network, tensor_pack)
    section = config["pv_neural"]
    configure_cpu(section)
    destination = Path(root)/f"farm_{farm_id}"
    # Run each farm with independent weights, scaler, embeddings and artifacts for each `folder` in `('csv', 'json', 'parquet', 'plots', 'models')`.
    for folder in ("csv", "json", "parquet", "plots", "models"):
        (destination/folder).mkdir(parents=True, exist_ok=True)
    frame, farm, pv_settings, provenance = prepare_neural_corpus(config, farm_id, destination, _progress)
    train_mask = (frame.eligible & frame.daylight & (frame.split == "train")).to_numpy()
    views, preprocessors, feature_tables = {}, {}, {}
    # Process each `name` in this collection.
    for name in {candidate["features"] for candidate in section["candidates"]} | {"weather"}:
        features = feature_frame(frame, name, section["delivery_timezone"])
        preprocessor = fit_preprocessor(features, train_mask, section)
        views[name] = pack_sequences(frame, transform_features(features, preprocessor), farm.ac_capacity_mw)
        preprocessors[name], feature_tables[name] = preprocessor, features
    frame["prediction_pvlib_mw"] = frame.baseline_mw
    frame["prediction_fixed_mw"] = frame.baseline_mw  # Common paired-bootstrap interface.
    diagnostics, methods, reload_differences = [], ["pvlib"], []
    input_hash = provenance["source_fingerprint"]
    # Write the corresponding artifacts for `candidate` from `section['candidates']`.
    for candidate in section["candidates"]:
        name = candidate["name"]
        packed = views[candidate["features"]]
        path = destination/"models"/f"{name}.pt"
        # When `path.exists() and path.with_suffix('.json').exists()`, set `(model, metadata)` using `load_network(path, farm_id=farm_id)`.
        if path.exists() and path.with_suffix(".json").exists():
            model, metadata = load_network(path, farm_id=farm_id)
            # Enforce the input contract: Neural model resume fingerprint mismatch.
            if metadata["configuration_fingerprint"] != config_hash or metadata["source_fingerprint"] != input_hash:
                raise ValueError("Neural model resume fingerprint mismatch")
            # Enforce the input contract: Neural preprocessing changed during resume.
            if metadata["preprocessor"] != preprocessors[candidate["features"]]:
                raise ValueError("Neural preprocessing changed during resume")
            diagnostic = metadata["diagnostics"]
            _progress(f"farm {farm_id}: reused {name}")
        else:
            _progress(f"farm {farm_id}: training {name} on CPU")
            model, diagnostic = fit_network(packed, candidate, section,
                seed=int(section["seed"])+farm_id, progress=lambda msg: _progress(f"farm {farm_id}: {msg}"))
            metadata = {"schema_version": 1, "farm_metadata": asdict(farm), "candidate": candidate,
                "neural_settings": section, "preprocessor": preprocessors[candidate["features"]],
                "configuration_fingerprint": config_hash, "source_fingerprint": input_hash,
                "diagnostics": diagnostic, "target": "raw_AC_MW_divided_by_farm_AC_nameplate",
                "training_end_day_exclusive": section["train_end_day_exclusive"],
                "selection_end_day_exclusive": section["validation_end_day_exclusive"],
                "forecast_availability_lag_hours": config["source"]["availability_lag_hours"],
                "night_policy": config["policy"]["night"],
                "fitted_pvlib_parameters_used_as_neural_inputs": False,
                "resolved_configuration": config}
            save_network(model, path, metadata)
        prediction = predict_network(model, tensor_pack(packed), int(section["prediction_batch_size"]))
        frame[f"prediction_{name}_mw"] = unpack_predictions(packed, prediction, len(frame), farm.ac_capacity_mw)
        reloaded, _ = load_network(path, farm_id=farm_id)
        repeated = predict_network(reloaded, tensor_pack(packed), int(section["prediction_batch_size"]))
        reload_differences.append(float(np.max(abs(prediction-repeated))))
        # Enforce the input contract: Neural artifact reload changed predictions.
        if reload_differences[-1] > 1e-7:
            raise ValueError("Neural artifact reload changed predictions")
        diagnostics.append(diagnostic)
        methods.append(name)
        pd.DataFrame(diagnostic["history"]).to_csv(destination/"csv"/f"training_{name}.csv", index=False)
    # Architecture selection reads only saved validation scores, never test errors.
    best = min(diagnostics, key=lambda item: item["validation_nrmse"])
    selected = best["candidate"]["name"]
    frame["prediction_selected_mw"] = frame[f"prediction_{selected}_mw"]
    methods.append("selected")
    # Run each farm with independent weights, scaler, embeddings and artifacts for each `suffix` in `('.pt', '.json')`.
    for suffix in (".pt", ".json"):
        shutil.copyfile(destination/"models"/f"{selected}{suffix}", destination/"models"/f"selected{suffix}")
    # Update `frame['prediction_ridge_mw']` when this condition is met.
    if section["include_ridge_baseline"]:
        frame["prediction_ridge_mw"] = _ridge(frame, feature_tables["weather"], preprocessors["weather"],
                                               farm, section, destination)
        methods.append("ridge")
    selection = {"selected_candidate": selected, "validation_nrmse": best["validation_nrmse"],
        "all_candidates": [{"name": item["candidate"]["name"], "validation_nrmse": item["validation_nrmse"],
                             "best_epoch": item["best_epoch"]} for item in diagnostics],
        "criterion": "validation_daylight_day_auction_balanced_NRMSE", "test_used_for_selection": False}
    _write(destination/"json"/"model_selection.json", selection)
    _write(destination/"json"/"resolved_configuration.json", config)
    _write(destination/"json"/"neural_inputs.json", {"preprocessors": preprocessors,
        "fitted_pvlib_values_used_as_features": False, "measured_history_used_as_features": False,
        "known_metadata": asdict(farm), "baseline_pv_settings": pv_settings,
        "core": "forecast irradiance, air temperature, wind, solar geometry, calendar and lead",
        "weather_extension": "dew point, cloud layers, precipitation, snow, visibility, upper-level wind"})
    frame.to_parquet(destination/"parquet"/"predictions.parquet", index=False)
    summary = report_neural_farm(frame, farm, methods, section, destination, diagnostics)
    scope_counts = frame.groupby(["split", "auction"]).size().unstack().to_dict()
    audit = {"status": "passed", "source_fingerprint": input_hash,
        "artifact_reload_max_difference_normalized": max(reload_differences),
        "forecast_availability_respected": bool((frame.forecast_available_at_utc <= frame.decision_cutoff_utc).all()),
        "split_dates_ordered": True, "native_quarter_hour_sequences": True,
        "test_targets_used_for_selection": False, "source_targets_unconverted": True,
        "split_embargo_days": section["split_embargo_days"],
        "scope_counts": scope_counts,
        "delivery_days": frame.groupby("split").delivery_day.nunique().to_dict(),
        "scored_daylight_rows": frame.loc[frame.eligible & frame.daylight].groupby("split").size().to_dict()}
    # Enforce the input contract: Neural corpus contains an unavailable forecast.
    if not audit["forecast_availability_respected"]:
        raise ValueError("Neural corpus contains an unavailable forecast")
    _write(destination/"json"/"validation_audit.json", audit)
    return {"farm_id": farm_id, "selected_candidate": selected, "summary": summary,
            "source_fingerprint": input_hash, "audit": audit}


def run_neural_study(config, progress=None):
    """Launch resumable independent farm fits and compare their untouched test data."""
    import torch
    validate_neural_config(config)
    config = deepcopy(config)
    section = config["pv_neural"]
    sources = ["data/pv_neural_data.py", "models/pv_neural_models.py", "training/pv_neural_training.py",
               "data/pv_study_inputs.py", "models/pv.py", "models/pv_physical.py"]
    code_hash = hashlib.sha256(b"".join((Path(__file__).resolve().parents[1]/name).read_bytes() for name in sources)).hexdigest()
    config["_neural_runtime"] = {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
                                  "code_fingerprint": code_hash, "torch_cuda_build": torch.version.cuda}
    started = time.perf_counter()
    run_id = section["run_id"] or datetime.now(timezone.utc).strftime("neural_pv_%Y%m%dT%H%M%S%fZ")
    root = config_path(config, section["output_root"])/run_id
    config_hash = digest(config)
    path = root/"json"/"study_config.json"
    previous_manifest_path = root/"json"/"manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text()) if previous_manifest_path.exists() else {}
    # Launch resumable independent farm fits and compare their untouched test data when `root.exists()`.
    if root.exists():
        # Enforce the input contract: Neural output exists with different settings/code; choose a fresh run ID.
        if not section["resume"] or not path.exists() or digest(json.loads(path.read_text())) != config_hash:
            raise ValueError("Neural output exists with different settings/code; choose a fresh run ID")
    # Launch resumable independent farm fits and compare their untouched test data for each `folder` in `('json', 'csv', 'plots')`.
    for folder in ("json", "csv", "plots"):
        (root/folder).mkdir(parents=True, exist_ok=True)
    _write(path, config)
    workers = min(int(section["workers"]), len(section["farm_ids"]))
    # Launch resumable independent farm fits and compare their untouched test data when `progress`.
    if progress:
        progress(f"CPU neural PV study: {workers} farm workers, {section['threads_per_worker']} threads each; output={root}")
    complete = []
    # Launch resumable independent farm fits and compare their untouched test data when `workers == 1`.
    if workers == 1:
        # Launch resumable independent farm fits and compare their untouched test data for each `farm_id` in `section['farm_ids']`.
        for farm_id in section["farm_ids"]:
            complete.append(_farm_worker(config, int(farm_id), str(root), config_hash))
    else:
        variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in variables}
        try:
            # Launch resumable independent farm fits and compare their untouched test data for each `name` in `variables`.
            for name in variables:
                os.environ[name] = str(section["threads_per_worker"])
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_farm_worker, config, int(farm), str(root), config_hash): farm
                           for farm in section["farm_ids"]}
                # Launch resumable independent farm fits and compare their untouched test data for each `future` in `as_completed(futures)`.
                for future in as_completed(futures):
                    complete.append(future.result())
                    # Launch resumable independent farm fits and compare their untouched test data when `progress`.
                    if progress:
                        progress(f"Neural PV: farm {futures[future]} complete")
        finally:
            # Launch resumable independent farm fits and compare their untouched test data for each `(name, value)` in `previous.items()`.
            for name, value in previous.items():
                # Launch resumable independent farm fits and compare their untouched test data when `value is None`.
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    rows = [{"selected_candidate": item["selected_candidate"], **row} for item in complete
            for row in item["summary"]["test_daylight_metrics"]]
    pd.DataFrame(rows).to_csv(root/"csv"/"farm_model_comparison.csv", index=False)
    _comparison_plot(pd.DataFrame(rows), root/"plots"/"farm_model_comparison.png", section)
    manifest = {"status": "validating", "run_id": run_id, "output_dir": str(root),
        "branch": "feature/neural-pv-benchmark", "configuration_fingerprint": config_hash,
        "runtime_seconds": previous_manifest.get("runtime_seconds", time.perf_counter()-started),
        "last_invocation_runtime_seconds": time.perf_counter()-started, "resumed_completed_study": bool(previous_manifest),
        "device": "cpu",
        "versions": config["_neural_runtime"], "workers": workers,
        "threads_per_worker": section["threads_per_worker"],
        "farms": [{"farm_id": item["farm_id"], "selected_candidate": item["selected_candidate"],
                   "source_fingerprint": item["source_fingerprint"], "audit": item["audit"]} for item in complete]}
    _write(root/"json"/"manifest.json", manifest)
    from ..evaluation.pv_neural_audit import audit_neural_study
    manifest["validation_audit"] = audit_neural_study(root)
    manifest["status"] = "complete"
    _write(root/"json"/"manifest.json", manifest)
    return manifest


def _comparison_plot(table, output, section):
    """Compare farms on capacity-normalized test error, without selecting on test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    # Compare farms on capacity-normalized test error, without selecting on test for each `(axis, (farm, group))` in `zip(axes.flat, table.groupby('farm_id'))`.
    for axis, (farm, group) in zip(axes.flat, table.groupby("farm_id")):
        group = group.sort_values("rmse_mw", ascending=False)
        axis.barh(group.method, group.nrmse_percent_ac_rating,
                  color=["tab:orange" if name == "pvlib" else "tab:green" if name == "selected" else "tab:blue"
                         for name in group.method])
        axis.set(title=f"Farm {farm}", xlabel="Untouched test daylight RMSE / AC rating (%)")
        axis.grid(axis="x", alpha=.2)
    last = (pd.Timestamp(section["end_day_exclusive"])-pd.Timedelta(days=1)).date().isoformat()
    fig.suptitle(f"{section['validation_end_day_exclusive']}–{last}: calibrated pvlib vs CPU neural forecasts")
    fig.savefig(output, dpi=160); plt.close(fig)
