"""CPU-parallel hyperparameter search for farm-specific PV regressors."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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

from ..data import config_path
from .pv_artifacts import digest
from .pv_fitting import balanced_weights
from .pv_neural_data import (feature_frame, fit_preprocessor, pack_sequences,
    prepare_neural_corpus, transform_features, unpack_predictions, validate_neural_config)

_WORKER_CPU_SETTINGS = None


def _write(path: Path, value):
    """Write study state atomically so interrupted tuning can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n")
    temporary.replace(path)


def _save_array(directory: Path, name: str, value):
    np.save(directory/f"{name}.npy", np.asarray(value), allow_pickle=False)


def _tune_candidates(neural: dict, tuning: dict, farm_id: int):
    """Generate a reproducible random search, anchoring each feature/architecture at its default."""
    space = tuning["search_space"]
    groups = {(row["architecture"], row["features"]): row for row in neural["candidates"]
              if tuning["include_core_and_extended_features"] or row["features"] == "weather"}
    result = []
    for group_number, ((architecture, features), baseline) in enumerate(sorted(groups.items())):
        rng = np.random.default_rng(int(tuning["seed"])+farm_id*1009+group_number)
        depths = space["mlp_depths"] if architecture == "mlp" else (
            space["residual_depths"] if architecture == "residual" else space["tcn_depths"])
        def draw(values):
            return values[int(rng.integers(0, len(values)))]
        def candidate_from(values, trial):
            candidate = {"architecture": architecture, "features": features,
                "name": f"{architecture}_{features}_{trial:02d}", **values}
            if architecture == "tcn":
                candidate["kernel_size"] = values.get("kernel_size", baseline["kernel_size"])
            return candidate
        default_values = {"width": baseline["width"], "depth": baseline["depth"],
            "dropout": neural["dropout"], "learning_rate": neural["learning_rate"],
            "weight_decay": neural["weight_decay"], "batch_size": neural["batch_size"],
            "loss": neural["loss"], "huber_delta": neural["huber_delta"],
            "month_embedding_dim": neural["month_embedding_dim"],
            "auction_embedding_dim": neural["auction_embedding_dim"]}
        if architecture == "tcn":
            default_values["kernel_size"] = baseline["kernel_size"]
        group = [candidate_from(default_values, 0)]
        seen = {digest(default_values)}
        attempts = 0
        while len(group) < int(tuning["trials_per_architecture_feature"]):
            attempts += 1
            if attempts > 10000:
                raise ValueError("The configured neural search space has too few unique settings")
            values = {"width": draw(space["widths"]), "depth": draw(depths),
                "dropout": draw(space["dropout"]), "learning_rate": draw(space["learning_rates"]),
                "weight_decay": draw(space["weight_decays"]), "batch_size": draw(space["batch_sizes"]),
                "loss": draw(space["losses"]), "huber_delta": draw(space["huber_deltas"]),
                "month_embedding_dim": draw(space["month_embedding_dims"]),
                "auction_embedding_dim": draw(space["auction_embedding_dims"])}
            if architecture == "tcn":
                values["kernel_size"] = draw(space["tcn_kernel_sizes"])
            key = digest(values)
            if key not in seen:
                seen.add(key)
                group.append(candidate_from(values, len(group)))
        result.extend(group)
    return result


def _prepare_arrays(frame: pd.DataFrame, farm, neural: dict, tuning: dict, farm_dir: Path, cache_dir: Path):
    """Freeze train-only feature processing and compact CPU arrays once per farm."""
    train_mask = (frame.eligible & frame.daylight & (frame.split == "train")).to_numpy()
    validation_mask = (frame.eligible & frame.daylight & (frame.split == "validation")).to_numpy()
    feature_names = ["core", "weather"] if tuning["include_core_and_extended_features"] else ["weather"]
    views, preprocessors = {}, {}
    for feature_name in feature_names:
        features = feature_frame(frame, feature_name, neural["delivery_timezone"])
        preprocessor = fit_preprocessor(features, train_mask, neural)
        x = transform_features(features, preprocessor)
        packed = pack_sequences(frame, x, farm.ac_capacity_mw)
        array_dir = cache_dir/f"farm_{farm.farm_id}"/feature_name
        array_dir.mkdir(parents=True, exist_ok=True)
        for name in ("x", "y", "mask", "daylight", "month", "auction", "indices", "split"):
            _save_array(array_dir, name, packed[name])
        _save_array(array_dir, "x_flat", x)
        _write(array_dir/"preprocessor.json", preprocessor)
        preprocessors[feature_name] = preprocessor
        views[feature_name] = {"features": features, "x": x, "array_dir": array_dir}
        train_rows = np.flatnonzero(train_mask)
        validation_rows = np.flatnonzero(validation_mask)
        x_train = x[train_rows].astype(np.float64)
        y_train = frame.measured_ac_mw.to_numpy(dtype=np.float64)[train_rows]/farm.ac_capacity_mw
        w_train = balanced_weights(frame.iloc[train_rows]).astype(np.float64)
        x_mean = w_train@x_train
        y_mean = float(w_train@y_train)
        centered = x_train-x_mean
        centered_y = y_train-y_mean
        gram = centered.T@(centered*w_train[:, None])
        rhs = centered.T@(centered_y*w_train)
        reg_dir = array_dir/"regression"
        reg_dir.mkdir(exist_ok=True)
        _save_array(reg_dir, "gram", gram)
        _save_array(reg_dir, "rhs", rhs)
        _save_array(reg_dir, "x_mean", x_mean)
        _save_array(reg_dir, "y_mean", np.asarray(y_mean))
        _save_array(reg_dir, "x_validation", x[validation_rows].astype(np.float64))
        _save_array(reg_dir, "y_validation", frame.measured_ac_mw.to_numpy(dtype=np.float64)[validation_rows]/farm.ac_capacity_mw)
        _save_array(reg_dir, "w_validation", balanced_weights(frame.iloc[validation_rows]).astype(np.float64))
    farm_dir.mkdir(parents=True, exist_ok=True)
    _write(farm_dir/"json"/"preprocessors.json", preprocessors)
    return views, preprocessors


def _regression_trials(tuning: dict):
    """Enumerate conventional weighted least-squares and shrinkage candidates."""
    settings = tuning["regression"]
    trials = []
    feature_sets = ("core", "weather") if tuning["include_core_and_extended_features"] else ("weather",)
    for features in feature_sets:
        trials.append({"family": "ols", "features": features, "alpha": 0.0, "l1_ratio": 0.0})
        for alpha in settings["ridge_alphas"]:
            trials.append({"family": "ridge", "features": features, "alpha": float(alpha), "l1_ratio": 0.0})
        for alpha in settings["lasso_alphas"]:
            trials.append({"family": "lasso", "features": features, "alpha": float(alpha), "l1_ratio": 1.0})
        for alpha in settings["elastic_net_alphas"]:
            for ratio in settings["elastic_net_l1_ratios"]:
                trials.append({"family": "elastic_net", "features": features,
                    "alpha": float(alpha), "l1_ratio": float(ratio)})
    for trial in trials:
        alpha_text = f"{trial['alpha']:.0e}".replace("-", "m").replace("+", "")
        ratio_text = f"_{trial['l1_ratio']:0.1f}" if trial["family"] == "elastic_net" else ""
        trial["name"] = f"{trial['family']}_{trial['features']}_a{alpha_text}{ratio_text}"
    return trials


def _soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value)*max(abs(value)-threshold, 0.0))


def _solve_regression(gram, rhs, family: str, alpha: float, l1_ratio: float, max_iterations: int, tolerance: float):
    """Solve a weighted OLS, Ridge, Lasso, or Elastic Net problem from sufficient statistics."""
    if family == "ols":
        return np.linalg.pinv(gram, rcond=1e-10)@rhs, 0
    if family == "ridge":
        return np.linalg.solve(gram+alpha*np.eye(len(rhs)), rhs), 0
    diagonal = np.diag(gram)+alpha*(1-l1_ratio)
    beta = np.zeros_like(rhs)
    for iteration in range(1, max_iterations+1):
        largest_change = 0.0
        for column in range(len(beta)):
            if diagonal[column] <= 1e-20:
                beta[column] = 0.0
                continue
            partial = rhs[column]-gram[column]@beta+gram[column, column]*beta[column]
            updated = _soft_threshold(partial, alpha*l1_ratio)/diagonal[column]
            largest_change = max(largest_change, abs(updated-beta[column]))
            beta[column] = updated
        if largest_change <= tolerance:
            return beta, iteration
    return beta, max_iterations


def _regression_worker(job: dict):
    """Fit and score one conventional regression trial using validation only."""
    task, root = job["trial"], Path(job["array_dir"])/"regression"
    def array(name):
        return np.load(root/f"{name}.npy", mmap_mode="r", allow_pickle=False)
    gram, rhs = array("gram"), array("rhs")
    beta, iterations = _solve_regression(gram, rhs, task["family"], task["alpha"], task["l1_ratio"],
        int(job["max_iterations"]), float(job["tolerance"]))
    x_mean, y_mean = array("x_mean"), float(array("y_mean"))
    intercept = y_mean-float(x_mean@beta)
    x_validation, y_validation = array("x_validation"), array("y_validation")
    weights = array("w_validation")
    prediction = np.clip(x_validation@beta+intercept, 0.0, 1.0)
    score = float(np.sqrt(np.dot(weights, (prediction-y_validation)**2)))
    record = {**task, "validation_nrmse": score, "intercept": intercept,
        "coefficients": beta.tolist(), "coordinate_descent_iterations": iterations,
        "converged": task["family"] in {"ols", "ridge"} or iterations < int(job["max_iterations"]),
        "test_used_for_fit_or_selection": False}
    return record


def _execute_job(job: dict):
    """Resume or execute one independent CPU training task."""
    if job["kind"] == "neural":
        return ("neural", job["farm_id"], job["candidate"]["name"], _neural_worker(job))
    path = Path(job["result_path"])
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("fingerprint") == job["fingerprint"]:
            return ("regression", job["farm_id"], job["trial"]["name"], saved["record"])
    record = _regression_worker(job)
    _write(path, {"fingerprint": job["fingerprint"], "record": record})
    return ("regression", job["farm_id"], job["trial"]["name"], record)


def _neural_worker(job: dict):
    """Train one neural configuration on a single CPU thread."""
    global _WORKER_CPU_SETTINGS
    from .pv_neural_training import configure_cpu, fit_network, load_network, save_network
    requested_settings = (job["threads"], job["interop_threads"], job["neural"]["deterministic"])
    if _WORKER_CPU_SETTINGS is None:
        configure_cpu(job["neural"] | {"threads_per_worker": job["threads"],
                                        "interop_threads_per_worker": job["interop_threads"]})
        _WORKER_CPU_SETTINGS = requested_settings
    elif _WORKER_CPU_SETTINGS != requested_settings:
        raise ValueError("A reused worker cannot change its PyTorch CPU thread settings")
    path = Path(job["model_path"])
    fingerprint = job["trial_fingerprint"]
    if path.exists() and path.with_suffix(".json").exists():
        try:
            _, metadata = load_network(path, farm_id=job["farm_id"])
            if metadata.get("tuning_fingerprint") == fingerprint:
                return metadata["diagnostics"]
        except (OSError, ValueError, KeyError, RuntimeError):
            pass  # A stale or incomplete checkpoint is safely retrained below.
    array_dir = Path(job["array_dir"])
    packed = {name: np.load(array_dir/f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("x", "y", "mask", "daylight", "month", "auction", "split")}
    model, diagnostic = fit_network(packed, job["candidate"], job["neural"], seed=job["seed"])
    metadata = {"farm_metadata": job["farm_metadata"], "candidate": job["candidate"],
        "neural_settings": job["neural"], "preprocessor": job["preprocessor"],
        "diagnostics": diagnostic, "tuning_fingerprint": fingerprint,
        "source_fingerprint": job["source_fingerprint"], "target": "raw_AC_MW_divided_by_AC_rating",
        "fitted_pvlib_values_used_as_neural_inputs": False, "device": "cpu"}
    save_network(model, path, metadata)
    return diagnostic


def _metrics(frame: pd.DataFrame, method: str, farm):
    group = frame.loc[frame.eligible & frame.daylight & (frame.split == "test")]
    actual = group.measured_ac_mw.to_numpy(dtype=float)
    prediction = group[f"prediction_{method}_mw"].to_numpy(dtype=float)
    error = prediction-actual
    weights = balanced_weights(group)
    rmse = float(np.sqrt(np.dot(weights, error**2)))
    return {"farm_id": farm.farm_id, "method": method, "pairs": len(group),
        "mae_mw": float(np.dot(weights, np.abs(error))), "rmse_mw": rmse,
        "nrmse_percent_ac_rating": float(100*rmse/farm.ac_capacity_mw),
        "bias_mw": float(np.dot(weights, error)), "ac_rating_mw": farm.ac_capacity_mw}


def _comparison_plot(frame: pd.DataFrame, output: Path, end_day: str):
    """Plot validation and untouched test comparisons without ranking on test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    preferred = ["pvlib", "selected", "best_mlp", "best_residual", "best_tcn",
                 "best_ols", "best_ridge", "best_lasso", "best_elastic_net"]
    for axis, (farm, group) in zip(axes.flat, frame.groupby("farm_id")):
        group = group.set_index("method").reindex(preferred).dropna().reset_index()
        colors = ["tab:orange" if method == "pvlib" else
                  "tab:green" if method == "selected" else
                  "tab:purple" if method.startswith("best_") and method in {"best_ols", "best_ridge", "best_lasso", "best_elastic_net"}
                  else "tab:blue" for method in group.method]
        axis.barh(group.method, group.nrmse_percent_ac_rating, color=colors)
        axis.set(title=f"Farm {farm}", xlabel="Untouched test daylight RMSE / AC rating (%)")
        axis.grid(axis="x", alpha=.2)
    through = (pd.Timestamp(end_day)-pd.Timedelta(days=1)).date().isoformat()
    fig.suptitle(f"Calibrated pvlib, tuned neural networks and conventional regressions through {through}")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def validate_tuning_config(config: dict):
    """Reject incomplete searches and accidental non-CPU training configurations."""
    validate_neural_config(config)
    tuning = config["pv_neural_tuning"]
    if tuning["workers"] is not None and int(tuning["workers"]) < 1:
        raise ValueError("pv_neural_tuning.workers must be null or a positive integer")
    if int(tuning["reserved_cpu_cores"]) < 0:
        raise ValueError("pv_neural_tuning.reserved_cpu_cores must be nonnegative")
    for name in ("threads_per_worker", "interop_threads_per_worker", "trials_per_architecture_feature"):
        if int(tuning[name]) < 1:
            raise ValueError(f"pv_neural_tuning.{name} must be positive")
    for key, values in tuning["search_space"].items():
        if not values:
            raise ValueError(f"Neural tuning search space {key} must not be empty")
    regression = tuning["regression"]
    for key in ("ridge_alphas", "lasso_alphas", "elastic_net_alphas", "elastic_net_l1_ratios"):
        if not regression[key] or any(float(value) < 0 for value in regression[key]):
            raise ValueError(f"Regression tuning values {key} must be nonempty and nonnegative")
    if any(float(value) > 1 for value in regression["elastic_net_l1_ratios"]):
        raise ValueError("Elastic Net l1 ratios must lie in [0, 1]")


def run_neural_tuning(config: dict, progress=None):
    """Tune all farm/model trials in parallel, then report held-out test performance."""
    from .pv_neural_training import load_network, predict_network, tensor_pack
    validate_tuning_config(config)
    study_started = time.perf_counter()
    config = deepcopy(config)
    neural, tuning = config["pv_neural"], config["pv_neural_tuning"]
    logical_cpus = os.cpu_count() or 1
    available_workers = max(1, logical_cpus-int(tuning["reserved_cpu_cores"]))
    workers = min(int(tuning["workers"] or available_workers), available_workers)
    if tuning["run_id"]:
        run_id = str(tuning["run_id"])
    else:
        run_id = datetime.now(timezone.utc).strftime("neural_hpo_%Y%m%dT%H%M%S%fZ")
    root = config_path(config, tuning["output_root"])/run_id
    config_path_json = root/"json"/"study_config.json"
    config_hash = digest(config)
    if root.exists():
        if not tuning["resume"] or not config_path_json.exists() or digest(json.loads(config_path_json.read_text())) != config_hash:
            raise ValueError("Tuning output exists with different settings; choose a fresh run ID")
    for folder in ("json", "csv", "plots"):
        (root/folder).mkdir(parents=True, exist_ok=True)
    _write(config_path_json, config)
    cache_dir = root/".training_cache"
    cache_dir.mkdir(exist_ok=True)
    code_files = ("pv_neural_tuning.py", "pv_neural_data.py", "pv_neural_models.py", "pv_neural_training.py",
                  "pv_study_inputs.py", "pv.py", "pv_physical.py")
    code_fingerprint = hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in code_files)).hexdigest()
    farm_contexts = {}
    source_root = config_path(config, neural["output_root"])
    source_id = tuning.get("source_corpus_run_id")
    for farm_id in neural["farm_ids"]:
        farm_dir = root/f"farm_{farm_id}"
        for folder in ("csv", "json", "parquet", "plots", "models", "models/trials"):
            (farm_dir/folder).mkdir(parents=True, exist_ok=True)
        if source_id:
            source_dir = source_root/source_id/f"farm_{farm_id}"
            if not (source_dir/"parquet"/"corpus.parquet").exists():
                raise FileNotFoundError(f"Configured source corpus is missing: {source_dir}")
        else:
            source_dir = farm_dir
        frame, farm, pv_settings, provenance = prepare_neural_corpus(config, int(farm_id), source_dir,
            (lambda message: progress(message)) if progress else None)
        frame.to_parquet(farm_dir/"parquet"/"corpus.parquet", index=False)
        _write(farm_dir/"json"/"input_provenance.json", provenance)
        views, preprocessors = _prepare_arrays(frame, farm, neural, tuning, farm_dir, cache_dir)
        farm_contexts[int(farm_id)] = {"frame": frame, "farm": farm, "pv_settings": pv_settings,
            "provenance": provenance, "farm_dir": farm_dir, "views": views,
            "preprocessors": preprocessors}

    nn_candidates = {farm_id: _tune_candidates(neural, tuning, farm_id)
                     for farm_id in map(int, neural["farm_ids"])}
    jobs, neural_job_keys, regression_jobs = [], {}, {}
    regression_trials = _regression_trials(tuning)
    for farm_id, context in farm_contexts.items():
        farm_dir, provenance = context["farm_dir"], context["provenance"]
        for trial_number, candidate in enumerate(nn_candidates[farm_id]):
            trial_path = farm_dir/"models"/"trials"/f"{candidate['name']}.pt"
            trial_fingerprint = digest({"config": config_hash, "code": code_fingerprint,
                "source": provenance["source_fingerprint"], "farm": farm_id, "candidate": candidate})
            array_dir = cache_dir/f"farm_{farm_id}"/candidate["features"]
            jobs.append({"kind": "neural", "farm_id": farm_id, "array_dir": str(array_dir),
                "model_path": str(trial_path), "trial_fingerprint": trial_fingerprint,
                "candidate": candidate, "neural": neural, "threads": int(tuning["threads_per_worker"]),
                "interop_threads": int(tuning["interop_threads_per_worker"]),
                "seed": int(tuning["seed"])+farm_id*1000+trial_number,
                "farm_metadata": context["provenance"]["farm_metadata"],
                "preprocessor": context["preprocessors"][candidate["features"]],
                "source_fingerprint": provenance["source_fingerprint"]})
            neural_job_keys[(farm_id, candidate["name"])] = (trial_path, trial_fingerprint)
        for trial in regression_trials:
            array_dir = cache_dir/f"farm_{farm_id}"/trial["features"]
            result_path = farm_dir/"json"/"regression_trials"/f"{trial['name']}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            fingerprint = digest({"config": config_hash, "code": code_fingerprint,
                "source": provenance["source_fingerprint"], "farm": farm_id, "trial": trial})
            jobs.append({"kind": "regression", "farm_id": farm_id, "array_dir": str(array_dir),
                "trial": trial, "result_path": str(result_path), "fingerprint": fingerprint,
                "max_iterations": tuning["regression"]["coordinate_descent_max_iterations"],
                "tolerance": tuning["regression"]["coordinate_descent_tolerance"]})
            regression_jobs[(farm_id, trial["name"])] = (result_path, fingerprint)

    # Cap numpy/BLAS thread pools so each independent configuration owns one CPU.
    variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in variables}
    for name in variables:
        os.environ[name] = str(tuning["threads_per_worker"])
    records = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_execute_job, job) for job in jobs]
            for number, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                if progress and (number == 1 or number % max(1, workers) == 0 or number == len(futures)):
                    progress(f"PV tuning: completed {number}/{len(futures)} fits; latest {record[0]} farm {record[1]} {record[2]}")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    results_by_farm = {int(farm): {"neural": [], "regression": []} for farm in neural["farm_ids"]}
    for kind, farm_id, name, record in records:
        results_by_farm[farm_id][kind].append(record)
    root_test_rows = []
    farm_audits = []
    for farm_id, context in farm_contexts.items():
        farm, frame, farm_dir = context["farm"], context["frame"].copy(), context["farm_dir"]
        neural_results = results_by_farm[farm_id]["neural"]
        regression_results = results_by_farm[farm_id]["regression"]
        neural_by_name = {item["candidate"]["name"]: item for item in neural_results}
        selected_neural = min(neural_results, key=lambda item: item["validation_nrmse"])
        selected_by_arch = {architecture: min((item for item in neural_results
            if item["candidate"]["architecture"] == architecture), key=lambda item: item["validation_nrmse"])
            for architecture in ("mlp", "residual", "tcn")}
        selected_regression = {}
        for family in ("ols", "ridge", "lasso", "elastic_net"):
            eligible = [item for item in regression_results if item["family"] == family and item["converged"]]
            if not eligible:
                raise ValueError(f"No converged {family} model for farm {farm_id}")
            selected_regression[family] = min(eligible, key=lambda item: item["validation_nrmse"])
        prediction_cache = {}
        for item in {selected_neural["candidate"]["name"]: selected_neural,
                     **{row["candidate"]["name"]: row for row in selected_by_arch.values()}}.values():
            name = item["candidate"]["name"]
            if name in prediction_cache:
                continue
            path, expected_fingerprint = neural_job_keys[(farm_id, name)]
            model, metadata = load_network(path, farm_id=farm_id)
            if metadata.get("tuning_fingerprint") != expected_fingerprint:
                raise ValueError("A selected neural tuning artifact has a stale fingerprint")
            packed = {key: np.load(cache_dir/f"farm_{farm_id}"/item["candidate"]["features"]/f"{key}.npy",
                mmap_mode="r", allow_pickle=False) for key in ("x", "y", "mask", "daylight", "month", "auction", "indices", "split")}
            prediction = predict_network(model, tensor_pack(packed), int(neural["prediction_batch_size"]))
            prediction_cache[name] = unpack_predictions(packed, prediction, len(frame), farm.ac_capacity_mw)
        frame["prediction_pvlib_mw"] = frame.baseline_mw
        frame["prediction_selected_mw"] = prediction_cache[selected_neural["candidate"]["name"]]
        selected_methods = {"pvlib": {"family": "calibrated_pvlib", "model": "fixed_global_2025_parameters"},
            "selected": {"family": "best_neural", "model": selected_neural["candidate"]["name"]}}
        for architecture, item in selected_by_arch.items():
            method = {"mlp": "best_mlp", "residual": "best_residual", "tcn": "best_tcn"}[architecture]
            frame[f"prediction_{method}_mw"] = prediction_cache[item["candidate"]["name"]]
            selected_methods[method] = {"family": architecture, "model": item["candidate"]["name"]}
        for family, item in selected_regression.items():
            method = f"best_{family}"
            array_dir = cache_dir/f"farm_{farm_id}"/item["features"]
            x = np.load(array_dir/"x_flat.npy", mmap_mode="r", allow_pickle=False)
            prediction = np.clip(x@np.asarray(item["coefficients"])+float(item["intercept"]), 0, 1)
            prediction *= frame.daylight.to_numpy(dtype=bool)
            frame[f"prediction_{method}_mw"] = prediction*farm.ac_capacity_mw
            selected_methods[method] = {"family": family, "features": item["features"],
                "model": item["name"], "alpha": item["alpha"], "l1_ratio": item["l1_ratio"]}
        methods = ["pvlib", "selected", "best_mlp", "best_residual", "best_tcn",
                   "best_ols", "best_ridge", "best_lasso", "best_elastic_net"]
        frame.to_parquet(farm_dir/"parquet"/"predictions.parquet", index=False)
        _write(farm_dir/"json"/"neural_selection.json", {"selected": selected_neural,
            "best_by_architecture": {key: value for key, value in selected_by_arch.items()},
            "criterion": "minimum validation day-auction-balanced daylight NRMSE",
            "test_used_for_selection": False})
        _write(farm_dir/"json"/"regression_selection.json", {"selected": selected_regression,
            "criterion": "minimum validation day-auction-balanced daylight NRMSE within each regression family",
            "test_used_for_selection": False})
        trials_table = pd.DataFrame([{**item["candidate"], "validation_nrmse": item["validation_nrmse"],
            "best_epoch": item["best_epoch"], "epochs_completed": item["epochs_completed"],
            "runtime_seconds": item["runtime_seconds"], "parameter_count": item["parameter_count"]}
            for item in neural_results])
        trials_table.to_csv(farm_dir/"csv"/"neural_trials.csv", index=False)
        pd.DataFrame(regression_results).to_csv(farm_dir/"csv"/"regression_trials.csv", index=False)
        chosen_diagnostics = [neural_by_name[item["candidate"]["name"]] for item in selected_by_arch.values()]
        from .pv_neural_reporting import report_neural_farm
        summary = report_neural_farm(frame, farm, methods, neural, farm_dir, chosen_diagnostics)
        for method, details in selected_methods.items():
            metrics = _metrics(frame, method, farm)
            metrics["selected_model"] = details.get("model")
            root_test_rows.append(metrics)
        audit = {"farm_id": farm_id, "source_fingerprint": context["provenance"]["source_fingerprint"],
            "neural_trials": len(neural_results), "regression_trials": len(regression_results),
            "selected_neural_is_validation_minimum": selected_neural["candidate"]["name"] == min(neural_results,
                key=lambda item: item["validation_nrmse"])["candidate"]["name"],
            "all_selection_precedes_test_evaluation": True,
            "baseline_equals_aligned_calibrated_pvlib": bool(np.array_equal(frame.prediction_pvlib_mw, frame.baseline_mw)),
            "all_outputs_ac_bounded": all(bool(np.isfinite(frame[f"prediction_{method}_mw"]).all()
                and (frame[f"prediction_{method}_mw"] >= 0).all()
                and (frame[f"prediction_{method}_mw"] <= farm.ac_capacity_mw+1e-7).all()) for method in methods),
            "forecast_availability_respected": bool((frame.forecast_available_at_utc <= frame.decision_cutoff_utc).all()),
            "test_delivery_days": int(frame.loc[frame.split == "test", "delivery_day"].nunique())}
        if not audit["selected_neural_is_validation_minimum"] or not audit["baseline_equals_aligned_calibrated_pvlib"] or not audit["all_outputs_ac_bounded"] or not audit["forecast_availability_respected"]:
            raise ValueError(f"Hyperparameter tuning audit failed for farm {farm_id}")
        _write(farm_dir/"json"/"tuning_audit.json", audit)
        farm_audits.append(audit)

    comparison = pd.DataFrame(root_test_rows)
    comparison.to_csv(root/"csv"/"test_comparison.csv", index=False)
    _comparison_plot(comparison, root/"plots"/"test_model_comparison.png", neural["end_day_exclusive"])
    elapsed = time.perf_counter()-study_started
    manifest = {"status": "complete", "run_id": run_id, "output_dir": str(root),
        "device": "cpu", "workers": workers, "threads_per_worker": tuning["threads_per_worker"],
        "total_parallel_fits": len(jobs), "runtime_seconds": elapsed,
        "search": "seeded random neural search plus weighted OLS, Ridge, Lasso and Elastic Net grids",
        "validation_used_for_selection": True, "test_used_for_hyperparameter_selection": False,
        "regression_implementation": "NumPy weighted least squares and coordinate descent; no new package installed",
        "code_fingerprint": code_fingerprint, "configuration_fingerprint": config_hash,
        "farms": farm_audits}
    _write(root/"json"/"manifest.json", manifest)
    shutil.rmtree(cache_dir)
    return manifest
