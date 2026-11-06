"""Checkpointed CPU price search with chronological validation and later-period holdouts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd

from .price_feature_data import digest, italian_holidays, prepare_matrices
from .price_feature_models import make_model, point_metrics, sample_parameters, stable_seed, training_objective


def write_json(path, value):
    """Atomically replace a checkpoint after each completed fit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def signature(value):
    """Hash scientific content without depending on Python's randomized hash seed."""
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def run_lock(directory):
    """Keep two cluster jobs from writing the same study concurrently."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this search output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _implementation():
    """Pin the actual implementation rather than unrelated worktree modifications."""
    directory = Path(__file__).parent
    paths = [directory / name for name in ["price_feature_search.py", "price_feature_data.py",
             "price_feature_models.py", "price_search_history.py"]]
    paths.append(directory.parents[1] / "data" / "price_search_inputs.py")
    return {path.name: digest(path) for path in paths}


def _software(settings):
    """Fail before launching workers if a configured model library is unavailable."""
    packages = {"numpy", "pandas", "pyarrow", "duckdb", "scikit-learn", "threadpoolctl", "joblib"}
    packages.update({family for family in settings["parameter_spaces"] if family in {"lightgbm", "catboost", "xgboost"}})
    return {"python": platform.python_version(), **{name: version(name) for name in sorted(packages)}}


def _validate(settings):
    """Reject overlapping periods, absent blocks and invalid resource choices before fitting."""
    if settings["workers"] < 1 or settings["trials_per_family"] < 1:
        raise ValueError("Worker and candidate counts must be positive")
    if not 0 < settings["inner_validation_fraction"] < 1 or settings["embargo_days"] < 0:
        raise ValueError("Invalid chronological holdout fraction or embargo")
    if not settings["parameter_spaces"]:
        raise ValueError("At least one model family must be configured")
    supported = {"ridge", "elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting", "lightgbm", "catboost", "xgboost", "mlp"}
    if set(settings["parameter_spaces"]) - supported:
        raise ValueError("Unrecognized model family")
    for group in settings["product_groups"].values():
        dates = [group[key] for key in ["train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end"]]
        if (dates != sorted(dates) or dates[0] == dates[1] or dates[2] == dates[3] or dates[4] == dates[5]
                or group["native_minutes"] not in {15, 60}):
            raise ValueError("Product groups require ordered nonempty native-resolution chronological splits")
    for blocks in settings["cohorts"].values():
        if not blocks or len(blocks) != len(set(blocks)) or set(blocks) - set(settings["feature_blocks"]):
            raise ValueError("Cohorts require distinct declared feature blocks")


def _arms(groups, settings):
    """Enumerate every subset on identical observations within each eligible cohort."""
    result = []
    for group_name, group in groups.items():
        for cohort_name, cohort in group["cohorts"].items():
            if not cohort["eligible"]:
                continue
            for mask in range(2 ** len(cohort["blocks"])):
                blocks = [name for bit, name in enumerate(cohort["blocks"]) if mask & (1 << bit)]
                columns = list(range(group["base_width"]))
                flags = [settings["feature_blocks"][block]["availability"] for block in blocks]
                for block in blocks:
                    columns.extend(group["block_columns"][block])
                outcome = group["market"] == "DA" and any(settings["feature_blocks"][block].get("contains_target_da_price") for block in blocks)
                availability = ("outcome_derived_da_price" if outcome else "retrospective_commercial" if "retrospective_commercial" in flags
                                else "retrospective_realized" if "retrospective_realized" in flags else "assumed_previous_day")
                result.append({"id": f"{group_name}__{cohort_name}__m{mask:03d}", "group": group_name,
                    "cohort": cohort_name, "mask": mask, "blocks": blocks, "columns": columns,
                    "availability_class": availability, "same_day_realized": any(flag.startswith("retrospective") for flag in flags),
                    "target_price_embedded_in_features": outcome})
    return result


def prepare_search(settings, input_settings, output, *, pinned_input_manifest=None):
    """Freeze DB source selections, arrays, candidate definitions, code and versions."""
    _validate(settings)
    output = Path(output)
    runtime_keys = {"database_path", "output_directory", "workers", "plot_cache_directory", "input_config"}
    scientific = {key: value for key, value in settings.items() if key not in runtime_keys}
    input_contract = {key: value for key, value in input_settings.items()
                      if key not in {"database_path", "output_directory", "database_threads"}}
    contract = signature({"study": scientific, "input_selection": input_contract})
    implementation = _implementation()
    software = _software(settings)
    prepared_path = output / "prepared.json"
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        if (prepared["scientific_config_sha256"] != contract or prepared["implementation_sha256"] != implementation
                or prepared["software"] != software):
            raise ValueError("Existing search has different settings, implementation or libraries; use its original environment/commit or a new output directory")
        # Resuming uses the frozen inputs; later changes in the source catalog cannot alter this study.
        for group in prepared["groups"].values():
            descriptors = list(group["files"].values())
            descriptors.extend(item for cohort in group["cohorts"].values() for item in cohort.get("indices", {}).values())
            for item in descriptors:
                if digest(output / item["path"]) != item["sha256"]:
                    raise ValueError(f"Prepared array checksum changed: {item['path']}")
        return prepared
    # Data imports are lazy so spawned fit workers do not import farm adapters or open DuckDB.
    from ...data.price_search_inputs import build_price_search_inputs
    print("Preparing snapshot-pinned DuckDB inputs and native-product matrices", flush=True)
    selection = dict(input_settings, database_path=settings["database_path"], output_directory=str(output / "inputs"))
    previous = output / "inputs" / "manifest.json"
    pinned = pinned_input_manifest or (str(previous) if previous.exists() else None)
    source_manifest = build_price_search_inputs({"price_search_inputs": selection}, pinned_manifest=pinned)
    groups = prepare_matrices(settings, output / "inputs", output)
    arms = _arms(groups, settings)
    if not arms:
        raise ValueError("No eligible matched study cohorts; inspect source selections and split coverage")
    tasks = []
    for arm in arms:
        for family, space in settings["parameter_spaces"].items():
            seed = stable_seed(settings["seed"], arm["group"], arm["cohort"], family)
            parameters = sample_parameters(space, seed, settings["trials_per_family"])
            task = {"id": f"{arm['id']}__{family}", "arm": arm, "family": family,
                    "candidates": [{"trial": trial, "parameters": values,
                                    "seed": stable_seed(seed, trial)} for trial, values in enumerate(parameters)]}
            task["signature"] = signature({"contract": contract, **task})
            tasks.append(task)
    prepared = {"status": "prepared", "scientific_config_sha256": contract,
        "scientific_settings": scientific, "implementation_sha256": implementation, "software": software,
        "source_selection_sha256": source_manifest["selection_sha256"],
        "source_rows": {name: details["selected_row_sha256"] for name, details in source_manifest["tables"].items()},
        "groups": groups, "arms": arms, "tasks": tasks, "inner_fits": sum(len(task["candidates"]) for task in tasks),
        "farm_files_required": False, "cuda_used": False, "threads_per_worker": 1,
        "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(prepared_path, prepared)
    return prepared


_ARRAY_CACHE = {}
_WORKER_OUTPUT = None
_WORKER_PREPARED = None


def _initialize_worker(output, groups):
    """Send small array descriptors once per process, instead of pickling the whole trial plan per fit."""
    global _WORKER_OUTPUT, _WORKER_PREPARED
    _WORKER_OUTPUT = output
    _WORKER_PREPARED = {"groups": groups}
    _ARRAY_CACHE.clear()


def _load_arrays(output, prepared, arm):
    """Share prepared files through read-only memory maps rather than IPC array copies."""
    key = (str(output), arm["group"], arm["cohort"])
    if key not in _ARRAY_CACHE:
        group = prepared["groups"][arm["group"]]
        cohort = group["cohorts"][arm["cohort"]]
        arrays = {name: np.load(Path(output) / item["path"], mmap_mode="r", allow_pickle=False)
                  for name, item in group["files"].items()}
        arrays.update({name: np.load(Path(output) / item["path"], mmap_mode="r", allow_pickle=False)
                       for name, item in cohort["indices"].items()})
        _ARRAY_CACHE[key] = arrays
    return _ARRAY_CACHE[key]


def _fit(family, candidate, x, y, test_x):
    """Fit one CPU estimator under both library and BLAS thread caps."""
    from threadpoolctl import threadpool_limits
    model = make_model(family, candidate["parameters"], candidate["seed"])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(x, y)
        # CatBoost prediction has its own thread_count argument independent of fit settings.
        prediction = model.predict(test_x, thread_count=1, task_type="CPU") if family == "catboost" else model.predict(test_x)
    messages = sorted({f"{type(item.message).__name__}: {item.message}"[:1000] for item in caught})
    return model, prediction, messages


def _trial_task(task, retry_failed):
    """Checkpoint each candidate within a family task and skip completed fits on resume."""
    output, prepared = _WORKER_OUTPUT, _WORKER_PREPARED
    path = Path(output) / "trials" / f"{task['id']}.json"
    state = json.loads(path.read_text()) if path.exists() else {"signature": task["signature"], "results": []}
    if state["signature"] != task["signature"]:
        raise ValueError("Trial checkpoint signature differs from the prepared search")
    completed = {item["trial"]: item for item in state["results"]}
    if (len(completed) == len(task["candidates"])
            and all(item["status"] == "complete" or not retry_failed for item in completed.values())):
        return {"id": task["id"], "successful": sum(item["status"] == "complete" for item in completed.values()),
                "failed": sum(item["status"] == "failed" for item in completed.values())}
    arrays = _load_arrays(output, prepared, task["arm"])
    older, later = arrays["inner_train"], arrays["inner_validation"]
    columns = task["arm"]["columns"]
    x = np.asarray(arrays["x"][np.ix_(older, columns)])
    y = arrays["y"][older] - arrays["anchor"][older]
    test_x = np.asarray(arrays["x"][np.ix_(later, columns)])
    for candidate in task["candidates"]:
        prior = completed.get(candidate["trial"])
        if prior is not None and (prior["status"] == "complete" or not retry_failed):
            continue
        began = time.monotonic()
        row = {**candidate, "family": task["family"], "training_objective": training_objective(task["family"], candidate["parameters"])}
        try:
            _, prediction, messages = _fit(task["family"], candidate, x, y, test_x)
            score = point_metrics(arrays["y"][later], prediction + arrays["anchor"][later], arrays["reference"][later])
            row.update(status="complete", metrics=score, warnings=messages)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["elapsed_seconds"] = time.monotonic() - began
        completed[candidate["trial"]] = row
        state["results"] = [completed[index] for index in sorted(completed)]
        write_json(path, state)
    return {"id": task["id"], "successful": sum(row["status"] == "complete" for row in state["results"]),
            "failed": sum(row["status"] == "failed" for row in state["results"])}


def _score_key(row):
    """Use absolute-price WAPE as the primary selection score, then MAE and stable IDs."""
    score = row["metrics"]
    return (float("inf") if score["wape_percent"] is None else score["wape_percent"],
            score["mae_eur_per_mwh"], row.get("family", row.get("candidate", {}).get("family", "")),
            row.get("trial", row.get("candidate", {}).get("trial", -1)), row.get("arm", {}).get("id", ""))


def _refit_task(job):
    """Refit a frozen candidate for validation or for its untouched holdout period."""
    output, prepared = _WORKER_OUTPUT, _WORKER_PREPARED
    path = Path(output) / "refits" / f"{job['stage']}__{job['arm']['id']}.json"
    expected = signature(job)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["signature"] != expected:
            raise ValueError("Refit checkpoint differs from the frozen candidate")
        if previous["status"] == "complete":
            for artifact in previous["artifacts"].values():
                if digest(Path(output) / artifact["path"]) != artifact["sha256"]:
                    raise ValueError("Refit artifact checksum mismatch")
            return previous
    arrays = _load_arrays(output, prepared, job["arm"])
    training = arrays["train" if job["stage"] == "validation" else "final_train"]
    evaluation = arrays["validation" if job["stage"] == "validation" else "test"]
    columns = job["arm"]["columns"]
    result = {"signature": expected, **job}
    try:
        model, prediction, messages = _fit(job["candidate"]["family"], job["candidate"],
            np.asarray(arrays["x"][np.ix_(training, columns)]), arrays["y"][training] - arrays["anchor"][training],
            np.asarray(arrays["x"][np.ix_(evaluation, columns)]))
        prediction = prediction + arrays["anchor"][evaluation]
        result.update(status="complete", metrics=point_metrics(arrays["y"][evaluation], prediction, arrays["reference"][evaluation]),
                      training_rows=len(training), warnings=messages, artifacts={})
        filename = f"{job['stage']}__{job['arm']['id']}"
        prediction_path = Path(output) / "predictions" / f"{filename}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"product_start": pd.to_datetime(arrays["starts"][evaluation], utc=True),
            "actual": arrays["y"][evaluation], "prediction": prediction, "reference": arrays["reference"][evaluation]}).to_parquet(prediction_path, index=False)
        result["artifacts"]["predictions"] = {"path": str(prediction_path.relative_to(output)), "sha256": digest(prediction_path)}
        if job["stage"] == "holdout":
            import joblib
            model_path = Path(output) / "models" / f"{filename}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
            result["artifacts"]["model"] = {"path": str(model_path.relative_to(output)), "sha256": digest(model_path)}
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(path, result)
    return result


def _pool_jobs(function, jobs, workers, progress, output, groups):
    """Run independent fits without collecting large matrices in the coordinator."""
    if not jobs:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_worker, initargs=(str(output), groups)) as executor:
        futures = [executor.submit(function, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            progress(results)
    return sorted(results, key=lambda row: row.get("id", row.get("arm", {}).get("id", "")))


def run_search(settings, input_settings, output, *, prepare_only=False, pinned_input_manifest=None, retry_failed=False):
    """Prepare once, resume inner fits, select on validation, and score frozen holdouts."""
    output = Path(output).resolve()
    with run_lock(output):
        prepared = prepare_search(settings, input_settings, output, pinned_input_manifest=pinned_input_manifest)
        if prepare_only:
            return prepared
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        if settings["workers"] > allowed:
            raise ValueError(f"Requested {settings['workers']} workers, but process affinity permits {allowed}; obtain the matching Slurm allocation")
        state = {"status": "running", "phase": "inner_search", "workers": settings["workers"], "threads_per_worker": 1,
                 "cuda_used": False, "inner_fits_expected": prepared["inner_fits"],
                 "scientific_config_sha256": prepared["scientific_config_sha256"], "software": prepared["software"],
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": datetime.now(timezone.utc).isoformat()}
        state_path = output / "status.json"
        def progress(rows):
            state["finished_tasks_in_phase"] = len(rows)
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, state)
            if len(rows) == 1 or len(rows) % 25 == 0:
                print(f"{state['phase']}: {len(rows)} tasks finished", flush=True)
        write_json(state_path, state)
        print(f"Searching {prepared['inner_fits']} candidates across {len(prepared['arms'])} feature subsets "
              f"with {settings['workers']} single-thread CPU workers", flush=True)
        tasks = [(task, retry_failed) for task in prepared["tasks"]]
        results = _pool_jobs(_trial_task, tasks, settings["workers"], progress, output, prepared["groups"])
        state["inner_fits_complete"] = sum(item["successful"] for item in results)
        state["inner_fits_failed"] = sum(item["failed"] for item in results)
        if state["inner_fits_failed"] or state["inner_fits_complete"] != prepared["inner_fits"]:
            state["status"] = "partial"
            write_json(state_path, state)
            return state
        # Compare family candidates on inner data only; validation never tunes their parameters.
        arm_winners = {}
        for task in prepared["tasks"]:
            rows = json.loads((output / "trials" / f"{task['id']}.json").read_text())["results"]
            winner = min((row for row in rows if row["status"] == "complete"), key=_score_key)
            key = task["arm"]["id"]
            if key not in arm_winners or _score_key(winner) < _score_key(arm_winners[key]):
                arm_winners[key] = winner
        validation_jobs = [{"stage": "validation", "arm": arm, "candidate": arm_winners[arm["id"]]} for arm in prepared["arms"]]
        state.update(phase="validation", finished_tasks_in_phase=0)
        validation = _pool_jobs(_refit_task, [(job,) for job in validation_jobs], settings["workers"], progress, output, prepared["groups"])
        if any(row["status"] != "complete" for row in validation):
            state.update(status="partial", refit_failures=[row for row in validation if row["status"] != "complete"])
            write_json(state_path, state)
            return state
        # Freeze separate winners for assumed-available, realized and outcome-derived feature classes.
        chosen = {}
        for row in validation:
            arm = row["arm"]
            key = (arm["group"], arm["cohort"], arm["availability_class"])
            if key not in chosen or _score_key(row) < _score_key(chosen[key]):
                chosen[key] = row
            if arm["mask"] == 0:
                chosen[(arm["group"], arm["cohort"], "price_only_baseline")] = row
        selected = {row["arm"]["id"]: row for row in chosen.values()}
        frozen = {"selection": "pooled validation absolute-price WAPE then MAE; no holdout score consulted",
                  "winners": list(selected.values())}
        write_json(output / "selected_models.json", frozen)
        state.update(phase="holdout", finished_tasks_in_phase=0)
        holdout_jobs = [{"stage": "holdout", "arm": row["arm"], "candidate": row["candidate"]} for row in selected.values()]
        holdout = _pool_jobs(_refit_task, [(job,) for job in holdout_jobs], settings["workers"], progress, output, prepared["groups"])
        state.update(status="complete" if all(row["status"] == "complete" for row in holdout) else "partial",
                     phase="finished", validation_models=len(validation), holdout_models=len(holdout),
                     completed_at=datetime.now(timezone.utc).isoformat())
        write_report(output, prepared, validation, holdout, state, settings)
        write_json(state_path, state)
        return state


def write_report(output, prepared, validation, holdout, state, settings):
    """Publish scores, matched baseline deltas, daily errors and inspection plots."""
    cache = Path(os.environ.get("MPLCONFIGDIR", settings["plot_cache_directory"]))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["# CPU price feature search", "", f"Status: **{state['status']}**.", "",
             "Parameters were selected on the chronological training tail; feature subsets were selected on development validation. "
             "Holdout results evaluate frozen selections. Realized and outcome-derived inputs are labeled separately. "
             "All models produce point forecasts; probabilistic calibration and trading-profit evaluation are outside this run.", ""]
    daily_rows = []
    for stage, rows in [("validation", validation), ("holdout", holdout)]:
        flattened = []
        for row in rows:
            arm = row["arm"]
            flat = {"group": arm["group"], "cohort": arm["cohort"], "mask": arm["mask"],
                    "blocks": ",".join(arm["blocks"]) or "price_only", "availability_class": arm["availability_class"],
                    "family": row["candidate"]["family"], "status": row["status"], **row.get("metrics", {})}
            flattened.append(flat)
            if stage == "holdout" and row["status"] == "complete":
                predictions = pd.read_parquet(output / row["artifacts"]["predictions"]["path"])
                local = pd.DatetimeIndex(predictions.product_start).tz_convert(settings["timezone"])
                predictions["day"] = local.strftime("%Y-%m-%d")
                predictions["hour"] = local.hour
                predictions["holiday"] = italian_holidays(pd.DatetimeIndex(predictions.product_start), settings).astype(bool)
                for day, day_frame in predictions.groupby("day"):
                    daily_rows.append({**flat, "day": day, "holiday": bool(day_frame.holiday.iloc[0]),
                                       **point_metrics(day_frame.actual, day_frame.prediction, day_frame.reference)})
                # The interval forecasts are retained for further weekday, holiday and price-regime diagnostics.
        frame = pd.DataFrame(flattened)
        if not frame.empty:
            baselines = frame.loc[frame["mask"].eq(0), ["group", "cohort", "mae_eur_per_mwh"]].rename(columns={"mae_eur_per_mwh": "price_only_mae"})
            frame = frame.merge(baselines, on=["group", "cohort"], how="left", validate="many_to_one")
            frame["mae_change_vs_matched_price_only"] = frame.mae_eur_per_mwh - frame.price_only_mae
        frame.to_csv(output / f"{stage}_scores.csv", index=False)
        if stage == "holdout":
            lines.extend(["| Product group | Cohort | Availability | Model | MAE EUR/MWh | WAPE % |", "|---|---|---|---|---:|---:|"])
            for row in flattened:
                if row["status"] == "complete":
                    wape = "n/a" if row["wape_percent"] is None else f"{row['wape_percent']:.3f}"
                    lines.append(f"| {row['group']} | {row['cohort']} | {row['availability_class']} | {row['family']} | {row['mae_eur_per_mwh']:.3f} | {wape} |")
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output / "daily_errors.csv", index=False)
    if not daily.empty:
        for (group, cohort), values in daily.groupby(["group", "cohort"]):
            figure, axis = plt.subplots(figsize=(12, 4))
            for mask, curve in values.groupby("mask"):
                label = "price only" if mask == 0 else curve.availability_class.iloc[0]
                axis.plot(pd.to_datetime(curve.day), curve.mae_eur_per_mwh, label=label, alpha=0.8)
            axis.set(title=f"{group} / {cohort}: holdout daily error", ylabel="MAE (EUR/MWh)")
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{group}__{cohort}__daily_mae.png", dpi=140)
            plt.close(figure)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
