"""Plan and run explicit continuous energy-market research contrasts.

Cases are declarations, not a parameter-search Cartesian product. Every job
resolves an independent copy of one raw configuration snapshot with its farm,
date window and case overrides, so asset sizes and references cannot leak.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import traceback

import pandas as pd

from ..config import _read_mapping, merge_mapping, resolve_config
from ..data import config_path
from ..markets import AuctionCalendar
from ..optimization.mpc import MPC_POLICIES, needs_mpc_ensemble
from ..parallel import configure_worker_threads
from .realtime_mpc_study import run_realtime_mpc_experiment, validate_mpc_study


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SCALES = {"battery_capacity": ("battery", "capacity_mwh"),
    "battery_power": ("battery", "max_power_mw"), "poi_import": ("plant", "poi_import_mw"),
    "poi_export": ("plant", "poi_export_mw")}


def _identifier(value, label):
    """Keep all run identifiers within their explicitly declared output directory."""
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a nonempty simple identifier without path separators")
    return value


def _positive_integer(value, label, maximum=None):
    """Validate worker limits without allowing booleans or silent rounding."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
        raise ValueError(f"{label} must be a positive integer" + (f" at most {maximum}" if maximum else ""))
    return value


def _sha256(path):
    """Hash a source or model artifact without reading entire databases into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    """Atomically replace one manifest while holding the parent study lease."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def _study_definition(path, raw=None):
    """Validate the finite case/window inventory before resolving any case config."""
    raw = _read_mapping(path) if raw is None else raw
    study = raw.get("energy_research_study", {})
    if not isinstance(study, dict):
        raise ValueError("energy_research_study must be a mapping")
    allowed = {"run_id", "output_root", "farms", "additional_farms", "windows", "cases", "workers", "native_threads",
        "solver_threads", "reporting", "factorial_cases", "forecast_flexibility_factorials", "mode"}
    if set(study) - allowed:
        raise ValueError(f"Unknown energy_research_study fields: {sorted(set(study) - allowed)}")
    if study.get("mode", "continuous") != "continuous":
        raise ValueError("Energy research studies use continuous optimization only")
    _identifier(study.get("run_id"), "Study run_id")
    if not isinstance(study.get("output_root"), str) or not study["output_root"]:
        raise ValueError("Study output_root must be a nonempty path")
    _positive_integer(study.get("workers", 42), "workers", maximum=42)
    _positive_integer(study.get("native_threads", 1), "native_threads")
    _positive_integer(study.get("solver_threads", 1), "solver_threads")
    farms = study.get("farms", [])
    if (not isinstance(farms, list) or not farms or any(isinstance(value, bool)
            or not isinstance(value, (int, str)) for value in farms)
            or len(set(map(str, farms))) != len(farms)):
        raise ValueError("Study farms must be unique selected farm IDs")
    for farm in farms:
        _identifier(str(farm), "Farm ID")
    # Additional farms are available only when explicitly selected; the
    # original default farm list remains unchanged for existing launchers.
    additional = study.get("additional_farms", [])
    if (not isinstance(additional, list) or any(isinstance(value, bool)
            or not isinstance(value, (int, str)) for value in additional)
            or len(set(map(str, additional))) != len(additional)
            or set(map(str, additional)) & set(map(str, farms))):
        raise ValueError("Additional farms must be unique and separate from default farms")
    for farm in additional:
        _identifier(str(farm), "Additional farm ID")
    for collection, fields in (("cases", {"id", "family", "baseline_case", "overrides", "asset_scales"}),
            ("windows", {"id", "start_date", "end_date", "overrides", "role"})):
        values = study.get(collection, [])
        if not isinstance(values, list) or not values:
            raise ValueError(f"Study {collection} must be a nonempty list")
        names = []
        for item in values:
            if not isinstance(item, dict) or set(item) - fields:
                raise ValueError(f"Invalid study {collection} entry")
            names.append(_identifier(item.get("id"), f"{collection} ID"))
            if not isinstance(item.get("overrides", {}), dict):
                raise ValueError(f"{collection} overrides must be mappings")
        if len(set(names)) != len(names):
            raise ValueError(f"Study {collection} IDs must be unique")
    case_ids = {case["id"] for case in study["cases"]}
    flex = study.get("forecast_flexibility_factorials", {})
    if not isinstance(flex, dict):
        raise ValueError("forecast_flexibility_factorials must be a mapping")
    for name, declaration in flex.items():
        _identifier(name, "Forecast/flexibility factorial ID")
        required = {"raw_base", "calibrated_base", "raw_modified", "calibrated_modified", "changed_fields"}
        if not isinstance(declaration, dict) or set(declaration) != required:
            raise ValueError("Forecast/flexibility factorial must declare four cases and changed_fields")
        selected = [declaration[key] for key in ("raw_base", "calibrated_base", "raw_modified", "calibrated_modified")]
        if len(set(selected)) != 4 or set(selected) - case_ids:
            raise ValueError("Forecast/flexibility cells must identify four distinct declared cases")
        fields = declaration["changed_fields"]
        if not isinstance(fields, list) or not fields or any(not isinstance(path, list) or len(path) != 2
                or path[0] not in {"battery", "plant"} or not isinstance(path[1], str) or not path[1]
                for path in fields):
            raise ValueError("Forecast/flexibility changed_fields must contain battery or plant field paths")
    for case in study["cases"]:
        if case.get("baseline_case") is not None and (case["baseline_case"] not in case_ids
                or case["baseline_case"] == case["id"]):
            raise ValueError("Each baseline_case must identify a different declared case")
        _identifier(case.get("family"), "Case family")
        scales = case.get("asset_scales", {})
        if not isinstance(scales, dict) or set(scales) - set(_SCALES):
            raise ValueError("Unknown asset_scales fields")
        for value in scales.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError("Asset scales must be finite and positive")
    return study


def _select(items, selected, label):
    """Select an explicit subset and reject misspelled names rather than dropping it."""
    selected = None if selected is None else set(map(str, selected))
    available = {str(item["id"]) for item in items}
    if selected is not None and (not selected or selected - available):
        raise ValueError(f"Unknown or empty selected {label}: {sorted((selected or set()) - available)}")
    return [item for item in items if selected is None or str(item["id"]) in selected]


def _artifact_requirements(config, checksums=None):
    """List mandatory files without pretending that absent research artifacts exist."""
    paths = {"pv_measurements_and_weather": config["inputs"]["zarr"],
        "plant_metadata": config["inputs"]["metadata_parquet"],
        "auction_catalog": config["gme_oracle_experiment"]["price_database"],
        "settlement_catalog": config["settlement"]["database_path"]}
    curated = config.get("forecast", {}).get("curated_realized", {})
    if curated.get("enabled"):
        root = Path(curated["dataset_root"])
        farm = config["forecast"]["farm_id"]
        paths["curated_pv_manifest"] = str(root / "manifest.json")
        paths["curated_pv_corpus"] = str(root / f"farm_{farm}" / "corpus.parquet")
        paths["curated_pv_modeling_view"] = str(root / f"farm_{farm}" / "modeling_views.parquet")
    replacement = config.get("pv", {}).get("forecast_replacement", {})
    selected = replacement.get("models_by_farm", {}).get(str(config["forecast"]["farm_id"]))
    if replacement.get("enabled") and selected and selected.get("kind") == "reviewed_hpo":
        artifact = Path(selected["artifact_path"])
        paths["pv_model_record"] = str(artifact)
        joblib_model = artifact.parent / "model.joblib"
        paths["pv_model_weights"] = str(joblib_model if config_path(config, joblib_model).is_file()
                                        else artifact.parent / "model.pt")
        paths["pv_model_preprocessor"] = str(artifact.parent / "preprocessor.json")
    if needs_mpc_ensemble(config):
        ensemble = config["continuous_optimization"]["pv_ensemble"]
        if ensemble.get("sources"):
            paths.update({f"weather_ensemble_{index}": source
                for index, source in enumerate(ensemble["sources"])})
        else:
            paths["weather_ensemble"] = ensemble.get("zarr")
    prices = config.get("energy_price_forecast", {})
    if prices.get("auction_enabled"):
        for auction in config["gme_oracle_experiment"].get("participating_auctions") or ["DA", "IDA1", "IDA2", "IDA3"]:
            paths[f"price_model_{auction}"] = prices.get("artifacts", {}).get(auction)
    if prices.get("settlement_enabled") or config.get("realtime_mpc", {}).get("terminal_inventory_mode") == "piecewise_linear":
        paths["price_model_settlement"] = prices.get("artifacts", {}).get("settlement")
        if prices.get("artifacts", {}).get("settlement_negative"):
            paths["price_model_settlement_negative"] = prices["artifacts"]["settlement_negative"]
    for name, value in prices.get("history_paths", {}).items():
        if f"price_model_{name}" in paths:
            paths[f"observed_price_history_{name}"] = value
    if prices.get("exogenous_path"):
        paths["price_exogenous_features"] = prices["exogenous_path"]
    scenarios = config.get("realtime_scenarios", {})
    if scenarios.get("dependence") == "ecc" or scenarios.get("calibrated_marginals_path"):
        paths["calibrated_pv_marginals"] = scenarios.get("calibrated_marginals_path")
    result = []
    checksums = {} if checksums is None else checksums
    for role, value in sorted(paths.items()):
        path = None if not value else config_path(config, value)
        present = path is not None and path.exists()
        checksum = None
        if present and path.is_file() and role.startswith(("price_model_", "pv_model_")):
            # Most jobs reuse the same fold/model. Hash each unchanged file once
            # per plan, while detecting replacement or modification mid-plan.
            before = path.stat()
            key = (str(path.resolve()), before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if key not in checksums:
                checksums[key] = _sha256(path)
                after = path.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Price model changed while hashing: {path}")
            checksum = checksums[key]
        result.append({"role": role, "path": None if path is None else str(path), "present": present,
            "sha256": checksum})
    return result


def expand_research_jobs(path, *, case_ids=None, farms=None, window_ids=None, project_root=None,
                         run_id=None, workers=None, solver_threads=None):
    """Resolve independent jobs from one raw YAML snapshot per expansion."""
    path = Path(path).resolve()
    raw = _read_mapping(path)
    study = deepcopy(_study_definition(path, raw))
    if run_id is not None:
        study["run_id"] = _identifier(run_id, "Study run_id")
    if workers is not None:
        study["workers"] = _positive_integer(workers, "workers", maximum=42)
    if solver_threads is not None:
        study["solver_threads"] = _positive_integer(solver_threads, "solver_threads")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    cases = _select(study["cases"], case_ids, "cases")
    windows = _select(study["windows"], window_ids, "windows")
    available_farms = study["farms"] if farms is None else study["farms"] + study.get("additional_farms", [])
    selected_farms = _select([{"id": str(farm)} for farm in available_farms], farms, "farms")
    output = Path(study["output_root"])
    output = output if output.is_absolute() else root / output
    batch = output / study["run_id"]
    jobs = []
    for window in windows:
        for selected_farm in selected_farms:
            farm = selected_farm["id"]
            for case in cases:
                job_id = f"{window['id']}__farm_{farm}__{case['id']}"
                overrides = merge_mapping(window.get("overrides", {}), case.get("overrides", {}))
                forced = {"farm_id": int(farm) if farm.isdigit() else farm,
                    "_project_root": str(root), "energy_research_study": {},
                    "continuous_optimization": {"start_date": window["start_date"], "end_date": window["end_date"],
                        "output_root": str(batch / "runs"), "run_id": job_id, "resume": False},
                    "realtime_mpc_comparison": {"mode": "continuous", "check_only": False, "frozen_source": None},
                    "solver": {"threads": study.get("solver_threads", 1)},
                    "experiment_outputs": {"overwrite_existing": False}}
                overrides = merge_mapping(overrides, forced)
                config = resolve_config(merge_mapping(raw, overrides))
                # Derive each scaling from this farm's fresh defaults, then reload
                # so every dependent typed reference sees the scaled numeric value.
                for name, multiplier in case.get("asset_scales", {}).items():
                    section, field = _SCALES[name]
                    overrides.setdefault(section, {})[field] = config[section][field] * multiplier
                if case.get("asset_scales"):
                    config = resolve_config(merge_mapping(raw, overrides))
                config.pop("energy_research_study", None)
                validate_mpc_study(config)
                controllers = config["realtime_mpc_comparison"]["controllers"]
                targets = [controller for controller in controllers if controller != "track_committed_grid"]
                if len(targets) != 1:
                    raise ValueError("Each research case must compare one controller with track_committed_grid")
                jobs.append({"job_id": job_id, "case_id": case["id"], "family": case["family"],
                    "baseline_case": case.get("baseline_case"), "farm_id": farm, "window_id": window["id"],
                    "window_role": window.get("role", "retrospective_development"),
                    "controller": targets[0], "config": config})
    effective_workers = min(study.get("workers", 42), len(jobs))
    native_budget = effective_workers * max(study.get("native_threads", 1), study.get("solver_threads", 1))
    if native_budget > 48:
        raise ValueError("Selected research workers and per-worker threads exceed the 48-core machine budget")
    return study, batch, jobs


def research_plan(path, **selection):
    """Count actual DST-aware work and expose missing inputs without fitting or solving."""
    study, destination, jobs = expand_research_jobs(path, **selection)
    rows = []
    checksums = {}
    for job in jobs:
        config = job["config"]
        settings = config["continuous_optimization"]
        calendar = AuctionCalendar.from_config(config)
        days = [stamp.date().isoformat() for stamp in pd.date_range(settings["start_date"], settings["end_date"], freq="D")]
        if not days:
            raise ValueError("Research date window is empty")
        intervals = sum(len(calendar.delivery_grid(day).to_domain().interval_ids) for day in days)
        active = set(config["gme_oracle_experiment"].get("participating_auctions") or ["DA", "IDA1", "IDA2", "IDA3"])
        auctions = sum(event.available and event.auction in active for day in days for event in calendar.schedule(day))
        requirements = _artifact_requirements(config, checksums)
        rows.append({key: value for key, value in job.items() if key != "config"} | {
            "days": days, "mode": "continuous", "controllers": config["realtime_mpc_comparison"]["controllers"],
            "physical_intervals_per_controller": intervals,
            "mpc_decisions": intervals * sum(controller in MPC_POLICIES for controller in config["realtime_mpc_comparison"]["controllers"]),
            "auction_submission_solves": auctions * len(config["realtime_mpc_comparison"]["controllers"]),
            "auction_price_information": config["gme_oracle_experiment"].get("price_information"),
            "current_pv_information": config.get("realtime_mpc", {}).get("current_pv_information", "forecast"),
            "mpc_solve_time_limit_seconds": config.get("realtime_mpc", {}).get("solver_time_limit_seconds", 10),
            "auction_solve_time_limit_seconds": config["solver"]["time_limit_seconds"],
            "requirements": requirements, "missing_requirements": [row["role"] for row in requirements if not row["present"]],
            "price_forecast": config.get("energy_price_forecast", {}), "information": config.get("research_information", {}),
            "battery": config["battery"], "poi": {name: config["plant"][name] for name in ("poi_import_mw", "poi_export_mw")}})
    return {"status": "plan_only", "mode": "continuous", "run_id": study["run_id"], "output_dir": str(destination),
        "job_count": len(rows), "case_count": len({job["case_id"] for job in jobs}),
        "farm_count": len({job["farm_id"] for job in jobs}), "window_count": len({job["window_id"] for job in jobs}),
        "mpc_decisions": sum(row["mpc_decisions"] for row in rows),
        "auction_submission_solves": sum(row["auction_submission_solves"] for row in rows),
        "physical_controller_trajectories": sum(len(row["controllers"]) for row in rows),
        "workers": min(study.get("workers", 42), len(jobs)), "requested_workers": study.get("workers", 42),
        "native_threads_per_worker": study.get("native_threads", 1),
        "solver_threads_per_worker": study.get("solver_threads", 1), "gpus": 0,
        "configured_cpu_threads": min(study.get("workers", 42), len(jobs)) * max(
            study.get("native_threads", 1), study.get("solver_threads", 1)),
        "missing_input_jobs": sum(bool(row["missing_requirements"]) for row in rows),
        "solves_started": 0, "models_fitted": 0,
        "count_scope": "nominal submissions and MPC decisions; fallbacks and solver retries are reported after execution",
        "jobs": rows}


def _worker(job, check_only):
    """Return every case failure explicitly while allowing independent jobs to finish."""
    config = deepcopy(job["config"])
    config["realtime_mpc_comparison"]["check_only"] = check_only
    result = {key: value for key, value in job.items() if key != "config"}
    try:
        result["result"] = run_realtime_mpc_experiment(config)
        result["status"] = result["result"]["status"]
    except Exception as error:
        result.update(status="failed", error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    return result


def _run_jobs(jobs, workers, native_threads, check_only, record):
    """Spawn bounded workers; interruption terminates active work and preserves manifests."""
    context = multiprocessing.get_context("spawn")
    # ProcessPoolExecutor reports dead workers instead of indefinitely awaiting
    # a task lost by an automatically replaced multiprocessing.Pool process.
    needs_torch = any(job["config"].get("pv", {}).get("forecast_replacement", {}).get("enabled", False)
        for job in jobs)
    pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context,
        initializer=configure_worker_threads, initargs=(native_threads, needs_torch))
    results = []
    try:
        pending = {pool.submit(_worker, job, check_only): job for job in jobs}
        for handle in as_completed(pending):
            try:
                value = handle.result()
            except Exception as error:
                value = {key: item for key, item in pending[handle].items() if key != "config"}
                value.update(status="failed", error=str(error), error_type=type(error).__name__)
            results.append(value)
            record(value)
        pool.shutdown(wait=True)
    except BaseException:
        # The project's current Python 3.14 exposes immediate worker termination.
        # Older interpreters can cancel queued work but cannot promise that an
        # already executing historical case stops immediately through this API.
        if hasattr(pool, "terminate_workers"):
            pool.terminate_workers()
        else:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    return results


def _reports(destination, results, jobs, study):
    """Compare each controller to its own continuous tracking baseline on common dates."""
    from ..reporting.research_study import (DEFAULT_METRICS, factorial_effects,
        forecast_flexibility_factorial, save_research_report)

    frames = []
    for result in results:
        if result["status"] != "complete":
            continue
        frame = pd.read_csv(Path(result["result"]["output_dir"]) / "paired_daily_metrics.csv")
        frame["case_id"] = result["case_id"]
        frame["farm_id"] = str(result["farm_id"])
        frame["window_id"] = result["window_id"]
        frames.append(frame)
    if not frames:
        return {"status": "no_completed_cases"}
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(destination / "daily_metrics.csv", index=False)
    settings = dict(study.get("reporting", {}))
    plots = settings.pop("plots", True)
    metrics = settings.pop("metrics", [metric for metric in DEFAULT_METRICS if metric in data])
    reports = []
    for case in sorted({job["case_id"] for job in jobs}):
        selected_jobs = [job for job in jobs if job["case_id"] == case]
        selected = data[data.case_id == case]
        if selected.empty:
            continue
        expected_days = sorted({day.date().isoformat() for job in selected_jobs for day in pd.date_range(
            job["config"]["continuous_optimization"]["start_date"], job["config"]["continuous_optimization"]["end_date"])})
        target = destination / "comparisons" / case
        save_research_report(selected, target, case, metrics=metrics, plots=plots,
            baseline_controller="track_committed_grid", expected_farms=sorted({job["farm_id"] for job in selected_jobs}),
            expected_days=expected_days, **settings)
        reports.append(str(target))
    # Cross-case contrasts retain the named target controller from each case.
    # These are full continuous-policy effects, since auction orders may differ.
    target_controllers = {job["case_id"]: job["controller"] for job in jobs}
    target_rows = data[data.controller == data.case_id.map(target_controllers)]
    contrast_reports = []
    for case in study["cases"]:
        reference = case.get("baseline_case")
        if case["id"] not in target_controllers or reference not in target_controllers:
            continue
        selected = target_rows[target_rows.case_id.isin([case["id"], reference])]
        if not {case["id"], reference}.issubset(set(selected.case_id)):
            continue
        target = destination / "case_contrasts" / case["id"]
        save_research_report(selected, target, reference, metrics=metrics, plots=plots,
            baseline_controller=target_controllers[reference], **settings)
        contrast_reports.append(str(target))
    factorial = study.get("factorial_cases", {})
    names = {"updates_on_all_auctions", "updates_off_all_auctions", "updates_on_da_only", "updates_off_da_only"}
    factorial_status = "not_all_cells_selected"
    if factorial and set(factorial) != names:
        raise ValueError("factorial_cases must name the four information/access cells")
    if factorial and set(factorial.values()).issubset(set(target_rows.case_id)):
        mapping = {(True, True): factorial["updates_on_all_auctions"],
            (False, True): factorial["updates_off_all_auctions"],
            (True, False): factorial["updates_on_da_only"],
            (False, False): factorial["updates_off_da_only"]}
        controls = {}
        for case in mapping.values():
            controls[case] = {job["farm_id"] + ":" + job["window_id"]: {
                name: job["config"][name] for name in ("battery", "plant", "state", "degradation", "realtime_mpc")}
                for job in jobs if job["case_id"] == case}
        factorial_settings = {key: value for key, value in settings.items() if key != "cvar_alpha"}
        value = factorial_effects(target_rows, mapping, controller=target_controllers[mapping[True, True]],
            physical_controls=controls, **factorial_settings)
        _write_json(destination / "information_auction_factorial.json", value)
        factorial_status = value.get("status", "complete")
    flexibility_statuses = {}
    for name, declaration in study.get("forecast_flexibility_factorials", {}).items():
        _identifier(name, "Forecast/flexibility factorial ID")
        fields = {"raw_base", "calibrated_base", "raw_modified", "calibrated_modified", "changed_fields"}
        if not isinstance(declaration, dict) or set(declaration) != fields:
            raise ValueError("Forecast/flexibility factorial must declare four cases and changed_fields")
        mapping = {(False, False): declaration["raw_base"], (True, False): declaration["calibrated_base"],
            (False, True): declaration["raw_modified"], (True, True): declaration["calibrated_modified"]}
        if not set(mapping.values()).issubset(set(target_rows.case_id)):
            flexibility_statuses[name] = "not_all_cells_selected"
            continue
        controllers = {target_controllers[case] for case in mapping.values()}
        if len(controllers) != 1:
            raise ValueError("Forecast/flexibility cells require the same controller")
        controls = {case: {job["farm_id"] + ":" + job["window_id"]: {
            key: job["config"][key] for key in ("battery", "plant", "state", "degradation", "realtime_mpc")}
            for job in jobs if job["case_id"] == case} for case in mapping.values()}
        factorial_settings = {key: value for key, value in settings.items() if key != "cvar_alpha"}
        value = forecast_flexibility_factorial(target_rows, mapping, controller=controllers.pop(),
            physical_controls=controls, changed_fields=declaration["changed_fields"], **factorial_settings)
        _write_json(destination / f"forecast_flexibility_{name}.json", value)
        flexibility_statuses[name] = value["status"]
    return {"status": "complete", "comparison_dirs": reports, "case_contrast_dirs": contrast_reports,
        "factorial_status": factorial_status, "forecast_flexibility_statuses": flexibility_statuses}


def run_research_study(path, *, check_only=False, case_ids=None, farms=None, window_ids=None,
                       project_root=None, run_id=None, workers=None, solver_threads=None, progress=None):
    """Execute only explicit selections; --check-only performs no training or solves."""
    selection = dict(case_ids=case_ids, farms=farms, window_ids=window_ids, project_root=project_root,
        run_id=run_id, workers=workers, solver_threads=solver_threads)
    plan = research_plan(path, **selection)
    study, destination, jobs = expand_research_jobs(path, **selection)
    if plan["missing_input_jobs"]:
        return {"status": "missing_inputs", "plan": plan, "solves_started": 0, "models_fitted": 0}
    if check_only:
        results = _run_jobs(jobs, study.get("workers", 42), study.get("native_threads", 1), True,
            lambda result: progress(f"Preflight {result['job_id']}: {result['status']}") if progress else None)
        return {"status": "preflight_passed" if all(row["status"] == "preflight_passed" for row in results) else "failed",
            "results": results, "plan": plan, "solves_started": 0, "models_fitted": 0}
    # Atomic creation plus an advisory lease prevent two launchers from replacing
    # or sharing this batch. A changed selection requires a new run identifier.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    import fcntl

    with (destination / "run.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_root = Path(__file__).resolve().parents[1]
        manifest = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": plan, "config_sha256": _sha256(path),
            "source_checksums": {str(item.relative_to(source_root)): _sha256(item)
                for item in sorted(source_root.rglob("*.py"))}, "results": []}
        _write_json(destination / "manifest.json", manifest)
        for job in jobs:
            _write_json(destination / f"config__{job['job_id']}.json", job["config"])
        def record(result):
            manifest["results"].append(result)
            _write_json(destination / "manifest.json", manifest)
            if progress:
                progress(f"Research {result['job_id']}: {result['status']}")
        try:
            results = _run_jobs(jobs, study.get("workers", 42), study.get("native_threads", 1), False, record)
            manifest["status"] = "complete" if all(row["status"] == "complete" for row in results) else "failed"
            manifest["reports"] = _reports(destination, results, jobs, study)
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(destination / "manifest.json", manifest)
    return {"status": manifest["status"], "output_dir": str(destination), "results": manifest["results"],
        "reports": manifest.get("reports")}
