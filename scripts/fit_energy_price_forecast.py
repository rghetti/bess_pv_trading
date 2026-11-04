"""Fit and evaluate declared chronological price models from the local DuckDB catalog."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.energy_prices import (
    CausalPriceForecaster, PriceForecastSettings, _day_grid, historical_decision_times,
    load_price_history, prepare_exogenous_features, prepare_price_history, price_forecast_scores,
)
from bess_pv_trading.forecasting.energy_price_features import load_catalog_features
from bess_pv_trading.parallel import configure_worker_threads


def _jobs(settings):
    """Expand only explicitly selected fold/model/target combinations."""
    rows = []
    models = {item["id"]: item for item in settings["models"]}
    for fold, split in settings["folds"].items():
        target_names = split.get("targets", list(settings["targets"]))
        model_names = split.get("models", list(models))
        if (set(target_names) - settings["targets"].keys() or set(model_names) - models.keys()
                or len(set(target_names)) != len(target_names) or len(set(model_names)) != len(model_names)):
            raise ValueError("Fold target/model selections must be unique configured names")
        for name in model_names:
            for target in target_names:
                rows.append((fold, name, target))
    return rows


def training_plan(config):
    """Describe all fitting choices and source assumptions before opening the database."""
    import pandas as pd

    settings = config["energy_price_training"]
    names = [model["id"] for model in settings["models"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Price model IDs must be unique and nonempty")
    if not settings["targets"] or not settings["folds"]:
        raise ValueError("Price training requires named targets and chronological folds")
    if set(model["feature_set"] for model in settings["models"]) - settings["feature_sets"].keys():
        raise ValueError("Unknown price model feature set")
    for target in settings["targets"].values():
        values = {**settings["fitting"], **target.get("settings", {})}
        model_settings = PriceForecastSettings(**values)
        for split in settings["folds"].values():
            first = _day_grid(split["evaluation_start"], model_settings.interval_minutes, model_settings.timezone,
                              model_settings.delivery_start_hour, model_settings.delivery_end_hour)
            fit_cutoff = pd.Timestamp(split["fit_cutoff"])
            if fit_cutoff.tzinfo is None or fit_cutoff > historical_decision_times(first, model_settings).min():
                raise ValueError("Price fitting cutoff must be aware and no later than the first evaluation decision")
            if pd.Timestamp(split["evaluation_start"]) >= pd.Timestamp(split["evaluation_end_exclusive"]):
                raise ValueError("Price evaluation start must precede exclusive end")
    for name, minimum in (("native_threads", 1), ("scenario_count", 1), ("plot_days", 0), ("workers", 1)):
        value = settings.get(name, 2 if name == "plot_days" else 1 if name == "workers" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Price training {name} must be an integer >= {minimum}")
    return {"status": "plan_only", "database": settings["database"], "targets": settings["targets"],
        "folds": settings["folds"], "models": settings["models"], "feature_sets": settings["feature_sets"],
        "fitting": settings["fitting"], "fit_count": len(_jobs(settings)),
        "training_loss": "standardized_ridge_squared_error_or_no_fit_seasonal_persistence",
        "calibration": "chronological_tail_predicted_by_model_fit_before_its_first_decision",
        "evaluation": "frozen_fit_per_fold_with_causally_filtered_updated_price_lags",
        "metrics": ["point_MAE", "empirical_CRPS", "coverage_and_exclusions"],
        "model_selection": "none_report_all_predeclared_models",
        "scenario_count": settings["scenario_count"], "native_threads": settings["native_threads"],
        "workers": settings.get("workers", 1), "gpus": 0,
        "outputs": ["checksummed_model_JSON", "history_Parquet", "exogenous_Parquet", "daily_predictions",
                    "daily_metrics", "summary", "representative_day_plots", "snapshot_provenance"]}


def _load_inputs(settings):
    """Read each target and common feature set once before any fitting loop."""
    import pandas as pd

    histories, evidence = {}, {}
    for name, target in settings["targets"].items():
        print(f"Reading pinned {name} history", file=sys.stderr, flush=True)
        source = dict(target["source"])
        histories[name], evidence[name] = load_price_history(settings["database"],
            start=target["history_start"], end=settings["history_end_exclusive"],
            timezone=settings["timezone"], interval_minutes=settings["interval_minutes"], **source)
    selections = {}
    for group in settings["feature_sets"].values():
        for selection in group:
            existing = selections.setdefault(selection["name"], selection)
            if existing != selection:
                raise ValueError("A feature name has different source definitions across feature sets")
    exogenous = None
    exogenous_evidence = None
    if selections:
        print("Reading projected fundamentals and optional weather", file=sys.stderr, flush=True)
        first = min(pd.Timestamp(item["history_start"]) for item in settings["targets"].values())
        exogenous, exogenous_evidence = load_catalog_features(settings["database"], list(selections.values()),
            start=first, end=settings["history_end_exclusive"], timezone=settings["timezone"],
            interval_minutes=settings["interval_minutes"], database_threads=settings["native_threads"])
    return histories, evidence, exogenous, exogenous_evidence


def _preflight(settings, histories, exogenous, exogenous_evidence):
    """Report exact retained row counts and chronological split support without fitting."""
    import pandas as pd

    rows = []
    for name, history in histories.items():
        for fold, split in settings["folds"].items():
            cutoff = pd.Timestamp(split["fit_cutoff"])
            first = pd.Timestamp(split["evaluation_start"], tz=settings["timezone"])
            stop = pd.Timestamp(split["evaluation_end_exclusive"], tz=settings["timezone"])
            train = (history.available_at <= cutoff) & (history.product_end <= cutoff)
            test = (history.product_start >= first) & (history.product_start < stop)
            rows.append({"target": name, "fold": fold, "retained_history_rows": len(history),
                "training_labels_available": int(train.sum()), "evaluation_labels": int(test.sum()),
                "observed_start": history.product_start.min().isoformat(),
                "observed_end": history.product_end.max().isoformat()})
    return {"status": "preflight_complete", "targets": rows,
            "exogenous_rows": 0 if exogenous is None else len(exogenous),
            "features": exogenous_evidence}


def _evaluate(model, history, split, settings, exogenous):
    """Score complete local days and report every missing-price or missing-feature exclusion."""
    import numpy as np
    import pandas as pd

    history_index = history.set_index("product_start")
    prepared_history = prepare_price_history(history, model.settings.timezone)
    records, daily, exclusions = [], [], []
    days = pd.date_range(split["evaluation_start"], split["evaluation_end_exclusive"], freq="D", inclusive="left")
    for day in days:
        starts = _day_grid(day.date().isoformat(), model.settings.interval_minutes, model.settings.timezone,
                           model.settings.delivery_start_hour, model.settings.delivery_end_hour)
        actual = history_index.reindex(starts).price_eur_per_mwh.to_numpy(float)
        if not np.isfinite(actual).all():
            exclusions.append({"day": str(day.date()), "reason": "incomplete_observed_target_day",
                               "missing_intervals": int((~np.isfinite(actual)).sum())})
            continue
        cutoff = historical_decision_times(starts, model.settings).min()
        # One fixed daily decision is explicit even for a rolling-settlement target.
        try:
            prediction = model.predict(starts, cutoff=cutoff, scenario_count=settings["scenario_count"],
                                       history=prepared_history, exogenous=exogenous)
        except ValueError as error:
            exclusions.append({"day": str(day.date()), "reason": str(error)})
            continue
        scores = price_forecast_scores(actual, prediction.scenarios_eur_per_mwh)
        scores["point_mae_eur_per_mwh"] = float(np.mean(np.abs(prediction.point_eur_per_mwh - actual)))
        daily.append({"day": str(day.date()), **scores})
        quantiles = np.quantile(prediction.scenarios_eur_per_mwh, [0.1, 0.5, 0.9], axis=0)
        records.append(pd.DataFrame({"product_start": starts, "decision_cutoff": cutoff, "actual": actual,
            "point": prediction.point_eur_per_mwh, "p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]}))
    return (pd.concat(records, ignore_index=True) if records else pd.DataFrame(), pd.DataFrame(daily), exclusions)


def _plots(frame, destination, count, timezone):
    """Group forecast ribbons and observations by representative delivery date."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame.empty or count == 0:
        return
    local = frame.product_start.dt.tz_convert(timezone)
    for day in list(dict.fromkeys(local.dt.date))[:count]:
        selected = frame.loc[local.dt.date == day]
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.fill_between(selected.product_start, selected.p10, selected.p90, alpha=0.2, label="empirical 10–90%")
        axis.plot(selected.product_start, selected.actual, label="observed")
        axis.plot(selected.product_start, selected.point, label="conditional point forecast")
        axis.set(ylabel="EUR/MWh", title=f"{day}: forecasts from declared daily cutoff", xlabel="delivery time (UTC)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / f"prices_{day}.png", dpi=140)
        plt.close(figure)


def write_training_report(destination):
    """Compare models on their common daily support without selecting a winner."""
    import pandas as pd

    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    paired, summaries = [], []
    for fold, split in manifest["plan"]["folds"].items():
        expected_models = split.get("models", [item["id"] for item in manifest["plan"]["models"]])
        for target in split.get("targets", manifest["plan"]["targets"]):
            tables = {}
            for model in expected_models:
                path = destination / fold / model / target / "daily_scores.csv"
                if path.exists() and path.stat().st_size > 1:
                    tables[model] = pd.read_csv(path)
            common = (set.intersection(*(set(frame.day) for frame in tables.values()))
                      if len(tables) == len(expected_models) else set())
            for model, frame in tables.items():
                selected = frame.loc[frame.day.isin(common)].copy()
                selected["fold"], selected["target"], selected["model"] = fold, target, model
                paired.append(selected)
                # Pool absolute errors and absolute observations across matched
                # intervals; unlike MAPE, individual zero prices do not explode.
                prediction_path = destination / fold / model / target / "predictions.csv"
                relative = None
                mean_abs_price = None
                if common and prediction_path.exists() and prediction_path.stat().st_size > 1:
                    predictions = pd.read_csv(prediction_path, parse_dates=["product_start"])
                    timezone = manifest["plan"].get("fitting", {}).get("timezone", "Europe/Rome")
                    local_days = predictions.product_start.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
                    matched = predictions.loc[local_days.isin(common)]
                    denominator = matched.actual.abs().sum()
                    if len(matched) and denominator > 0:
                        relative = float(100 * (matched.point - matched.actual).abs().sum() / denominator)
                        mean_abs_price = float(matched.actual.abs().mean())
                summaries.append({"fold": fold, "target": target, "model": model,
                    "paired_days": len(selected),
                    "point_mae_eur_per_mwh": None if selected.empty else float(selected.point_mae_eur_per_mwh.mean()),
                    "crps_eur_per_mwh": None if selected.empty else float(selected.crps_eur_per_mwh.mean()),
                    "wape_percent": relative, "mean_absolute_actual_eur_per_mwh": mean_abs_price})
    if paired:
        pd.concat(paired, ignore_index=True).to_csv(destination / "paired_daily_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(destination / "paired_scores.csv", index=False)
    lines = ["# Chronological price forecasting study", "",
        "These are retrospective comparisons under explicitly configured publication assumptions. "
        "Training labels and feature vintages remain separated from evaluation by the declared cutoff. "
        "Current snapshots do not reconstruct every historical revision.", "",
        "The ridge objective is squared prediction error plus a standardized coefficient penalty. "
        "Evaluation reports point MAE and empirical CRPS separately from downstream trading profit. "
        "No hyperparameter search or winner selection is performed.", "",
        "The table uses only delivery dates scored by every declared model for that market and fold. "
        "Per-model exclusions remain in each output folder; gaps are never replaced by zero. "
        "WAPE is 100 times the sum of absolute point errors divided by the sum of absolute observed prices "
        "on those days; it is undefined if the denominator is zero. It is not mean individual percentage error.", "",
        "| Fold | Market | Model | Common days | Point MAE (EUR/MWh) | WAPE (%) | CRPS (EUR/MWh) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        mae = "unavailable" if row["point_mae_eur_per_mwh"] is None else f"{row['point_mae_eur_per_mwh']:.3f}"
        crps = "unavailable" if row["crps_eur_per_mwh"] is None else f"{row['crps_eur_per_mwh']:.3f}"
        relative = "unavailable" if row["wape_percent"] is None else f"{row['wape_percent']:.2f}"
        lines.append(f"| {row['fold']} | {row['target']} | {row['model']} | {row['paired_days']} | {mae} | {relative} | {crps} |")
    lines.extend(["", "The current folds cover their configured months only. "
        "A future complete-2026 ensemble archive can be evaluated by adding chronological folds and explicit input coverage. "
        "The present partial-year results do not establish annual performance.", ""])
    (destination / "summary.md").write_text("\n".join(lines))
    return summaries


def _fit_job(job, settings, histories, history_evidence, prepared_features, exogenous_evidence, destination):
    """Write one isolated model folder and return only its small outcome record."""
    fold, model_id, target = job
    split = settings["folds"][fold]
    model_spec = next(item for item in settings["models"] if item["id"] == model_id)
    target_spec = settings["targets"][target]
    features = settings["feature_sets"][model_spec["feature_set"]]
    output = Path(destination) / fold / model_id / target
    output.mkdir(parents=True, exist_ok=False)
    selected = {**settings["fitting"], **target_spec.get("settings", {}), "method": model_spec["method"],
                "exogenous_features": tuple(item["name"] for item in features)}
    print(f"Fitting {fold}/{model_id}/{target}", file=sys.stderr, flush=True)
    try:
        model = CausalPriceForecaster.fit(histories[target], cutoff=split["fit_cutoff"],
            settings=PriceForecastSettings(**selected), exogenous=prepared_features,
            provenance={"target": target, "source": history_evidence[target],
                        "features": exogenous_evidence if features else None, "split": split,
                        "study_information": "retrospective_with_explicit_availability_assumptions"})
        model.save(output / "model.json")
        predictions, daily, exclusions = _evaluate(model, histories[target], split, settings, prepared_features)
        predictions.to_csv(output / "predictions.csv", index=False)
        daily.to_csv(output / "daily_scores.csv", index=False)
        (output / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
        _plots(predictions, output, settings.get("plot_days", 2), settings["timezone"])
        summary = {"fold": fold, "model": model_id, "target": target,
            "artifact": str(output / "model.json"), "artifact_sha256": model.artifact_sha256,
            "evaluated_days": len(daily), "excluded_days": len(exclusions),
            "point_mae_eur_per_mwh": None if daily.empty else float(daily.point_mae_eur_per_mwh.mean()),
            "crps_eur_per_mwh": None if daily.empty else float(daily.crps_eur_per_mwh.mean())}
        return {"summary": summary, "failure": {**summary, "reason": "No complete evaluation days"} if daily.empty else None}
    except Exception as error:
        # Keep every failed job explicit while unrelated workers finish their declared work.
        failure = {"fold": fold, "model": model_id, "target": target,
                   "reason": f"{type(error).__name__}: {error}"}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        return {"summary": None, "failure": failure}


_WORKER_INPUTS = None


def _initialize_price_worker(settings, destination, history_paths, history_evidence, exogenous_path, exogenous_evidence):
    """Read immutable local Parquet once per spawned worker; never open DuckDB here."""
    import pandas as pd

    global _WORKER_INPUTS
    configure_worker_threads(settings["native_threads"], configure_torch=False)
    os.environ["MPLCONFIGDIR"] = str(Path(destination) / "matplotlib_cache" / f"worker_{os.getpid()}")
    histories = {name: pd.read_parquet(path) for name, path in history_paths.items()}
    features = None if exogenous_path is None else prepare_exogenous_features(pd.read_parquet(exogenous_path))
    _WORKER_INPUTS = (settings, histories, history_evidence, features, exogenous_evidence, destination)


def _price_worker(job):
    """Return an isolated fit result without serializing fitted models between processes."""
    if _WORKER_INPUTS is None:
        raise RuntimeError("Price worker was not initialized")
    return _fit_job(job, *_WORKER_INPUTS)


def _save_manifest(path, manifest):
    """Publish parent-owned progress atomically, including failure evidence."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train_from_config(config, *, check_only=False):
    """Run one explicit offline fitting batch; never alter the source catalog."""
    import pandas as pd

    settings = config["energy_price_training"]
    plan = training_plan(config)
    configure_worker_threads(settings["native_threads"], configure_torch=False)
    histories, history_evidence, exogenous, exogenous_evidence = _load_inputs(settings)
    preflight = _preflight(settings, histories, exogenous, exogenous_evidence)
    if check_only:
        return preflight
    destination = Path(settings["output_root"]) / settings["run_id"]
    if not destination.is_absolute():
        destination = Path(config.get("_project_root", ROOT)) / destination
    destination.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(destination / "matplotlib_cache"))
    (destination / "history").mkdir()
    history_paths = {}
    history_checksums = {}
    for name, history in histories.items():
        path = destination / "history" / f"{name}.parquet"
        history.to_parquet(path, index=False)
        history_paths[name] = str(path)
        history_checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    prepared_features = None
    exogenous_path = None
    exogenous_checksum = None
    if exogenous is not None:
        path = destination / "exogenous.parquet"
        exogenous.to_parquet(path, index=False)
        exogenous_path = str(path)
        exogenous_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared_features = prepare_exogenous_features(exogenous)
    manifest = {"plan": plan, "preflight": preflight, "history_paths": history_paths,
                "history_sha256": history_checksums, "exogenous_path": exogenous_path,
                "exogenous_sha256": exogenous_checksum, "source_provenance": history_evidence,
                "feature_provenance": exogenous_evidence, "models": [], "failures": []}
    def record(outcome):
        # Only the parent process updates shared progress and aggregate results.
        if outcome["summary"] is not None:
            manifest["models"].append(outcome["summary"])
        if outcome["failure"] is not None:
            manifest["failures"].append(outcome["failure"])
        manifest["status"] = "running"
        _save_manifest(destination / "manifest.json", manifest)

    jobs = _jobs(settings)
    workers = min(settings.get("workers", 1), len(jobs))
    if workers == 1:
        for job in jobs:
            record(_fit_job(job, settings, histories, history_evidence,
                            prepared_features, exogenous_evidence, destination))
    else:
        # Spawn avoids inheriting BLAS pools or source-database connections.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_price_worker,
                initargs=(settings, destination, history_paths, history_evidence,
                          exogenous_path, exogenous_evidence)) as executor:
            pending = {executor.submit(_price_worker, job): job for job in jobs}
            for future in as_completed(pending):
                record(future.result())
    manifest["models"].sort(key=lambda row: (row["fold"], row["model"], row["target"]))
    manifest["status"] = "complete" if not manifest["failures"] else "partial"
    manifest["output_directory"] = str(destination)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(manifest["models"]).to_csv(destination / "scores.csv", index=False)
    write_training_report(destination)
    return manifest


def main(argv=None):
    """Default to a no-I/O plan; explicit flags select preflight or fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/price_forecast_training.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    config["_project_root"] = str(ROOT)
    plan = training_plan(config)
    print(json.dumps(plan, indent=2))
    if not arguments.run and not arguments.check_only:
        return 0
    result = train_from_config(config, check_only=arguments.check_only)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
