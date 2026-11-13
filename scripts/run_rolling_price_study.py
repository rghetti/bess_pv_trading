"""Run a checkpointed rolling auction-price replay from a pinned snapshot."""

from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bess_rolling_price_matplotlib")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import product
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import tempfile

import numpy as np
import pandas as pd
import sklearn

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.rolling_price_data import build_examples
from bess_pv_trading.forecasting.studies.rolling_price_models import InsufficientHistory, RollingModelEngine
from bess_pv_trading.forecasting.studies.rolling_price_reporting import rolling_price_report, compare_october_pointwise


EXAMPLES = None
BY_DAY = None
STUDY = None
DESTINATION = None
STOP_EVENT = None
MARKETS = ("DA", "IDA1", "IDA2", "IDA3")


def digest(path):
    """Hash a file in bounded chunks for restart and provenance checks."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    """Atomically install a complete metadata or daily checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _published_reference(day_examples, gate, cutoff, starts, da_prediction):
    """Align the newest earlier published price, or use the DA prediction."""
    requested = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    for market in reversed(MARKETS[:MARKETS.index(gate)]):
        source = day_examples.get((market, "DA" if market == "DA" else market))
        if source is None or pd.Timestamp(source["target_publication_at"]) > cutoff:
            continue
        if source.get("metadata", {}).get("cancelled", False):
            continue
        values = np.asarray(source["target_values"], dtype=float)
        source_starts = pd.DatetimeIndex(pd.to_datetime(source["product_starts_utc"], utc=True))
        index = source_starts.get_indexer(requested)
        if (index >= 0).all() and np.isfinite(values[index]).all():
            return values[index], market, str(source["target_publication_at"])
    if da_prediction is not None:
        index = pd.DatetimeIndex(pd.to_datetime(da_prediction["product_starts_utc"], utc=True)).get_indexer(requested)
        if (index >= 0).all():
            return np.asarray(da_prediction["prediction"])[index], "DA_forecast", da_prediction["forecast_issue_at"]
    return None, None, None


def _job(spec):
    """Replay one independent model setting in chronological order."""
    variant, family, window, target = spec
    job_name = f"{variant}__{family}__{window}d__{target}"
    destination = DESTINATION / "jobs" / job_name
    destination.mkdir(parents=True, exist_ok=True)
    options = {**STUDY, "variant": variant, "family": family, "window_days": window, "ida_target": target}
    engine = RollingModelEngine(EXAMPLES, options)
    started = datetime.now(timezone.utc)
    completed = 0
    for day in sorted(BY_DAY):
        # Stop only between atomic daily checkpoints, leaving resumable files intact.
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            break
        if not STUDY["start"] <= day < STUDY["end_exclusive"]:
            continue
        marker = destination / f"{day}.json"
        parquet = destination / f"{day}.parquet"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if not parquet.is_file() or digest(parquet) != saved["parquet_sha256"]:
                raise ValueError(f"Checkpoint is incomplete or changed: {marker}")
            completed += 1
            continue
        examples = BY_DAY[day]
        da_example = examples.get(("DA", "DA"))
        if da_example is None:
            raise ValueError(f"Missing DA decision example for {day}")
        fitted = engine.fit_delivery_day(day, da_example["decision_at"])
        da_forecast = None
        stored = {}
        outcomes = []
        errors = []
        # Earlier actual prices are read only after their recorded publication.
        for gate_index, gate in enumerate(MARKETS):
            gate_example = examples.get((gate, "DA" if gate == "DA" else gate))
            if gate_example is None:
                continue
            cutoff = pd.Timestamp(gate_example["decision_at"])
            for market in MARKETS[gate_index:]:
                target_example = examples.get((market, "DA" if market == "DA" else market))
                if target_example is None:
                    continue
                starts = target_example["product_starts_utc"]
                baseline, baseline_source, baseline_time = _published_reference(examples, gate, cutoff, starts, da_forecast)
                if gate == "DA" and market == "DA":
                    baseline = np.asarray(da_example["price_reference"], dtype=float)
                    baseline_source, baseline_time = "previous_DA", da_example["decision_at"]
                if gate == "DA" or gate == market:
                    example = examples.get((market, gate))
                    if example is None:
                        continue
                    if "weather" in engine.blocks and not example.get("metadata", {}).get("weather_usable", False):
                        errors.append({"market": market, "gate": gate, "reason": "weather_coverage_incomplete"})
                        continue
                    da_values = baseline if gate == "DA" and market != "DA" else None
                    if gate == "DA" and market != "DA" and da_values is None:
                        errors.append({"market": market, "gate": gate, "reason": "DA_forecast_unavailable"})
                        continue
                    if gate != "DA" and target == "spread":
                        reference = np.asarray(example["price_reference"], dtype=float)
                        if len(reference) != len(starts) or not np.isfinite(reference).all():
                            errors.append({"market": market, "gate": gate, "reason": "published_price_reference_incomplete"})
                            continue
                    try:
                        prediction_example = {key: value for key, value in example.items()
                                              if key not in {"target_values", "target_publication_at"}}
                        result = engine.predict(fitted, prediction_example, da_prediction=da_values)
                    except InsufficientHistory as error:
                        errors.append({"market": market, "gate": gate, "reason": str(error)})
                        continue
                    if gate == "DA":
                        stored[market] = result
                        if market == "DA":
                            da_forecast = result
                    prediction = np.asarray(result["prediction"], dtype=float)
                    method = "rolling_model"
                    coverage = result["metadata"].get("feature_block_coverage", {})
                    inputs_complete = all(coverage.get(name, {}).get("complete", False)
                                          for name in engine.blocks if name != "core")
                    detail = {"market": market, "gate": gate, "method": method,
                              "model": result["model"], "metadata": result["metadata"]}
                else:
                    prior = stored.get(market)
                    prior_time = pd.Timestamp(prior["forecast_issue_at"]) if prior else None
                    if baseline is not None and baseline_time is not None and (prior_time is None or pd.Timestamp(baseline_time) > prior_time):
                        prediction, method = baseline, "carry_published_clearing"
                        inputs_complete = True
                    elif prior is not None:
                        index = pd.DatetimeIndex(pd.to_datetime(prior["product_starts_utc"], utc=True)).get_indexer(pd.to_datetime(starts, utc=True))
                        if (index < 0).any():
                            errors.append({"market": market, "gate": gate, "reason": "fallback_product_mismatch"})
                            continue
                        prediction, method = np.asarray(prior["prediction"])[index], "carry_da_gate_forecast"
                        coverage = prior["metadata"].get("feature_block_coverage", {})
                        inputs_complete = all(coverage.get(name, {}).get("complete", False)
                                              for name in engine.blocks if name != "core")
                    else:
                        errors.append({"market": market, "gate": gate, "reason": "no_published_or_forecast_fallback"})
                        continue
                    detail = {"market": market, "gate": gate, "method": method,
                              "reference_source": baseline_source, "reference_available_at": baseline_time}
                if baseline is None or len(baseline) != len(starts):
                    baseline = np.full(len(starts), np.nan)
                actual = np.asarray(target_example["target_values"], dtype=float)
                if len(actual) != len(starts) or len(prediction) != len(starts):
                    raise ValueError(f"Product count mismatch for {day} {gate}/{market}")
                holiday = bool(target_example["metadata"].get("is_holiday", False))
                for stamp, truth, value, reference in zip(starts, actual, prediction, baseline):
                    outcomes.append({"delivery_day": day, "product_start": stamp, "auction": market, "gate": gate,
                        "variant": variant, "family": family, "window_days": window, "ida_target": target,
                        "actual": truth, "prediction": value, "reference": reference,
                        "method": method, "is_holiday": holiday, "feature_inputs_complete": inputs_complete,
                        "target_cancelled_notice": bool(target_example.get("metadata", {}).get("cancelled", False))})
                errors.append(detail)
        frame = pd.DataFrame(outcomes, columns=["delivery_day", "product_start", "auction", "gate", "variant", "family",
            "window_days", "ida_target", "actual", "prediction", "reference", "method", "is_holiday", "feature_inputs_complete",
            "target_cancelled_notice"])
        with tempfile.NamedTemporaryFile(dir=destination, suffix=".parquet", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, parquet)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(marker, {"delivery_day": day, "parquet_sha256": digest(parquet), "rows": len(frame),
            "model_unavailable": fitted.unavailable, "forecast_details": errors})
        completed += 1
        if completed == 1 or completed % 25 == 0:
            print(f"{job_name}: {completed} delivery days checkpointed; latest {day}", flush=True)
    files = sorted(destination.glob("????-??-??.parquet"))
    if files and not (STOP_EVENT is not None and STOP_EVENT.is_set()):
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        if not frame.empty:
            rolling_price_report(frame, destination / "metrics", timezone=STUDY["timezone"],
                                 evaluation_end_exclusive=STUDY["end_exclusive"])
    return {"job": job_name, "days": len(files), "interrupted": bool(STOP_EVENT and STOP_EVENT.is_set()),
            "seconds": (datetime.now(timezone.utc) - started).total_seconds()}


def main():
    """Plan without fitting, or run explicitly on the frozen source snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/rolling_price_study.yaml")
    parser.add_argument("--snapshot", required=True, help="Prepared merged input snapshot directory")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--pointwise-dir", default="outputs/price_staged_proxies_2026")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Inspect pinned gate and weather coverage without fitting")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    global EXAMPLES, BY_DAY, STUDY, DESTINATION, STOP_EVENT
    STUDY = load_config(args.config)["rolling_price_study"]
    DESTINATION = Path(args.output or STUDY["output_directory"]).resolve()
    snapshot = Path(args.snapshot).resolve()
    workers = args.workers if args.workers is not None else STUDY["workers"]
    if workers < 1 or workers > len(os.sched_getaffinity(0)):
        raise ValueError("Worker count must fit the available CPU affinity")
    variants, families = args.variants or list(STUDY["feature_blocks"]), args.families or STUDY["families"]
    if set(variants) - set(STUDY["feature_blocks"]) or set(families) - set(STUDY["families"]):
        raise ValueError("Unknown variant or model family")
    jobs = list(product(variants, families, STUDY["windows"], STUDY["ida_targets"]))
    if not (args.run or args.check_only):
        print(json.dumps({"action": "plan_only", "snapshot": str(snapshot), "output": str(DESTINATION),
                          "workers": workers, "jobs": len(jobs), "settings": STUDY}, indent=2))
        return
    if STUDY["end_exclusive"] > "2026-11-01":
        raise ValueError("November is reserved and cannot enter this development replay")
    snapshot_manifest = snapshot / "manifest.json"
    input_settings = load_config(STUDY["inputs_config"])["rolling_price_inputs"]
    if args.check_only:
        examples = build_examples(snapshot, input_settings)
        counts = {}
        for example in examples:
            day = str(example["delivery_day"])
            if not STUDY["start"] <= day < STUDY["end_exclusive"]:
                continue
            key = (day[:7], example["market"], example["gate"])
            item = counts.setdefault(key, {"examples": 0, "complete_target_days": 0,
                "cancelled_notices": 0, "complete_weather_days": 0})
            item["examples"] += 1
            item["complete_target_days"] += int(np.isfinite(np.asarray(example["target_values"], dtype=float)).all())
            item["cancelled_notices"] += int(example["metadata"].get("cancelled", False))
            item["complete_weather_days"] += int(example["metadata"].get("weather_usable", False))
        summary = [{"month": key[0], "auction": key[1], "gate": key[2], **value}
                   for key, value in sorted(counts.items())]
        print(json.dumps({"action": "check_only", "snapshot_sha256": digest(snapshot_manifest),
            "model_fits": 0, "coverage": summary}, indent=2), flush=True)
        return
    identity = {"snapshot_sha256": digest(snapshot_manifest), "config": STUDY, "input_config": input_settings,
        "effective_base_search": load_config(input_settings["base_search_config"])["price_feature_search"],
        "pointwise_sources": {path.name: digest(path) for path in
                              sorted(Path(args.pointwise_dir).glob("*_quarter_hourly__main.parquet"))},
        "implementation": {str(path): digest(path) for path in (Path(__file__),
            Path("src/forecasting/studies/rolling_price_data.py"), Path("src/forecasting/studies/rolling_price_models.py"),
            Path("src/forecasting/studies/rolling_price_reporting.py"), Path("src/forecasting/studies/price_feature_data.py"),
            Path("src/forecasting/studies/price_full_year_features.py"),
            Path("src/forecasting/studies/price_holiday_features.py"), Path(input_settings["base_search_config"]))},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": version("scipy"), "pyarrow": version("pyarrow"),
                     "threadpoolctl": version("threadpoolctl")}}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "manifest.json"
    with (DESTINATION / ".runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another rolling replay already owns this output directory") from error
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != identity:
                raise ValueError("Cannot resume: input, code, settings or software identity changed")
        else:
            write_json(manifest_path, identity)
        write_json(DESTINATION / "status.json", {"status": "running", "selected_jobs": len(jobs), "workers": workers})
        print("Building gate-specific daily examples from the pinned snapshot", flush=True)
        EXAMPLES = build_examples(snapshot, input_settings)
        BY_DAY = {}
        for example in EXAMPLES:
            key = (example["market"], example["gate"])
            group = BY_DAY.setdefault(str(example["delivery_day"]), {})
            if key in group:
                raise ValueError(f"Duplicate gate example {example['delivery_day']} {key}")
            group[key] = example
        print(f"Prepared {len(EXAMPLES)} examples; running {len(jobs)} independent jobs with {workers} workers", flush=True)
        # Fork shares immutable prepared arrays; Ctrl-C requests a stop after each
        # worker finishes its current day's atomic output.
        context = multiprocessing.get_context("fork")
        STOP_EVENT = context.Event()
        previous_handler = signal.getsignal(signal.SIGINT)
        def interrupt_handler(signum, frame):
            STOP_EVENT.set()
            print("Interrupt requested: finishing current daily checkpoints", flush=True)
        signal.signal(signal.SIGINT, interrupt_handler)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_job, job) for job in jobs]
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        if STOP_EVENT.is_set():
            write_json(DESTINATION / "status.json", {"status": "interrupted", "selected_jobs": len(jobs)})
            print("Interrupted after current daily checkpoints; rerun the same command to resume", flush=True)
            return
        for family, window, target in product(families, STUDY["windows"], STUDY["ida_targets"]):
            paths = []
            for variant in variants:
                paths.extend(sorted((DESTINATION / "jobs" / f"{variant}__{family}__{window}d__{target}").glob("????-??-??.parquet")))
            if not paths:
                continue
            group = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
            if group.empty:
                continue
            report_dir = DESTINATION / "comparisons" / f"{family}__{window}d__{target}"
            rolling_price_report(group, report_dir, timezone=STUDY["timezone"],
                                 evaluation_end_exclusive=STUDY["end_exclusive"], pair_baseline="core",
                                 expected_variants=variants, plots=True)
            if Path(args.pointwise_dir).is_dir():
                compare_october_pointwise(group, args.pointwise_dir, report_dir / "pointwise_october.csv")
        write_json(DESTINATION / "status.json", {"status": "complete_selected_jobs", "selected_jobs": len(jobs),
            "variants": variants, "families": families})


if __name__ == "__main__":
    main()
