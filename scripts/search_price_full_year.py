"""Plan, prepare or run the full-year price search with audited source fallbacks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

# Limit each CPU estimator and its native libraries before numerical imports.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

from bess_pv_trading.config import load_config
from bess_pv_trading.data.price_search_inputs import _digest
from bess_pv_trading.forecasting.studies.price_feature_search import run_search
from scripts.prepare_full_year_price_inputs import prepare_merged_inputs


def run_inner_only(settings, input_settings, output, engine):
    """Checkpoint inner trials without selecting on or scoring later periods."""
    output = Path(output).resolve()
    with engine.run_lock(output):
        prepared = engine.prepare_search(settings, input_settings, output)
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        if settings["workers"] > allowed:
            raise ValueError(f"Requested {settings['workers']} workers, but affinity permits {allowed}")
        state = {"status": "running", "phase": "inner_search", "workers": settings["workers"],
            "threads_per_worker": 1, "cuda_used": False, "inner_fits_expected": prepared["inner_fits"],
            "scientific_config_sha256": prepared["scientific_config_sha256"],
            "started_at": datetime.now(timezone.utc).isoformat()}
        path = output / "status.json"
        def progress(rows):
            state["finished_tasks_in_phase"] = len(rows)
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            engine.write_json(path, state)
            if len(rows) == 1 or len(rows) % 25 == 0:
                print(f"inner_search: {len(rows)} tasks finished", flush=True)
        engine.write_json(path, state)
        jobs = [(task, False) for task in prepared["tasks"]]
        results = engine._pool_jobs(engine._trial_task, jobs, settings["workers"], progress, output, prepared["groups"])
        state["inner_fits_complete"] = sum(item["successful"] for item in results)
        state["inner_fits_failed"] = sum(item["failed"] for item in results)
        state["status"] = ("inner_complete" if state["inner_fits_complete"] == prepared["inner_fits"]
                           and state["inner_fits_failed"] == 0 else "partial")
        state["phase"] = "inner_finished"
        engine.write_json(path, state)
        return state


def main():
    """Require an explicit action and reuse the original checkpointed search engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_feature_selection_full_year.yaml")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--inner-only", action="store_true", help="Fit inner trials without validation or test")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["price_feature_search"]
    settings["database_path"] = args.database
    if args.output:
        settings["output_directory"] = args.output
    if args.workers:
        settings["workers"] = args.workers
    # The adapter is part of the study identity even though the old search is untouched.
    settings["full_year_adapter_sha256"] = {str(path): _digest(path) for path in (
        Path(__file__), Path("scripts/prepare_full_year_price_inputs.py"),
        Path("src/data/price_source_fallback.py"), Path("src/forecasting/studies/price_full_year_features.py"),
        Path("src/forecasting/studies/price_holiday_features.py"), Path(settings["holiday_feature_config"]))}
    input_settings = load_config(settings["input_config"])["price_search_inputs"]
    if not (args.prepare_only or args.inner_only or args.run):
        print(json.dumps({"action": "plan_only", "groups": list(settings["product_groups"]),
            "training_windows": {name: [group["train_start"], group["train_end"]]
                                 for name, group in settings["product_groups"].items()},
            "cohorts": settings["cohorts"], "holiday_feature_config": settings["holiday_feature_config"],
            "maximum_inner_fits_before_coverage_checks": (len(settings["product_groups"]) *
                sum(2 ** len(blocks) for blocks in settings["cohorts"].values()) *
                len(settings["parameter_spaces"]) * settings["trials_per_family"]),
            "workers": settings["workers"], "output": settings["output_directory"],
            "source_policy": "GME before Energy Charts; consolidated RTE before realtime"}, indent=2))
        return
    # Patch only this coordinator's first-time source builder; existing search code remains unchanged.
    from bess_pv_trading.data import price_search_inputs
    from bess_pv_trading.forecasting.studies import price_feature_search
    from bess_pv_trading.forecasting.studies.price_full_year_features import annotate_cohort_coverage, append_calendar_features
    original_builder = price_search_inputs.build_price_search_inputs
    original_validate = price_feature_search._validate
    original_matrices = price_feature_search.prepare_matrices
    def build_merged(config, *, pinned_manifest=None):
        if pinned_manifest is not None:
            raise ValueError("Pin this study using its saved raw and merged source manifests")
        merged = Path(config["price_search_inputs"]["output_directory"])
        return prepare_merged_inputs(config, merged.parent / "raw_sources", merged,
                                     source_builder=original_builder)
    price_search_inputs.build_price_search_inputs = build_merged
    def validate_full_year(value):
        """Allow one explicitly named empty price-only cohort without changing the old search."""
        empty = {name for name, blocks in value["cohorts"].items() if not blocks}
        if empty - {"price_only"}:
            raise ValueError("Only the price_only cohort may have no external blocks")
        checked = {**value, "cohorts": {name: blocks for name, blocks in value["cohorts"].items() if blocks}}
        original_validate(checked)
    price_feature_search._validate = validate_full_year
    def prepare_full_year_matrices(study_settings, source, destination):
        """Allow inner training before lagged validation/test sources are published."""
        threshold = int(study_settings["minimum_evaluation_rows"])
        if study_settings.get("allow_training_without_evaluation_coverage", False):
            effective = {**study_settings, "minimum_evaluation_rows": 0}
        else:
            effective = study_settings
        groups = original_matrices(effective, source, destination)
        # Training eligibility stays distinct from whether future evaluation rows exist.
        annotate_cohort_coverage(groups, threshold)
        holiday = load_config(study_settings["holiday_feature_config"])["price_holiday_expansion"]
        return append_calendar_features(groups, destination, study_settings, holiday)
    price_feature_search.prepare_matrices = prepare_full_year_matrices
    try:
        if args.inner_only:
            result = run_inner_only(settings, input_settings, Path(settings["output_directory"]), price_feature_search)
        else:
            # Do not score a partially populated snapshot as a completed final evaluation.
            prepared_path = Path(settings["output_directory"]) / "prepared.json"
            if args.run:
                if not prepared_path.is_file():
                    raise ValueError("Run --prepare-only and inspect evaluation coverage before full evaluation")
                prepared = json.loads(prepared_path.read_text())
                pending = [f"{name}/{cohort_name}" for name, group in prepared["groups"].items()
                           for cohort_name, cohort in group["cohorts"].items()
                           if cohort.get("training_eligible") and not cohort.get("evaluation_minimum_met")]
                if pending:
                    raise ValueError("Validation or test has fewer than the minimum rows for: " + ", ".join(pending))
            result = run_search(settings, input_settings, Path(settings["output_directory"]),
                                prepare_only=args.prepare_only)
    finally:
        price_search_inputs.build_price_search_inputs = original_builder
        price_feature_search._validate = original_validate
        price_feature_search.prepare_matrices = original_matrices
    print(json.dumps({"status": result["status"], "output": settings["output_directory"],
                      "inner_fits": result.get("inner_fits", result.get("inner_fits_complete"))}, indent=2))


if __name__ == "__main__":
    main()
