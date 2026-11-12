"""Staged, matched-row feature screening for full-year auction price studies."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_config
from . import price_feature_data, price_feature_search as engine
from .price_feature_data import digest
from .price_feature_models import point_metrics, sample_parameters, stable_seed, training_objective


def staged_settings(base, options, output):
    """Build a frozen contract with only approved previous-day and realized features."""
    settings = deepcopy(base)
    blocks = options["initial_blocks"]
    if len(blocks) != len(set(blocks)) or set(blocks) != {"nord_forecasts", "gas", "france", "switzerland"}:
        raise ValueError("The staged study requires the four declared previous-day blocks")
    count = int(options["configurations_per_family"])
    if not 1 <= count <= 4 or int(options["promoted_families"]) != 4:
        raise ValueError("The study allows one to four configurations and promotes four families")
    existing = settings["feature_blocks"]
    source_allowlist = {"nord_forecasts": {"nord_forecasts"}, "gas": {"psv_gas"},
                        "france": {"france_balance"},
                        "switzerland": {"swiss_energy_statistics", "swiss_reservoirs"}}
    forecast_datasets = {"load_forecast", "generation_forecast", "solar_forecast", "wind_onshore_forecast"}
    # Restrict the primary stage to forecasts and lagged observations, never target-derived values.
    for block in blocks:
        specification = existing[block]
        if specification.get("availability") != "assumed_previous_day" or specification.get("contains_target_da_price"):
            raise ValueError(f"Main block {block} is not a previous-day feature block")
        for feature in specification["features"]:
            name = feature["name"].lower()
            selectors = " ".join(str(value).lower() for value in feature.get("selectors", {}).values())
            if ("spread" in name or "same_day" in name or "spread" in selectors
                    or feature.get("contains_target_da_price")
                    or feature.get("source") not in source_allowlist[block]
                    or (block != "gas" and "price" in str(feature.get("column", "")).lower())
                    or "price" in str(feature.get("component", "")).lower()):
                raise ValueError(f"Target-derived or same-day main feature is prohibited: {feature['name']}")
            if block == "nord_forecasts" and feature.get("selectors", {}).get("dataset") not in forecast_datasets:
                raise ValueError(f"Unapproved NORD forecast source for {feature['name']}")
            if block == "gas" and int(feature.get("lag_days", 0)) < 1:
                raise ValueError(f"Gas feature needs a previous-day lag: {feature['name']}")
            if block in {"france", "switzerland"} and int(feature.get("lag_hours", 0)) < 24:
                raise ValueError(f"Observed feature needs a previous-day lag: {feature['name']}")
    realized = {feature["name"]: feature for feature in existing["realized_flows"]["features"]}
    realized.update({
        "fr_nuclear_same_day": {"name": "fr_nuclear_same_day", "source": "france_balance",
            "column": "generation_components_json", "component": "nucleaire"},
        "fr_hydro_same_day": {"name": "fr_hydro_same_day", "source": "france_balance",
            "column": "generation_components_json", "component": "hydraulique"},
        "fr_generation_same_day": {"name": "fr_generation_same_day", "source": "france_balance",
            "column": "generation_mw"},
        "fr_consumption_same_day": {"name": "fr_consumption_same_day", "source": "france_balance",
            "column": "source_record", "component": "consommation"},
    })
    allowed = set(options["retrospective_features"])
    safe = {"ch_it_physical_same_day", "ch_fr_physical_same_day", "ch_generation_same_day",
            "ch_consumption_same_day", "fr_physical_all_borders_same_day", "fr_nuclear_same_day",
            "fr_hydro_same_day", "fr_generation_same_day", "fr_consumption_same_day"}
    if allowed != safe or len(allowed) != len(options["retrospective_features"]):
        raise ValueError("Retrospective features must be the approved quantity-only whitelist")
    if not allowed <= set(realized):
        raise ValueError("An approved retrospective source feature is missing from the base configuration")
    # Fix each quantity's source and field so an allowed name cannot hide a price signal.
    expected_quantities = {
        "ch_it_physical_same_day": ("swiss_energy_statistics", "physical_net_export_it_mw", None),
        "ch_fr_physical_same_day": ("swiss_energy_statistics", "physical_net_export_fr_mw", None),
        "ch_generation_same_day": ("swiss_energy_statistics", "generation_mw", None),
        "ch_consumption_same_day": ("swiss_energy_statistics", "consumption_mw", None),
        "fr_physical_all_borders_same_day": ("france_balance", "source_record", "ech_physiques"),
        "fr_nuclear_same_day": ("france_balance", "generation_components_json", "nucleaire"),
        "fr_hydro_same_day": ("france_balance", "generation_components_json", "hydraulique"),
        "fr_generation_same_day": ("france_balance", "generation_mw", None),
        "fr_consumption_same_day": ("france_balance", "source_record", "consommation"),
    }
    for name in allowed:
        feature = realized[name]
        expected = expected_quantities[name]
        allowed_keys = {"name", "source", "column"} | ({"component"} if expected[2] else set())
        actual = (feature.get("source"), feature.get("column"), feature.get("component"))
        if set(feature) != allowed_keys or actual != expected:
            raise ValueError(f"Forbidden source, selector or field for retrospective feature {name}")
    settings["feature_blocks"] = {name: deepcopy(existing[name]) for name in blocks}
    settings["feature_blocks"]["retrospective"] = {"availability": "retrospective_realized",
        "features": [deepcopy(realized[name]) for name in options["retrospective_features"]]}
    # Core is retained as a diagnostic when common previous-day coverage is inadequate.
    settings["cohorts"] = {"stage_core": [], "stage_main": list(blocks),
                           "stage_retrospective": [*blocks, "retrospective"]}
    settings["trials_per_family"] = count
    settings["output_directory"] = str(output)
    settings["workers"] = 1
    snapshot_end = str(options.get("snapshot_end_exclusive", "2027-01-01"))
    for group in settings["product_groups"].values():
        if pd.Timestamp(snapshot_end) < pd.Timestamp(group["test_end"]):
            raise ValueError("The staged snapshot must cover every configured evaluation period")
        group["native_end_exclusive"] = snapshot_end
    implementation_paths = [Path(__file__), Path("scripts/search_price_staged.py"),
        Path("scripts/search_price_full_year.py"), Path("scripts/prepare_full_year_price_inputs.py"),
        Path("src/data/price_source_fallback.py"), Path("src/forecasting/studies/price_full_year_features.py"),
        Path("src/forecasting/studies/price_holiday_features.py"),
        Path("src/forecasting/studies/price_cancellation_features.py"), Path(settings["holiday_feature_config"])]
    reporting = Path("src/forecasting/studies/price_staged_reporting.py")
    if reporting.is_file():
        implementation_paths.append(reporting)
    settings["staged_implementation_sha256"] = {str(path): digest(path) for path in implementation_paths}
    settings["staged_contract"] = {"blocks": list(blocks), "configurations_per_family": count,
        "promoted_families": 4, "retrospective_features": list(options["retrospective_features"]),
        "snapshot_end_exclusive": snapshot_end}
    return settings


def _available_feature(frames, feature):
    """Check whether a selected source has at least one finite quantity anywhere."""
    frame = frames[feature["source"]]
    selected = np.ones(len(frame), dtype=bool)
    for key, value in feature.get("selectors", {}).items():
        selected &= frame[key].eq(value).to_numpy()
    source = frame.loc[selected]
    if source.empty:
        return False
    if feature.get("component"):
        def component(raw):
            if pd.isna(raw):
                return None
            value = json.loads(raw) if isinstance(raw, str) else raw
            return value.get(feature["component"]) if isinstance(value, dict) else None
        values = source[feature["column"]].map(component)
    else:
        values = source[feature["column"]]
    return bool(np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(float)).any())


def feature_coverage(root, group, settings):
    """Count each feature and block subset against core-eligible rows in each split."""
    root = Path(root)
    matrix_descriptor = group["files"]["x"]
    matrix_path = root / matrix_descriptor["path"]
    if digest(matrix_path) != matrix_descriptor["sha256"]:
        raise ValueError("Prepared feature matrix checksum changed")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    blocks = list(settings["feature_blocks"])
    core = group["cohorts"]["stage_core"]
    descriptors = core.get("indices", {})
    coverage = {}
    # Core split identities are fixed once, making every exclusion count comparable.
    for split in ("train", "inner_train", "inner_validation", "validation", "test", "final_train"):
        if split not in descriptors:
            coverage[split] = {"status": "unavailable", "reason": "No eligible core split indices"}
            continue
        descriptor = descriptors[split]
        path = root / descriptor["path"]
        if digest(path) != descriptor["sha256"]:
            raise ValueError(f"Prepared core split checksum changed: {split}")
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        total = len(rows)
        feature_details, block_details = {}, {}
        fitted_masks, configured_masks = {}, {}
        # Distinguish an omitted configured feature from a fitted feature with row gaps.
        for block in blocks:
            available_columns = {group["feature_names"][column]: column
                                 for column in group["block_columns"].get(block, [])}
            configured = [feature["name"] for feature in settings["feature_blocks"][block]["features"]]
            masks = []
            for name in configured:
                if name in available_columns:
                    mask = np.isfinite(matrix[rows, available_columns[name]])
                    status = "included"
                    masks.append(mask)
                else:
                    mask = np.zeros(total, dtype=bool)
                    status = "omitted"
                feature_details[name] = {"block": block, "status": status,
                                         "available_rows": int(mask.sum()),
                                         "excluded_rows": int(total - mask.sum())}
            omitted = [name for name in configured if name not in available_columns]
            fitted = [name for name in configured if name in available_columns]
            fitted_mask = np.logical_and.reduce(masks) if masks else None
            configured_mask = (np.zeros(total, dtype=bool) if omitted or not configured else fitted_mask)
            fitted_masks[block] = fitted_mask
            configured_masks[block] = configured_mask
            block_details[block] = {
                "status": "unavailable" if not fitted else "partial" if omitted else "complete",
                "configured_features": configured,
                "fitted_features": fitted,
                "omitted_features": omitted,
                "available_rows": int(fitted_mask.sum()) if fitted_mask is not None else None,
                "excluded_rows": int(total - fitted_mask.sum()) if fitted_mask is not None else None,
                "configured_available_rows": int(configured_mask.sum()),
                "configured_excluded_rows": int(total - configured_mask.sum()),
            }
        # Report every previous-day subset and its optional retrospective augmentation.
        combinations_coverage = []
        for size in range(len(blocks) + 1):
            for selected in combinations(blocks, size):
                unavailable = [block for block in selected if fitted_masks[block] is None]
                if unavailable:
                    fitted_mask = None
                else:
                    fitted_mask = np.ones(total, dtype=bool)
                    for block in selected:
                        fitted_mask &= fitted_masks[block]
                configured_mask = np.ones(total, dtype=bool)
                for block in selected:
                    configured_mask &= configured_masks[block]
                combinations_coverage.append({
                    "blocks": list(selected),
                    "status": "unavailable" if unavailable else "available",
                    "unavailable_blocks": unavailable,
                    "available_rows": int(fitted_mask.sum()) if fitted_mask is not None else None,
                    "excluded_rows": int(total - fitted_mask.sum()) if fitted_mask is not None else None,
                    "configured_available_rows": int(configured_mask.sum()),
                    "configured_excluded_rows": int(total - configured_mask.sum()),
                })
        coverage[split] = {"status": "available", "core_eligible_rows": total,
                           "features": feature_details, "blocks": block_details,
                           "combinations": combinations_coverage}
    return coverage


def prepare_staged(base, options, output):
    """Freeze an updated catalog snapshot using the full-year cancellation adapter."""
    from ...data import price_search_inputs
    from .price_cancellation_features import build_cancellation_aware_base, cancellation_days
    from .price_full_year_features import annotate_cohort_coverage, append_calendar_features
    from scripts.prepare_full_year_price_inputs import prepare_merged_inputs

    output = Path(output).resolve()
    settings = staged_settings(base, options, output)
    inputs = staged_inputs(settings)
    original_builder = price_search_inputs.build_price_search_inputs
    original_validate = engine._validate
    original_matrices = engine.prepare_matrices
    original_external = price_feature_data._external_grid
    omitted = {}

    def merged_builder(config, *, pinned_manifest=None):
        """Pin a fresh merged snapshot before any matrix construction."""
        if pinned_manifest is not None:
            raise ValueError("A new staged study requires a fresh merged snapshot")
        merged = Path(config["price_search_inputs"]["output_directory"])
        return prepare_merged_inputs(config, merged.parent / "raw_sources", merged,
                                     source_builder=original_builder)

    def validate(value):
        """Allow the diagnostic core-only cohort in the generic validator."""
        original_validate({**value, "cohorts": {name: blocks for name, blocks in value["cohorts"].items() if blocks}})

    def external(frames, grid, value):
        """Omit entirely unavailable optional columns and record every omission."""
        effective = deepcopy(value)
        for block, specification in effective["feature_blocks"].items():
            selected = []
            for feature in specification["features"]:
                if _available_feature(frames, feature):
                    selected.append(feature)
                else:
                    omitted[feature["name"]] = "no finite selected source values in the frozen snapshot"
            specification["features"] = selected
        return original_external(frames, grid, effective)

    def matrices(value, source, destination):
        """Apply the same cancellation and holiday policy as the full-year workflow."""
        manifest = json.loads((Path(source) / "manifest.json").read_text())
        if "gme_cancellation_notices" not in manifest:
            raise ValueError("The snapshot must include source-backed cancellation notices")
        prices = pd.read_parquet(Path(source) / manifest["tables"]["nord_auctions"]["path"])
        cancelled = cancellation_days(manifest["gme_cancellation_notices"], value["auctions"], prices, value["timezone"])
        original_base = price_feature_data._base_matrix
        def cancellation_aware(history, histories, starts, settings_, group):
            return build_cancellation_aware_base(original_base, histories, starts, settings_, group,
                                                cancelled, int(settings_["cancellation_fallback_days"]))
        price_feature_data._base_matrix = cancellation_aware
        price_feature_data._external_grid = external
        try:
            # Outer periods may be incomplete, but inner training can still be prepared.
            effective = {**value, "minimum_evaluation_rows": 0}
            groups = original_matrices(effective, source, destination)
        finally:
            price_feature_data._base_matrix = original_base
            price_feature_data._external_grid = original_external
        annotate_cohort_coverage(groups, int(value["minimum_evaluation_rows"]))
        holiday = load_config(value["holiday_feature_config"])["price_holiday_expansion"]
        return append_calendar_features(groups, destination, value, holiday)

    with engine.run_lock(output):
        price_search_inputs.build_price_search_inputs = merged_builder
        engine._validate = validate
        engine.prepare_matrices = matrices
        try:
            prepared = engine.prepare_search(settings, inputs, output)
        finally:
            price_search_inputs.build_price_search_inputs = original_builder
            engine._validate = original_validate
            engine.prepare_matrices = original_matrices
        report = output / "feature_coverage.json"
        if not report.exists():
            engine.write_json(report, {"omitted_features": omitted, "groups": {
                name: {"blocks": {block: [group["feature_names"][i] for i in columns]
                                   for block, columns in group["block_columns"].items()},
                       "cohorts": {key: {"counts": cohort["counts"],
                                         "training_eligible": cohort.get("training_eligible", False)}
                                   for key, cohort in group["cohorts"].items()},
                       "coverage_by_split": feature_coverage(output, group, settings)}
                for name, group in prepared["groups"].items()},
                "coverage_basis": "Each count is relative to stage_core eligible rows in that split; "
                                  "stage_main and stage_retrospective may have different chronological inner splits.",
                "outer_periods": "Not evaluated; row-count thresholds do not establish complete coverage."})
    return prepared


def staged_inputs(settings):
    """Exclude the unrelated target-derived Swiss commercial data source entirely."""
    inputs = load_config(settings["input_config"])["price_search_inputs"]
    inputs["end_exclusive"] = settings["staged_contract"]["snapshot_end_exclusive"]
    inputs["sources"] = [source for source in inputs["sources"] if source["name"] != "swiss_commercial_border"]
    return inputs


def candidates(settings, group, family):
    """Use the same bounded reproducible configurations for every feature set."""
    seed = stable_seed(settings["seed"], "staged", group, family)
    count = int(settings["staged_contract"]["configurations_per_family"])
    parameters = sample_parameters(settings["parameter_spaces"][family], seed, count)
    return [{"trial": index, "family": family, "parameters": values,
             "seed": stable_seed(seed, index)} for index, values in enumerate(parameters)]


def arm(group_name, group, cohort, columns, blocks, availability):
    """Describe the exact input columns and matched cohort of a candidate."""
    return {"group": group_name, "cohort": cohort, "columns": list(columns),
            "blocks": list(blocks), "availability_class": availability}


def block_arm(group_name, group, blocks, cohort="stage_main"):
    """Combine the required core with the requested ordered external blocks."""
    columns = list(range(group["base_width"]))
    for block in blocks:
        columns.extend(group["block_columns"][block])
    availability = "retrospective_realized" if "retrospective" in blocks else "assumed_previous_day"
    return arm(group_name, group, cohort, columns, blocks, availability)


def _staged_fit(job, retry_failed=False):
    """Checkpoint a single candidate against its exact feature and row identity."""
    root = Path(engine._WORKER_OUTPUT)
    selected = job["arm"]
    key = engine.signature(job)
    path = root / "staged_trials" / f"{key}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["signature"] != key:
            raise ValueError("Staged fit checkpoint identity mismatch")
        if prior["status"] == "complete" or not retry_failed:
            return prior
    arrays = engine._load_arrays(root, engine._WORKER_PREPARED, selected)
    older = arrays["inner_train"]
    later = arrays["inner_validation"]
    columns = selected["columns"]
    candidate = job["candidate"]
    result = {"id": key, "signature": key, "arm": selected, "candidate": candidate,
              "train_rows": len(older), "validation_rows": len(later),
              "feature_names": [engine._WORKER_PREPARED["groups"][selected["group"]]["feature_names"][column]
                                for column in columns],
              "training_objective": training_objective(candidate["family"], candidate["parameters"])}
    try:
        _, prediction, warnings = engine._fit(candidate["family"], candidate,
            np.asarray(arrays["x"][np.ix_(older, columns)]), arrays["y"][older] - arrays["anchor"][older],
            np.asarray(arrays["x"][np.ix_(later, columns)]))
        result.update(status="complete", warnings=warnings,
            metrics=point_metrics(arrays["y"][later], prediction + arrays["anchor"][later], arrays["reference"][later]))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    engine.write_json(path, result)
    return result


def score(row):
    """Rank completed fits by WAPE, MAE and deterministic scientific identifiers."""
    metrics = row["metrics"]
    return (float("inf") if metrics["wape_percent"] is None else metrics["wape_percent"],
            metrics["mae_eur_per_mwh"], row["candidate"]["family"], row["candidate"]["trial"],
            tuple(row["arm"]["columns"]))


def best(rows):
    """Select only successfully completed finite-score candidates."""
    eligible = [row for row in rows if row["status"] == "complete" and row["metrics"]["wape_percent"] is not None]
    return min(eligible, key=score) if eligible else None


def _run_jobs(stage, jobs, output, prepared, workers, retry_failed):
    """Execute and persist independent fits with one progress update per completion."""
    unique = {}
    prepared_hash = digest(output / "prepared.json")
    for item in jobs:
        # Retry is execution policy, not part of a model's scientific identity.
        selected = item["arm"]
        group = prepared["groups"][selected["group"]]
        columns = set(selected["columns"])
        retained = [block for block in selected["blocks"]
                    if any(column in columns for column in group["block_columns"][block])]
        selected = {**selected, "blocks": retained}
        job = {**item, "arm": selected, "prepared_sha256": prepared_hash}
        unique[engine.signature(job)] = job
    def progress(rows):
        engine.write_json(output / "status.json", {"status": "running", "phase": stage,
            "workers": workers, "threads_per_worker": 1, "completed_in_phase": len(rows),
            "expected_in_phase": len(unique), "last_update": datetime.now(timezone.utc).isoformat()})
    rows = engine._pool_jobs(_staged_fit, [(item, retry_failed) for item in unique.values()],
                             workers, progress, output, prepared["groups"])
    engine.write_json(output / "stages" / f"{stage}.json", {"stage": stage, "results": rows})
    from .price_staged_reporting import write_stage_reports
    write_stage_reports(output, stage, rows, prepared["groups"])
    return rows


def _failed_stage(stage, rows, output, prepared, workers):
    """Stop selection until every expected candidate in this stage has succeeded."""
    failures = [{"id": row["id"], "error": row.get("error", "incomplete fit")}
                for row in rows if row.get("status") != "complete"]
    if not failures:
        return None
    result = {"status": "partial", "phase": stage, "failed_candidates": failures,
              "prepared_sha256": digest(output / "prepared.json"),
              "scientific_config_sha256": prepared["scientific_config_sha256"], "groups": {}}
    engine.write_json(output / "winners.json", result)
    engine.write_json(output / "status.json", {"status": "partial", "phase": stage,
        "workers": workers, "threads_per_worker": 1, "failed_candidates": len(failures),
        "last_update": datetime.now(timezone.utc).isoformat()})
    return result


def run_staged(base, options, output, workers, *, retry_failed=False):
    """Screen families, combinations and feature ablations without outer scoring."""
    output = Path(output).resolve()
    settings = staged_settings(base, options, output)
    inputs = staged_inputs(settings)
    allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if workers < 1 or (allowed and workers > allowed):
        raise ValueError(f"Requested workers exceed available affinity ({allowed})")
    with engine.run_lock(output):
        # Verify every matrix and all software versions without opening the catalog.
        original_validate = engine._validate
        def validate(value):
            original_validate({**value, "cohorts": {name: blocks for name, blocks in value["cohorts"].items() if blocks}})
        engine._validate = validate
        try:
            prepared = engine.prepare_search(settings, inputs, output)
        finally:
            engine._validate = original_validate
        groups = prepared["groups"]
        families = list(settings["parameter_spaces"])
        blocks = list(options["initial_blocks"])
        params = {(name, family): candidates(settings, name, family)
                  for name in groups for family in families}
        screening = []
        skipped = {}
        for name, group in groups.items():
            if not group["cohorts"]["stage_main"].get("training_eligible"):
                skipped[name] = "insufficient matched previous-day training or inner-validation rows"
                continue
            available = [block for block in blocks if group["block_columns"].get(block)]
            for subset in [[], *[[block] for block in available]]:
                selected = block_arm(name, group, subset)
                for family in families:
                    screening.extend({"arm": selected, "candidate": candidate} for candidate in params[name, family])
        screened = _run_jobs("screen", screening, output, prepared, workers, retry_failed)
        failure = _failed_stage("screen", screened, output, prepared, workers)
        if failure:
            return failure
        promoted = {}
        for name in groups:
            ranked = []
            for family in families:
                winner = best([row for row in screened if row["arm"]["group"] == name
                               and row["candidate"]["family"] == family])
                if winner is not None:
                    ranked.append(winner)
            promoted[name] = sorted(ranked, key=score)[:4]
            if name not in skipped and len(promoted[name]) < 4:
                skipped[name] = "fewer than four model families completed screening successfully"
                promoted[name] = []
        combinations_jobs = []
        for name, winners in promoted.items():
            group = groups[name]
            available = [block for block in blocks if group["block_columns"].get(block)]
            for size in range(len(available) + 1):
                for subset in combinations(available, size):
                    selected = block_arm(name, group, subset)
                    combinations_jobs.extend({"arm": selected, "candidate": winner["candidate"]} for winner in winners)
        combined = _run_jobs("combinations", combinations_jobs, output, prepared, workers, retry_failed)
        failure = _failed_stage("combinations", combined, output, prepared, workers)
        if failure:
            return failure
        family_best = []
        for name, winners in promoted.items():
            for winner in winners:
                family = winner["candidate"]["family"]
                selected = best([row for row in combined if row["arm"]["group"] == name
                                 and row["candidate"]["family"] == family])
                if selected is not None:
                    family_best.append(selected)
        ablation_jobs = []
        for winner in family_best:
            selected = winner["arm"]
            group = groups[selected["group"]]
            for column in selected["columns"]:
                if column >= group["base_width"]:
                    reduced = {**selected, "columns": [item for item in selected["columns"] if item != column]}
                    ablation_jobs.append({"arm": reduced, "candidate": winner["candidate"]})
        ablated = _run_jobs("ablation", ablation_jobs, output, prepared, workers, retry_failed)
        failure = _failed_stage("ablation", ablated, output, prepared, workers)
        if failure:
            return failure
        tune_jobs = []
        for winner in family_best:
            name = winner["arm"]["group"]
            family = winner["candidate"]["family"]
            variants = [winner, *[row for row in ablated if row["arm"]["group"] == name
                                   and row["candidate"]["family"] == family]]
            selected = best(variants)
            if selected is not None:
                tune_jobs.extend({"arm": selected["arm"], "candidate": candidate} for candidate in params[name, family])
        tuned = _run_jobs("refinement", tune_jobs, output, prepared, workers, retry_failed)
        failure = _failed_stage("refinement", tuned, output, prepared, workers)
        if failure:
            return failure
        winners = {}
        retrospective_jobs = []
        for name, group in groups.items():
            main = best([row for row in tuned if row["arm"]["group"] == name])
            if main is None:
                continue
            winners[name] = {"main": main, "retrospective": None}
            retro = group["cohorts"]["stage_retrospective"]
            extra = group["block_columns"].get("retrospective", [])
            if retro.get("training_eligible") and extra:
                common = {**main["arm"], "cohort": "stage_retrospective"}
                augmented = {**common, "columns": [*common["columns"], *extra],
                    "blocks": [*common["blocks"], "retrospective"], "availability_class": "retrospective_realized"}
                retrospective_jobs.extend([{"arm": common, "candidate": main["candidate"]},
                                           {"arm": augmented, "candidate": main["candidate"]}])
        retrospective = _run_jobs("retrospective", retrospective_jobs, output, prepared, workers, retry_failed)
        failure = _failed_stage("retrospective", retrospective, output, prepared, workers)
        if failure:
            return failure
        for name, winner in winners.items():
            rows = [row for row in retrospective if row["arm"]["group"] == name]
            baseline = next((row for row in rows if row["arm"]["availability_class"] == "assumed_previous_day"), None)
            extended = next((row for row in rows if row["arm"]["availability_class"] == "retrospective_realized"), None)
            if baseline and extended and baseline["status"] == extended["status"] == "complete":
                winner["retrospective"] = extended
                winner["retrospective_comparison"] = {"matched_baseline": baseline["metrics"],
                    "extended": extended["metrics"], "train_rows": baseline["train_rows"],
                    "validation_rows": baseline["validation_rows"]}
            else:
                winner["retrospective_comparison"] = {"status": "unavailable_or_failed"}
        result = {"status": "complete" if len(winners) == len(groups) else "partial",
            "prepared_sha256": digest(output / "prepared.json"),
            "scientific_config_sha256": prepared["scientific_config_sha256"],
            "selection": "inner chronological WAPE then MAE; outer periods not evaluated",
            "groups": winners, "skipped": skipped,
            "promoted_families": {name: [row["candidate"]["family"] for row in rows]
                                  for name, rows in promoted.items()}}
        engine.write_json(output / "winners.json", result)
        engine.write_json(output / "status.json", {"status": result["status"], "phase": "inner_finished",
            "workers": workers, "threads_per_worker": 1, "last_update": datetime.now(timezone.utc).isoformat()})
        return result
