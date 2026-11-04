"""Search native-resolution price models on earlier chronological validation data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.energy_prices import (PreparedExogenous, PriceForecastSettings, historical_decision_times,
                                                       load_price_history, prepare_exogenous_features)
from bess_pv_trading.forecasting.energy_price_features import load_catalog_features
from bess_pv_trading.forecasting.price_search import (complete_shared_days, daily_price_curve_features,
                                                      explain_incomplete_days, point_metrics,
                                                      prepare_search_matrix, project_search_matrix,
                                                      restrict_to_common_training_rows)
from bess_pv_trading.forecasting.price_search_models import make_price_candidate
from bess_pv_trading.parallel import configure_worker_threads


def _sha256(path):
    """Verify the immutable feature table inherited from the prior catalog read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rolling_blocks(starts, decisions_ns, available_ns, product_end_ns, timezone, frequency_days, gap_days):
    """Return nonoverlapping evaluation blocks and strictly earlier training rows."""
    for name, value in (("frequency_days", frequency_days), ("gap_days", gap_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Rolling {name} must be a positive integer")
    timestamps = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    if not len(timestamps) or not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Rolling forecasts require unique increasing product starts")
    decisions_ns = np.asarray(decisions_ns, dtype=np.int64)
    available_ns = np.asarray(available_ns, dtype=np.int64)
    product_end_ns = np.asarray(product_end_ns, dtype=np.int64)
    if any(values.shape != (len(timestamps),) for values in (decisions_ns, available_ns, product_end_ns)):
        raise ValueError("Rolling label and decision clocks must align with every product")
    days = timestamps.tz_convert(timezone).date
    first, last = days[0], days[-1]
    result = []
    begin = first
    while begin <= last:
        end = begin + timedelta(days=frequency_days)
        # Exclude the immediately preceding configured number of local days;
        # the older whole-day labels are the only added training observations.
        train_before = begin - timedelta(days=gap_days)
        test = np.flatnonzero((days >= begin) & (days < end))
        if len(test):
            cutoff = decisions_ns[test].min()
            past = np.flatnonzero((days < train_before) & (available_ns <= cutoff) & (product_end_ns <= cutoff))
            result.append((begin, end, past, test))
        begin = end
    return result


def _candidate_predict(spec, train_x, train_y, test_x, *, train_starts=None, test_starts=None,
                       timezone=None, seed, threads):
    """Fit a pooled model or independently regularized local delivery slots."""
    family = spec["family"]
    options = dict(spec["parameters"])
    if not family.startswith("slot_"):
        estimator = make_price_candidate(family, options, seed=seed, threads=threads)
        estimator.fit(train_x, train_y)
        return estimator.predict(test_x)
    minimum = options.pop("min_rows_per_slot", 30)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("Per-slot model needs positive min_rows_per_slot")
    if train_starts is None or test_starts is None or timezone is None:
        raise ValueError("Per-slot models require aligned local delivery timestamps")
    train_local = pd.DatetimeIndex(pd.to_datetime(train_starts, utc=True)).tz_convert(timezone)
    test_local = pd.DatetimeIndex(pd.to_datetime(test_starts, utc=True)).tz_convert(timezone)
    if len(train_local) != len(train_y) or len(test_local) != len(test_x):
        raise ValueError("Per-slot timestamps must match the corresponding feature rows")
    train_slots = train_local.hour * 60 + train_local.minute
    test_slots = test_local.hour * 60 + test_local.minute
    prediction = np.full(len(test_x), np.nan)
    for slot in np.unique(test_slots):
        training = train_slots == slot
        evaluation = test_slots == slot
        if training.sum() < minimum:
            raise ValueError(f"Insufficient training rows for local delivery slot {slot}")
        estimator = make_price_candidate(family.removeprefix("slot_"), options, seed=seed, threads=threads)
        estimator.fit(train_x[training], train_y[training])
        prediction[evaluation] = estimator.predict(test_x[evaluation])
    return prediction


def _validate_slot_coverage(train_starts, test_starts, timezone, minimum):
    """Fail preflight if any evaluated local slot lacks enough eligible examples."""
    train_local = pd.DatetimeIndex(pd.to_datetime(train_starts, utc=True)).tz_convert(timezone)
    test_local = pd.DatetimeIndex(pd.to_datetime(test_starts, utc=True)).tz_convert(timezone)
    train_slots = train_local.hour * 60 + train_local.minute
    test_slots = np.unique(test_local.hour * 60 + test_local.minute)
    counts = {int(slot): int(np.sum(train_slots == slot)) for slot in test_slots}
    insufficient = {slot: count for slot, count in counts.items() if count < minimum}
    if insufficient:
        raise ValueError(f"Insufficient per-slot training coverage: {insufficient}")
    return min(counts.values()) if counts else 0


def _fit_candidate(job):
    """Fit one estimator and retain its interval-level validation predictions."""
    configure_worker_threads(job["threads"], configure_torch=False)
    data = np.load(job["dataset"], allow_pickle=False)
    spec = job["candidate"]
    # A spread model fits the residual to an already-cleared price, then adds
    # that price back so every score remains in actual EUR/MWh.
    residual_index = job.get("residual_feature_index")
    train_anchor = 0 if residual_index is None else data["train_x"][:, residual_index]
    validation_anchor = 0 if residual_index is None else data["validation_x"][:, residual_index]
    rolling = job.get("rolling_refit")
    slot_model = spec["family"].startswith("slot_")
    initial_starts = data["train_starts"] if slot_model else None
    refits = []
    if rolling is None:
        prediction = _candidate_predict(spec, data["train_x"], data["train_y"] - train_anchor, data["validation_x"],
            train_starts=initial_starts, test_starts=data["starts"] if slot_model else None,
            timezone=job.get("timezone"), seed=job["seed"], threads=job["threads"]) + validation_anchor
    else:
        prediction = np.full(len(data["validation_y"]), np.nan)
        for begin, end, past, test in _rolling_blocks(data["starts"], data["decisions_ns"], data["available_ns"],
                                                     data["product_end_ns"], job["timezone"],
                                                     rolling["frequency_days"], rolling["gap_days"]):
            # All target features were generated at their own historical
            # decision time. Only already completed older labels are appended.
            train_x = np.concatenate([data["train_x"], data["validation_x"][past]])
            initial_target = data["train_y"] - train_anchor
            past_anchor = 0 if residual_index is None else data["validation_x"][past, residual_index]
            train_y = np.concatenate([initial_target, data["validation_y"][past] - past_anchor])
            test_anchor = 0 if residual_index is None else data["validation_x"][test, residual_index]
            train_starts = np.concatenate([initial_starts, data["starts"][past]]) if slot_model else None
            prediction[test] = _candidate_predict(spec, train_x, train_y, data["validation_x"][test],
                train_starts=train_starts, test_starts=data["starts"][test] if slot_model else None,
                timezone=job.get("timezone"), seed=job["seed"], threads=job["threads"]) + test_anchor
            refits.append({"start_day": begin.isoformat(), "end_day_exclusive": end.isoformat(),
                           "added_prior_intervals": len(past), "training_intervals": len(train_y),
                           "predicted_intervals": len(test)})
    metrics = point_metrics(data["validation_y"], prediction, data["reference"])
    output = Path(job["output"])
    pd.DataFrame({"product_start": data["starts"], "actual": data["validation_y"],
                  "point": prediction, "seasonal_naive": data["reference"]}).to_csv(output, index=False)
    if rolling is not None:
        output.with_suffix(".refits.json").write_text(json.dumps(refits, indent=2) + "\n")
    return {"group": job["group"], "feature_set": job["feature_set"], "candidate": spec["id"],
            "family": spec["family"], "training_rows": len(data["train_y"]),
            "rolling_refit_count": len(refits), **metrics}


def _validation_selection(study, root):
    """Freeze top validation candidates and named baselines before any holdout fit."""
    selection = study.get("selection")
    if not selection:
        return None, None
    path = root / selection["source_manifest"]
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Selection requires a complete, failure-free validation search")
    if not isinstance(selection["top_n"], int) or isinstance(selection["top_n"], bool) or selection["top_n"] < 1:
        raise ValueError("Selection top_n must be a positive integer")
    source_study = manifest["config"]
    for name in ("lag_hours", "feature_sets", "seed", "timezone", "source_study"):
        if study[name] != source_study[name]:
            raise ValueError(f"Holdout {name} differs from validation search")
    for name in ("feature_set_aliases", "add_market_price_features_to", "residual_target_features",
                 "daily_curve_lag_days", "add_daily_curve_features_to", "common_training_rows",
                 "supplemental_features", "add_supplemental_features_to"):
        if study.get(name) != source_study.get(name):
            raise ValueError(f"Holdout {name} differs from validation search")
    expected_candidates = {item["id"]: item for item in source_study["candidates"]}
    requested_candidates = {item["id"]: item for item in study["candidates"]}
    if expected_candidates != requested_candidates:
        raise ValueError("Holdout candidates must match the validation search exactly")
    chosen = {}
    for group, split in study["product_groups"].items():
        previous = source_study["product_groups"][group]
        validation_end = pd.Timestamp(previous["validation_end"], tz=study["timezone"]).tz_convert("UTC")
        availability = manifest["source_lineage"][group]["availability"]
        if availability["kind"] == "assumed_lag_after_product_end":
            selection_ready = validation_end + pd.Timedelta(hours=availability["lag_hours"])
        elif availability["kind"] == "assumed_delivery_day_publication_clock":
            # Conservatively wait through the end of any publication day
            # after the last delivery day, even if its clock is earlier.
            delay = max(0, availability["day_offset"])
            local_end = validation_end.tz_convert(availability["timezone"])
            selection_ready = (local_end + pd.DateOffset(days=delay)).tz_convert("UTC")
        else:
            raise ValueError("Selection needs an auditable validation-label publication policy")
        if pd.Timestamp(split["cutoff"]) < selection_ready:
            raise ValueError("Holdout model cutoff precedes the end of candidate selection")
        if split["native_minutes"] != previous["native_minutes"] or split.get("target", "DA") != previous.get("target", "DA"):
            raise ValueError("Holdout and validation must use the same target and native resolution")
        for name in ("market_price_features", "market_neighbor_offsets_minutes", "lag_hours"):
            if split.get(name) != previous.get(name):
                raise ValueError(f"Holdout {name} differs from validation search")
        rows = [row for row in manifest["results"] if row["group"] == group]
        previous_features = previous.get("feature_sets", source_study["feature_sets"])
        if split.get("feature_sets", study["feature_sets"]) != previous_features:
            raise ValueError("Holdout feature sets differ from validation search")
        expected = {(feature, name) for feature in previous_features for name in expected_candidates}
        observed = {(row["feature_set"], row["candidate"]) for row in rows}
        if len(rows) != len(expected) or observed != expected:
            raise ValueError(f"Incomplete or duplicate validation results for {group}")
        if any(not np.isfinite(row["mae_eur_per_mwh"]) for row in rows):
            raise ValueError("Validation selection requires finite MAE scores")
        ordered = sorted(rows, key=lambda row: (row["mae_eur_per_mwh"], row["feature_set"], row["candidate"]))
        chosen[group] = {(row["feature_set"], row["candidate"]) for row in ordered[:selection["top_n"]]}
        for feature in previous_features:
            for baseline in selection.get("baseline_candidates", []):
                if baseline not in expected_candidates:
                    raise ValueError("Selection baseline is not a declared candidate")
                chosen[group].add((feature, baseline))
    evidence = {"source_manifest": str(path), "source_manifest_sha256": _sha256(path),
                "criterion": "pooled_validation_mae", "top_n": selection["top_n"],
                "selected": {group: [list(pair) for pair in sorted(pairs)] for group, pairs in chosen.items()}}
    return chosen, evidence


def _verify_supplemental_selection_sources(study, validation_manifest):
    """Verify selection-period supplemental rows before extending a holdout."""
    source_study = validation_manifest["config"]
    source_evidence = validation_manifest.get("supplemental_features")
    if not source_evidence or not source_evidence.get("features"):
        raise ValueError("Validation search lacks supplemental feature provenance")
    first = min(pd.Timestamp(split["history_start"]) for split in source_study["product_groups"].values())
    last = max(pd.Timestamp(split["history_end"]) for split in source_study["product_groups"].values())
    holdout_first = min(pd.Timestamp(split["history_start"]) for split in study["product_groups"].values())
    holdout_last = max(pd.Timestamp(split["history_end"]) for split in study["product_groups"].values())
    if holdout_first > first or holdout_last < last:
        raise ValueError("Holdout supplemental range must contain the selection range")

    # The extended range can add snapshots, but all old projected rows and
    # their immutable source hashes must still match the selection run.
    _, current = load_catalog_features(study["database"], study["supplemental_features"],
        start=first, end=last, timezone=study["timezone"], interval_minutes=15,
        database_threads=study["threads_per_worker"])
    if current != source_evidence:
        raise ValueError("Supplemental feature source snapshots or selected rows changed since validation")
    return {"selection_start": first.isoformat(), "selection_end": last.isoformat(),
            "verification": "exact_selection_period_feature_lineage_and_projected_row_hash"}


def _write_report(destination, manifest):
    """Publish readable scores and compact paired-sample comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = pd.DataFrame(manifest["results"])
    lines = ["# Native-resolution price model comparison", "",
             f"Status: **{manifest['status']}**. Candidates are evaluated on shared complete delivery days.",
             "WAPE is total absolute error divided by total absolute observed price. "
             "Relative MAE compares the same intervals with the available seasonal lag. "
             "These results are retrospective and contain no trading-profit evaluation.", ""]
    if manifest.get("selection"):
        lines.extend(["Candidates were fixed using the earlier validation manifest; "
                      "this period is an evaluation of that selection, not a new selection stage.", ""])
    for group in sorted(results.group.unique()) if not results.empty else []:
        table = results.loc[results.group == group].sort_values(["feature_set", "candidate"])
        lines.extend([f"## {group}", "", "| Feature set | Model | Training rows | MAE EUR/MWh | WAPE % | Relative MAE |",
                      "|---|---|---:|---:|---:|---:|"])
        for row in table.itertuples(index=False):
            wape = "n/a" if pd.isna(row.wape_percent) else f"{row.wape_percent:.2f}"
            relative = "n/a" if pd.isna(row.relative_mae_to_naive) else f"{row.relative_mae_to_naive:.3f}"
            lines.append(f"| {row.feature_set} | {row.candidate} | {row.training_rows} | "
                         f"{row.mae_eur_per_mwh:.3f} | {wape} | {relative} |")
        lines.append("")
        figure, axis = plt.subplots(figsize=(10, max(4, len(table) * 0.25)))
        labels = table.feature_set + " / " + table.candidate
        axis.barh(labels, table.mae_eur_per_mwh)
        axis.set(xlabel="Point MAE (EUR/MWh)", title=f"{group}: matched validation days")
        figure.tight_layout()
        figure.savefig(destination / f"{group}__mae.png", dpi=130)
        plt.close(figure)
    if manifest["failures"]:
        lines.extend(["## Failed fits", "", "Failure details are retained in the manifest.", ""])
    (destination / "summary.md").write_text("\n".join(lines))


def _eligible_auction_features(target, markets):
    """Reject current or later auction targets before loading cleared prices."""
    allowed = {"IDA1": {"DA"}, "IDA2": {"DA", "IDA1"}, "IDA3": {"DA", "IDA1", "IDA2"}}
    order = {"DA": 0, "IDA1": 1, "IDA2": 2}
    if (target not in allowed or len(markets) != len(set(markets)) or set(markets) - allowed[target]
            or markets != sorted(markets, key=order.get)):
        raise ValueError(f"Invalid earlier-auction features for {target}")
    return [f"cleared_{market.lower()}" for market in markets]


def _neighbor_price_feature(prior_values, name, offset):
    """Join a nearby delivery product at the target while preserving its release time."""
    neighbor = prior_values.copy()
    neighbor["product_start"] -= pd.Timedelta(minutes=offset)
    return neighbor.assign(feature=name)


def run_search(config, *, check_only=False):
    """Prepare source-native matrices, then fit candidate workers on shared days."""
    study = config["price_model_search"]
    if not isinstance(study.get("common_training_rows", False), bool):
        raise ValueError("common_training_rows must be a boolean")
    root = Path(config.get("_project_root", ROOT))
    source = root / study["source_study"]
    source_manifest = json.loads((source / "manifest.json").read_text())
    feature_path = source / "exogenous.parquet"
    if _sha256(feature_path) != source_manifest["exogenous_sha256"]:
        raise ValueError("Pinned feature table checksum mismatch")
    configure_worker_threads(study["threads_per_worker"], configure_torch=False)
    base_exogenous = pd.read_parquet(feature_path)
    exogenous = prepare_exogenous_features(base_exogenous)
    feature_sets = source_manifest["plan"]["feature_sets"]
    selected_candidates, selection_evidence = _validation_selection(study, root)
    supplemental_specs = study.get("supplemental_features", [])
    if supplemental_specs and selection_evidence:
        validation_manifest = json.loads(Path(selection_evidence["source_manifest"]).read_text())
        selection_evidence["supplemental_source_verification"] = _verify_supplemental_selection_sources(
            study, validation_manifest)
    supplemental_evidence = None
    supplemental_names = tuple(item["name"] for item in supplemental_specs)
    if supplemental_specs:
        first = min(pd.Timestamp(split["history_start"]) for split in study["product_groups"].values())
        last = max(pd.Timestamp(split["history_end"]) for split in study["product_groups"].values())
        supplemental, supplemental_evidence = load_catalog_features(study["database"], supplemental_specs,
            start=first, end=last, timezone=study["timezone"], interval_minutes=15,
            database_threads=study["threads_per_worker"])
        supplement_index = prepare_exogenous_features(supplemental)
        if set(exogenous.groups) & set(supplement_index.groups):
            raise ValueError("Supplemental feature names collide with existing catalog features")
        exogenous = PreparedExogenous({**exogenous.groups, **supplement_index.groups})
    if selection_evidence:
        validation_manifest = json.loads(Path(selection_evidence["source_manifest"]).read_text())
        if validation_manifest["feature_sha256"] != source_manifest["exogenous_sha256"]:
            raise ValueError("Holdout and validation feature snapshots differ")
    destination = root / study["output_root"] / study["run_id"]
    if destination.exists():
        raise FileExistsError(f"Search output already exists: {destination}")
    if not check_only:
        destination.mkdir(parents=True)
    jobs, coverage, incomplete_days, lineage = [], [], {}, {}
    earlier_price_cache = {}
    for group, split in study["product_groups"].items():
        resolution = split["native_minutes"]
        group_lags = tuple(split.get("lag_hours", study["lag_hours"]))
        target_name = split.get("target", "DA")
        target = source_manifest["plan"]["targets"][target_name]
        # Retain each market's actual delivery window and declared decision and
        # publication assumptions while splitting by native product duration.
        history, evidence = load_price_history(study["database"], **target["source"],
            interval_minutes=resolution, native_resolution_minutes=resolution, timezone=study["timezone"],
            start=split["history_start"], end=split["history_end"])
        lineage[group] = evidence
        previous_markets = split.get("market_price_features", [])
        extra_names = _eligible_auction_features(target_name, previous_markets) if previous_markets else []
        group_exogenous = exogenous
        if extra_names:
            additions = []
            prior_lineage = {}
            offsets = split.get("market_neighbor_offsets_minutes", [])
            if (len(offsets) != len(set(offsets)) or any(isinstance(value, bool) or not isinstance(value, int)
                    or value == 0 or value % resolution for value in offsets)):
                raise ValueError("Auction neighbor offsets must be distinct nonzero multiples of the product resolution")
            expanded_names = list(extra_names)
            for market, name in zip(previous_markets, extra_names):
                key = (market, resolution, split["history_start"], split["history_end"])
                if key not in earlier_price_cache:
                    previous_source = source_manifest["plan"]["targets"][market]["source"]
                    prior, provenance = load_price_history(study["database"], **previous_source,
                        interval_minutes=resolution, timezone=study["timezone"],
                        start=split["history_start"], end=split["history_end"])
                    earlier_price_cache[key] = (prior, provenance)
                prior, prior_evidence = earlier_price_cache[key]
                prior_values = prior.rename(columns={"price_eur_per_mwh": "value"})[
                    ["product_start", "available_at", "value"]]
                additions.append(prior_values.assign(feature=name))
                # Offset describes the source product relative to the target.
                # Keep the source publication clock when moving its join key.
                for offset in offsets:
                    neighbor_name = f"{name}_neighbor_{'m' if offset < 0 else 'p'}{abs(offset)}"
                    additions.append(_neighbor_price_feature(prior_values, neighbor_name, offset))
                    expanded_names.append(neighbor_name)
                prior_lineage[market] = prior_evidence
            earlier_features = prepare_exogenous_features(pd.concat(additions, ignore_index=True))
            if set(exogenous.groups) & set(earlier_features.groups):
                raise ValueError("Earlier-auction feature names collide with catalog features")
            # Reuse the validated large catalog index and build only the small
            # market-specific additions rather than repeatedly copying it.
            group_exogenous = PreparedExogenous({**exogenous.groups, **earlier_features.groups})
            lineage[group] = {**evidence, "earlier_auction_features": prior_lineage,
                              "market_neighbor_offsets_minutes": offsets}
            extra_names = expanded_names
        datasets = {}
        residual_indices = {}
        group_features = {}
        for feature_set in split.get("feature_sets", study["feature_sets"]):
            base_name = study.get("feature_set_aliases", {}).get(feature_set, feature_set)
            names = tuple(item["name"] for item in feature_sets[base_name])
            if feature_set in study.get("add_market_price_features_to", []):
                names += tuple(extra_names)
            if feature_set in study.get("add_supplemental_features_to", []):
                names += supplemental_names
            residual_feature = study.get("residual_target_features", {}).get(feature_set)
            if residual_feature == "latest_cleared_auction":
                if not previous_markets:
                    raise ValueError("Latest-cleared spread requires an earlier auction")
                residual_feature = f"cleared_{previous_markets[-1].lower()}"
            if residual_feature is not None and residual_feature not in names:
                raise ValueError(f"Spread target anchor {residual_feature} is absent from {feature_set}")
            residual_index = (None if residual_feature is None else
                              len(group_lags) + 7 + names.index(residual_feature))
            residual_indices[feature_set] = residual_index
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate feature names in {feature_set}")
            group_features[feature_set] = names
        # Build each target's expensive lags, clocks and feature joins once.
        # Individual feature groups keep their own missing-data masks.
        all_names = tuple(dict.fromkeys(name for names in group_features.values() for name in names))
        settings = PriceForecastSettings(lag_hours=group_lags, interval_minutes=resolution,
            timezone=study["timezone"], exogenous_features=all_names, daily_lag_fallback="target_clock",
            **target.get("settings", {}))
        full = prepare_search_matrix(history, settings, group_exogenous, cutoff=split["cutoff"],
                                     validation_start=split["validation_start"], validation_end=split["validation_end"])
        base_count = len(group_lags) + 7
        feature_index = {name: base_count + index for index, name in enumerate(all_names)}
        curve_groups = set(study.get("add_daily_curve_features_to", [])) & set(group_features)
        curves = None
        if curve_groups:
            decisions = historical_decision_times(full["starts"], settings)
            curves = daily_price_curve_features(history, full["starts"], decisions, settings,
                                                 study.get("daily_curve_lag_days", []))
        for feature_set, names in group_features.items():
            columns = list(range(base_count)) + [feature_index[name] for name in names]
            datasets[feature_set] = project_search_matrix(full, columns)
            if feature_set in curve_groups:
                dataset = datasets[feature_set]
                dataset["matrix"] = np.column_stack([dataset["matrix"], curves])
                finite = np.isfinite(dataset["matrix"]).all(axis=1)
                dataset["training"] = finite & dataset["training_clock"]
                dataset["validation"] = finite & dataset["validation_clock"]
        if study.get("common_training_rows", False):
            datasets = restrict_to_common_training_rows(datasets)
        days = complete_shared_days(list(datasets.values()), start=split["validation_start"],
                                    end=split["validation_end"], settings=settings)
        # Retain reasons for omitted delivery days so scores are not mistaken
        # for the full requested period and missing sources can be repaired.
        column_names = {}
        for feature_set, data in datasets.items():
            names = ([f"price_lag_{lag}h" for lag in group_lags]
                     + [f"calendar_{index}" for index in range(7)] + list(group_features[feature_set]))
            names += [f"daily_curve_{index}" for index in range(data["matrix"].shape[1] - len(names))]
            column_names[feature_set] = names
        incomplete_days[group] = explain_incomplete_days(datasets, column_names,
            start=split["validation_start"], end=split["validation_end"], settings=settings)
        if not days:
            raise ValueError(f"No shared complete validation days for {group}")
        for feature_set, data in datasets.items():
            selected = data["validation"] & np.isin(data["days"], days)
            training_rows = int(data["training"].sum())
            if training_rows < 100:
                raise ValueError(f"Insufficient native training rows for {group}/{feature_set}")
            slot_candidates = [candidate for candidate in study["candidates"] if candidate["family"].startswith("slot_")]
            slot_minimum = None
            if slot_candidates:
                required = max(candidate["parameters"].get("min_rows_per_slot", 30) for candidate in slot_candidates)
                slot_minimum = _validate_slot_coverage(data["starts"][data["training"]], data["starts"][selected],
                                                       study["timezone"], required)
            coverage.append({"group": group, "target": target_name, "feature_set": feature_set, "native_minutes": resolution,
                             "training_rows": training_rows, "shared_days": len(days),
                             "validation_rows": int(selected.sum()), "fallback_count": data["fallback_count"],
                             "minimum_training_rows_per_evaluated_slot": slot_minimum})
            if check_only:
                continue
            dataset = destination / f"{group}__{feature_set}.npz"
            np.savez(dataset, train_x=data["matrix"][data["training"]], train_y=data["target"][data["training"]],
                     train_starts=np.asarray(data["starts"][data["training"]].astype(str), dtype=str),
                     validation_x=data["matrix"][selected], validation_y=data["target"][selected],
                     reference=data["matrix"][selected, 0],
                     decisions_ns=data["decisions"].as_unit("ns").asi8[selected],
                     available_ns=data["label_available_at"].as_unit("ns").asi8[selected],
                     product_end_ns=data["label_product_end"].as_unit("ns").asi8[selected],
                     starts=np.asarray(data["starts"][selected].astype(str), dtype=str))
            for candidate in study["candidates"]:
                if selected_candidates is not None and (feature_set, candidate["id"]) not in selected_candidates[group]:
                    continue
                jobs.append({"dataset": str(dataset), "output": str(destination / f"{group}__{feature_set}__{candidate['id']}.csv"),
                    "candidate": candidate, "group": group, "feature_set": feature_set,
                    "seed": study["seed"], "threads": study["threads_per_worker"],
                    "residual_feature_index": residual_indices[feature_set], "timezone": study["timezone"],
                    "rolling_refit": study.get("rolling_refit")})
    if check_only:
        return {"status": "preflight_complete", "coverage": coverage, "incomplete_days": incomplete_days,
                "feature_checksum": _sha256(feature_path),
                "supplemental_features": supplemental_evidence,
                "candidate_count": len(study["candidates"]), "selection": selection_evidence}
    manifest = {"status": "running", "config": study, "coverage": coverage, "incomplete_days": incomplete_days,
                "source_lineage": lineage,
                "feature_sha256": _sha256(feature_path), "supplemental_features": supplemental_evidence,
                "selection": selection_evidence, "results": [], "failures": []}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    with ProcessPoolExecutor(max_workers=study["workers"], mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(_fit_candidate, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                manifest["results"].append(future.result())
            except Exception as error:
                manifest["failures"].append({"group": job["group"], "feature_set": job["feature_set"],
                    "candidate": job["candidate"]["id"], "error": f"{type(error).__name__}: {error}"})
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    manifest["status"] = "complete" if not manifest["failures"] else "partial"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    pd.DataFrame(manifest["results"]).to_csv(destination / "validation_scores.csv", index=False)
    _write_report(destination, manifest)
    return manifest


def main():
    """A no-data plan is the default; run or check-only require an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/price_model_search.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_search(config, check_only=args.check_only) if args.run or args.check_only else config["price_model_search"]
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
