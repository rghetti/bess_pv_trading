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
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

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
        if family.startswith("recency_"):
            # Observation-based decay requires strictly ordered native products.
            if train_starts is None or len(train_starts) != len(train_y):
                raise ValueError("Recency-weighted models require aligned training timestamps")
            times = pd.DatetimeIndex(pd.to_datetime(train_starts, utc=True))
            if times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
                raise ValueError("Recency-weighted training timestamps must be unique and chronological")
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


def _training_window(starts, window_days):
    """Select the latest elapsed training window without looking at test labels."""
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
        raise ValueError("Training window must be a positive integer number of days")
    times = pd.DatetimeIndex(pd.to_datetime(starts, utc=True))
    if times.empty or times.hasnans:
        raise ValueError("Training window requires nonempty valid timestamps")
    return np.asarray(times >= times.max() - pd.Timedelta(days=window_days))


def _fit_candidate_impl(job):
    """Fit one estimator and retain its interval-level validation predictions."""
    configure_worker_threads(job["threads"], configure_torch=False)
    data = np.load(job["dataset"], allow_pickle=False)
    spec = job["candidate"]
    # A spread model fits the residual to an already-cleared price, then adds
    # that price back so every score remains in actual EUR/MWh.
    residual_index = job.get("residual_feature_index")
    if spec["family"] == "zero_spread" and residual_index is None:
        raise ValueError("Zero-spread candidate requires a cleared-price anchor")
    train_anchor = 0 if residual_index is None else data["train_x"][:, residual_index]
    validation_anchor = 0 if residual_index is None else data["validation_x"][:, residual_index]
    rolling = job.get("rolling_refit")
    slot_model = spec["family"].startswith("slot_")
    recency_model = spec["family"].startswith("recency_")
    window_days = spec.get("train_window_days")
    initial_starts = data["train_starts"] if slot_model or recency_model or window_days is not None else None
    initial_mask = (_training_window(initial_starts, window_days) if window_days is not None
                    else np.ones(len(data["train_y"]), dtype=bool))
    if window_days is not None and initial_mask.sum() < 100:
        raise ValueError("Training window leaves fewer than 100 observations")
    refits = []
    if rolling is None:
        target = (data["train_y"] - train_anchor)[initial_mask]
        prediction = _candidate_predict(spec, data["train_x"][initial_mask], target, data["validation_x"],
            train_starts=initial_starts[initial_mask] if slot_model or recency_model else None,
            test_starts=data["starts"] if slot_model else None,
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
            train_starts = (np.concatenate([initial_starts, data["starts"][past]])
                            if slot_model or recency_model or window_days is not None else None)
            mask = (_training_window(train_starts, window_days) if window_days is not None
                    else np.ones(len(train_y), dtype=bool))
            if window_days is not None and mask.sum() < 100:
                raise ValueError("Training window leaves fewer than 100 observations")
            prediction[test] = _candidate_predict(spec, train_x[mask], train_y[mask], data["validation_x"][test],
                train_starts=train_starts[mask] if slot_model or recency_model else None,
                test_starts=data["starts"][test] if slot_model else None,
                timezone=job.get("timezone"), seed=job["seed"], threads=job["threads"]) + test_anchor
            refits.append({"start_day": begin.isoformat(), "end_day_exclusive": end.isoformat(),
                           "added_prior_intervals": int(mask[len(initial_target):].sum()),
                           "training_intervals": int(mask.sum()),
                           "predicted_intervals": len(test)})
    metrics = point_metrics(data["validation_y"], prediction, data["reference"])
    output = Path(job["output"])
    pd.DataFrame({"product_start": data["starts"], "actual": data["validation_y"],
                  "point": prediction, "seasonal_naive": data["reference"]}).to_csv(output, index=False)
    if rolling is not None:
        output.with_suffix(".refits.json").write_text(json.dumps(refits, indent=2) + "\n")
    return {"group": job["group"], "feature_set": job["feature_set"], "candidate": spec["id"],
            "family": spec["family"], "training_rows": int(initial_mask.sum()), "train_window_days": window_days,
            "rolling_refit_count": len(refits), **metrics}


def _fit_candidate(job):
    """Record optimizer convergence warnings alongside each candidate score."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = _fit_candidate_impl(job)
    messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    for item in caught:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    result["convergence_warnings"] = messages
    return result


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
    if not isinstance(selection.get("matched_window_baselines", False), bool):
        raise ValueError("matched_window_baselines must be a boolean")
    if not isinstance(selection.get("matched_harmonic_baselines", False), bool):
        raise ValueError("matched_harmonic_baselines must be a boolean")
    source_study = manifest["config"]
    for name in ("lag_hours", "feature_sets", "seed", "timezone", "source_study"):
        if study[name] != source_study[name]:
            raise ValueError(f"Holdout {name} differs from validation search")
    for name in ("feature_set_aliases", "add_market_price_features_to", "residual_target_features",
                 "daily_curve_lag_days", "add_daily_curve_features_to", "common_training_rows",
                 "supplemental_features", "add_supplemental_features_to", "add_lagged_market_features_to",
                 "derived_features", "add_derived_features_to"):
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
        for name in ("market_price_features", "market_neighbor_offsets_minutes", "market_neighbor_boundary",
                     "lagged_market_price_features", "lag_hours"):
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
        eligible = [row for row in rows if not row.get("convergence_warnings")]
        if len(eligible) < selection["top_n"]:
            raise ValueError("Not enough converged validation candidates for selection")
        ordered = sorted(eligible, key=lambda row: (row["mae_eur_per_mwh"], row["feature_set"], row["candidate"]))
        chosen[group] = {(row["feature_set"], row["candidate"]) for row in ordered[:selection["top_n"]]}
        for feature in previous_features:
            for baseline in selection.get("baseline_candidates", []):
                if baseline not in expected_candidates:
                    raise ValueError("Selection baseline is not a declared candidate")
                chosen[group].add((feature, baseline))
        # Carry selected hyperparameters to each declared matching feature set
        # so the later comparison remains paired rather than reselecting models.
        paired_features = set()
        for pair in selection.get("matched_feature_pairs", []):
            if (not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]
                    or any(feature not in previous_features or feature in paired_features for feature in pair)):
                raise ValueError("Matched selection pairs must contain distinct declared feature sets")
            paired_features.update(pair)
            for feature, candidate in tuple(chosen[group]):
                if feature in pair:
                    counterpart = pair[1] if feature == pair[0] else pair[0]
                    chosen[group].add((counterpart, candidate))
        if selection.get("matched_window_baselines", False):
            # Each selected recent-history candidate gets its exact full-history
            # estimator counterpart on every selected feature set.
            for feature, candidate in tuple(chosen[group]):
                specification = expected_candidates[candidate]
                if specification.get("train_window_days") is not None:
                    matches = [entry["id"] for entry in expected_candidates.values()
                               if entry.get("train_window_days") is None
                               and entry["family"] == specification["family"]
                               and entry["parameters"] == specification["parameters"]]
                    if len(matches) != 1:
                        raise ValueError("A selected training window requires one matching full-history candidate")
                    chosen[group].add((feature, matches[0]))
        if selection.get("matched_harmonic_baselines", False):
            # Match each selected higher-order basis to its one-harmonic model
            # without changing estimator parameters or the training window.
            for feature, candidate in tuple(chosen[group]):
                specification = expected_candidates[candidate]
                parameters = specification["parameters"]
                if parameters.get("harmonics", 1) > 1:
                    base_parameters = {key: value for key, value in parameters.items() if key != "harmonics"}
                    matches = [entry["id"] for entry in expected_candidates.values()
                               if entry["family"] == specification["family"]
                               and entry.get("train_window_days") == specification.get("train_window_days")
                               and {key: value for key, value in entry["parameters"].items() if key != "harmonics"} == base_parameters
                               and entry["parameters"].get("harmonics", 1) == 1]
                    if len(matches) != 1:
                        raise ValueError("A selected harmonic model requires one matching first-harmonic candidate")
                    chosen[group].add((feature, matches[0]))
    evidence = {"source_manifest": str(path), "source_manifest_sha256": _sha256(path),
                "criterion": "pooled_validation_mae", "top_n": selection["top_n"],
                "matched_feature_pairs": selection.get("matched_feature_pairs", []),
                "matched_window_baselines": selection.get("matched_window_baselines", False),
                "matched_harmonic_baselines": selection.get("matched_harmonic_baselines", False),
                "selected": {group: [list(pair) for pair in sorted(pairs)] for group, pairs in chosen.items()}}
    return chosen, evidence


def _comparable_catalog_evidence(value, database):
    """Normalize only a catalog's own location while retaining snapshot evidence."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "database" and item == database:
                result[key] = "<catalog>"
            elif key in {"manifest", "directory"} and isinstance(item, str) and item.startswith(database + "::"):
                result[key] = "<catalog>" + item[len(database):]
            else:
                result[key] = _comparable_catalog_evidence(item, database)
        return result
    if isinstance(value, list):
        return [_comparable_catalog_evidence(item, database) for item in value]
    return value


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
    if (_comparable_catalog_evidence(current, current["database"])
            != _comparable_catalog_evidence(source_evidence, source_evidence["database"])):
        raise ValueError("Supplemental feature source snapshots or selected rows changed since validation")
    return {"selection_start": first.isoformat(), "selection_end": last.isoformat(),
            "verification": "exact_selection_period_feature_lineage_and_projected_row_hash"}


def _verify_price_selection_sources(study, validation_manifest, source_targets):
    """Check selection-period auction snapshots before using them in a holdout."""
    selection_study = validation_manifest["config"]
    verified = {}
    for group, split in selection_study["product_groups"].items():
        resolution = split["native_minutes"]
        target = split.get("target", "DA")
        old = validation_manifest["source_lineage"][group]

        def verify(market, start, end, expected, native=None):
            """Requery the original range and compare its complete snapshot evidence."""
            _, current = load_price_history(study["database"], **source_targets[market]["source"],
                interval_minutes=resolution, native_resolution_minutes=native, timezone=study["timezone"],
                start=start, end=end)
            old_database = expected.get("database")
            new_database = current.get("database")
            left = _comparable_catalog_evidence(current, new_database) if new_database else current
            right = _comparable_catalog_evidence(expected, old_database) if old_database else expected
            if json.dumps(left, sort_keys=True, default=str) != json.dumps(right, sort_keys=True, default=str):
                raise ValueError(f"Auction price snapshots changed since selection for {group}/{market}")

        # Old manifests add feature provenance alongside target provenance;
        # compare the original target fields without those nested additions.
        extensions = {"earlier_auction_features", "market_neighbor_offsets_minutes", "market_neighbor_boundary",
                      "lagged_auction_features"}
        target_evidence = {key: value for key, value in old.items() if key not in extensions}
        verify(target, split["history_start"], split["history_end"], target_evidence, native=resolution)
        for market in split.get("market_price_features", []):
            verify(market, split["history_start"], split["history_end"], old["earlier_auction_features"][market])
        for item in split.get("lagged_market_price_features", []):
            market, lag = item["market"], item["lag_hours"]
            name = _lagged_auction_name(market, lag)
            verify(market, pd.Timestamp(split["history_start"]) - pd.Timedelta(hours=lag),
                   pd.Timestamp(split["history_end"]) - pd.Timedelta(hours=lag),
                   old["lagged_auction_features"][name])
        verified[group] = {"target": target, "earlier_auctions": split.get("market_price_features", []),
                           "lagged_auctions": len(split.get("lagged_market_price_features", []))}
    return verified


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
    warned = [row for row in manifest["results"] if row.get("convergence_warnings")]
    if warned:
        lines.extend([f"Optimizer convergence warnings occurred in **{len(warned)}** candidate fits; "
                      "details are retained in the manifest. Those fits are excluded from later candidate ranking.", ""])
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


def _neighbor_price_feature(prior_values, name, offset, *, boundary="unbounded", timezone="Europe/Rome"):
    """Join a nearby delivery product at the target while preserving its release time."""
    if boundary not in {"unbounded", "same_product"}:
        raise ValueError("Unknown neighboring-auction delivery-day boundary policy")
    neighbor = prior_values.copy()
    neighbor["product_start"] -= pd.Timedelta(minutes=offset)
    if boundary == "same_product":
        # Keep only shifted products from the same local delivery day. At its
        # edges, use the already-cleared same product with its original clock.
        source_days = pd.DatetimeIndex(prior_values.product_start).tz_convert(timezone).date
        target_days = pd.DatetimeIndex(neighbor.product_start).tz_convert(timezone).date
        neighbor = neighbor.loc[source_days == target_days]
        target_times = pd.DatetimeIndex(prior_values.product_start)
        target_days = target_times.tz_convert(timezone).date
        requested_days = (target_times + pd.Timedelta(minutes=offset)).tz_convert(timezone).date
        edge = prior_values.loc[target_days != requested_days]
        neighbor = pd.concat([neighbor, edge], ignore_index=True)
    return neighbor.assign(feature=name)


def _lagged_auction_name(market, lag_hours):
    """Validate and name a strictly historical market-price feature."""
    if market not in {"DA", "IDA1", "IDA2", "IDA3", "settlement"}:
        raise ValueError("Unknown lagged price market")
    if isinstance(lag_hours, bool) or not isinstance(lag_hours, int) or lag_hours < 24:
        raise ValueError("Lagged auction features require an integer lag of at least 24 hours")
    return f"past_{market.lower()}_{lag_hours}h"


def _lagged_auction_feature(prior_values, market, lag_hours):
    """Move past market deliveries to later products, retaining publication time."""
    name = _lagged_auction_name(market, lag_hours)
    feature = prior_values.copy()
    feature["product_start"] += pd.Timedelta(hours=lag_hours)
    return feature.assign(feature=name)


def _linear_derived_features(matrix, feature_names, specifications):
    """Combine named decision-time inputs without filling missing components."""
    index = {name: position for position, name in enumerate(feature_names)}
    if len(index) != len(feature_names):
        raise ValueError("Derived feature input names must be unique")
    names, columns = [], []
    for spec in specifications:
        if set(spec) != {"name", "weights"} or not isinstance(spec["name"], str) or not spec["name"]:
            raise ValueError("Derived features require a name and component weights")
        weights = spec["weights"]
        if (not isinstance(weights, dict) or not weights or spec["name"] in index or spec["name"] in names
                or any(name not in index for name in weights)
                or any(isinstance(weight, bool) or not isinstance(weight, (int, float))
                       or not np.isfinite(weight) for weight in weights.values())):
            raise ValueError("Derived feature names, components or weights are invalid")
        names.append(spec["name"])
        columns.append(sum(matrix[:, index[name]] * weight for name, weight in weights.items()))
    return names, np.column_stack(columns) if columns else np.empty((len(matrix), 0))


def _verify_resumed_dataset(path, expected):
    """Require saved training and evaluation arrays to match rebuilt inputs."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f"Resume dataset fields differ: {path}")
        for name, values in expected.items():
            if not np.array_equal(saved[name], values):
                raise ValueError(f"Resume dataset differs for {path.name}/{name}")


def _verify_resumed_results(manifest, jobs):
    """Validate recorded forecasts and scores before retaining completed fits."""
    job_map = {(job["group"], job["feature_set"], job["candidate"]["id"]): job for job in jobs}
    if len(job_map) != len(jobs):
        raise ValueError("Resume contains duplicate planned candidate jobs")
    completed = set()
    for record in manifest["results"]:
        key = (record["group"], record["feature_set"], record["candidate"])
        if key not in job_map or key in completed:
            raise ValueError("Resume manifest has duplicate or unexpected candidate results")
        job = job_map[key]
        if job.get("rolling_refit"):
            refits = json.loads(Path(job["output"]).with_suffix(".refits.json").read_text())
            if len(refits) != record["rolling_refit_count"]:
                raise ValueError(f"Resume rolling refit records differ for {key}")
        frame = pd.read_csv(job["output"])
        with np.load(job["dataset"], allow_pickle=False) as dataset:
            saved_times = pd.DatetimeIndex(pd.to_datetime(dataset["starts"], utc=True))
            times = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True))
            if (not times.equals(saved_times) or not np.allclose(frame.actual, dataset["validation_y"], rtol=0, atol=1e-10)
                    or not np.allclose(frame.seasonal_naive, dataset["reference"], rtol=0, atol=1e-10)):
                raise ValueError(f"Resume prediction sample differs for {key}")
            scores = point_metrics(frame.actual, frame.point, frame.seasonal_naive)
        for name, score in scores.items():
            previous = record[name]
            if score is None or previous is None:
                if score is not None or previous is not None:
                    raise ValueError(f"Resume score differs for {key}/{name}")
            elif not np.isclose(score, previous, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Resume score differs for {key}/{name}")
        completed.add(key)
    return [job for key, job in job_map.items() if key not in completed]


def run_search(config, *, check_only=False, resume=False, allow_legacy_resume=False):
    """Prepare source-native matrices, then fit candidate workers on shared days."""
    study = config["price_model_search"]
    if not isinstance(study.get("common_training_rows", False), bool):
        raise ValueError("common_training_rows must be a boolean")
    # Reject a zero-spread baseline on any direct-price feature set before loading data.
    if any(candidate["family"] == "zero_spread" for candidate in study["candidates"]):
        anchors = study.get("residual_target_features", {})
        for split in study["product_groups"].values():
            for feature_set in split.get("feature_sets", study["feature_sets"]):
                if feature_set not in anchors:
                    raise ValueError("Zero-spread candidate requires a cleared-price anchor for every feature set")
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
    if selection_evidence:
        validation_manifest = json.loads(Path(selection_evidence["source_manifest"]).read_text())
        selection_evidence["price_source_verification"] = _verify_price_selection_sources(
            study, validation_manifest, source_manifest["plan"]["targets"])
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
    previous_manifest = None
    implementation = {"search_script": _sha256(Path(__file__)),
                      "candidate_factory": _sha256(ROOT / "src/forecasting/price_search_models.py")}
    if resume:
        previous_manifest = json.loads((destination / "manifest.json").read_text())
        if previous_manifest.get("status") not in {"running", "partial"}:
            raise ValueError("Resume requires an incomplete search manifest")
        if previous_manifest["config"] != study:
            raise ValueError("Resume configuration differs from the original search")
        previous_implementation = previous_manifest.get("implementation_sha256")
        if previous_implementation is None and not allow_legacy_resume:
            raise ValueError("Legacy resume requires explicit acknowledgement of missing implementation hashes")
        if previous_implementation is not None and previous_implementation != implementation:
            raise ValueError("Resume implementation differs from the original search")
    elif destination.exists():
        raise FileExistsError(f"Search output already exists: {destination}")
    if not check_only and not resume:
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
            boundary = split.get("market_neighbor_boundary", "unbounded")
            if boundary not in {"unbounded", "same_product"}:
                raise ValueError("Unknown neighboring-auction delivery-day boundary policy")
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
                    additions.append(_neighbor_price_feature(prior_values, neighbor_name, offset,
                                                             boundary=boundary, timezone=study["timezone"]))
                    expanded_names.append(neighbor_name)
                prior_lineage[market] = prior_evidence
            earlier_features = prepare_exogenous_features(pd.concat(additions, ignore_index=True))
            if set(exogenous.groups) & set(earlier_features.groups):
                raise ValueError("Earlier-auction feature names collide with catalog features")
            # Reuse the validated large catalog index and build only the small
            # market-specific additions rather than repeatedly copying it.
            group_exogenous = PreparedExogenous({**exogenous.groups, **earlier_features.groups})
            lineage[group] = {**evidence, "earlier_auction_features": prior_lineage,
                              "market_neighbor_offsets_minutes": offsets, "market_neighbor_boundary": boundary}
            extra_names = expanded_names
        lagged_specs = split.get("lagged_market_price_features", [])
        lagged_names = []
        if lagged_specs:
            additions, lagged_lineage = [], {}
            for spec in lagged_specs:
                if set(spec) != {"market", "lag_hours"}:
                    raise ValueError("Lagged auction feature requires market and lag_hours")
                market, lag = spec["market"], spec["lag_hours"]
                # Validate before querying, and ensure every feature name is unique.
                name = _lagged_auction_name(market, lag)
                if name in lagged_names:
                    raise ValueError("Duplicate lagged auction feature")
                source_spec = source_manifest["plan"]["targets"][market]["source"]
                past, provenance = load_price_history(study["database"], **source_spec,
                    interval_minutes=resolution, timezone=study["timezone"],
                    start=pd.Timestamp(split["history_start"]) - pd.Timedelta(hours=lag),
                    end=pd.Timestamp(split["history_end"]) - pd.Timedelta(hours=lag))
                past_values = past.rename(columns={"price_eur_per_mwh": "value"})[
                    ["product_start", "available_at", "value"]]
                additions.append(_lagged_auction_feature(past_values, market, lag))
                lagged_lineage[name] = provenance
                lagged_names.append(name)
            lagged_features = prepare_exogenous_features(pd.concat(additions, ignore_index=True))
            if set(group_exogenous.groups) & set(lagged_features.groups):
                raise ValueError("Lagged auction feature names collide with existing features")
            group_exogenous = PreparedExogenous({**group_exogenous.groups, **lagged_features.groups})
            lineage[group] = {**lineage[group], "lagged_auction_features": lagged_lineage}
        datasets = {}
        residual_indices = {}
        group_features = {}
        for feature_set in split.get("feature_sets", study["feature_sets"]):
            base_name = study.get("feature_set_aliases", {}).get(feature_set, feature_set)
            names = tuple(item["name"] for item in feature_sets[base_name])
            if feature_set in study.get("add_market_price_features_to", []):
                names += tuple(extra_names)
            if feature_set in study.get("add_lagged_market_features_to", []):
                names += tuple(lagged_names)
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
        derived_names_by_set = {}
        for feature_set, names in group_features.items():
            columns = list(range(base_count)) + [feature_index[name] for name in names]
            datasets[feature_set] = project_search_matrix(full, columns)
            derived_names_by_set[feature_set] = []
            if feature_set in study.get("add_derived_features_to", []):
                # Derivations use inputs already selected as of each historical
                # decision time; missing component values remain missing.
                derived_names, derived = _linear_derived_features(
                    datasets[feature_set]["matrix"][:, base_count:], names, study.get("derived_features", []))
                derived_names_by_set[feature_set] = derived_names
                datasets[feature_set]["matrix"] = np.column_stack([datasets[feature_set]["matrix"], derived])
                finite = np.isfinite(datasets[feature_set]["matrix"]).all(axis=1)
                datasets[feature_set]["training"] = finite & datasets[feature_set]["training_clock"]
                datasets[feature_set]["validation"] = finite & datasets[feature_set]["validation_clock"]
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
                     + [f"calendar_{index}" for index in range(7)] + list(group_features[feature_set])
                     + derived_names_by_set[feature_set])
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
            if check_only and not resume:
                continue
            dataset = destination / f"{group}__{feature_set}.npz"
            arrays = dict(train_x=data["matrix"][data["training"]], train_y=data["target"][data["training"]],
                     train_starts=np.asarray(data["starts"][data["training"]].astype(str), dtype=str),
                     validation_x=data["matrix"][selected], validation_y=data["target"][selected],
                     reference=data["matrix"][selected, 0],
                     decisions_ns=data["decisions"].as_unit("ns").asi8[selected],
                     available_ns=data["label_available_at"].as_unit("ns").asi8[selected],
                     product_end_ns=data["label_product_end"].as_unit("ns").asi8[selected],
                     starts=np.asarray(data["starts"][selected].astype(str), dtype=str))
            if resume:
                _verify_resumed_dataset(dataset, arrays)
            else:
                np.savez(dataset, **arrays)
            for candidate in study["candidates"]:
                if selected_candidates is not None and (feature_set, candidate["id"]) not in selected_candidates[group]:
                    continue
                jobs.append({"dataset": str(dataset), "output": str(destination / f"{group}__{feature_set}__{candidate['id']}.csv"),
                    "candidate": candidate, "group": group, "feature_set": feature_set,
                    "seed": study["seed"], "threads": study["threads_per_worker"],
                    "residual_feature_index": residual_indices[feature_set], "timezone": study["timezone"],
                    "rolling_refit": study.get("rolling_refit")})
    if resume:
        rebuilt = {"coverage": coverage, "source_lineage": lineage, "feature_sha256": _sha256(feature_path),
                   "supplemental_features": supplemental_evidence}
        for key, value in rebuilt.items():
            if json.dumps(value, sort_keys=True, default=str) != json.dumps(previous_manifest[key], sort_keys=True, default=str):
                raise ValueError(f"Resume {key} differs from the original search")
        jobs = _verify_resumed_results(previous_manifest, jobs)
    if check_only:
        return {"status": "preflight_complete", "coverage": coverage, "incomplete_days": incomplete_days,
                "feature_checksum": _sha256(feature_path),
                "supplemental_features": supplemental_evidence,
                "candidate_count": len(study["candidates"]), "selection": selection_evidence,
                "resume_pending_fits": len(jobs) if resume else None,
                "resume_legacy_implementation_unverified": bool(resume and previous_manifest.get("implementation_sha256") is None)}
    manifest = {"status": "running", "config": study, "coverage": coverage, "incomplete_days": incomplete_days,
                "source_lineage": lineage,
                "feature_sha256": _sha256(feature_path), "supplemental_features": supplemental_evidence,
                "selection": selection_evidence, "implementation_sha256": implementation, "results": [], "failures": []}
    if resume:
        manifest["results"] = previous_manifest["results"]
        manifest["previous_failures"] = previous_manifest.get("previous_failures", []) + previous_manifest.get("failures", [])
        manifest["resume_note"] = ("Earlier fits may lack per-candidate convergence-warning records; "
            "legacy implementation hashes were unavailable." if previous_manifest.get("implementation_sha256") is None
            else "Earlier fits may lack per-candidate convergence-warning records.")
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
    parser.add_argument("--project-root", type=Path, help="Root containing the selected source and output directories")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--resume-check-only", action="store_true")
    parser.add_argument("--allow-legacy-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.project_root is not None:
        config["_project_root"] = str(args.project_root.resolve())
    result = run_search(config, check_only=args.check_only or args.resume_check_only,
                        resume=args.resume or args.resume_check_only,
                        allow_legacy_resume=args.allow_legacy_resume) if args.run or args.check_only or args.resume or args.resume_check_only else config["price_model_search"]
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
