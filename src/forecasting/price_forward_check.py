"""Frozen forward evaluation of one preregistered native DA price model.

The original feature bytes and selected hyperparameters stay fixed. Weekly
coefficient refits and rolling error updates use strictly earlier eligible
observations. New labels never select a model or a Gaussian policy. Inputs are
read exclusively from an explicitly configured DuckDB snapshot catalog.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..data.api_catalog import load_api_cache_records
from .gaussian_adaptive import _aggregate, evaluate_adaptive_gaussian
from .gaussian_prices import GaussianPriceCalibration, evaluate_gaussian_prices
from .linear_price_study import BASELINES, LIMITATION, _predict_candidate, _scores
from .price_experiment import _write_frame, prepare_native_prices
from .price_features import build_linear_price_features
from .price_pipeline import _plot_paths, _sample_day, _write_json


SCHEMA = "frozen-price-forward-check-v1"
PROTOCOL_SCHEMA = "frozen-price-validation-protocol-v1"
PINNED_PROTOCOL_SCHEMA = "frozen-price-validation-protocol-v2"


def _hashed_json(path: Path, expected: str | None = None) -> tuple[dict, bytes, str]:
    """Read one immutable source and reject any changed pinned bytes."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen source SHA256 mismatch: {path}")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Frozen protocol and manifests must be JSON objects")
    return result, raw, digest


def _source_file(manifest: Mapping[str, Any], directory: Path, name: str, default: str) -> Path:
    """Resolve source-owned artifact references without changing their bytes."""
    path = Path(manifest.get(name, default))
    return path if path.is_absolute() else directory / path


def _regime(manifest: Mapping[str, Any]) -> dict:
    """Require one unambiguous selected native quarter-hour regime."""
    regimes = [value for value in manifest.get("regimes", [])
               if value.get("native_interval_minutes") == 15 and value.get("status") == "evaluated"]
    if len(regimes) != 1:
        raise ValueError("Frozen forward checks require exactly one evaluated native 15-minute source regime")
    return regimes[0]


def _frame(path: Path, expected: str | None = None) -> tuple[pd.DataFrame, str]:
    """Hash before reading frozen Parquet so no altered bytes enter a replay."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen source SHA256 mismatch: {path}")
    return pd.read_parquet(BytesIO(raw)), digest


def _days(first: str, stop: str) -> list[str]:
    """Expand an exclusive civil-date interval without assuming 96 rows per day."""
    begin, end = date.fromisoformat(first), date.fromisoformat(stop)
    if begin >= end:
        raise ValueError("Frozen forward dates must define a positive exclusive interval")
    return [value.date().isoformat() for value in pd.date_range(begin, end, inclusive="left")]


def _missing_native_days(native: pd.DataFrame, requested_days: list[str], timezone_name: str) -> list[str]:
    """Require every exact native product; gaps and missing labels stay missing."""
    missing = []
    groups = {} if native.empty else dict(tuple(native.groupby("delivery_day", sort=False)))
    for day in requested_days:
        midnight = pd.Timestamp(day, tz=timezone_name)
        following = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=timezone_name)
        expected = pd.date_range(midnight.tz_convert("UTC"), following.tz_convert("UTC"),
                                 freq="15min", inclusive="left").as_unit("ns")
        values = groups.get(day)
        if values is None:
            missing.append(day)
            continue
        values = values.sort_values("product_start")
        starts = pd.DatetimeIndex(values.product_start).as_unit("ns")
        ends = pd.DatetimeIndex(values.product_end).as_unit("ns")
        if (not starts.equals(expected) or not ends.equals(expected + pd.Timedelta(minutes=15))
                or not values.native_interval_minutes.eq(15).all()
                or not np.isfinite(values.price_eur_mwh.to_numpy(float)).all()):
            missing.append(day)
    return missing


def _compare_predictions(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    """Verify that replaying the unchanged past reproduces its fitted decisions."""
    keys = ["product_start", "product_end"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if len(actual) != len(expected) or actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Frozen replay overlap product coverage changed")
    for name in keys + ["fit_cutoff", "forecast_cutoff"]:
        left, right = pd.to_datetime(actual[name], utc=True), pd.to_datetime(expected[name], utc=True)
        if not np.array_equal(left.array.as_unit("ns").asi8, right.array.as_unit("ns").asi8):
            raise ValueError(f"Frozen replay overlap changed {name}")
    if actual.training_max_delivery_day.astype(str).tolist() != expected.training_max_delivery_day.astype(str).tolist():
        raise ValueError("Frozen replay training history or weekly refit anchor changed")
    delta = np.abs(actual.prediction.to_numpy(float) - expected.prediction.to_numpy(float))
    if not np.array_equal(actual.actual.to_numpy(float), expected.actual.to_numpy(float)) or not np.all(delta <= 1e-9):
        raise ValueError("Frozen replay overlap predictions or source labels changed")
    return {"rows": len(actual), "maximum_prediction_difference_eur_mwh": float(delta.max(initial=0)),
            "comparison_tolerance_eur_mwh": 1e-9, "cutoffs_and_training_days_identical": True}


def _common_columns(groups: Mapping[str, list[str]], settings: Mapping[str, Any]) -> list[str]:
    """Recreate the original common-product mask without selecting candidates."""
    names = ["price_eur_mwh", "lag_1d", "lag_7d"]
    for selected in settings.get("feature_sets", {"lags": ["calendar", "autoregressive"]}).values():
        for group in selected:
            names.extend(groups[group])
    return list(dict.fromkeys(names))


def _plot_check(fresh: pd.DataFrame, scores: pd.DataFrame, path: Path) -> None:
    """Show fresh point errors and frozen-policy uncertainty coverage by day."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    point = fresh.assign(error=(fresh.actual - fresh.prediction).abs()).groupby("delivery_day").error.mean()
    coverage = scores.groupby("delivery_day").covered_95.mean()
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(pd.to_datetime(point.index), point.values)
    axes[0].set(ylabel="Point MAE (EUR/MWh)", title="Frozen protocol: new delivery days only")
    axes[1].plot(pd.to_datetime(coverage.index), coverage.values, label="Adaptive Gaussian 95% coverage")
    axes[1].axhline(.95, color="black", linestyle="--", linewidth=.8)
    axes[1].set(ylabel="Coverage", xlabel="Delivery day", ylim=(0, 1.02))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run_frozen_price_check(config: Mapping[str, Any], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Replay a pinned protocol, refusing to fit until every new label is ready."""
    settings = dict(config.get("price_forward_check", {}))
    allowed = {"protocol", "cache_directory", "output_directory", "plots", "sample_count", "seed"}
    if set(settings) - allowed or not settings.get("protocol"):
        raise ValueError("price_forward_check requires a protocol; model/policy overrides are forbidden")
    protocol_path = Path(settings["protocol"]).resolve()
    protocol, protocol_bytes, protocol_hash = _hashed_json(protocol_path)
    if protocol.get("schema_version") not in (PROTOCOL_SCHEMA, PINNED_PROTOCOL_SCHEMA):
        raise ValueError("Unsupported frozen forward price protocol")
    ancillary_pinned = protocol["schema_version"] == PINNED_PROTOCOL_SCHEMA
    ancillary_fields = ("source_mean_settings_sha256", "source_mean_feature_groups_sha256", "source_static_gaussian_sha256")
    if ancillary_pinned and any(not isinstance(protocol.get(name), str) or len(protocol[name]) != 64 for name in ancillary_fields):
        raise ValueError("Version 2 forward protocols must pin settings, feature groups and static Gaussian SHA256s")
    database = Path(settings.get("cache_directory", protocol.get("database", "data/api/market.duckdb")))
    if database.suffix.lower() != ".duckdb" or database.is_dir():
        raise ValueError("Frozen forward inputs require a DuckDB database; legacy file caches are not permitted")
    source_path = Path(protocol["source_mean_study"]).resolve()
    uncertainty_path = Path(protocol["source_uncertainty_study"]).resolve()
    source, source_bytes, source_hash = _hashed_json(source_path, protocol["source_mean_sha256"])
    uncertainty, uncertainty_bytes, uncertainty_hash = _hashed_json(uncertainty_path, protocol["source_uncertainty_sha256"])
    if (source.get("status") not in {"complete", "completed"} or uncertainty.get("status") not in {"complete", "completed"}
            or uncertainty.get("source_manifest_sha256") != source_hash):
        raise ValueError("Frozen mean and uncertainty studies must be completed and share the pinned source")
    mean_regime, noise_regime = _regime(source), _regime(uncertainty)
    candidate, policy = protocol["selected_mean_specification"], protocol["selected_gaussian_policy"]
    if candidate != mean_regime["selected_candidate"] or policy != noise_regime["selected_policy"]:
        raise ValueError("Frozen protocol model specification or Gaussian policy differs from its source studies")
    if policy.get("native_interval_minutes") != 15 or not policy.get("fixed_after_selection"):
        raise ValueError("The frozen Gaussian policy must explicitly cover native quarter-hour prices")
    maximum = str(protocol["maximum_seen_delivery_day"])
    first, stop = str(protocol["new_test_start"]), str(protocol["new_test_end_exclusive"])
    requested_days = _days(first, stop)
    if source.get("last_delivery_day") != maximum or date.fromisoformat(first) != date.fromisoformat(maximum) + timedelta(days=1):
        raise ValueError("New evaluation must append directly after the frozen maximum seen delivery day")
    source_settings, source_settings_bytes, settings_hash = _hashed_json(source_path.parent / "settings.json",
        protocol.get("source_mean_settings_sha256"))
    groups, groups_bytes, groups_hash = _hashed_json(source_path.parent / "feature_groups.json",
        protocol.get("source_mean_feature_groups_sha256"))
    static_path = uncertainty_path.parent / "15min" / "static" / "gaussian_calibration.json"
    _, _, static_hash = _hashed_json(static_path, protocol.get("source_static_gaussian_sha256"))
    static = GaussianPriceCalibration.load(static_path)
    if hashlib.sha256(static_path.read_bytes()).hexdigest() != static_hash:
        raise ValueError("Static Gaussian source changed while loading its verified parameters")
    timezone_name = str(source_settings.get("timezone", "Europe/Rome"))
    if source_settings.get("refit_every_days", 7) != 7 or policy.get("timezone", "Europe/Rome") != timezone_name:
        raise ValueError("Frozen protocol requires the original seven-day refit schedule and market timezone")
    if (source_settings.get("fundamental_features") or source_settings.get("feature_settings", {}).get("fundamental_columns")
            or set(candidate.get("feature_groups", [])) - {"calendar", "calendar_extended", "autoregressive", "autoregressive_extended", "daily_summary", "rolling_history", "daily_profile_lags"}):
        raise ValueError("Initial frozen forward checks support price-only features; new fundamentals need a separate protocol")
    old_features, feature_hash = _frame(source_path.parent / "features.parquet", source["frozen_features_sha256"])
    frozen_hashes = uncertainty["frozen_input_sha256"]
    calibration_path = _source_file(source, source_path.parent, "calibration_predictions", "calibration_predictions.parquet")
    test_path = _source_file(source, source_path.parent, "test_predictions", "test_predictions.parquet")
    calibration, calibration_hash = _frame(calibration_path, frozen_hashes["calibration_predictions.parquet"])
    original_test, original_test_hash = _frame(test_path, frozen_hashes["test_predictions.parquet"])
    selected_original, selected_hash = _frame(source_path.parent / "selected_predictions.parquet")
    calibration = calibration.loc[calibration.native_interval_minutes.eq(15)].copy()
    original_test = original_test.loc[original_test.native_interval_minutes.eq(15)].copy()
    selected_original = selected_original.loc[selected_original.native_interval_minutes.eq(15)].copy()
    if old_features.delivery_day.max() != maximum or calibration.empty or original_test.empty or selected_original.empty:
        raise ValueError("Frozen feature and prediction boundaries are inconsistent")
    hashes = {"protocol.json": protocol_hash, "source_mean_manifest.json": source_hash,
              "source_uncertainty_manifest.json": uncertainty_hash, "features.parquet": feature_hash,
              "calibration_predictions.parquet": calibration_hash, "original_test_predictions.parquet": original_test_hash,
              "selected_predictions.parquet": selected_hash, "settings.json": settings_hash,
              "feature_groups.json": groups_hash, "static_gaussian_calibration.json": static_hash}
    if frozen_hashes.get("source_manifest.json") != source_hash:
        raise ValueError("Uncertainty input manifest bytes differ from the pinned mean study")
    count = settings.get("sample_count", 100)
    seed = settings.get("seed", uncertainty.get("settings", {}).get("seed", 41))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Forward path sample_count/seed must be valid nonnegative integer settings")

    def waiting(missing: list[str], reason: str) -> dict[str, Any]:
        """Leave all sources and output locations untouched while data arrives."""
        return {"schema_version": SCHEMA, "status": "waiting_for_data", "missing_days": missing,
                "requested_days": len(requested_days), "reason": reason,
                "protocol": str(protocol_path), "protocol_sha256": protocol_hash,
                "database": str(database), "fits_performed": 0, "outputs_created": False,
                "operational_eligible": False}

    if progress:
        progress("Checking only new native DA snapshots in DuckDB; no API requests")
    start_utc = pd.Timestamp(first, tz=timezone_name).tz_convert("UTC")
    end_utc = pd.Timestamp(stop, tz=timezone_name).tz_convert("UTC")
    records, snapshots = load_api_cache_records(database, provider=source_settings.get("provider", "entsoe"),
        dataset=source_settings.get("dataset", "day_ahead_prices"), zone=source_settings.get("zone", "NORD"),
        start=start_utc, end=end_utc, verify_checksums=True)
    if records.empty:
        return waiting(requested_days, "New DA snapshots are not yet cached")
    if not snapshots or any(value.get("storage", {}).get("backend") != "duckdb" or not value.get("storage", {}).get("snapshot_id") for value in snapshots):
        raise ValueError("New price inputs must have verified pinned DuckDB snapshot identifiers")
    expected_labels = {"provider": source_settings.get("provider", "entsoe"),
                       "dataset": source_settings.get("dataset", "day_ahead_prices"),
                       "zone": source_settings.get("zone", "NORD"), "auction": "DA"}
    if expected_labels["dataset"] != "day_ahead_prices":
        raise ValueError("Frozen forward checks support actual day-ahead prices only")
    for field, expected in expected_labels.items():
        if field not in records or not records[field].eq(expected).all():
            raise ValueError(f"New price {field} differs from the frozen source market")
        if field in old_features and not old_features[field].eq(expected).all():
            raise ValueError(f"Frozen feature {field} differs from its source settings")
    finite = pd.to_numeric(records.price_eur_mwh, errors="coerce")
    try:
        extension = prepare_native_prices(records.loc[np.isfinite(finite)].copy(), source_settings)
    except ValueError as error:
        return waiting(requested_days, f"New native price products are incomplete or invalid: {error}")
    if extension.delivery_day.lt(first).any() or extension.delivery_day.ge(stop).any():
        raise ValueError("DuckDB extension read returned prices outside the preregistered new date window")
    missing = _missing_native_days(extension, requested_days, timezone_name)
    if missing:
        return waiting(missing, "Every requested day needs all exact finite native quarter-hour prices")
    # Rebuild lags using frozen historical values, and assert that appended days
    # cannot revise any already-frozen feature. Use the exact old rows in fits.
    feature_settings = dict(source_settings.get("feature_settings", {}))
    feature_settings.setdefault("fundamental_columns", [])
    rebuilt, _ = build_linear_price_features(pd.concat([old_features, extension], ignore_index=True),
                                             timezone_name, settings=feature_settings)
    historical = rebuilt.loc[rebuilt.delivery_day.le(maximum), old_features.columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(historical, old_features.reset_index(drop=True), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError:
        raise ValueError("Appending new prices changed frozen historical feature values") from None
    features = pd.concat([old_features, rebuilt.loc[rebuilt.delivery_day.ge(first)]], ignore_index=True)
    portion = features.loc[features.native_interval_minutes.eq(15)].sort_values("product_start").reset_index(drop=True)
    columns = _common_columns(groups, source_settings)
    if set(candidate["feature_names"]) - set(columns):
        raise ValueError("The selected frozen mean features differ from the original common-product mask")
    common = pd.Series(np.isfinite(portion[columns].to_numpy(float)).all(axis=1), index=portion.index)
    missing_features = sorted(portion.loc[portion.delivery_day.ge(first) & ~common, "delivery_day"].unique())
    if missing_features:
        return waiting(missing_features, "Frozen lag history cannot form complete new prediction features")
    splits = mean_regime["split"]
    replay_days = sorted(portion.loc[portion.delivery_day.ge(splits["calibration_start_day"]), "delivery_day"].unique())

    def phase(day: str) -> str:
        if day >= first:
            return "forward_test"
        return "calibration" if day < splits["maturity_gap_start_day"] else "maturity_gap" if day < splits["test_start_day"] else "test"

    if progress:
        progress("Replaying the frozen mean specification from its original weekly refit anchor")
    predictions, models = _predict_candidate(portion, candidate, replay_days, source_settings, common, phase)
    overlap = predictions.loc[predictions.delivery_day.le(maximum)].copy()
    overlap_audit = _compare_predictions(overlap, selected_original)
    fresh = predictions.loc[predictions.delivery_day.ge(first)].copy()
    if len(fresh) != len(extension):
        raise ValueError("Frozen selected model does not predict every new native product")
    baselines = []
    for name in BASELINES:
        specification = {"id": f"baseline_{name}", "kind": "baseline", "baseline": name,
                         "feature_names": ["lag_1d", "lag_7d"]}
        values, _ = _predict_candidate(portion, specification, requested_days, source_settings, common, lambda day: "forward_test")
        if len(values) != len(fresh) or not values.product_start.reset_index(drop=True).equals(fresh.product_start.reset_index(drop=True)):
            raise ValueError("Frozen point model and baselines must use identical fresh native products")
        baselines.append(values)
    baseline_rows = pd.concat(baselines, ignore_index=True)
    point_metrics = {"selected": _scores(fresh), **{name: _scores(values) for name, values in zip(BASELINES, baselines)}}
    for name in BASELINES:
        baseline_mae = point_metrics[name]["mae_eur_mwh"]
        point_metrics[name]["selected_mae_skill"] = (None if baseline_mae == 0 else
            1 - point_metrics["selected"]["mae_eur_mwh"] / baseline_mae)
    combined_test = predictions.loc[predictions.delivery_day.ge(splits["test_start_day"])].copy()
    _compare_predictions(combined_test.loc[combined_test.delivery_day.le(maximum)], original_test)
    if progress:
        progress("Updating only earlier mature Gaussian errors under the frozen policy; scoring new days only")
    adaptive = evaluate_adaptive_gaussian(calibration, combined_test, policy)
    daily = []
    for day, values in fresh.groupby("delivery_day", sort=True):
        if day not in adaptive["daily_calibrations"]:
            raise ValueError(f"Frozen Gaussian policy cannot score fresh day {day}; no new selection is allowed")
        daily.append(evaluate_gaussian_prices(adaptive["daily_calibrations"][day], values))
    fresh_scores, adaptive_metrics = _aggregate(daily)
    if static.auction != "DA" or pd.Timestamp(static.calibration_cutoff) != pd.to_datetime(calibration.product_end, utc=True).max():
        raise ValueError("Frozen static Gaussian artifact does not match the original calibration boundary")
    static_scored = evaluate_gaussian_prices(static, fresh)
    sample_day = str(fresh.delivery_day.min())
    sample = _sample_day(adaptive["daily_calibrations"][sample_day], fresh, fresh_scores, count, seed)
    # Publish only completed evidence; missing-data checks above create nothing.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parent = Path(settings.get("output_directory", "outputs/price_forward_checks"))
    directory, pending = parent / run_id, parent / f".{run_id}.pending"
    pending.mkdir(parents=True, exist_ok=False)
    for filename, value in (("protocol.json", protocol_bytes), ("source_mean_manifest.json", source_bytes),
                            ("source_uncertainty_manifest.json", uncertainty_bytes),
                            ("source_settings.json", source_settings_bytes), ("source_feature_groups.json", groups_bytes)):
        (pending / filename).write_bytes(value)
    for name, frame in (("new_native_prices", extension), ("features", features),
                        ("selected_predictions", predictions), ("fresh_predictions", fresh),
                        ("bridge_predictions", combined_test.loc[combined_test.delivery_day.lt(first)]),
                        ("baseline_predictions", baseline_rows), ("calibration_predictions", calibration),
                        ("adaptive_fresh_scores", fresh_scores), ("static_fresh_scores", static_scored["scored_predictions"])):
        _write_frame(frame, pending, name)
    for day, artifact in adaptive["daily_calibrations"].items():
        artifact.save(pending / "daily_calibrations" / f"{day}.json")
    static.save(pending / "static" / "gaussian_calibration.json")
    _write_json(pending / "new_source_snapshots.json", snapshots)
    _write_json(pending / "selected_specification.json", candidate)
    _write_json(pending / "frozen_gaussian_policy.json", policy)
    _write_json(pending / "weekly_model_refits.json", models)
    _write_json(pending / "daily_gaussian_audit.json", adaptive["daily_audit"])
    _write_json(pending / "point_metrics.json", point_metrics)
    _write_json(pending / "adaptive_metrics.json", adaptive_metrics)
    _write_json(pending / "static_metrics.json", static_scored["metrics"])
    np.savez_compressed(pending / f"paths_{sample_day}.npz", prices_eur_mwh=sample["paths"],
        innovations_eur_mwh=sample["innovations"], point_forecast_eur_mwh=sample["point"],
        bias_eur_mwh=sample["bias"], gaussian_mean_eur_mwh=sample["mean"], covariance_eur2_mwh2=sample["covariance"],
        probabilities=np.full(count, 1 / count))
    _write_frame(sample["products"], pending, f"products_{sample_day}")
    plots = []
    if settings.get("plots", True):
        _plot_check(fresh, fresh_scores, pending / "fresh_daily_metrics.png")
        _plot_paths(sample["products"], sample["paths"], sample["mean"], pending / f"paths_{sample_day}.png")
        plots = ["fresh_daily_metrics.png", f"paths_{sample_day}.png"]
    result = {"schema_version": SCHEMA, "status": "completed", "run_id": run_id,
        "output_dir": str(directory.resolve()), "manifest": str((directory / "manifest.json").resolve()),
        "protocol": str(protocol_path), "protocol_sha256": protocol_hash, "frozen_source_sha256": hashes,
        "static_artifact_sha256": static_hash, "source_ancillary_hashes_pinned_in_protocol": ancillary_pinned,
        "database": str(database), "new_snapshot_count": len(snapshots), "new_source_snapshots": snapshots,
        "maximum_previously_seen_delivery_day": maximum, "new_test_start": first, "new_test_end_exclusive": stop,
        "fresh_days": int(fresh.delivery_day.nunique()), "fresh_rows": len(fresh),
        "bridge_days": int(combined_test.loc[combined_test.delivery_day.lt(first), "delivery_day"].nunique()),
        "selected_mean_specification": candidate, "selected_gaussian_policy": policy,
        "refit_every_days": 7, "original_refit_anchor": splits["calibration_start_day"], "overlap_verification": overlap_audit,
        "point_metrics": point_metrics, "adaptive_metrics": adaptive_metrics, "static_metrics": static_scored["metrics"],
        "test_labels_used_for_selection": False, "source_historical_features_unchanged": True,
        "evaluation_design": "Previously uninspected historical extension; protocol saved before extension outcomes were inspected, not a real-time prospective forecast",
        "sample_day": sample_day, "sample_count": count, "seed": seed, "plots": plots,
        "operational_eligible": False, "historical_publication_verified": False, "limitation": LIMITATION}
    _write_json(pending / "manifest.json", result)
    pending.rename(directory)
    return result
