"""Frozen forward checks use synthetic artifacts and a mocked DuckDB catalog."""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting import price_forward_check as forward
from bess_pv_trading.forecasting.gaussian_prices import calibrate_gaussian_prices
from bess_pv_trading.forecasting.linear_price_study import _predict_candidate
from bess_pv_trading.forecasting.price_experiment import prepare_native_prices
from bess_pv_trading.forecasting.price_features import build_linear_price_features


def write_json(path, value):
    """Write independent source fixtures with stable bytes for SHA assertions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_prices(first="2025-11-01", stop="2025-12-19"):
    """Generate ordinary-day native prices with calendar-dependent residuals."""
    starts = pd.date_range(pd.Timestamp(first, tz="Europe/Rome").tz_convert("UTC"),
        pd.Timestamp(stop, tz="Europe/Rome").tz_convert("UTC"), freq="15min", inclusive="left")
    local = starts.tz_convert("Europe/Rome")
    day = np.array([(value.date() - date(2025, 11, 1)).days for value in local])
    clock = np.asarray(local.hour + local.minute / 60)
    values = 80 + .4 * day + 15 * np.cos(clock / 24 * 2 * np.pi) + 3 * np.sin(day * .7) * np.cos(clock / 12)
    return pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
        "price_eur_mwh": values, "retrieved_at": pd.Timestamp("2026-01-01T00:00Z"),
        "provider": "entsoe", "dataset": "day_ahead_prices", "zone": "NORD", "auction": "DA",
        "unit": "EUR/MWh", "historical_feature_eligible": False})


@pytest.fixture
def frozen_source(tmp_path):
    """Create and freeze a past experiment before exposing new synthetic labels."""
    source_dir, noise_dir = tmp_path / "original_mean", tmp_path / "original_noise"
    source_dir.mkdir()
    noise_dir.mkdir()
    source_settings = {"timezone": "Europe/Rome", "provider": "entsoe", "dataset": "day_ahead_prices",
        "zone": "NORD", "cache_directory": str(tmp_path / "market.duckdb"),
        "min_training_days": 10, "refit_every_days": 7,
        "native_product_regimes": [{"start": "2025-10-01", "end": None, "minutes": 15}],
        "feature_sets": {"lags": ["calendar", "autoregressive"]},
        "feature_settings": {"lag_days": [1, 7], "daily_summary_lags": [], "rolling_windows": [],
                             "daily_profile_lags": [], "fundamental_columns": []}}
    all_prices = synthetic_prices()
    old = all_prices.loc[all_prices.product_start < pd.Timestamp("2025-12-16", tz="Europe/Rome")].copy()
    native = prepare_native_prices(old, source_settings)
    features, groups = build_linear_price_features(native, settings=source_settings["feature_settings"])
    candidate = {"id": "frozen_synthetic_ridge", "kind": "ridge", "feature_set": "lags",
        "feature_groups": ["calendar", "autoregressive"],
        "feature_names": list(dict.fromkeys(groups["calendar"] + groups["autoregressive"])),
        "alpha": 10., "training_window_days": 30, "target": "level", "model_structure": "pooled"}
    common = pd.Series(np.isfinite(features[["price_eur_mwh", *candidate["feature_names"]]].to_numpy(float)).all(axis=1), index=features.index)
    replay_days = [value.date().isoformat() for value in pd.date_range("2025-12-01", "2025-12-15")]
    phase = lambda day: "calibration" if day < "2025-12-06" else "maturity_gap" if day < "2025-12-07" else "test"
    predictions, _ = _predict_candidate(features, candidate, replay_days, source_settings, common, phase)
    calibration = predictions.loc[predictions.phase.eq("calibration")].copy()
    tests = predictions.loc[predictions.phase.eq("test")].copy()
    for name, frame in (("features", features), ("selected_predictions", predictions),
                        ("calibration_predictions", calibration), ("test_predictions", tests)):
        frame.to_parquet(source_dir / f"{name}.parquet", index=False)
    write_json(source_dir / "settings.json", source_settings)
    write_json(source_dir / "feature_groups.json", groups)
    split = {"calibration_start_day": "2025-12-01", "calibration_end_day": "2025-12-05",
             "maturity_gap_start_day": "2025-12-06", "test_start_day": "2025-12-07", "test_end_day": "2025-12-15"}
    source = {"status": "completed", "last_delivery_day": "2025-12-15",
        "frozen_features_sha256": digest(source_dir / "features.parquet"),
        "regimes": [{"native_interval_minutes": 15, "status": "evaluated", "selected_candidate": candidate, "split": split}]}
    write_json(source_dir / "manifest.json", source)
    static = calibrate_gaussian_prices(calibration, calibration_cutoff=calibration.product_end.max(), minimum_complete_days=2)
    static_path = noise_dir / "15min/static/gaussian_calibration.json"
    static.save(static_path)
    policy = {"schema": "adaptive-gaussian-price-policy-v1", "window_days": 14,
        "covariance_shrinkage": .5, "minimum_complete_days": 2,
        "minimum_variance_eur2_mwh2": 1e-6, "timezone": "Europe/Rome", "native_interval_minutes": 15,
        "selection_source": "calibration_predictions_only", "selection_max_product_end": calibration.product_end.max().isoformat(),
        "fixed_after_selection": True}
    uncertainty = {"status": "completed", "source_manifest_sha256": digest(source_dir / "manifest.json"),
        "frozen_input_sha256": {"source_manifest.json": digest(source_dir / "manifest.json"),
            "calibration_predictions.parquet": digest(source_dir / "calibration_predictions.parquet"),
            "test_predictions.parquet": digest(source_dir / "test_predictions.parquet")},
        "settings": {"seed": 41}, "regimes": [{"native_interval_minutes": 15, "status": "evaluated", "selected_policy": policy}]}
    write_json(noise_dir / "manifest.json", uncertainty)
    protocol = {"schema_version": "frozen-price-validation-protocol-v2",
        "database": str(tmp_path / "market.duckdb"), "source_mean_study": str(source_dir / "manifest.json"),
        "source_uncertainty_study": str(noise_dir / "manifest.json"),
        "source_mean_sha256": digest(source_dir / "manifest.json"),
        "source_uncertainty_sha256": digest(noise_dir / "manifest.json"),
        "source_mean_settings_sha256": digest(source_dir / "settings.json"),
        "source_mean_feature_groups_sha256": digest(source_dir / "feature_groups.json"),
        "source_static_gaussian_sha256": digest(static_path),
        "maximum_seen_delivery_day": "2025-12-15", "new_test_start": "2025-12-16",
        "new_test_end_exclusive": "2025-12-19", "selected_mean_specification": candidate, "selected_gaussian_policy": policy}
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    extension = all_prices.loc[all_prices.product_start >= pd.Timestamp("2025-12-16", tz="Europe/Rome")].copy()
    config = {"price_forward_check": {"protocol": str(protocol_path), "cache_directory": str(tmp_path / "market.duckdb"),
        "output_directory": str(tmp_path / "forward_outputs"), "plots": False}}
    return {"config": config, "protocol": protocol, "protocol_path": protocol_path,
            "source_dir": source_dir, "noise_dir": noise_dir, "extension": extension, "tmp_path": tmp_path}


def mock_catalog(monkeypatch, fixture, records=None):
    """Expose synthetic new rows only and assert verified DuckDB-bound selection."""
    extension = fixture["extension"] if records is None else records
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        assert Path(path).suffix == ".duckdb"
        assert kwargs["provider"] == "entsoe" and kwargs["dataset"] == "day_ahead_prices"
        assert kwargs["verify_checksums"] is True
        assert kwargs["start"] == pd.Timestamp("2025-12-16", tz="Europe/Rome").tz_convert("UTC")
        assert kwargs["end"] == pd.Timestamp("2025-12-19", tz="Europe/Rome").tz_convert("UTC")
        return extension.copy(), [{"storage": {"backend": "duckdb", "snapshot_id": "synthetic-snapshot-1"},
                                    "checksums_verified": True}]

    monkeypatch.setattr(forward, "load_api_cache_records", load)
    return calls


def test_frozen_extension_replays_original_anchor_and_saves_fresh_only_metrics(frozen_source, monkeypatch):
    fixture = frozen_source
    calls = mock_catalog(monkeypatch, fixture)
    source_hash = digest(fixture["source_dir"] / "features.parquet")
    result = forward.run_frozen_price_check(fixture["config"])
    directory = Path(result["output_dir"])
    assert result["status"] == "completed" and len(calls) == 1
    assert result["fresh_days"] == 3 and result["fresh_rows"] == 288 and result["bridge_days"] == 9
    assert result["original_refit_anchor"] == "2025-12-01"
    assert result["selected_mean_specification"] == fixture["protocol"]["selected_mean_specification"]
    assert result["selected_gaussian_policy"] == fixture["protocol"]["selected_gaussian_policy"]
    assert result["adaptive_metrics"]["scored_rows"] == 288 and result["static_metrics"]["scored_rows"] == 288
    assert result["overlap_verification"]["maximum_prediction_difference_eur_mwh"] <= 1e-9
    assert result["source_ancillary_hashes_pinned_in_protocol"]
    assert not result["test_labels_used_for_selection"] and not result["operational_eligible"]
    assert digest(fixture["source_dir"] / "features.parquet") == source_hash
    fresh = pd.read_parquet(directory / "fresh_predictions.parquet")
    assert fresh.delivery_day.min() == "2025-12-16" and fresh.delivery_day.max() == "2025-12-18"
    assert fresh.fit_cutoff.nunique() == 1
    assert fresh.fit_cutoff.iloc[0] == pd.Timestamp("2025-12-14T11:00Z")
    assert len(pd.read_parquet(directory / "baseline_predictions.parquet")) == 3 * 288
    sample = np.load(directory / "paths_2025-12-16.npz")
    assert sample["prices_eur_mwh"].shape == (100, 96)
    assert len(list((directory / "daily_calibrations").glob("*.json"))) == 12


def test_future_label_changes_cannot_change_earlier_predictions_or_gaussian_parameters(frozen_source, monkeypatch):
    fixture = frozen_source
    mock_catalog(monkeypatch, fixture)
    first = forward.run_frozen_price_check(fixture["config"])
    changed = fixture["extension"].copy()
    mask = changed.product_start.dt.tz_convert("Europe/Rome").dt.date == date(2025, 12, 18)
    changed.loc[mask, "price_eur_mwh"] += 500
    mock_catalog(monkeypatch, fixture, changed)
    second = forward.run_frozen_price_check(fixture["config"])
    a, b = Path(first["output_dir"]), Path(second["output_dir"])
    original = pd.read_parquet(a / "fresh_predictions.parquet")
    altered = pd.read_parquet(b / "fresh_predictions.parquet")
    assert np.array_equal(original.prediction, altered.prediction)
    assert not np.array_equal(original.actual, altered.actual)
    for day in ("2025-12-16", "2025-12-17", "2025-12-18"):
        assert (a / "daily_calibrations" / f"{day}.json").read_bytes() == (b / "daily_calibrations" / f"{day}.json").read_bytes()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_incomplete_data_waits_without_fits_or_outputs(frozen_source, monkeypatch, empty):
    fixture = frozen_source
    data = fixture["extension"].iloc[:0] if empty else fixture["extension"].drop(index=fixture["extension"].index[0])
    mock_catalog(monkeypatch, fixture, data)
    monkeypatch.setattr(forward, "_predict_candidate", lambda *args, **kwargs: pytest.fail("Incomplete data must never trigger a fit"))
    result = forward.run_frozen_price_check(fixture["config"])
    assert result["status"] == "waiting_for_data"
    assert "2025-12-16" in result["missing_days"]
    assert result["fits_performed"] == 0 and not result["outputs_created"]
    assert not Path(fixture["config"]["price_forward_check"]["output_directory"]).exists()


@pytest.mark.parametrize("name", ["settings.json", "feature_groups.json", "features.parquet", "manifest.json", "static"])
def test_changed_pinned_sources_are_rejected_before_database_access(frozen_source, monkeypatch, name):
    fixture = frozen_source
    path = fixture["noise_dir"] / "15min/static/gaussian_calibration.json" if name == "static" else fixture["source_dir"] / name
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(forward, "load_api_cache_records", lambda *args, **kwargs: pytest.fail("Changed sources must fail before DB access"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        forward.run_frozen_price_check(fixture["config"])


def test_legacy_cache_and_model_overrides_are_refused(frozen_source, monkeypatch):
    fixture = frozen_source
    monkeypatch.setattr(forward, "load_api_cache_records", lambda *args, **kwargs: pytest.fail("No legacy cache reads"))
    fixture["config"]["price_forward_check"]["cache_directory"] = str(fixture["tmp_path"] / "legacy_cache")
    with pytest.raises(ValueError, match="DuckDB"):
        forward.run_frozen_price_check(fixture["config"])
    fixture["config"]["price_forward_check"]["alpha"] = 1
    with pytest.raises(ValueError, match="overrides are forbidden"):
        forward.run_frozen_price_check(fixture["config"])


def test_wrong_market_extension_is_rejected_and_v2_requires_ancillary_pins(frozen_source, monkeypatch):
    fixture = frozen_source
    mock_catalog(monkeypatch, fixture, fixture["extension"].assign(zone="CNOR"))
    with pytest.raises(ValueError, match="zone differs"):
        forward.run_frozen_price_check(fixture["config"])
    del fixture["protocol"]["source_mean_settings_sha256"]
    write_json(fixture["protocol_path"], fixture["protocol"])
    with pytest.raises(ValueError, match="Version 2"):
        forward.run_frozen_price_check(fixture["config"])
