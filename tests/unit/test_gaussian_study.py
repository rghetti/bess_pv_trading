"""Frozen-file Gaussian studies, artifact reuse, and test-label isolation."""

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.gaussian_prices import (
    GaussianPriceCalibration, calibrate_gaussian_prices, evaluate_gaussian_prices,
)
from bess_pv_trading.forecasting.gaussian_study import run_gaussian_study


def _predictions(start, days):
    """Create hourly OOS predictions with a varying correlated daily error."""
    frames = []
    first = pd.Timestamp(start).date()
    for offset in range(days):
        day = first + timedelta(days=offset)
        midnight = pd.Timestamp(day, tz="Europe/Rome")
        following = pd.Timestamp(day + timedelta(days=1), tz="Europe/Rome")
        starts = pd.date_range(midnight, following, inclusive="left", freq="60min").tz_convert("UTC")
        slots = np.arange(len(starts))
        index = (day - date(2025, 1, 1)).days
        prediction = 70 + 10 * np.sin(slots / 4)
        error = (index % 7 - 3) * (1 + .3 * np.cos(slots / 5)) + .2 * np.sin(slots + index)
        frames.append(pd.DataFrame({"delivery_day": day.isoformat(), "product_start": starts,
            "product_end": starts + pd.Timedelta(hours=1), "native_interval_minutes": 60,
            "actual": prediction + error, "prediction": prediction,
            "fit_cutoff": pd.Timestamp(day - timedelta(days=2), tz="Europe/Rome").tz_convert("UTC"),
            "forecast_cutoff": (pd.Timestamp(day - timedelta(days=1), tz="Europe/Rome") + pd.Timedelta(hours=12)).tz_convert("UTC"),
            "training_max_delivery_day": (day - timedelta(days=3)).isoformat(),
            "prediction_provenance": "chronological_out_of_sample"}))
    return pd.concat(frames, ignore_index=True)


def _source(directory, *, test_shift=0.):
    """Freeze 30 calibration days, a Jan 31 gap, and two February test days."""
    directory.mkdir()
    calibration, test = _predictions("2025-01-01", 30), _predictions("2025-02-01", 2)
    test.loc[test.delivery_day.eq("2025-02-02"), "actual"] += test_shift
    calibration.to_parquet(directory / "calibration_predictions.parquet", index=False)
    test.to_parquet(directory / "test_predictions.parquet", index=False)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "settings": {"timezone": "Europe/Rome"},
        "prior_evaluation_overlap": "Synthetic unit fixture only"}), encoding="utf-8")
    return manifest


def _config(source, output, **settings):
    return {"price_scenarios": {"study_manifest": str(source), "output_directory": str(output),
        "sample_count": 100, "seed": 13, "minimum_complete_days": 3, "covariance_shrinkage": .5,
        "adaptive": {"window_days_candidates": [7, 14], "covariance_shrinkage_candidates": [.1, .5],
                     "minimum_complete_days": 3, "inner_validation_days": 3, "minimum_inner_scored_days": 3},
        **settings}}


@pytest.fixture(autouse=True)
def _skip_plot(monkeypatch):
    """Avoid plotting overhead in numerical artifact integration tests."""
    monkeypatch.setattr("bess_pv_trading.forecasting.gaussian_study._plot_paths", lambda *args: None)


def test_independent_study_freezes_inputs_saves_reloadable_daily_artifacts_and_static_comparison(tmp_path):
    source = _source(tmp_path / "source")
    source_before = {name: (source.parent / name).read_bytes() for name in
                     ("manifest.json", "calibration_predictions.parquet", "test_predictions.parquet")}
    messages = []
    result = run_gaussian_study(_config(source, tmp_path / "output"), progress=messages.append)
    output = Path(result["output_dir"])
    assert result["status"] == "completed" and result["operational_eligible"] is False
    assert output.is_dir() and not list(output.parent.glob("*.pending"))
    assert json.loads((output / "manifest.json").read_text()) == result
    assert result["source_manifest_sha256"] == hashlib.sha256(source_before["manifest.json"]).hexdigest()
    assert (output / "source_manifest.json").read_bytes() == source_before["manifest.json"]
    for name in ("calibration_predictions.parquet", "test_predictions.parquet"):
        assert (output / name).read_bytes() == source_before[name]
        assert result["frozen_input_sha256"][name] == hashlib.sha256(source_before[name]).hexdigest()
    assert all((source.parent / name).read_bytes() == data for name, data in source_before.items())
    assert not (source.parent / "uncertainty").exists()
    regime = result["regimes"][0]
    assert regime["sample_count"] == 100 and regime["sample_day"] == "2025-02-01"
    assert regime["adaptive_metrics"]["scored_rows"] == 48
    assert regime["static_metrics"]["scored_rows"] == 48
    assert len(messages) >= 3
    policy = json.loads((output / "60min" / "policy.json").read_text())
    assert policy["selection_source"] == "calibration_predictions_only"
    candidate_scores = pd.read_parquet(output / "60min" / "inner_candidate_scores.parquet")
    assert len(candidate_scores) == 4 and candidate_scores.selected.sum() == 1
    artifacts = sorted((output / "60min" / "daily_calibrations").glob("*.json"))
    assert [path.stem for path in artifacts] == ["2025-02-01", "2025-02-02"]
    artifact = GaussianPriceCalibration.load(artifacts[0])
    products = pd.read_parquet(output / "60min" / "products_2025-02-01.parquet")
    with np.load(regime["sample_paths"]) as arrays:
        assert arrays["prices_eur_mwh"].shape == (100, 24)
        assert np.allclose(arrays["probabilities"], .01)
        restored = artifact.sample_future(products.prediction.to_numpy(), products, 100,
            decision_cutoff=products.forecast_cutoff.max(), seed=13, require_operational=False)
        assert np.array_equal(arrays["prices_eur_mwh"], restored)
    original_calibration = pd.read_parquet(source.parent / "calibration_predictions.parquet")
    original_test = pd.read_parquet(source.parent / "test_predictions.parquet")
    static = calibrate_gaussian_prices(original_calibration,
        calibration_cutoff=original_calibration.product_end.max(), minimum_complete_days=3, covariance_shrinkage=.5)
    assert regime["static_metrics"] == evaluate_gaussian_prices(static, original_test)["metrics"]


def test_final_test_labels_never_choose_policy_or_change_earlier_sample_paths(tmp_path):
    first = run_gaussian_study(_config(_source(tmp_path / "first"), tmp_path / "outputs"))
    second = run_gaussian_study(_config(_source(tmp_path / "second", test_shift=1000), tmp_path / "outputs"))
    a, b = first["regimes"][0], second["regimes"][0]
    assert a["selected_policy"] == b["selected_policy"]
    for name in ("policy.json", "inner_candidate_scores.parquet", "daily_calibrations/2025-02-01.json",
                 "daily_calibrations/2025-02-02.json"):
        assert (Path(first["output_dir"]) / "60min" / name).read_bytes() == (Path(second["output_dir"]) / "60min" / name).read_bytes()
    with np.load(a["sample_paths"]) as x, np.load(b["sample_paths"]) as y:
        for name in x.files:
            assert np.array_equal(x[name], y[name])
    assert b["adaptive_metrics"]["mae_eur_mwh"] > a["adaptive_metrics"]["mae_eur_mwh"] + 400
    assert first["frozen_input_sha256"]["test_predictions.parquet"] != second["frozen_input_sha256"]["test_predictions.parquet"]


def test_invalid_settings_fail_without_publishing_any_output_directory(tmp_path):
    source = _source(tmp_path / "source")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="sample_count"):
        run_gaussian_study(_config(source, output, sample_count=False))
    assert not output.exists()
    bad = _config(source, output)
    bad["price_scenarios"]["adaptive"]["selection_metric"] = "test_coverage"
    with pytest.raises(ValueError, match="selection_metric"):
        run_gaussian_study(bad)
    assert not output.exists()
