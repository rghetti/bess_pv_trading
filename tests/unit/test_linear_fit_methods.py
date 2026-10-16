"""Exercise estimator selection, reusable forecasts and frozen old features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting import linear_price_study
from bess_pv_trading.forecasting.linear_price_study import (
    LinearPriceForecast, _candidates, _predict_candidate, evaluate_linear_price_study,
    run_linear_price_study,
)
from test_linear_price_study import _features, _settings


@pytest.mark.parametrize("method,options", [
    ("ols", {}), ("weighted_ridge", {"half_life_days": 7, "alpha": 1.}),
    ("huber", {"alpha": 1., "epsilon": 1.35}),
    ("lasso", {"alpha": .1}),
    ("elastic_net", {"alpha": .1, "l1_ratio": .5}),
])
def test_each_fitting_method_reuses_coefficients_without_future_labels(tmp_path, method, options):
    frame = _features()
    settings = _settings(additional_estimators=[{"name": "alternative", "method": method, **options}])
    candidate = next(item for item in _candidates({"lags": ["lag_1d", "signal"]}, settings)
                     if item["kind"] == "linear" and item["target"] == "delta_day")
    days = sorted(frame.delivery_day.unique())[30:39]
    common = pd.Series(True, index=frame.index)
    predictions, refits = _predict_candidate(frame, candidate, days, settings, common, lambda day: "test")
    changed = frame.copy()
    changed.loc[changed.delivery_day.eq(days[-1]), "price_eur_mwh"] += 99999
    repeated, _ = _predict_candidate(changed, candidate, days, settings, common, lambda day: "test")
    np.testing.assert_allclose(predictions.prediction, repeated.prediction)
    assert predictions.training_max_delivery_day.lt(predictions.delivery_day).all()
    payload = dict(refits[-1])
    payload.pop("refit_delivery_day")
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps(payload))
    artifact = LinearPriceForecast.load(path)
    target_day = days[-1]
    rows = frame.loc[frame.delivery_day.eq(target_day)]
    expected = predictions.loc[predictions.delivery_day.eq(target_day), "prediction"]
    np.testing.assert_allclose(artifact.predict(rows), expected)
    assert artifact.to_dict()["schema_version"] == "linear-price-point-model-v3"
    assert not artifact.operational_eligible


def test_new_estimators_leave_default_ridge_grid_identical():
    settings = _settings()
    original = _candidates({"lags": ["lag_1d", "signal"]}, settings)
    expanded = _candidates({"lags": ["lag_1d", "signal"]}, {
        **settings, "additional_estimators": [{"name": "least_squares", "method": "ols"}]})
    assert expanded[:len(original)] == original
    assert {item["kind"] for item in original} == {"baseline", "ridge"}
    assert len(expanded) == len(original) + 2


def test_nonconverged_candidates_cannot_win_or_silently_fall_back(monkeypatch):
    from bess_pv_trading.forecasting import linear_estimators
    frame = _features()
    settings = _settings(additional_estimators=[{"name": "failed", "method": "lasso", "alpha": .1}])

    def fail(*args, **kwargs):
        raise linear_estimators.LinearEstimatorConvergenceError("fixture iteration limit")

    monkeypatch.setattr(linear_estimators, "fit_linear_estimator", fail)
    result = evaluate_linear_price_study(frame, {"lags": ["lag_1d", "signal"]}, settings)
    failed = [item for item in result["validation_results"] if item["candidate"]["kind"] == "linear"]
    assert failed and all(item["status"] == "estimator_did_not_converge" for item in failed)
    assert result["regimes"][0]["selected_candidate"]["kind"] != "linear"


def test_estimator_study_reuses_frozen_features_without_opening_database(tmp_path, monkeypatch):
    rows = _features()
    provenance = [{"checksums_verified": True, "storage": {"backend": "duckdb", "snapshot_id": "fixture"}}]
    monkeypatch.setattr(linear_price_study, "load_api_cache_records", lambda *a, **k: (rows, provenance))
    settings = _settings(output_directory=str(tmp_path / "first"),
        analysis_directory=str(tmp_path / "first_analysis"),
        native_product_regimes=[{"start": "2025-01-01", "end": None, "minutes": 60}],
        feature_sets={"lags": ["calendar", "autoregressive"]})
    first = run_linear_price_study({"linear_price_study": settings})

    def no_database(*args, **kwargs):
        raise AssertionError("Frozen estimator studies must not reopen the current database")

    monkeypatch.setattr(linear_price_study, "load_api_cache_records", no_database)
    settings.update(output_directory=str(tmp_path / "second"),
        analysis_directory=str(tmp_path / "second_analysis"),
        frozen_feature_manifest=first["manifest"],
        additional_estimators=[{"name": "ols", "method": "ols"}])
    second = run_linear_price_study({"linear_price_study": settings})
    assert second["status"] == "completed"
    assert second["frozen_features_sha256"] == first["frozen_features_sha256"]
    assert second["frozen_feature_source"]["features_sha256"] == first["frozen_features_sha256"]
    assert second["source_cache_count"] == 1
    assert set(second["frozen_feature_source"]["ancillary_sha256"]) == {
        "settings.json", "feature_groups.json", "source_caches.json", "feature_coverage.json"}
    source_path = Path(first["output_dir"]) / "features.parquet"
    source_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_linear_price_study({"linear_price_study": settings})
