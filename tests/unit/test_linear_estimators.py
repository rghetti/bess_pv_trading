"""Linear families recover signal, resist outliers and preserve saved predictors."""

import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.linear_estimators import (
    LinearEstimatorConvergenceError, fit_linear_estimator, validate_estimator,
)
from bess_pv_trading.forecasting.price_experiment import _fit_retrospective_ridge
from bess_pv_trading.forecasting.prices import RidgePriceModel


def _training(rows=160):
    """Create independent signals and realistic row/day grouping deterministically."""
    random = np.random.default_rng(31)
    x = random.normal(size=(rows, 4))
    frame = pd.DataFrame(x, columns=["signal", "second", "noise_a", "noise_b"])
    frame["price_eur_mwh"] = -10 + 4 * x[:, 0] - 2 * x[:, 1] + random.normal(scale=.05, size=rows)
    frame["delivery_day"] = (pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(rows) // 4, unit="D")).strftime("%Y-%m-%d")
    return frame


def _fit(train, method, settings=None, names=None):
    return fit_linear_estimator(train,
        feature_names=names or ["signal", "second", "noise_a", "noise_b"],
        fit_cutoff="2025-03-01T12:00:00+01:00", method=method, settings=settings or {})


def _restore(model):
    """Reconstruct the existing public coefficient artifact from JSON numbers."""
    values = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    return RidgePriceModel(tuple(values["feature_names"]), np.array(values["center"]), np.array(values["scale"]),
        np.array(values["coefficients"]), values["intercept"], values["alpha"],
        values["fit_cutoff"], values["training_rows"], values["target"])


def test_ols_handles_collinearity_constant_features_and_negative_prices():
    frame = _training()
    frame["twice_signal"] = 2 * frame.signal
    frame["constant"] = 7.
    frame["price_eur_mwh"] = -10 + 4 * frame.signal - 2 * frame.second
    names = ["constant", "twice_signal", "signal", "second"]
    model = _fit(frame, "ols", names=names)
    assert np.max(abs(model.predict(frame) - frame.price_eur_mwh)) < 1e-10
    assert np.isfinite(model.coefficients).all() and model.scale[0] == 1
    assert model.feature_names == tuple(names)
    assert model.alpha == 0 and model.training_rows == len(frame)
    assert (model.predict(frame) < 0).any()


def test_ridge_reuses_exact_existing_scaler_coefficients_and_penalty_semantics():
    frame = _training()
    names = ["second", "signal", "noise_b"]
    model = _fit(frame, "ridge", {"alpha": 25}, names)
    established = _fit_retrospective_ridge(frame, 25., "2025-03-01T12:00:00+01:00", names)
    assert model.to_dict() == established.to_dict()
    ols = _fit(frame, "ols", names=names)
    stronger = _fit(frame, "ridge", {"alpha": 10000}, names)
    assert np.linalg.norm(stronger.coefficients) < np.linalg.norm(ols.coefficients) / 5


def test_recency_weights_use_calendar_days_and_weighted_training_only_scaling():
    frame = pd.DataFrame({"signal": [1., 5., 7., 11.], "price_eur_mwh": [3., 11., 15., 23.],
                          "delivery_day": ["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-04"]})
    model = _fit(frame, "weighted_ridge", {"alpha": 0, "half_life_days": 2}, ["signal"])
    weights = np.exp2(-np.array([3., 2., 0., 0.]) / 2)
    expected_mean = np.average(frame.signal, weights=weights)
    expected_scale = np.sqrt(np.average(np.square(frame.signal - expected_mean), weights=weights))
    assert model.center[0] == pytest.approx(expected_mean)
    assert model.scale[0] == pytest.approx(expected_scale)
    assert model.intercept == pytest.approx(np.average(frame.price_eur_mwh, weights=weights))
    assert np.allclose(model.predict(frame), frame.price_eur_mwh)
    frozen = model.to_dict()
    assert np.isfinite(model.predict(pd.DataFrame({"signal": [-10000, 10000]}))).all()
    assert model.to_dict() == frozen


def test_recency_weighting_tracks_a_recent_intercept_change_better_than_uniform_fit():
    frame = _training(160)
    clean = -10 + 4 * frame.signal
    frame["price_eur_mwh"] = clean
    frame.loc[frame.index[-40:], "price_eur_mwh"] += 20
    weighted = _fit(frame, "weighted_ridge", {"alpha": 1, "half_life_days": 3}, ["signal"])
    uniform = _fit(frame, "ridge", {"alpha": 1}, ["signal"])
    latest = frame.iloc[-40:]
    assert abs(latest.price_eur_mwh - weighted.predict(latest)).mean() < abs(latest.price_eur_mwh - uniform.predict(latest)).mean() / 2


def test_huber_irls_reduces_vertical_outlier_distortion_without_clipping_targets():
    frame = _training(240)
    clean = frame.price_eur_mwh.copy()
    contaminated = frame.copy()
    outliers = np.random.default_rng(44).choice(len(frame), 18, replace=False)
    contaminated.loc[outliers, "price_eur_mwh"] += 100
    original = contaminated.price_eur_mwh.copy()
    ols = _fit(contaminated, "ols")
    robust = _fit(contaminated, "huber", {"alpha": 0, "epsilon": 1.35, "max_iter": 500, "tolerance": 1e-8})
    assert abs(clean - robust.predict(frame)).mean() < abs(clean - ols.predict(frame)).mean() / 4
    pd.testing.assert_series_equal(contaminated.price_eur_mwh, original)


def test_lasso_suppresses_irrelevant_predictors_and_high_penalty_keeps_only_intercept():
    frame = _training(400)
    sparse = _fit(frame, "lasso", {"alpha": .15, "tolerance": 1e-8})
    assert np.max(abs(sparse.coefficients[2:])) < 1e-10
    assert np.all(abs(sparse.coefficients[:2]) > 1)
    assert abs(frame.price_eur_mwh - sparse.predict(frame)).mean() < .5
    intercept_only = _fit(frame, "lasso", {"alpha": 100})
    assert np.array_equal(intercept_only.coefficients, np.zeros(4))
    assert np.allclose(intercept_only.predict(frame), frame.price_eur_mwh.mean())


def test_elastic_net_l2_limit_matches_ridge_after_documented_penalty_conversion():
    frame = _training()
    elastic = _fit(frame, "elastic_net", {"alpha": .3, "l1_ratio": 0, "tolerance": 1e-10})
    ridge = _fit(frame, "ridge", {"alpha": .3 * len(frame)})
    assert np.allclose(elastic.predict(frame), ridge.predict(frame), atol=1e-8)
    mixed = _fit(frame, "elastic_net", {"alpha": .1, "l1_ratio": .5})
    assert np.isfinite(mixed.coefficients).all()


@pytest.mark.parametrize("method,settings", [
    ("ols", {}), ("ridge", {"alpha": 10}),
    ("weighted_ridge", {"alpha": 10, "half_life_days": 10}),
    ("huber", {"alpha": 0}), ("lasso", {"alpha": .1}),
    ("elastic_net", {"alpha": .1, "l1_ratio": .5}),
])
def test_every_family_reloads_through_existing_predictor_artifact(method, settings):
    frame = _training()
    model = _fit(frame, method, settings)
    restored = _restore(model)
    assert np.array_equal(restored.predict(frame), model.predict(frame))
    assert restored.feature_names == model.feature_names
    assert restored.fit_cutoff == "2025-03-01T12:00:00+01:00"


def test_iterative_convergence_failures_are_explicit_without_a_ridge_fallback():
    frame = _training()
    frame.loc[frame.index[:10], "price_eur_mwh"] += 100
    with pytest.raises(LinearEstimatorConvergenceError, match="Huber IRLS"):
        _fit(frame, "huber", {"alpha": 0, "max_iter": 1, "tolerance": 1e-15})
    with pytest.raises(LinearEstimatorConvergenceError, match="coordinate descent"):
        _fit(frame, "elastic_net", {"alpha": .001, "l1_ratio": .5, "max_iter": 1, "tolerance": 1e-15})


@pytest.mark.parametrize("settings", [
    {"method": "unknown"}, {"method": "ols", "alpha": 1},
    {"method": "ridge", "alpha": -1}, {"method": "weighted_ridge", "half_life_days": 0},
    {"method": "lasso", "l1_ratio": .5}, {"method": "elastic_net", "l1_ratio": 2},
    {"method": "huber", "epsilon": 0}, {"method": "huber", "max_iter": True},
    {"method": "elastic_net", "tolerance": 0}, {"method": "ridge", "epsilon": 1.35},
])
def test_invalid_estimator_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_estimator(settings)


def test_nonfinite_inputs_naive_fit_times_and_missing_weight_days_are_rejected():
    frame = _training()
    frame.loc[0, "signal"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _fit(frame, "ols")
    with pytest.raises(ValueError, match="timezone-aware"):
        fit_linear_estimator(_training(), feature_names=["signal"], fit_cutoff="2025-03-01", method="ols", settings={})
    with pytest.raises(ValueError, match="delivery_day"):
        _fit(_training().drop(columns="delivery_day"), "weighted_ridge")
