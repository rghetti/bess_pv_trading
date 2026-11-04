"""Research candidate factories fit deterministic synthetic nonlinear data."""

import numpy as np
import pytest

from bess_pv_trading.forecasting.price_search_models import make_price_candidate


@pytest.mark.parametrize("family,options", [
    ("ridge", {"alpha": 1}),
    ("lasso", {"alpha": 0.01, "max_iter": 1000}),
    ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000}),
    ("huber", {"epsilon": 1.35, "alpha": 0.01, "max_iter": 500}),
    ("sgd_mae", {"alpha": 0.001, "max_iter": 500, "tol": 1e-3}),
    ("hist_gradient_boosting", {"max_iter": 20, "min_samples_leaf": 4}),
    ("random_forest", {"n_estimators": 10, "min_samples_leaf": 2}),
    ("mlp", {"hidden_layer_sizes": (8,), "max_iter": 50}),
    ("lightgbm", {"n_estimators": 10, "min_child_samples": 4}),
    ("catboost", {"iterations": 10, "depth": 3}),
])
def test_candidate_fits_and_returns_finite_predictions(family, options):
    features = np.arange(120, dtype=float).reshape(60, 2) / 20
    target = features[:, 0] ** 2 + features[:, 1]
    model = make_price_candidate(family, options, seed=4, threads=1)
    model.fit(features[:50], target[:50])
    assert np.isfinite(model.predict(features[50:])).all()


@pytest.mark.parametrize("family,options", [
    ("interaction_ridge", {"lag_count": 1, "alpha": 1}),
    ("interaction_lasso", {"lag_count": 1, "alpha": 0.01, "max_iter": 1000}),
    ("interaction_elastic_net", {"lag_count": 1, "alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000}),
    ("interaction_huber", {"lag_count": 1, "epsilon": 1.35, "alpha": 0.01, "max_iter": 500}),
    ("interaction_sgd_mae", {"lag_count": 1, "alpha": 0.001, "max_iter": 500, "tol": 1e-3}),
])
def test_interaction_candidates_fit_with_declared_calendar_layout(family, options):
    """Calendar-modulated lag effects remain finite and CPU-compatible."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, 9))
    target = features[:, 0] * features[:, 1] + features[:, 8]
    model = make_price_candidate(family, options, seed=4, threads=1)
    model.fit(features[:60], target[:60])
    assert np.isfinite(model.predict(features[60:])).all()


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown price candidate"):
        make_price_candidate("unknown", {}, seed=1)


def test_huber_candidate_resists_a_large_training_outlier():
    """Robust candidate stays closer to the central price than ordinary ridge."""
    features = np.zeros((40, 1))
    target = np.full(40, 100.0)
    target[-1] = 1000.0
    robust = make_price_candidate("huber", {"epsilon": 1.35, "max_iter": 500}, seed=1)
    ridge = make_price_candidate("ridge", {"alpha": 1.0}, seed=1)
    robust.fit(features, target)
    ridge.fit(features, target)
    assert abs(robust.predict([[0]])[0] - 100) < abs(ridge.predict([[0]])[0] - 100)


def test_sgd_mae_uses_absolute_error_and_returns_original_units():
    """The stochastic robust model fits the median while retaining price units."""
    features = np.zeros((200, 1))
    target = np.full(200, 100.0)
    target[-20:] = 1000.0
    model = make_price_candidate("sgd_mae", {"alpha": 0.001, "max_iter": 2000, "tol": 1e-4,
                                             "average": True}, seed=1)
    model.fit(features, target)
    assert abs(model.predict([[0]])[0] - 100) < 40
