"""Research candidate factories fit deterministic synthetic nonlinear data."""

import numpy as np
import pytest

from bess_pv_trading.forecasting.price_search_models import make_price_candidate


@pytest.mark.parametrize("family,options", [
    ("ridge", {"alpha": 1}),
    ("lasso", {"alpha": 0.01, "max_iter": 1000}),
    ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000}),
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


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown price candidate"):
        make_price_candidate("unknown", {}, seed=1)
