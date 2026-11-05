"""Point-model candidates for chronological electricity-price model searches.

These estimators are research candidates, not the serialized MPC price model.
Their inputs must already be built with the same decision-time feature rules.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class IntradayShapeInteractions(BaseEstimator, TransformerMixin):
    """Allow lag and exogenous effects to vary smoothly with delivery hour."""

    def __init__(self, lag_count, harmonics=1):
        self.lag_count = lag_count
        self.harmonics = harmonics

    def fit(self, features, target=None):
        """Validate the calendar layout without learning from validation data."""
        values = np.asarray(features)
        if isinstance(self.lag_count, bool) or not isinstance(self.lag_count, int) or self.lag_count < 1:
            raise ValueError("Interaction lag_count must be a positive integer")
        if isinstance(self.harmonics, bool) or not isinstance(self.harmonics, int) or self.harmonics < 1:
            raise ValueError("Interaction harmonics must be a positive integer")
        if values.ndim != 2 or values.shape[1] < self.lag_count + 7:
            raise ValueError("Interaction input lacks the declared seven calendar columns")
        return self

    def transform(self, features):
        """Interact non-calendar signals with configured delivery-time harmonics."""
        values = np.asarray(features, dtype=float)
        signal = np.column_stack([values[:, :self.lag_count], values[:, self.lag_count + 7:]])
        sine = values[:, self.lag_count:self.lag_count + 1]
        cosine = values[:, self.lag_count + 1:self.lag_count + 2]
        columns = [values]
        harmonic_sine = sine
        harmonic_cosine = cosine
        for _ in range(self.harmonics):
            # Recurrence preserves the existing sine/cosine encoding and
            # generates higher harmonics without an inverse-angle operation.
            columns.extend([signal * harmonic_sine, signal * harmonic_cosine])
            harmonic_sine, harmonic_cosine = (harmonic_sine * cosine + harmonic_cosine * sine,
                                               harmonic_cosine * cosine - harmonic_sine * sine)
        return np.column_stack(columns)


def make_price_candidate(family, parameters, *, seed, threads=1):
    """Create an unfitted, isolated estimator from explicit study parameters."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Candidate thread count must be a positive integer")
    options = dict(parameters)
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**options))
    if family == "lasso":
        return make_pipeline(StandardScaler(), Lasso(random_state=seed, **options))
    if family == "elastic_net":
        return make_pipeline(StandardScaler(), ElasticNet(random_state=seed, **options))
    if family == "huber":
        estimator = make_pipeline(StandardScaler(), HuberRegressor(**options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family == "sgd_mae":
        estimator = make_pipeline(StandardScaler(), SGDRegressor(loss="epsilon_insensitive", epsilon=0,
                                                                   random_state=seed, **options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family in {"interaction_ridge", "interaction_lasso", "interaction_elastic_net", "interaction_huber",
                  "interaction_sgd_mae"}:
        lag_count = options.pop("lag_count")
        harmonics = options.pop("harmonics", 1)
        if family == "interaction_ridge":
            estimator = Ridge(**options)
        elif family == "interaction_lasso":
            estimator = Lasso(random_state=seed, **options)
        elif family == "interaction_elastic_net":
            estimator = ElasticNet(random_state=seed, **options)
        elif family == "interaction_huber":
            estimator = HuberRegressor(**options)
        else:
            estimator = SGDRegressor(loss="epsilon_insensitive", epsilon=0, random_state=seed, **options)
        pipeline = make_pipeline(IntradayShapeInteractions(lag_count, harmonics), StandardScaler(), estimator)
        if family in {"interaction_huber", "interaction_sgd_mae"}:
            return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        return pipeline
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **options)
    if family == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=threads, **options)
    if family == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=threads, **options)
    if family == "knn":
        # Scale each training feature before searching for comparable products.
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_jobs=threads, **options))
    if family == "mlp":
        # Scale targets as well as predictors; inverse transformation returns EUR/MWh.
        estimator = make_pipeline(StandardScaler(), MLPRegressor(random_state=seed, **options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=threads, verbosity=-1, **options)
    if family == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, n_jobs=threads, **options)
    if family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=seed, thread_count=threads, verbose=False,
                                 allow_writing_files=False, **options)
    raise ValueError(f"Unknown price candidate family: {family}")
