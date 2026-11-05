"""Point-model candidates for chronological electricity-price model searches.

These estimators are research candidates, not the serialized MPC price model.
Their inputs must already be built with the same decision-time feature rules.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, QuantileRegressor, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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


class WeekdayLagInteractions(BaseEstimator, TransformerMixin):
    """Allow selected recent-price lag effects to vary by local weekday."""

    def __init__(self, lag_count, signal_count):
        self.lag_count = lag_count
        self.signal_count = signal_count

    def fit(self, features, target=None):
        """Validate the declared lag and deterministic calendar layout."""
        values = np.asarray(features)
        if (isinstance(self.lag_count, bool) or not isinstance(self.lag_count, int) or self.lag_count < 1
                or isinstance(self.signal_count, bool) or not isinstance(self.signal_count, int)
                or not 1 <= self.signal_count <= self.lag_count):
            raise ValueError("Weekday interactions require valid lag and signal counts")
        if values.ndim != 2 or values.shape[1] < self.lag_count + 7:
            raise ValueError("Weekday interaction input lacks the declared calendar columns")
        return self

    def transform(self, features):
        """Decode weekday phase and append one-hot terms and lag interactions."""
        values = np.asarray(features, dtype=float)
        sine = values[:, self.lag_count + 2]
        cosine = values[:, self.lag_count + 3]
        if not np.isfinite(sine).all() or not np.isfinite(cosine).all() or not np.allclose(sine**2 + cosine**2, 1):
            raise ValueError("Weekday sine and cosine must encode a valid phase")
        phase = np.mod(np.arctan2(sine, cosine), 2 * np.pi)
        weekdays = np.rint(phase * 7 / (2 * np.pi)).astype(int) % 7
        one_hot = np.eye(7)[weekdays]
        interactions = (one_hot[:, :, None] * values[:, None, :self.signal_count]).reshape(len(values), -1)
        return np.column_stack([values, one_hot, interactions])


class RecencyWeightedLinear(RegressorMixin, BaseEstimator):
    """Fit a linear model with a half-life measured in ordered observations."""

    def __init__(self, family, half_life_observations, parameters, seed, lag_count=None, harmonics=1):
        self.family = family
        self.half_life_observations = half_life_observations
        self.parameters = parameters
        self.seed = seed
        self.lag_count = lag_count
        self.harmonics = harmonics

    def fit(self, features, target):
        """Weight recent rows more heavily in scaling and regression."""
        count = len(target)
        if (isinstance(self.half_life_observations, bool) or not isinstance(self.half_life_observations, int)
                or self.half_life_observations < 1 or not count):
            raise ValueError("Recency half-life must be a positive observation count")
        weights = np.exp2(-np.arange(count - 1, -1, -1, dtype=float) / self.half_life_observations)
        self.interactions_ = None
        if self.lag_count is not None:
            self.interactions_ = IntradayShapeInteractions(self.lag_count, self.harmonics)
            features = self.interactions_.fit_transform(features)
        self.scaler_ = StandardScaler().fit(features, sample_weight=weights)
        if self.family == "ridge":
            self.estimator_ = Ridge(**self.parameters)
        elif self.family == "elastic_net":
            self.estimator_ = ElasticNet(random_state=self.seed, **self.parameters)
        else:
            raise ValueError("Unsupported recency-weighted linear family")
        self.estimator_.fit(self.scaler_.transform(features), target, sample_weight=weights)
        return self

    def predict(self, features):
        """Return predictions in the original target units."""
        if self.interactions_ is not None:
            features = self.interactions_.transform(features)
        return self.estimator_.predict(self.scaler_.transform(features))


def make_price_candidate(family, parameters, *, seed, threads=1):
    """Create an unfitted, isolated estimator from explicit study parameters."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Candidate thread count must be a positive integer")
    options = dict(parameters)
    if family == "zero_spread":
        # The search worker adds the configured cleared-price anchor back.
        if options:
            raise ValueError("Zero-spread candidate accepts no estimator parameters")
        return DummyRegressor(strategy="constant", constant=0.0)
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**options))
    if family == "lasso":
        return make_pipeline(StandardScaler(), Lasso(random_state=seed, **options))
    if family == "elastic_net":
        return make_pipeline(StandardScaler(), ElasticNet(random_state=seed, **options))
    if family in {"recency_ridge", "recency_elastic_net"}:
        half_life = options.pop("half_life_observations")
        return RecencyWeightedLinear(family.removeprefix("recency_"), half_life, options, seed)
    if family == "recency_interaction_elastic_net":
        half_life = options.pop("half_life_observations")
        lag_count = options.pop("lag_count")
        harmonics = options.pop("harmonics", 1)
        return RecencyWeightedLinear("elastic_net", half_life, options, seed, lag_count, harmonics)
    if family == "median_regression":
        # HiGHS receives the explicit worker limit through its solver options.
        solver_options = dict(options.pop("solver_options", {}))
        if "threads" in solver_options and solver_options["threads"] != threads:
            raise ValueError("Median-regression solver threads must match the worker limit")
        solver_options["threads"] = threads
        return make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, solver="highs",
                                                                   solver_options=solver_options, **options))
    if family == "huber":
        estimator = make_pipeline(StandardScaler(), HuberRegressor(**options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family == "sgd_mae":
        estimator = make_pipeline(StandardScaler(), SGDRegressor(loss="epsilon_insensitive", epsilon=0,
                                                                   random_state=seed, **options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family in {"weekday_ridge", "weekday_elastic_net"}:
        lag_count = options.pop("lag_count")
        signal_count = options.pop("signal_count")
        estimator = Ridge(**options) if family == "weekday_ridge" else ElasticNet(random_state=seed, **options)
        return make_pipeline(WeekdayLagInteractions(lag_count, signal_count), StandardScaler(), estimator)
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
    if family == "svr":
        # Scale targets as well as inputs so epsilon has a consistent meaning.
        estimator = make_pipeline(StandardScaler(), SVR(**options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
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
