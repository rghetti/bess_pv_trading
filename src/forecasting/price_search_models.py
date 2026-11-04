"""Point-model candidates for chronological electricity-price model searches.

These estimators are research candidates, not the serialized MPC price model.
Their inputs must already be built with the same decision-time feature rules.
"""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **options)
    if family == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=threads, **options)
    if family == "mlp":
        # Scale targets as well as predictors; inverse transformation returns EUR/MWh.
        estimator = make_pipeline(StandardScaler(), MLPRegressor(random_state=seed, **options))
        return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=threads, verbosity=-1, **options)
    if family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=seed, thread_count=threads, verbose=False,
                                 allow_writing_files=False, **options)
    raise ValueError(f"Unknown price candidate family: {family}")
