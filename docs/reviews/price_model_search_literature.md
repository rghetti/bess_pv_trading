# Expanded electricity-price model search

The initial January/August study used one pooled ridge specification with a
fixed penalty and was a feature ablation, not an exhaustive model search.
The open benchmark by [Lago et al. (2021)](https://doi.org/10.1016/j.apenergy.2021.116983)
motivates a seasonal-naive reference, sparse autoregressive models, neural
networks, different estimation windows, model combinations and long test
periods. Its LEAR uses whole daily lagged price and exogenous curves; a pooled
Lasso on this library's existing interval-level features is a useful candidate
but is not a reproduction of LEAR. The benchmark also documents why individual
percentage errors are unstable near zero electricity prices.

For this dataset, search ridge, Lasso, elastic net, feed-forward neural nets,
and gradient-boosted trees with several depths, learning rates and regularization
settings, along with persistence and forest references. Vary training windows
where coverage permits and compare inputs in groups. Reserve an untouched 2026
test period after chronological 2025 validation, using 2024 history for day-ahead
training; refits and any scaling must use only earlier rows. The available
January–August 2026 span is shorter than the year-long test recommended in the
benchmark and cannot establish year-round generalization. Intraday 2024 prices
have coarser source resolution; settlement begins later. Dates and data coverage
must be checked before declaring folds, rather than manufacturing missing rows.

Report MAE and pooled WAPE (sum absolute errors divided by sum absolute
observations), the error ratio to a seasonal-naive forecast on identical
intervals, RMSE, per-month results and forecast coverage. Negative prices are
preserved; WAPE is undefined when all observed prices are zero. Probabilistic
scores and trading outcomes are separate from point-model selection. Compare
models on shared target days, disclose training-sample differences, and account
for multiple comparisons before claiming statistical superiority. The user's
requested previous-day realized features are retrospective; their exact
historical publication times are not a prerequisite at this stage.

`price_search_models.py` provides candidate estimators only. No expanded search
has yet been fitted or validated, and these estimators are not yet integrated
as portable MPC forecast artifacts. Do not label them LEAR or claim a selected
model or improved score until the complete chronological search is implemented
and evaluated.
