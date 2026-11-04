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
where coverage permits and compare inputs in groups. In response to the product
resolution split, train and validate hourly day-ahead models using native hourly
prices, and quarter-hour models using only native quarter-hour prices. The
current initial search selects on January–June 2025 for hourly products and
February–April 2026 for quarter-hour products; later periods remain held out.
Their different forecast eras make the two resolution models separate tasks,
not directly comparable estimates of a treatment effect. Intraday and settlement
models use their own native-resolution data and folds in
`configs/research/price_model_search_intraday.yaml`. More history may improve
the hourly task, but cannot provide quarter-hour labels before those products.
The available 2026 span is shorter than the year-long test recommended in the
benchmark and cannot establish year-round generalization.

Report MAE and pooled WAPE (sum absolute errors divided by sum absolute
observations), the error ratio to a seasonal-naive forecast on identical
intervals, RMSE, per-month results and forecast coverage. Negative prices are
preserved; WAPE is undefined when all observed prices are zero. Probabilistic
scores and trading outcomes are separate from point-model selection. Compare
models on shared target days, disclose training-sample differences, and account
for multiple comparisons before claiming statistical superiority. The user's
requested previous-day realized features are retrospective; their exact
historical publication times are not a prerequisite at this stage.

`scripts/search_energy_price_models.py` prepares native-resolution data and
shared complete validation days for the candidate estimators. Its current
configurations are first validation searches, not a complete study over
every auction, training window, feature set, hyperparameter or held-out period.
These estimators are not yet integrated as portable MPC forecast artifacts.
Do not label them LEAR or claim a selected model or improved score until the
chronological search and held-out evaluation have been completed.
