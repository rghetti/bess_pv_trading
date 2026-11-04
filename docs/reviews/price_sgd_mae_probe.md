# Stochastic absolute-error day-ahead probe

A CPU-only pilot tested nine interaction models using the
`ida_curve_fundamentals` feature set on the same 82 complete validation days
as `nord_da_2026_previous_ida_prices_v1`. The objective was zero-width
epsilon-insensitive loss on a standardized target, with averaged stochastic
updates. The pilot varied regularization over `1e-5`, `1e-4` and `1e-3`, and
training history over the full available sample, 60 days and 90 days. Each
fit used seed 17, one CPU thread, a 1,000-epoch limit and tolerance `1e-4`.

The best pilot was the full-history model with regularization `1e-3`: 10.86%
WAPE and €13.92/MWh MAE. The full-history models ranged from 10.86% to 11.01%
WAPE, 90-day models from 16.15% to 16.24%, and 60-day models from 18.27% to
25.14%. The worker recorded no optimizer convergence warning in these nine
fits. Their outputs are temporary pilot files under `/tmp/sgd_mae_grid_probe.log`
and `/tmp/interaction_sgd_mae_*_probe.csv`.

The selected earlier interaction-ridge candidate scored 9.46% WAPE on the
same validation products. Both the pilot grid and that candidate were
inspected on the same period, so these are exploratory comparisons, not an
independent test. The pilot covers one feature set and does not establish
performance for the prepared wider configuration.
