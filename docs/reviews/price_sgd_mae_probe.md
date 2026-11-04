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
independent test. The pilot itself covers only one feature set.

## Completed wider comparison

The wider configuration subsequently completed all 80 fits without reported
failures or convergence warnings. It used the same 82 validation days and
four matched feature sets, comparing stochastic absolute-error models with
interaction-ridge and interaction-elastic-net references. The best observed
stochastic candidate scored 10.27% WAPE on `curve_combined`, while the
interaction-ridge reference on `ida_curve_fundamentals` scored 9.46%.
These are minimum scores selected on the same validation period, not a later
test. The stronger 9.01% recent-window interaction-elastic-net result came
from a separate matched search on the same evaluation products.

The full run, including candidate settings and warning records, is in
`outputs/energy_price_forecasts/nord_da_2026_sgd_mae_search_v1/`.
