# Recency-weighted DA forecasts

A CPU comparison applies exponentially decaying weights to training rows in
both feature scaling and interaction Elastic Net fitting. Half-lives are
counts of available chronological product observations, not elapsed days;
missing products mean 2,880 observations are only approximately 30 days.
It uses prior DA curves, optionally prior IDA1/IDA2 prices and regional
weather; training minimizes weighted squared error plus Elastic Net penalties,
while candidates are selected on MAE. Historical weather vintage is
operational-or-hindcast-unverified and some auction release clocks are assumed.

The early selection uses 7,868 products on 82 February–April 2026 days. With
weather, the 2,880-observation candidate scored 11.796 EUR/MWh MAE and 9.202%
WAPE without past IDAs, versus 12.278 and 9.578% unweighted. With prior IDAs
it scored 11.711 and 9.136%, versus 12.133 and 9.465%. All 16 fits completed
without failures or convergence warnings. These selection dates have already
been inspected repeatedly.

## May–August weekly replay

The selected candidates were refitted weekly with a one-day gap. All six
fits finished without recorded failures or convergence warnings. The table
uses 10,272 shared quarter-hourly products on 107 complete days, previously
examined in other studies. WAPE is total absolute error divided by total
absolute actual price; all prices are EUR/MWh.

| Features | Model | MAE | WAPE |
| --- | --- | ---: | ---: |
| Price curves and weather | Unweighted interaction Elastic Net | 16.226 | 11.171% |
| Price curves and weather | Recency-weighted, 2,880 observations | 16.363 | 11.265% |
| Price curves, prior IDAs and weather | Unweighted interaction Elastic Net | 15.949 | 10.980% |
| Price curves, prior IDAs and weather | Recency-weighted, 2,880 observations | 16.049 | 11.049% |

Paired baseline-minus-weighted MAE differences were -0.138 and -0.099
EUR/MWh. Their 95% independent-day bootstrap percentile intervals from 5,000
resamples with seed 173 were respectively -0.558–0.286 and -0.510–0.315.
These intervals omit serial correlation and model-selection uncertainty.
The early recency gain therefore did not carry over in this replay; the
comparison cannot establish a reliable advantage in either direction.
September–October replay is running. Matched forecasts, refit records,
source coverage and hashes are saved in
`outputs/energy_price_forecasts/nord_da_2026_recency_weighted_may_aug_rolling_v1/`.

## September–October replay and convergence audit

The initial September–October replay completed six fits without recorded
failures on 4,800 products over 50 shared days. The unweighted weather model
scored 16.306 EUR/MWh MAE and 7.617% WAPE, versus 17.182 and 8.026% for its
recency-weighted counterpart. With prior IDA prices as well, the respective
scores were 16.078 and 7.510% versus 16.916 and 7.902%. Paired
baseline-minus-weighted MAE differences were -0.876 and -0.838 EUR/MWh,
with 95% independent-day bootstrap percentile intervals -1.654–-0.141 and
-1.601–-0.114 (5,000 resamples, seed 173).

However, the weighted model emitted convergence warnings in three refits
without prior IDAs and four with them at the 5,000-iteration limit. The
unweighted fits had no recorded warnings. The apparent disadvantage of
weighting is therefore diagnostic, not a defensible ranking of converged
models. A matched rerun of both weather feature arms, fixing the same
preselected half-life and raising the weighted model limit to 30,000
iterations, has been started. The samples are retrospective and the
bootstrap intervals omit serial dependence and selection uncertainty.

The higher-iteration audit has now completed all four matched fits without
recorded failures or convergence warnings, on the same 50 days and 4,800
products. Without prior IDAs, unweighted versus weighted MAE/WAPE were
16.306 EUR/MWh/7.617% versus 17.183/8.026%. With prior IDAs, they were
16.078/7.510% versus 16.917/7.902%. Baseline-minus-weighted MAE differences
were -0.877 and -0.839 EUR/MWh, with independent-day 95% bootstrap intervals
-1.655–-0.142 and -1.603–-0.114, respectively (5,000 resamples, seed 173).
This convergence check supports the observed disadvantage for these samples,
subject to the retrospective selection, assumed source clocks, and omitted
serial and model-selection uncertainty. Audit predictions and warnings are in
`outputs/energy_price_forecasts/nord_da_2026_recency_convergence_sep_oct_rolling_v1/`.
