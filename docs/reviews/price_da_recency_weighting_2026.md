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
