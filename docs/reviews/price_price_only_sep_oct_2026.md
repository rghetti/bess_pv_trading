# Price-only later-period evaluation, September–October 2026

This study selects one model per market by February–April 2026 validation MAE
from five candidates: ridge, Lasso, Elastic Net and two Extra Trees settings.
It then refits that model weekly with the configured one-day training gap and
scores the later September–October period. DA uses previous DA prices and daily
curves. IDAs use past prices and previously cleared auctions, either directly
or as a spread to cleared DA. The study excludes external fundamentals because
several catalog series end in August. Early candidate and feature definitions,
source snapshots and the selection checks are preserved in the corresponding
`price_model_search_*_price_only_early.yaml` and
`price_model_rolling_*_price_only_sep_oct.yaml` configurations and manifests.

| Market | Selected model | Complete days | Products | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| DA | Elastic Net, DA curves | 52 | 4,992 | 16.850 | 7.852% |
| IDA1 | Elastic Net, spread to DA | 50 | 4,800 | 8.781 | 4.116% |
| IDA2 | Extra Trees, cleared prices | 49 | 4,704 | 10.319 | 4.817% |
| IDA3 | Ridge, spread to DA | 50 | 2,400 | 14.964 | 6.996% |

DA target coverage ends October 22 in this study; October 23–31 are excluded.
IDA1 lacks targets on October 8 and October 22–31. IDA2 excludes October 8–9
because cleared-auction inputs are missing and October 22–31 because targets
are missing. IDA3 excludes October 8 for missing inputs and October 22–31 for
missing targets. The IDA3 horizon contains 48 quarter-hour products per complete
day, whereas DA and IDA1/2 use 96 over these evaluated dates. The workflow's
configured target-clock fallback is used for some daily lag features; the
manifests record fallback counts and source provenance. All four selected runs
completed without recorded failures or convergence warnings.

Selection used early-period results, not the later prices; the later data were
not part of the earlier model comparisons reported in this study. This is a
more temporally separated assessment but is not a fully independent prospective
backtest: models, feature design and methods were developed after the historical
dates, and publication clocks and forecast availability are partly assumed.
Evaluation coverage differs between markets and excludes missing days. WAPE
also depends on the level of prices, so its lower percentage relative to earlier
periods does not necessarily indicate a lower error in EUR/MWh.

## Naive comparison and day-level uncertainty

The saved seasonal-naive forecasts use the same product observations as the
selected models. A paired bootstrap of complete delivery days (5,000 resamples,
seed 173) gives the following descriptive 95% percentile intervals for WAPE.

| Market | Selected WAPE | Bootstrap interval | Naive WAPE | Naive MAE EUR/MWh |
| --- | ---: | ---: | ---: | ---: |
| DA | 7.852% | 7.088–8.668% | 10.177% | 21.838 |
| IDA1 | 4.116% | 3.907–4.328% | 10.545% | 22.498 |
| IDA2 | 4.817% | 4.496–5.174% | 11.972% | 25.644 |
| IDA3 | 6.996% | 6.176–7.936% | 15.238% | 32.590 |

The selected models outperform this particular naive forecast over the observed
products. The intervals resample days as independent units, preserve within-day
products, and do not account for serial dependence, selection uncertainty,
missing-day bias or historical availability assumptions. The IDA1 interval
includes 4%, but its observed WAPE is above 4%; an interval including the target
does not establish that the target has been achieved.

## Interpretation of lag fallback counts

For a lagged same-auction price, the feature builder first looks for the exact
elapsed-hour lag and checks that its recorded or assumed availability precedes
the decision. If unavailable, it searches earlier prices at the required local
clock slot, no later than the requested lag time, subject to the same
availability check and a configured maximum age. For whole-day lags the local
clock slot is based on the target product; other lags use the requested
elapsed-hour timestamp. If no eligible value exists the feature remains missing
and the row can be excluded.

The manifest's `fallback_count` is accumulated over all lag features and all
rows used to build the historical matrix, including training and evaluation
rows. It is not a count of affected validation products or a percentage of test
forecasts. Availability checks enforce the catalog's supplied or assumed
publication clocks; they do not prove that those clocks were historically
observed. The exact implementation is `_features` in
`src/forecasting/energy_prices.py`, and the count is collected by
`prepare_search_matrix` in `src/forecasting/price_search.py`.

For the selected later-period forecasts, a separate reconstruction compared
each requested same-auction lag against the catalog's price timestamps and
availability clocks. The stored validation lag columns were verified finite;
therefore, on these retained products, an unavailable exact lag implies use of
the configured fallback. This count excludes daily-curve features and any
separately loaded earlier-auction features.

| Market | Products | Products with at least one fallback | Fraction | Fallback lag entries |
| --- | ---: | ---: | ---: | ---: |
| DA | 4,992 | 208 | 4.2% | 208 |
| IDA1 | 4,800 | 664 | 13.8% | 1,252 |
| IDA2 | 4,704 | 384 | 8.2% | 580 |
| IDA3 | 2,400 | 0 | 0.0% | 0 |

The DA fallbacks were all for the 23-hour lag. These are evaluation-only
counts, unlike the aggregate matrix counts above. They are conditional on the
selected complete days and the assumed or supplied publication clocks; they
do not measure forecast accuracy with fallback disabled.
