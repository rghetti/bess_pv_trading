# IDA spreads and neighboring cleared-auction products

This exploratory CPU selection predicts spreads to each auction's latest
cleared price. Inputs include earlier-auction products at -60, -15, +15 and
+60 minutes around the delivery product, subject to their configured
publication clocks. At a delivery boundary, the configured `same_product`
policy uses the same product instead of inventing a missing neighbor.
Weather and no-weather arms share training and evaluation products within
each auction; the zero-spread model is the latest-cleared-price baseline.

All 36 fits completed without recorded failures. Best early candidates were
selected on already inspected February–April 2026 products. IDA2 has 87
complete days under this boundary policy, one more than the earlier
one-direction study, so aggregate scores across those studies do not by
themselves isolate the effect of additional neighbors.

| Auction | Days | Products | Selected early candidate | MAE EUR/MWh | WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| IDA1 | 88 | 8,444 | Elastic Net with weather | 5.929 | 4.656% | 5.986 | 4.701% |
| IDA2 | 87 | 8,348 | Extra Trees with weather | 6.885 | 5.377% | 6.902 | 5.390% |
| IDA3 | 86 | 4,128 | Ridge with weather | 11.442 | 9.103% | 11.592 | 9.223% |

The paired MAE gains over the baseline were 0.057, 0.017 and 0.151 EUR/MWh.
The 95% percentile intervals from 5,000 independent-day bootstrap resamples
with seed 173 were -0.079–0.187, -0.067–0.101 and -0.107–0.409 EUR/MWh.
They omit serial dependence and selection uncertainty and all include zero.
These selection results do not establish a reliable gain or the 4% target;
historical auction and weather release clocks remain assumptions. Forecasts,
coverage and source hashes are in
`outputs/energy_price_forecasts/nord_ida_2026_bidirectional_latest_spread_early_v1/`.

Matched later replays were configured for the selected candidates and the
zero-spread references; later dates have also been examined in related work.

## May–August rolling replay

The selected candidates and zero-spread references were refitted weekly with
a one-day gap from 3 May through 31 August 2026. All 12 fits completed without
recorded failures or convergence warnings. Each comparison uses identical
products within the auction; the boundary policy and common-input requirements
give IDA2 114 complete days, versus 109 in the earlier one-direction weather
study, so their aggregate scores are not a direct neighbor ablation.

| Auction | Days | Products | Selected weather model MAE EUR/MWh | Model WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 115 | 11,040 | 6.678 | 4.557% | 6.677 | 4.557% |
| IDA2 | 114 | 10,944 | 7.873 | 5.350% | 7.834 | 5.323% |
| IDA3 | 113 | 5,424 | 13.171 | 9.073% | 13.053 | 8.992% |

The selected models are respectively Elastic Net, Extra Trees and ridge. Their
paired MAE gains over the baseline were approximately 0.000, -0.040 and
-0.118 EUR/MWh. Independent-day bootstrap 95% percentile intervals, using
5,000 resamples and seed 173, were -0.139–0.132, -0.213–0.135 and
-0.496–0.273 EUR/MWh. They omit serial dependence and selection uncertainty;
all span zero. These previously inspected dates and assumed information clocks
do not establish prospective improvement or the 4% target. Predictions and
refits are in
`outputs/energy_price_forecasts/nord_ida_2026_bidirectional_latest_spread_may_aug_v1/`.

## September–October rolling replay

The same early-selected models were evaluated with weekly refits and a
one-day gap. All 12 fits completed without recorded failures or convergence
warnings. Each auction has 50 shared days; IDA3 covers 48 products per day.

| Auction | Products | Weather model MAE EUR/MWh | Model WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 4,800 | 8.762 | 4.107% | 8.885 | 4.164% |
| IDA2 | 4,800 | 10.296 | 4.796% | 9.933 | 4.627% |
| IDA3 | 2,400 | 14.889 | 6.961% | 14.998 | 7.013% |

The paired baseline-minus-model MAE gains were +0.122, -0.363 and +0.110
EUR/MWh, respectively. Independent-day bootstrap 95% percentile intervals
from 5,000 resamples with seed 173 were -0.084–0.332, -0.763–0.016 and
-0.544–0.787 EUR/MWh. All include zero and omit serial dependence and model
selection uncertainty. The selection and later periods have been repeatedly
inspected; clocks and forecast vintage retain the caveats above. This does not
establish robust improvement over the latest-cleared reference or a 4% result
across auctions. Predictions and refits are in
`outputs/energy_price_forecasts/nord_ida_2026_bidirectional_latest_spread_sep_oct_v1/`.
