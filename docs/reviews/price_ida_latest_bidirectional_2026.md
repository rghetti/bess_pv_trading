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

Matched later replays have been queued for the selected candidates and the
zero-spread references; later dates have also been examined in related work.
