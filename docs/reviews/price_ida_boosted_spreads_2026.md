# Baseline-inclusive boosted IDA spread search

This CPU grid compared XGBoost, LightGBM, CatBoost and histogram gradient
boosting against Elastic Net and an explicit zero-spread candidate. Inputs were
prior prices and cleared-auction prices; targets were spreads to DA or the
latest cleared auction. February–April 2026 MAE selected the latest-cleared
baseline for both markets. Exact settings and data lineage are in
`price_model_search_ida_boosted_spreads.yaml`,
`price_model_rolling_ida_boosted_spreads.yaml` and their manifests.

| Market | Early products | Best boosted MAE EUR/MWh | Best boosted WAPE | Latest-cleared baseline MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA2 | 8,252 | 6.946 | 5.414% | 6.910 | 5.386% |
| IDA3 | 4,128 | 11.999 | 9.547% | 11.592 | 9.223% |

Because the baseline ranked first, the later September–October replay evaluated
that baseline: 9.939 EUR/MWh and 4.640% WAPE on 4,704 IDA2 products, and
14.998 EUR/MWh and 7.013% on 2,400 IDA3 products. The boosted candidates were
not selected for that later replay. All 28 early and four later fits completed
without recorded failures. Both periods were previously examined, source
publication clocks are partly assumed, and missing days are excluded; these
results do not rule out other boosted models or establish a prospective 4%
forecast.
