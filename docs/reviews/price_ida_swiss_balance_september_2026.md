# Swiss balance in September IDA spread forecasts

The study tests the previous-day Swissgrid national consumption-minus-production
balance alongside lagged prices and cleared auction prices. It predicts spreads
to the latest cleared auction: IDA1 for IDA2 and IDA2 for IDA3. The Swiss
balance is a national proxy, not measured flow to Italy, and the configured
historical publication clock is assumed. Candidates were selected by
February–April MAE and refitted weekly in September. Each with/without-Swiss
pair has the same training-row policy, product timestamps and actual prices.
September was examined in related studies, so this is exploratory.

| Auction | Products | No Swiss MAE EUR/MWh | With Swiss MAE EUR/MWh | No Swiss WAPE | With Swiss WAPE | Latest-cleared baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA2 | 2,880 | 9.822 | 9.825 | 4.808% | 4.809% | 4.782% |
| IDA3 | 1,440 | 14.880 | 14.833 | 7.385% | 7.362% | 7.511% |

There are 30 complete days per auction. The selected models are Elastic Net
for IDA2 and ridge for IDA3. A paired independent-day bootstrap with 5,000
resamples and seed 173 gives 95% percentile intervals for MAE gain from adding
Swiss balance of -0.009 to +0.004 EUR/MWh for IDA2 and -0.081 to +0.173 for
IDA3. Against the latest-cleared-price baseline, the intervals for the with-Swiss
models are -0.182 to +0.076 and -0.360 to +0.925 EUR/MWh respectively. These
intervals omit serial dependence and model-selection uncertainty and do not
establish an improvement. All early and later fits completed without recorded
failures; the later fits had no recorded convergence warnings. The exact
feature, source snapshots and selection are in
`price_model_search_ida_latest_spread_swiss.yaml`,
`price_model_rolling_ida_latest_spread_swiss.yaml` and their manifests.
