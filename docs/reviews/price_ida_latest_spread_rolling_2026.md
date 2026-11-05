# Latest-cleared-auction spread anchor in rolling IDA forecasts

The earlier February–April validation selected Elastic Net for IDA2 and Lasso
for IDA3 from the existing nearest-auction spread search. For each auction,
the same model and feature family were refitted weekly over May–August 2026,
predicting a spread either to DA or to the most recently cleared auction:
IDA1 for IDA2 and IDA2 for IDA3. Inputs include prior prices, cleared-auction
prices and configured fundamentals. The split, clock assumptions and exact
candidate parameters are in `price_model_rolling_ida_nearest_spread.yaml` and
its output manifest.

| Auction | Products | DA anchor MAE EUR/MWh | DA anchor WAPE | Latest anchor MAE EUR/MWh | Latest anchor WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA2 | 10,560 | 9.584 | 6.570% | 7.799 | 5.347% |
| IDA3 | 5,472 | 13.056 | 9.021% | 13.001 | 8.983% |

The paired prediction files have identical timestamps and actual prices. The
IDA2 latest-anchor forecast also aligns exactly with the previously evaluated
Extra Trees IDA2 forecast, which scored 7.879 EUR/MWh and 5.401% WAPE on these
products. All four fits completed without recorded failures or convergence
warnings.

For the paired anchor comparison, a 5,000-resample independent-delivery-day
bootstrap (seed 173) gives a 95% percentile interval of +1.501 to +2.057
EUR/MWh for the IDA2 MAE improvement and +0.005 to +0.107 for IDA3. These
intervals omit serial dependence and model-selection uncertainty. The early
and later periods have been examined in other analyses, and historical
publication clocks are partly assumed. This is exploratory evidence for the
anchor, not an untouched prospective result or evidence that 4% is met.

## Later price-only follow-up

A separate price-only study selected models on February–April and evaluated
weekly refits on available September–October 2026 products. It included the
same DA and latest-auction anchors plus direct-price models. Because the later
period had already been examined, this is exploratory. For IDA2, the
validation-selected `elastic_05` had 10.643 EUR/MWh MAE (4.969% WAPE) as a
direct-price model, 11.559 (5.396%) with a DA spread anchor and 9.927 (4.635%)
with the IDA1 anchor. These variants have identical timestamps and actuals
across 4,704 products. The earlier baseline-candidate `lasso_01` scored 9.834
(4.591%) with a DA anchor and 9.882 (4.614%) with an IDA1 anchor on the same
products; choosing this lower later-period score after inspection is not a
fresh model selection. IDA3's early-selected latest-anchor ridge scored
14.964 EUR/MWh and 6.997% WAPE over 2,400 products.

A paired independent-day bootstrap with 5,000 resamples and seed 173 gives a
95% percentile interval of +1.106 to +2.152 EUR/MWh for the IDA2 Elastic Net
MAE improvement from the latest anchor, but -0.149 to +0.045 for the same
comparison with the Lasso reference. Intervals exclude serial dependence and
model-selection uncertainty. This indicates that anchor effects depend on the
model; it does not prove a universal benefit. All 42 early fits and 13 later
fits completed without recorded failures. Historical publication clocks and
missing-day limitations remain; details are in the price-only latest-spread
configs and manifests.

## Joint training-window and anchor follow-up

A subsequent price-only early search compared 180-day, 365-day and full-history
ridge, Lasso and Elastic Net candidates for both spread anchors. Early
February–April MAE selected a 180-day `elastic_05` model for IDA2 and a
full-history `ridge_20` model for IDA3. Weekly refits on the available
September–October products gave the following results:

| Auction | Anchor | Training window | Products | MAE EUR/MWh | WAPE |
| --- | --- | --- | ---: | ---: | ---: |
| IDA2 | IDA1 | 180 days | 4,704 | 9.972 | 4.656% |
| IDA2 | IDA1 | Full history | 4,704 | 9.927 | 4.635% |
| IDA2 | DA | 180 days | 4,704 | 11.696 | 5.460% |
| IDA2 | DA | Full history | 4,704 | 11.559 | 5.396% |
| IDA3 | IDA2 | Full history | 2,400 | 14.964 | 6.997% |
| IDA3 | DA | Full history | 2,400 | 14.964 | 6.996% |

The IDA2 180-day and full-history latest-anchor predictions have identical
product timestamps and actual values. A 5,000-resample independent-day
bootstrap with seed 173 gives a 95% interval of -0.031 to +0.124 EUR/MWh for
the full-history MAE improvement. It omits temporal dependence and selection
uncertainty. All early and later fits completed without recorded failures; the
later fits had no recorded convergence warnings. Both periods have been
inspected, and publication-clock and missing-day caveats remain. Configurations
are `price_model_search_ida_latest_spread_windows.yaml` and
`price_model_rolling_ida_latest_spread_windows.yaml`.
