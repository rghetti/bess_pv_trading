# Exploratory IDA tree and linear blends

This diagnostic combines the early-selected weekly rolling Extra Trees and
linear forecasts from `price_ida_extra_trees_rolling_2026.md`. Within each
auction, their product timestamps and actual prices were checked to be identical.
A grid of tree weights from 0 to 1 in increments of 0.05 was selected by pooled
MAE on May–June 2026; the fixed weight was then evaluated on July–August. No
model fitting used July–August targets for choosing the blend weight. Both time
periods have been examined during wider model development, so the comparison is
exploratory rather than a fresh holdout. Historical information-availability
assumptions remain those of the source studies.

| Auction | Tree weight | Test products | Blend MAE EUR/MWh | Blend WAPE | Tree WAPE | Linear WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 0.75 | 5,472 | 7.233 | 4.398% | 4.445% | 4.407% |
| IDA2 | 0.80 | 4,992 | 8.355 | 5.083% | 5.097% | 5.311% |
| IDA3 | 0.10 | 2,640 | 13.694 | 8.347% | 8.615% | 8.342% |

The blend yields small gains for IDA1 and IDA2 on these later products and is
slightly worse than the linear forecast for IDA3. The gains are too small and
the periods too extensively inspected to establish robust improvement or the
4% target. A subsequent assessment needs a genuinely unused evaluation period.

A paired delivery-day bootstrap (5,000 resamples, seed 2026) gives the following
95% percentile intervals for the improvement in MAE, calculated as comparator
error minus blend error. Positive values favor the blend.

| Auction | Days | Gain versus tree EUR/MWh (95% interval) | Gain versus linear EUR/MWh (95% interval) |
| --- | ---: | ---: | ---: |
| IDA1 | 57 | 0.078 (0.030, 0.127) | 0.014 (-0.146, 0.170) |
| IDA2 | 52 | 0.023 (-0.045, 0.092) | 0.375 (0.111, 0.635) |
| IDA3 | 55 | 0.440 (-0.029, 0.893) | -0.008 (-0.058, 0.043) |

The intervals resample complete delivery days, preserving within-day products,
but do not account for dependence between days, model-selection uncertainty or
the broader repeated inspection of this evaluation period. They do not establish
an out-of-sample improvement. In particular, the blend is not clearly better
than the stronger individual model for any of these three auctions.
