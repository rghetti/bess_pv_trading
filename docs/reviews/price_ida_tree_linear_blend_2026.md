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
