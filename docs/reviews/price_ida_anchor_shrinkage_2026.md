# Shrinking IDA forecasts toward the latest cleared price

For each IDA, this diagnostic takes the early-selected price-only model from
`price_price_only_sep_oct_2026.md` and blends its forecast with the latest
cleared same-product price. A model weight from 0 to 1 in increments of 0.05
is chosen by MAE on the same February–April period used for model selection;
the fixed weight is then scored on the available September–October products.
The source model predictions, targets and anchors are aligned by timestamp, and
the anchor columns were separately verified against catalog prices. No model
is retrained for the blend. Selection on the same early period can overfit,
the later period was already examined, and historical publication clocks are
partly assumed.

| Market | Model weight | Products | Blend MAE EUR/MWh | Blend WAPE | Cleared-price baseline WAPE | Original model WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 0.70 | 4,800 | 8.774 | 4.112% | 4.164% | 4.116% |
| IDA2 | 0.25 | 4,704 | 9.920 | 4.631% | 4.640% | 4.817% |
| IDA3 | 0.65 | 2,400 | 14.919 | 6.976% | 7.013% | 6.996% |

A paired independent-day bootstrap (5,000 resamples, seed 173) gives 95%
percentile intervals for MAE gain over the cleared-price baseline of -0.047 to
+0.274 EUR/MWh for IDA1, -0.048 to +0.084 for IDA2 and -0.169 to +0.322 for
IDA3. These intervals do not account for serial dependence, model or weight
selection, or missing-day bias. None establishes a reliable improvement over
the baseline, and this diagnostic does not demonstrate the 4% target across
markets or periods.

## Six-hour block diagnostic

A follow-up allowed a separate model weight of 0, 0.5 or 1 in each six-hour
local delivery block. Weights were chosen by February–April MAE and then fixed
for the same later products above. The IDA3 delivery horizon covers only the
12:00–17:59 and 18:00–23:59 blocks in this dataset.

| Market | Weights for 00–05, 06–11, 12–17, 18–23 | Later MAE EUR/MWh | Later WAPE | Latest-cleared baseline MAE EUR/MWh |
| --- | --- | ---: | ---: | ---: |
| IDA1 | 1, 0, 0.5, 1 | 8.851 | 4.149% | 8.885 |
| IDA2 | 1, 0, 0, 0.5 | 9.942 | 4.642% | 9.939 |
| IDA3 | not applicable, not applicable, 1, 0.5 | 14.927 | 6.979% | 14.998 |

A paired independent-day bootstrap with 5,000 resamples and seed 173 gives 95%
percentile intervals for MAE gain over the cleared-price baseline of -0.143 to
+0.225 EUR/MWh for IDA1, -0.180 to +0.168 for IDA2 and -0.270 to +0.407 for
IDA3. These intervals omit serial dependence and selection uncertainty. The
blockwise method does not show a clear improvement, and uses periods already
inspected in other studies. Configured publication clocks remain partly assumed.
