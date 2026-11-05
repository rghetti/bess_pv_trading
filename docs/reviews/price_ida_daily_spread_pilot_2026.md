# Joint daily IDA spread-curve pilot

This CPU pilot predicts each auction's delivery-window spread curve jointly
around the latest-cleared-price curve. It stacks the existing price/calendar
feature matrix by delivery product, fits input standardization and target
spread PCA on training days only, then fits multi-output ridge to PCA scores.
Ranks 4, 8 and 16 and ridge penalties 100, 1,000 and 10,000 were considered.
The source's baseline CSV is checked against the saved target and latest-price
anchor; products in each evaluation curve share a decision time.

IDA1 and IDA2 use full 96-product windows; IDA3 uses its 48-product noon-to-
midnight window. DST-length local days and incomplete windows are excluded
explicitly, even when a half-day DST window itself has 48 products. The early
February–April selection period has been examined in other studies; auction
publication clocks are assumptions.

| Auction | Training days | Selection days | Products | Selected rank / penalty | Curve MAE EUR/MWh | Curve WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| IDA1 | 370 | 87 | 8,352 | 4 / 100 | 7.061 | 5.541% | 5.979 | 4.692% |
| IDA2 | 362 | 85 | 8,160 | 4 / 10,000 | 6.948 | 5.414% | 6.860 | 5.345% |
| IDA3 | 366 | 85 | 4,080 | 4 / 10,000 | 11.658 | 9.255% | 11.649 | 9.248% |

Paired MAE gains of the selected curve model over the baseline were -1.082,
-0.088 and -0.009 EUR/MWh. Independent-day bootstrap 95% percentile intervals,
using 5,000 resamples and seed 173, were -1.456 to -0.725, -0.182 to 0.009,
and -0.160 to 0.146 EUR/MWh. They omit serial dependence and model-selection
uncertainty. The tested curve models did not beat the latest-cleared baseline
on their matched selection products, so no later replay was started. This does
not rule out other joint or neural curve methods or establish the 4% target.

Source and implementation hashes, exclusions, complete candidate scores,
selected predictions and first-day plots for each auction are saved in
`outputs/energy_price_forecasts/nord_ida_2026_daily_spread_curves_early_v1/`.
The study configuration is `configs/research/price_ida_daily_spread_curves_early.yaml`.
