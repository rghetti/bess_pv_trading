# Joint daily DA curve pilot

This CPU pilot predicts all 96 quarter-hour prices of an ordinary local day
jointly. It stacks the source experiment's per-product information-set matrix
into one daily input, standardizes inputs on training days, fits PCA to training
target curves, and fits multi-output ridge to the resulting target scores.
Ranks 4, 8 and 16 and ridge penalties 100, 1,000 and 10,000 were compared for
two feature arms. Both target PCA and input scaling are fitted only on training
days. Every evaluation product within a daily curve shares the source's
forecast decision time.

The input matrices come from the native quarter-hourly DA prior-IDA/weather
factorial study. They use its historical feature-vintage and publication-time
assumptions. Only exact 96-product local-day grids were used: the October 2025
DST day and a partial January 2026 training day were excluded, as was the
92-product March 2026 DST selection day. Each arm therefore has 99 complete
training days and 81 already inspected February–April selection days, or 7,776
products. This pilot does not evaluate variable-length DST days.

| Feature arm | Best selection rank / penalty | Curve MAE EUR/MWh | Curve WAPE | Matched interaction MAE EUR/MWh | Interaction WAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Price curves | 8 / 1,000 | 13.789 | 10.749% | 13.039 | 10.164% |
| Price, prior IDAs and weather | 8 / 10,000 | 12.894 | 10.051% | 11.950 | 9.316% |

For the best overall curve candidate, the matched interaction model's MAE gain
was 0.944 EUR/MWh. A 5,000-resample independent-delivery-day bootstrap with
seed 173 gave a 95% percentile interval of 0.304–1.522 EUR/MWh. It excludes
serial dependence and model-selection uncertainty. Since both curve candidates
trailed their matched references on this selection sample, no later replay was
started. This does not rule out other joint or neural curve architectures and
does not establish the 4% target.

The source manifest, dataset and baseline hashes, excluded days, full grid,
selected predictions and a first-day plot are retained in
`outputs/energy_price_forecasts/nord_da_2026_daily_curves_early_v1/`. The
configuration is `configs/research/price_da_daily_curves_early.yaml`.
