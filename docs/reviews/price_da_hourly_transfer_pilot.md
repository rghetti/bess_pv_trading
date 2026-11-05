# Hourly-to-quarter-hourly DA transfer pilot

This exploratory price-only pilot uses 14,577 hourly examples from 2024 through
September 2025 to fit an hourly ridge model. A separate quarter-hourly ridge is
trained on 10,128 quarter-hourly examples after the resolution boundary, to
predict the quarter-hourly price less a fixed fraction of the hourly model's
prediction. The two source manifests have identical lag definitions and timezone
and declare 60- and 15-minute native products respectively; the pilot uses only
the 11 lag and seven calendar columns common to those matrices. It does not
interpolate hourly labels into quarter-hourly training labels.

The grid varies the hourly and quarter-hourly ridge penalties over 20 and 200,
and the hourly prediction weight over 0, 0.5 and 1. A zero weight is the
matched quarter-hourly-only reference. On the 8,252 already examined
February–April 2026 products, the best grid result was 16.943 EUR/MWh MAE and
13.322% WAPE, versus 16.968 and 13.342% for the matched zero-weight model
with the same quarter-hourly penalty. These are selection-period scores; the
tiny difference does not establish transfer benefit. The models use only
price-lag and calendar inputs and are not directly comparable to weather or
other feature-rich results on different product sets.

The reproducible pilot is `scripts/evaluate_hourly_transfer_pilot.py` with
`configs/research/price_hourly_transfer_pilot.yaml`. It checks native resolutions,
lag definitions, timezone, completed source manifests, the hourly boundary and
chronological separation; hourly validation labels must be published and their
products ended by the boundary. It emits source-array checksums with results.
The unit test verifies that a post-boundary hourly label cannot affect scores
and that overlapping training dates are rejected. Source availability clocks
remain assumptions, and the saved hourly sample has lag warm-up exclusions.

## Nonlinear transfer and later replay

A follow-up replaces the hourly ridge with Extra Trees and fits a separate
quarter-hourly ridge to the residual. The hourly model still uses only hourly
labels available by the resolution boundary. The February–April selection
grid considered two minimum leaf sizes, two residual penalties, and three
hourly prediction weights. The selected model used 150 trees, minimum leaf
size 3, maximum feature fraction 0.6, ridge penalty 200 and hourly weight 1.
Its matched reference has weight 0 and the same quarter-hourly penalty.

| Price-only evaluation | Products | Transfer MAE, EUR/MWh | Transfer WAPE | Reference MAE, EUR/MWh | Reference WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| February–April 2026 selection | 8,252 | 14.670 | 11.535% | 16.968 | 13.342% |
| September–October 2026 rolling replay | 4,992 | 24.024 | 11.195% | 17.410 | 8.113% |

The later replay uses eight weekly refits and a one-day gap; each refit only
adds previously completed and published quarter-hourly labels. The final nine
to ten October days are absent from the available source. Both models are
scored on the same products within each period, but the later period had
already been inspected in other searches and is not an untouched test set.
The average hourly training price was about 111 EUR/MWh, versus 215 EUR/MWh
in the later evaluation; the shift may contribute to the poor transfer but
does not establish its cause. These results do not establish that every
hourly-to-quarter-hourly transfer approach will fail.

The configurations are `configs/research/price_hourly_tree_transfer_early.yaml`
and `configs/research/price_hourly_tree_transfer_later.yaml`. The same script
emits input checksums and scores for both; the test also checks that a future
target cannot affect earlier rolling predictions. Weather and other external
features are absent, and historical publication times remain assumptions.
