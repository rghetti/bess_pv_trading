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
A more expressive transfer model and an untouched later period would be needed
to assess whether 2024 hourly history can materially improve quarter-hourly DA.
