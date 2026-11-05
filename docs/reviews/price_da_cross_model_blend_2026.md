# DA forecast combination diagnostic

This exploratory comparison combines two earlier-selected weekly rolling DA
forecasts: an interaction Elastic Net using prior IDAs, fundamentals and
regional weather, and an interaction Elastic Net using price curves and
regional weather without those extra fundamentals. The saved source runs are
complete, and their common product timestamps and realized prices were checked.
The intersection contains 10,272 products; source inputs, exact prediction
file hashes and manifest hashes are retained in the output.

A fixed convex weight from 0 to 1 in steps of 0.05 was selected by pooled
MAE on 5,376 May–June products. It selected 85% weight on the model with
fundamentals and 15% on the price-and-weather model. The fixed blend was then
scored on 4,896 shared products across 51 July–August days.

| July–August forecast | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Model with fundamentals | 14.896 | 9.106% |
| Price-and-weather model | 15.882 | 9.709% |
| Selected blend | 14.773 | 9.031% |

The paired MAE gain versus the stronger individual model was 0.123 EUR/MWh;
an independent-delivery-day bootstrap with 5,000 resamples and seed 173 gives
a 95% percentile interval of -0.024 to 0.277 EUR/MWh. This excludes serial
dependence and model/weight-selection uncertainty. Both periods were already
examined in other studies; historical auction clocks and weather vintage remain
assumed or unverified. The small gain does not establish improved prospective
performance or the 4% target.

The reproducible configuration is
`configs/research/price_da_cross_model_blend_2026.yaml`, with the evaluator
`scripts/evaluate_two_price_forecasts.py`. Its complete manifest and matched
test predictions are in
`outputs/energy_price_forecasts/nord_da_2026_cross_model_blend_v1/`.
