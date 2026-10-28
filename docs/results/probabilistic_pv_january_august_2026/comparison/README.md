# Matched forecast-vintage comparison

Both studies use the same 49 training and 12 validation delivery days and matched measured AC targets. The preceding-day study uses the previous 00 UTC run throughout. The same-day study uses the current 00 UTC run retrospectively after 00 UTC and the previous run before 00 UTC.

Same-day weather is unavailable at that delivery day’s DA and IDA1 gates under the six-hour availability assumption. The model for each objective is selected independently on the same validation days, so comparisons are exploratory.

| Farm | Selection | Score | Prior model | Same-day model | Matched quarters | Prior (% AC) | Same day (% AC) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | all | rmse_mw | pvlib | pvlib | 1152 | 5.03 | 5.25 |
| 1 | all | mae_mw | pvlib | pvlib | 1152 | 2.81 | 2.89 |
| 1 | all | crps_mw | pvlib | pvlib | 1152 | 1.88 | 1.96 |
| 1 | member_mean | rmse_mw | tcn_core_01 | tcn_core_01 | 1152 | 5.07 | 5.10 |
| 1 | member_median | mae_mw | tcn_core_02 | extra_trees_weather_03 | 1152 | 2.66 | 2.66 |
| 1 | calibrated_quantiles | crps_mw | huber_core_03 | huber_core_03 | 1152 | 2.05 | 1.89 |
| 345 | all | rmse_mw | pvlib | pvlib | 1152 | 11.48 | 11.54 |
| 345 | all | mae_mw | pvlib | pvlib | 1152 | 6.59 | 6.63 |
| 345 | all | crps_mw | pvlib | pvlib | 1152 | 2.48 | 2.51 |
| 345 | member_mean | rmse_mw | tcn_weather_02 | tcn_core_02 | 1152 | 6.73 | 6.68 |
| 345 | member_median | mae_mw | tcn_weather_02 | tcn_weather_02 | 1152 | 3.35 | 3.26 |
| 345 | calibrated_quantiles | crps_mw | tcn_weather_02 | tcn_weather_01 | 1152 | 2.42 | 2.55 |
| 901 | all | rmse_mw | pvlib | pvlib | 1152 | 10.04 | 9.70 |
| 901 | all | mae_mw | pvlib | pvlib | 1152 | 5.66 | 5.44 |
| 901 | all | crps_mw | pvlib | pvlib | 1152 | 4.01 | 3.89 |
| 901 | member_mean | rmse_mw | tcn_core_01 | tcn_weather_02 | 1152 | 10.07 | 9.66 |
| 901 | member_median | mae_mw | tcn_core_04 | tcn_weather_02 | 1152 | 5.51 | 5.13 |
| 901 | calibrated_quantiles | crps_mw | tcn_weather_01 | tcn_weather_02 | 1152 | 4.31 | 3.86 |
| 910 | all | rmse_mw | pvlib | pvlib | 1101 | 9.28 | 9.37 |
| 910 | all | mae_mw | pvlib | pvlib | 1101 | 5.27 | 5.30 |
| 910 | all | crps_mw | pvlib | pvlib | 1101 | 3.08 | 3.04 |
| 910 | member_mean | rmse_mw | tcn_core_01 | tcn_weather_01 | 1101 | 5.72 | 5.29 |
| 910 | member_median | mae_mw | tcn_core_01 | tcn_weather_01 | 1101 | 2.83 | 2.77 |
| 910 | calibrated_quantiles | crps_mw | tcn_core_04 | tcn_core_04 | 1101 | 2.29 | 2.09 |

Scores use all matched eligible quarters, including nighttime. Percentages divide by installed farm AC capacity. Calibrated CRPS is calculated from marginal quantiles by numerical quadrature. Raw member labels stitched across runs do not establish joint trajectory identity.

Daily plots show the curated observation in black, preceding-day mean/median and p10–p90 in blue, and same-day mean/median and p10–p90 in orange. Gray shading marks the preceding-cycle midnight fill; the dotted vertical line marks the assumed same-day cycle availability six hours after issue. The same-day values before that line are retrospective.
