# Probabilistic PV ensemble study

Status: **complete**, 256/256 model configurations complete.

49 delivery days through 2026-08-19 are training; 12 days from 2026-08-20 through 2026-08-31 are validation used for model selection. There is no independent test. The first validation day uses completed prior-day targets retrospectively after its day-ahead gate.

Declared forecast delivery windows: 2026-01-01 through 2026-01-31; 2026-08-02 through 2026-08-31.

The preceding-day 00 UTC cycle is used for the entire delivery day, with cycle availability assumed six hours after issue. Deterministic weather is held to the same 00 UTC cycle. January 1 has missing instantaneous weather in its first local hour. Source NWP identity and member weights are unspecified.

Raw member mean, median and quantiles summarize 50 member-wise PV paths; equal empirical weights do not establish calibrated probabilities. Linear quantile models use the configured chronological out-of-fold predictions from training only. Calibration metrics on the training period are in-sample, and validation-selected results are exploratory.

Calibrated quantiles are sorted and bounded; raw quantiles and adjustment flags are saved. The calibrated mean integrates a piecewise linear quantile curve with constant tails. Hourly quantiles are not calculated from marginal quarter-hour quantiles.

| Farm | Objective | Selected candidate | Score (MW) | Score (% AC) |
| --- | --- | --- | ---: | ---: |
| 1 | member_mean / rmse_mw | tcn_core_01 | 0.04835 | 5.07 |
| 1 | member_median / mae_mw | tcn_core_02 | 0.02534 | 2.66 |
| 1 | calibrated_quantiles / crps_mw | huber_core_03 | 0.01952 | 2.05 |
| 345 | member_mean / rmse_mw | tcn_weather_02 | 0.10256 | 6.73 |
| 345 | member_median / mae_mw | tcn_weather_02 | 0.05114 | 3.35 |
| 345 | calibrated_quantiles / crps_mw | tcn_weather_02 | 0.03687 | 2.42 |
| 901 | member_mean / rmse_mw | tcn_core_01 | 0.41893 | 10.07 |
| 901 | member_median / mae_mw | tcn_core_04 | 0.22920 | 5.51 |
| 901 | calibrated_quantiles / crps_mw | tcn_weather_01 | 0.17916 | 4.31 |
| 910 | member_mean / rmse_mw | tcn_core_01 | 0.37983 | 5.72 |
| 910 | member_median / mae_mw | tcn_core_01 | 0.18817 | 2.83 |
| 910 | calibrated_quantiles / crps_mw | tcn_core_04 | 0.15187 | 2.29 |

Each farm folder contains pvlib metrics, paired predictions, calibration diagnostics, daily plots and per-trial outputs. `validation_rankings.csv` records every candidate. Quarter-hour CRPS for the raw ensemble is exact; calibrated quantile CRPS is numerical quadrature.

## Physical baseline on the validation period

| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | deterministic | 1152 | 0.02808 | 2.95 | 0.05157 | 5.41 |
| 1 | member_mean | 1152 | 0.02739 | 2.87 | 0.04799 | 5.03 |
| 1 | member_median | 1152 | 0.02679 | 2.81 | 0.04738 | 4.97 |
| 1 | calibrated_mean | 1152 | 0.02558 | 2.68 | 0.04628 | 4.85 |
| 1 | calibrated_median | 1152 | 0.02431 | 2.55 | 0.04581 | 4.81 |
| 345 | deterministic | 1152 | 0.10032 | 6.58 | 0.17242 | 11.31 |
| 345 | member_mean | 1152 | 0.10188 | 6.68 | 0.17504 | 11.48 |
| 345 | member_median | 1152 | 0.10050 | 6.59 | 0.17273 | 11.33 |
| 345 | calibrated_mean | 1152 | 0.05521 | 3.62 | 0.10171 | 6.67 |
| 345 | calibrated_median | 1152 | 0.05157 | 3.38 | 0.10249 | 6.72 |
| 901 | deterministic | 1152 | 0.24467 | 5.88 | 0.43642 | 10.49 |
| 901 | member_mean | 1152 | 0.23731 | 5.70 | 0.41784 | 10.04 |
| 901 | member_median | 1152 | 0.23563 | 5.66 | 0.41998 | 10.10 |
| 901 | calibrated_mean | 1152 | 0.23338 | 5.61 | 0.41372 | 9.95 |
| 901 | calibrated_median | 1152 | 0.23327 | 5.61 | 0.41729 | 10.03 |
| 910 | deterministic | 1101 | 0.38292 | 5.77 | 0.71117 | 10.71 |
| 910 | member_mean | 1101 | 0.33551 | 5.05 | 0.61590 | 9.28 |
| 910 | member_median | 1101 | 0.35001 | 5.27 | 0.64198 | 9.67 |
| 910 | calibrated_mean | 1101 | 0.29424 | 4.43 | 0.52225 | 7.87 |
| 910 | calibrated_median | 1101 | 0.28059 | 4.23 | 0.51111 | 7.70 |
