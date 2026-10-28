# Probabilistic PV ensemble study

Status: **complete**, 256/256 model configurations complete.

49 delivery days through 2026-08-19 are training; 12 days from 2026-08-20 through 2026-08-31 are validation used for model selection. There is no independent test. The first validation day uses completed prior-day targets retrospectively after its day-ahead gate.

Declared forecast delivery windows: 2026-01-01 through 2026-01-31; 2026-08-02 through 2026-08-31.

The same-day 00 UTC cycle is used retrospectively after 00 UTC, with the preceding-day cycle filling local midnight intervals before 00 UTC. Cycle availability is assumed six hours after issue, so same-day weather was not available at the DA or IDA1 gate for that delivery day. Stitched member labels are not proven to represent one trajectory across runs. January 1 has missing instantaneous weather in its first local hour. Source NWP identity and member weights are unspecified.

Raw member mean, median and quantiles summarize 50 member-wise PV paths; equal empirical weights do not establish calibrated probabilities. Linear quantile models use the configured chronological out-of-fold predictions from training only. Calibration metrics on the training period are in-sample, and validation-selected results are exploratory.

Calibrated quantiles are sorted and bounded; raw quantiles and adjustment flags are saved. The calibrated mean integrates a piecewise linear quantile curve with constant tails. Hourly quantiles are not calculated from marginal quarter-hour quantiles.

| Farm | Objective | Selected candidate | Score (MW) | Score (% AC) |
| --- | --- | --- | ---: | ---: |
| 1 | member_mean / rmse_mw | tcn_core_01 | 0.04865 | 5.10 |
| 1 | member_median / mae_mw | extra_trees_weather_03 | 0.02539 | 2.66 |
| 1 | calibrated_quantiles / crps_mw | huber_core_03 | 0.01806 | 1.89 |
| 345 | member_mean / rmse_mw | tcn_core_02 | 0.10182 | 6.68 |
| 345 | member_median / mae_mw | tcn_weather_02 | 0.04975 | 3.26 |
| 345 | calibrated_quantiles / crps_mw | tcn_weather_01 | 0.03882 | 2.55 |
| 901 | member_mean / rmse_mw | tcn_weather_02 | 0.40197 | 9.66 |
| 901 | member_median / mae_mw | tcn_weather_02 | 0.21344 | 5.13 |
| 901 | calibrated_quantiles / crps_mw | tcn_weather_02 | 0.16053 | 3.86 |
| 910 | member_mean / rmse_mw | tcn_weather_01 | 0.35097 | 5.29 |
| 910 | member_median / mae_mw | tcn_weather_01 | 0.18354 | 2.77 |
| 910 | calibrated_quantiles / crps_mw | tcn_core_04 | 0.13902 | 2.09 |

Each farm folder contains pvlib metrics, paired predictions, calibration diagnostics, daily plots and per-trial outputs. `validation_rankings.csv` records every candidate. Quarter-hour CRPS for the raw ensemble is exact; calibrated quantile CRPS is numerical quadrature.

## Physical baseline on the validation period

| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | deterministic | 1152 | 0.02735 | 2.87 | 0.05190 | 5.44 |
| 1 | member_mean | 1152 | 0.02772 | 2.91 | 0.05005 | 5.25 |
| 1 | member_median | 1152 | 0.02758 | 2.89 | 0.05061 | 5.31 |
| 1 | calibrated_mean | 1152 | 0.02708 | 2.84 | 0.04926 | 5.17 |
| 1 | calibrated_median | 1152 | 0.02502 | 2.62 | 0.04752 | 4.98 |
| 345 | deterministic | 1152 | 0.10077 | 6.61 | 0.17378 | 11.40 |
| 345 | member_mean | 1152 | 0.10226 | 6.71 | 0.17593 | 11.54 |
| 345 | member_median | 1152 | 0.10108 | 6.63 | 0.17417 | 11.42 |
| 345 | calibrated_mean | 1152 | 0.05693 | 3.73 | 0.10549 | 6.92 |
| 345 | calibrated_median | 1152 | 0.05160 | 3.38 | 0.10504 | 6.89 |
| 901 | deterministic | 1152 | 0.22644 | 5.44 | 0.40911 | 9.83 |
| 901 | member_mean | 1152 | 0.22627 | 5.44 | 0.40339 | 9.70 |
| 901 | member_median | 1152 | 0.22634 | 5.44 | 0.40634 | 9.77 |
| 901 | calibrated_mean | 1152 | 0.22279 | 5.36 | 0.39801 | 9.57 |
| 901 | calibrated_median | 1152 | 0.22016 | 5.29 | 0.39687 | 9.54 |
| 910 | deterministic | 1101 | 0.36223 | 5.46 | 0.66377 | 10.00 |
| 910 | member_mean | 1101 | 0.33640 | 5.07 | 0.62177 | 9.37 |
| 910 | member_median | 1101 | 0.35172 | 5.30 | 0.64480 | 9.71 |
| 910 | calibrated_mean | 1101 | 0.29459 | 4.44 | 0.50993 | 7.68 |
| 910 | calibrated_median | 1101 | 0.27994 | 4.22 | 0.49744 | 7.49 |
