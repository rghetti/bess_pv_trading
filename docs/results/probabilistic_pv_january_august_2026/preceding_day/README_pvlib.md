# Physical PV ensemble baseline

The same farm-specific 2025 parameters are treated as prior metadata for both weather sources. Forecasts use the preceding 00 UTC cycle and the existing six-hour availability assumption.

Training ends 2026-08-19; validation ends 2026-08-31; there is no independent test. Calibration uses the configured training-only folds. The first validation day is retrospective at its historical day-ahead gate. January 1 has a missing first local hour.

| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) | Bias (% AC) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deterministic | 1152 | 0.02808 | 2.95 | 0.05157 | 5.41 | -0.61 |
| 1 | member_mean | 1152 | 0.02739 | 2.87 | 0.04799 | 5.03 | -0.90 |
| 1 | member_median | 1152 | 0.02679 | 2.81 | 0.04738 | 4.97 | -0.71 |
| 1 | calibrated_mean | 1152 | 0.02558 | 2.68 | 0.04628 | 4.85 | -0.94 |
| 1 | calibrated_median | 1152 | 0.02431 | 2.55 | 0.04581 | 4.81 | -0.86 |
| 345 | deterministic | 1152 | 0.10032 | 6.58 | 0.17242 | 11.31 | -5.48 |
| 345 | member_mean | 1152 | 0.10188 | 6.68 | 0.17504 | 11.48 | -5.73 |
| 345 | member_median | 1152 | 0.10050 | 6.59 | 0.17273 | 11.33 | -5.59 |
| 345 | calibrated_mean | 1152 | 0.05521 | 3.62 | 0.10171 | 6.67 | -0.36 |
| 345 | calibrated_median | 1152 | 0.05157 | 3.38 | 0.10249 | 6.72 | 0.46 |
| 901 | deterministic | 1152 | 0.24467 | 5.88 | 0.43642 | 10.49 | 0.29 |
| 901 | member_mean | 1152 | 0.23731 | 5.70 | 0.41784 | 10.04 | -0.25 |
| 901 | member_median | 1152 | 0.23563 | 5.66 | 0.41998 | 10.10 | 0.12 |
| 901 | calibrated_mean | 1152 | 0.23338 | 5.61 | 0.41372 | 9.95 | -0.49 |
| 901 | calibrated_median | 1152 | 0.23327 | 5.61 | 0.41729 | 10.03 | -0.62 |
| 910 | deterministic | 1101 | 0.38292 | 5.77 | 0.71117 | 10.71 | 4.68 |
| 910 | member_mean | 1101 | 0.33551 | 5.05 | 0.61590 | 9.28 | 3.82 |
| 910 | member_median | 1101 | 0.35001 | 5.27 | 0.64198 | 9.67 | 4.17 |
| 910 | calibrated_mean | 1101 | 0.29424 | 4.43 | 0.52225 | 7.87 | -1.48 |
| 910 | calibrated_median | 1101 | 0.28059 | 4.23 | 0.51111 | 7.70 | -0.38 |

The raw ensemble uses equal empirical member weights; this is not a demonstrated calibrated probability distribution. All percent errors use the farm AC nameplate as denominator. Calibrated curves are marginal quantiles, not calibrated joint trajectories. The calibrated mean integrates a piecewise linear quantile curve with constant tails.

Per-farm folders contain all-study and validation metrics, daily plots, paired predictions, coverage and quantile-correction counts. Training-period calibration scores are in-sample.
