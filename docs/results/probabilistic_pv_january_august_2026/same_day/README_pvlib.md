# Physical PV ensemble baseline

The same farm-specific 2025 parameters are treated as prior metadata for both weather sources. The same-day 00 UTC cycle is used retrospectively after 00 UTC, with the preceding-day cycle filling local midnight intervals before 00 UTC. Cycle availability is assumed six hours after issue, so same-day weather was not available at the DA or IDA1 gate for that delivery day. Stitched member labels are not proven to represent one trajectory across runs.

Training ends 2026-08-19; validation ends 2026-08-31; there is no independent test. Calibration uses the configured training-only folds. The first validation day is retrospective at its historical day-ahead gate. January 1 has a missing first local hour.

| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) | Bias (% AC) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deterministic | 1152 | 0.02735 | 2.87 | 0.05190 | 5.44 | -0.26 |
| 1 | member_mean | 1152 | 0.02772 | 2.91 | 0.05005 | 5.25 | -0.86 |
| 1 | member_median | 1152 | 0.02758 | 2.89 | 0.05061 | 5.31 | -0.69 |
| 1 | calibrated_mean | 1152 | 0.02708 | 2.84 | 0.04926 | 5.17 | -1.10 |
| 1 | calibrated_median | 1152 | 0.02502 | 2.62 | 0.04752 | 4.98 | -0.79 |
| 345 | deterministic | 1152 | 0.10077 | 6.61 | 0.17378 | 11.40 | -5.45 |
| 345 | member_mean | 1152 | 0.10226 | 6.71 | 0.17593 | 11.54 | -5.70 |
| 345 | member_median | 1152 | 0.10108 | 6.63 | 0.17417 | 11.42 | -5.58 |
| 345 | calibrated_mean | 1152 | 0.05693 | 3.73 | 0.10549 | 6.92 | -0.63 |
| 345 | calibrated_median | 1152 | 0.05160 | 3.38 | 0.10504 | 6.89 | 0.38 |
| 901 | deterministic | 1152 | 0.22644 | 5.44 | 0.40911 | 9.83 | 0.27 |
| 901 | member_mean | 1152 | 0.22627 | 5.44 | 0.40339 | 9.70 | -0.46 |
| 901 | member_median | 1152 | 0.22634 | 5.44 | 0.40634 | 9.77 | -0.09 |
| 901 | calibrated_mean | 1152 | 0.22279 | 5.36 | 0.39801 | 9.57 | -0.58 |
| 901 | calibrated_median | 1152 | 0.22016 | 5.29 | 0.39687 | 9.54 | -0.03 |
| 910 | deterministic | 1101 | 0.36223 | 5.46 | 0.66377 | 10.00 | 4.42 |
| 910 | member_mean | 1101 | 0.33640 | 5.07 | 0.62177 | 9.37 | 3.93 |
| 910 | member_median | 1101 | 0.35172 | 5.30 | 0.64480 | 9.71 | 4.22 |
| 910 | calibrated_mean | 1101 | 0.29459 | 4.44 | 0.50993 | 7.68 | -1.16 |
| 910 | calibrated_median | 1101 | 0.27994 | 4.22 | 0.49744 | 7.49 | -0.19 |

The raw ensemble uses equal empirical member weights; this is not a demonstrated calibrated probability distribution. All percent errors use the farm AC nameplate as denominator. Calibrated curves are marginal quantiles, not calibrated joint trajectories. The calibrated mean integrates a piecewise linear quantile curve with constant tails.

Per-farm folders contain all-study and validation metrics, daily plots, paired predictions, coverage and quantile-correction counts. Training-period calibration scores are in-sample.
