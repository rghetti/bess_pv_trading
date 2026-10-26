# Five worst held-out days per farm

The ranking uses May–August 2026 test profiles. For each day, `daily_mean_auction_view_nrmse_percent` is the square root of the mean squared normalized RMSE across available auction views. The plot shows the worst auction view that day. Error is normalized by the farm AC rating. Each image includes measured AC, the selected pilot forecast, the same model with satellite GHI, pvlib forecasts, and GHI.

## Farm 1

| Day | Worst auction | Daily mean error | Worst profile error | Plot |
|---|---|---:|---:|---|
| 2026-05-14 | IDA3 | 16.02% | 18.26% | [PNG](farm_1/2026-05-14_IDA3.png) |
| 2026-05-15 | IDA3 | 15.53% | 16.72% | [PNG](farm_1/2026-05-15_IDA3.png) |
| 2026-06-02 | IDA3 | 14.23% | 15.57% | [PNG](farm_1/2026-06-02_IDA3.png) |
| 2026-08-20 | IDA3 | 13.67% | 14.41% | [PNG](farm_1/2026-08-20_IDA3.png) |
| 2026-05-11 | IDA3 | 13.40% | 16.22% | [PNG](farm_1/2026-05-11_IDA3.png) |

## Farm 345

| Day | Worst auction | Daily mean error | Worst profile error | Plot |
|---|---|---:|---:|---|
| 2026-06-05 | IDA1 | 20.96% | 26.00% | [PNG](farm_345/2026-06-05_IDA1.png) |
| 2026-05-12 | DA | 20.26% | 21.44% | [PNG](farm_345/2026-05-12_DA.png) |
| 2026-06-01 | IDA3 | 18.87% | 20.76% | [PNG](farm_345/2026-06-01_IDA3.png) |
| 2026-08-17 | IDA2 | 18.36% | 20.13% | [PNG](farm_345/2026-08-17_IDA2.png) |
| 2026-07-18 | IDA1 | 18.07% | 20.14% | [PNG](farm_345/2026-07-18_IDA1.png) |

## Farm 901

| Day | Worst auction | Daily mean error | Worst profile error | Plot |
|---|---|---:|---:|---|
| 2026-05-06 | IDA3 | 25.07% | 30.12% | [PNG](farm_901/2026-05-06_IDA3.png) |
| 2026-06-02 | IDA3 | 23.44% | 27.61% | [PNG](farm_901/2026-06-02_IDA3.png) |
| 2026-07-21 | IDA1 | 22.89% | 25.79% | [PNG](farm_901/2026-07-21_IDA1.png) |
| 2026-05-11 | IDA3 | 22.52% | 25.09% | [PNG](farm_901/2026-05-11_IDA3.png) |
| 2026-05-04 | IDA3 | 21.83% | 25.15% | [PNG](farm_901/2026-05-04_IDA3.png) |

## Farm 910

| Day | Worst auction | Daily mean error | Worst profile error | Plot |
|---|---|---:|---:|---|
| 2026-05-12 | IDA3 | 31.33% | 34.89% | [PNG](farm_910/2026-05-12_IDA3.png) |
| 2026-08-02 | IDA3 | 30.66% | 32.39% | [PNG](farm_910/2026-08-02_IDA3.png) |
| 2026-06-01 | IDA3 | 26.97% | 33.55% | [PNG](farm_910/2026-06-01_IDA3.png) |
| 2026-05-19 | IDA3 | 25.07% | 32.65% | [PNG](farm_910/2026-05-19_IDA3.png) |
| 2026-05-16 | DA | 22.18% | 22.95% | [PNG](farm_910/2026-05-16_DA.png) |

Scores use cached selected-pilot predictions; no retraining was done. The day ranking weights each available auction view equally and then plots the day’s worst view.
