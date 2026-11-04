# IDA model retraining cadence

For each auction, the table compares the same earlier-selected model and features
across three refit frequencies in the May–August 2026 evaluation window. The
prediction files have identical product timestamps and actual values within each
auction. The one-day training gap, price features and historical input data are
held fixed. IDA3 has fewer products because its configured prediction coverage is
more limited; scores are not directly comparable across auctions as if they used
the same delivery intervals.

| Auction | Products | Daily WAPE | Weekly WAPE | 28-day WAPE | Daily MAE EUR/MWh | Weekly MAE EUR/MWh | 28-day MAE EUR/MWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 11,136 | 4.543% | 4.540% | 4.546% | 6.641 | 6.638 | 6.646 |
| IDA2 | 10,560 | 5.610% | 5.611% | 5.617% | 8.182 | 8.185 | 8.193 |
| IDA3 | 5,472 | 9.028% | 9.020% | 9.014% | 13.066 | 13.054 | 13.046 |

IDA1 uses `elastic_01` on the spread to DA with cleared prices and combined
fundamentals; IDA2 uses `lasso_1` with cleared prices; IDA3 uses `lasso_01`
with cleared prices and fundamentals. The daily and 28-day runs recorded no
failures or convergence warnings. The older weekly manifest did not record
per-fit convergence warnings. Differences here are small and do not establish a
universal preferred frequency. The evaluation dates were previously examined and
some input publication clocks remain assumptions, so this is exploratory rather
than an untouched deployment test. Study configurations and manifests preserve
the exact selection and feature settings.
