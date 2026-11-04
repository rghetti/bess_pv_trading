# Quarter-hourly DA model retraining cadence

For the earlier-selected `interaction_ridge_200` model with previous IDA prices,
price-curve and fundamental features, the following runs compare refit intervals
on exactly the same 10,272 products from May through August 2026. The saved
prediction files have identical product timestamps and actual values. All runs
use the same configured one-day gap, training history and features; only the
refit interval changes.

| Refit interval | MAE (EUR/MWh) | WAPE |
| --- | ---: | ---: |
| 1 day | 15.797 | 10.875% |
| 7 days | 16.021 | 11.029% |
| 28 days | 16.626 | 11.446% |

The daily and 28-day runs completed without recorded failures or convergence
warnings. The older weekly manifest does not contain per-fit warning records.
The same evaluation period has been inspected in other studies and the source
features include assumed historical availability; treat this as exploratory,
not a fresh holdout or proof of deployment performance. These results concern
one selected model and do not establish an optimal refit interval for every
price model. The configured sources are `price_model_rolling_da_refit_1d.yaml`,
`price_model_rolling_da_previous_idas.yaml`, and
`price_model_rolling_da_refit_28d.yaml` in `configs/research/`.
