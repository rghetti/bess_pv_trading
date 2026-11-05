# Prior IDA prices in later DA forecasts

The early February–April 2026 validation selected `elastic_01` from the
configured price-curve models. The same candidate was evaluated with weekly
refits and a one-day gap over September–October, both with and without previous
IDA1 and IDA2 prices at 24- and 48-hour lags. Both variants use DA price
curves and exclude external fundamentals. Exact feature definitions, model
candidates and lineage are in the early-search and later-rolling
`price_model_*_da_price_curves_prior_ida.yaml` configs and output manifests.

| Features | Complete days | Products | MAE EUR/MWh | WAPE |
| --- | ---: | ---: | ---: | ---: |
| DA curves | 50 | 4,800 | 16.925 | 7.906% |
| DA curves and prior IDAs | 50 | 4,800 | 16.734 | 7.817% |

The saved predictions have identical product timestamps and targets. Both fits
completed without recorded failures or convergence warnings. The shared set
excludes October 9–10 for missing required lagged IDA inputs and October 23–31
for missing DA targets. A paired bootstrap of delivery days, with 5,000
resamples and seed 173, gives a 95% percentile interval of -0.079 to +0.478
EUR/MWh for MAE improvement from adding prior IDAs. This interval resamples
complete days independently and does not include serial dependence or model
selection uncertainty.

This is an exploratory follow-up because the later dates were previously
inspected in the price-only study. Historical auction publication clocks are
assumptions and target-clock lag fallbacks are recorded in the manifests. The
comparison does not establish a reliable gain from prior IDA features.
