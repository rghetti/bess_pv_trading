# DA Extra Trees with previous IDA prices

This comparison uses the native quarter-hourly DA regime, with training through
January 2026 and February–April validation. Models share 7,868 evaluation
products and matched training rows. Both feature sets contain lagged DA prices,
daily DA curves and fundamentals; the extended set adds prior IDA1 and IDA2
prices at 24- and 48-hour lags. Historical price publication timing and other
source vintages follow the study's stated assumptions. The period has been
inspected in other experiments and is not an untouched holdout.

| Model | WAPE without IDA | WAPE with IDA | MAE without IDA (EUR/MWh) | MAE with IDA (EUR/MWh) |
| --- | ---: | ---: | ---: | ---: |
| Interaction ridge, alpha 200 | 9.638% | 9.456% | 12.354 | 12.121 |
| Extra Trees, leaf 3, max features 0.6 | 12.318% | 11.882% | 15.790 | 15.231 |
| Extra Trees, leaf 3, max features 1.0 | 12.655% | 12.273% | 16.222 | 15.733 |
| Extra Trees, leaf 15, max features 0.6 | 12.623% | 12.209% | 16.180 | 15.651 |
| Extra Trees, leaf 15, max features 1.0 | 12.731% | 12.531% | 16.319 | 16.063 |

All ten fits completed without recorded failures or convergence warnings.
Earlier IDA prices improved each tested model here, but the Extra Trees models
remained behind the interaction-ridge reference. This contrasts with their
stronger historical hourly DA result and does not imply a general ranking for
other periods or feature sets. The exact configuration is
`configs/research/price_model_search_da_previous_idas_extra_trees.yaml`.
