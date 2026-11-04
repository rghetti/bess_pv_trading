# Hourly day-ahead Extra Trees rolling comparison

This study addresses the historical hourly DA product regime, not the later
quarter-hourly regime. The Extra Trees configurations were selected on an earlier
validation period and evaluated with weekly refits over July–September 2025.
The comparison includes 88 available delivery days and 2,112 hourly products;
missing days are not treated as zero. Inputs include lagged DA prices, daily
price curves and the configured fundamentals. Some historical information
availability and forecast vintages remain assumptions. These dates have also been
examined in other studies, so the comparison is exploratory rather than an
untouched final test.

| Model and features | MAE (EUR/MWh) | WAPE | Products |
| --- | ---: | ---: | ---: |
| Extra Trees, price curves and fundamentals, max features 0.6 | 7.242 | 6.647% | 2,112 |
| Extra Trees, price curves and fundamentals, max features 1.0 | 7.369 | 6.764% | 2,112 |
| CatBoost robust, price curves and fundamentals | 7.662 | 7.032% | 2,112 |
| Random forest, price curves and fundamentals | 7.740 | 7.104% | 2,112 |
| LightGBM L1, price curves and fundamentals | 8.058 | 7.396% | 2,112 |

Each Extra Trees model uses 150 trees and a minimum leaf size of three. Candidate
selection and rolling refit settings are recorded in
`configs/research/price_model_rolling_da_extra_trees.yaml` and the output
manifest. The CatBoost prediction file is from the earlier
`nord_da_curves_rolling_v1` run. Joining its saved predictions to the Extra Trees
predictions confirms identical timestamps and actual values across all 2,112
products. The Extra Trees run completed with no recorded failures or convergence
warnings. This comparison supports further evaluation of Extra Trees; it does not
establish a 4% error or performance in the quarter-hourly regime.
