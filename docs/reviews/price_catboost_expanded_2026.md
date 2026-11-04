# Expanded CatBoost search in the quarter-hourly regime

The CPU search evaluated regularized CatBoost models with depths 4 and 8,
absolute-error and squared-error objectives, and 600–800 boosting iterations.
DA features included daily price curves and fundamentals, with or without prior
IDA prices. IDA features included previously cleared auction prices and
fundamentals, predicting either the price directly or its spread to cleared DA.
Training ended in January 2026; evaluation covers February–April. See
`configs/research/price_model_search_da_catboost_expanded.yaml` and
`configs/research/price_model_search_ida_catboost_expanded.yaml` for exact
parameters and input definitions.

| Market | Products | Best CatBoost MAE EUR/MWh | Best CatBoost WAPE | Earlier reference MAE EUR/MWh | Earlier reference WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DA | 7,868 | 16.230 | 12.661% | 12.121 | 9.456% |
| IDA1 | 8,444 | 6.128 | 4.812% | 6.037 | 4.741% |
| IDA2 | 8,252 | 7.245 | 5.647% | 7.027 | 5.477% |
| IDA3 | 4,128 | 12.325 | 9.806% | 11.828 | 9.411% |

The DA reference is interaction ridge and the IDA references are Extra Trees.
Their saved predictions match the corresponding CatBoost timestamps and actuals.
All 32 CatBoost fits completed with no recorded failures; the selected fits had
no recorded convergence warnings. The best configurations were selected on this
same previously examined period, so their scores are exploratory. Historical
publication timing and source forecast vintage are partly assumed. These results
do not exclude other boosting configurations or model families.
