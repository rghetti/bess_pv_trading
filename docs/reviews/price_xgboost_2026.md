# CPU XGBoost price search

The quarter-hourly search trained 32 CPU histogram XGBoost models, with depths
3 and 6, varying minimum child weight and regularization. All use squared-error
training and 600 boosting iterations. DA models use daily price curves and
fundamentals with or without prior IDA prices; IDA models use previously cleared
prices and fundamentals and predict either prices or spreads to cleared DA.
Training ends in January 2026 and evaluation is February–April 2026. Exact
hyperparameters and features are in the DA and IDA XGBoost research configs.

| Market | Products | Best XGBoost MAE EUR/MWh | Best XGBoost WAPE | Earlier reference MAE EUR/MWh | Earlier reference WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DA | 7,868 | 15.540 | 12.123% | 12.121 | 9.456% |
| IDA1 | 8,444 | 6.320 | 4.963% | 6.037 | 4.741% |
| IDA2 | 8,252 | 7.320 | 5.706% | 7.027 | 5.477% |
| IDA3 | 4,128 | 12.667 | 10.078% | 11.828 | 9.411% |

The DA reference is interaction ridge and the IDA references are Extra Trees.
Each pair has identical saved product timestamps and actual prices. All 32 fits
completed without recorded failures; selected fits recorded no convergence
warnings. Candidate minima were chosen on this already inspected validation
period, so the results are exploratory. Historical publication clocks and
forecast vintages are partly assumed. These results do not rule out other
XGBoost settings or training methods.

The user authorized installation of the CPU-only `xgboost-cpu` 3.4.1 package in
the existing `bess` environment. The optional adapter's price-model unit tests
passed, including a one-thread CPU prediction test.

## Absolute-error training follow-up

A second 32-fit CPU grid uses XGBoost's `reg:absoluteerror` objective with
depths 3 and 6, two child-weight and regularization settings, 600 iterations,
and the same evaluation dates and feature families. Its selected predictions
have identical product timestamps and actual values to the squared-error grid's
selected predictions in each market. All fits completed without recorded
failures; the selected fits had no recorded convergence warnings.

| Market | Best absolute-error WAPE | Best absolute-error MAE EUR/MWh | Best earlier squared-error WAPE | Earlier reference WAPE |
| --- | ---: | ---: | ---: | ---: |
| DA | 12.261% | 15.717 | 12.123% | 9.456% |
| IDA1 | 4.798% | 6.110 | 4.963% | 4.741% |
| IDA2 | 5.587% | 7.168 | 5.706% | 5.477% |
| IDA3 | 9.785% | 12.298 | 10.078% | 9.411% |

Absolute-error training improved the best XGBoost result for each IDA but not
DA in these tested grids; all remain behind the earlier matched references.
The grids differ slightly in hyperparameters, so this is not a controlled
objective-only comparison. The same previously inspected validation period and
input-availability limitations apply.
