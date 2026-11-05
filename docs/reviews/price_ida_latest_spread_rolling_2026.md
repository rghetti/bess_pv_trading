# Latest-cleared-auction spread anchor in rolling IDA forecasts

The earlier February–April validation selected Elastic Net for IDA2 and Lasso
for IDA3 from the existing nearest-auction spread search. For each auction,
the same model and feature family were refitted weekly over May–August 2026,
predicting a spread either to DA or to the most recently cleared auction:
IDA1 for IDA2 and IDA2 for IDA3. Inputs include prior prices, cleared-auction
prices and configured fundamentals. The split, clock assumptions and exact
candidate parameters are in `price_model_rolling_ida_nearest_spread.yaml` and
its output manifest.

| Auction | Products | DA anchor MAE EUR/MWh | DA anchor WAPE | Latest anchor MAE EUR/MWh | Latest anchor WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA2 | 10,560 | 9.584 | 6.570% | 7.799 | 5.347% |
| IDA3 | 5,472 | 13.056 | 9.021% | 13.001 | 8.983% |

The paired prediction files have identical timestamps and actual prices. The
IDA2 latest-anchor forecast also aligns exactly with the previously evaluated
Extra Trees IDA2 forecast, which scored 7.879 EUR/MWh and 5.401% WAPE on these
products. All four fits completed without recorded failures or convergence
warnings.

For the paired anchor comparison, a 5,000-resample independent-delivery-day
bootstrap (seed 173) gives a 95% percentile interval of +1.501 to +2.057
EUR/MWh for the IDA2 MAE improvement and +0.005 to +0.107 for IDA3. These
intervals omit serial dependence and model-selection uncertainty. The early
and later periods have been examined in other analyses, and historical
publication clocks are partly assumed. This is exploratory evidence for the
anchor, not an untouched prospective result or evidence that 4% is met.
