# Exact median-regression IDA spread search

This CPU search used standardized linear median regression, solved as a
regularized linear program with HiGHS, to predict spreads to DA or the latest
cleared auction. It compared three regularization strengths with an explicit
zero-spread cleared-price baseline. Training used historical prices through
January 2026; the February–April validation period was already inspected in
related work. Exact inputs, anchors, feature lineage and parameters are in
`price_model_search_ida_median_spreads.yaml` and its manifest.

| Market | Products | Best median MAE EUR/MWh | Best median WAPE | Latest-cleared baseline MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA2 | 8,252 | 6.911 | 5.387% | 6.910 | 5.386% |
| IDA3 | 4,128 | 11.523 | 9.168% | 11.592 | 9.223% |

Both best median candidates use the latest-cleared-auction spread anchor. All
16 fits completed without recorded failures or convergence warnings. SciPy
emits an `OptimizeWarning` stating that the explicit one-thread option is
forwarded to HiGHS; this is distinct from a convergence warning. Model
selection and scoring use the same already-inspected period, so these are
exploratory. The selected IDA3 model and cleared-price baseline are configured
for a later weekly replay in `price_model_rolling_ida3_median_spreads.yaml`;
that replay's manifest is authoritative for completion and later scores.
Historical publication clocks remain partly assumed.

## Later IDA3 replay

The selected latest-anchor median model completed the later weekly replay
without recorded failures or convergence warnings. On 2,400 matched products
across 50 complete September–October days, it scored 14.944 EUR/MWh MAE and
6.987% WAPE; the explicit IDA2 clearing-price baseline scored 14.998 and
7.013%. Predictions and actuals align by product timestamp. A paired
5,000-resample independent-day bootstrap with seed 173 gives a 95% percentile
interval of -0.163 to +0.265 EUR/MWh for the model's MAE gain. The interval
excludes serial dependence and model-selection uncertainty, and the periods
were already inspected. The small observed gain does not establish an
improvement over the cleared-price baseline or the 4% target.
