# Residual-demand price-feature comparisons

The matched 2026 comparisons added two configurable combinations of inputs
already available to the existing models: load forecast minus solar and
onshore-wind forecasts, and generation forecast minus load forecast. These
are derived predictors, not measured residual demand. The inputs retain the
source forecast-vintage and assumed availability policies, and missing
components are not filled. Training and evaluation rows are matched within
each comparison.

The day-ahead search completed 36 fits without reported failures or
convergence warnings on 82 shared February–April validation days. Adding the
derived inputs improved MAE for 0 of 9 candidates without prior IDA prices
and 2 of 9 candidates with them. The median MAE change was +€0.386/MWh and
+€0.362/MWh respectively; positive means worse. The best matched DA model
without the derived inputs scored 9.46% WAPE when prior IDA prices were used,
versus 9.74% for the best model with those additional derived inputs.

The IDA search completed 36 fits without reported failures or convergence
warnings. Adding the derived inputs improved 2 of 6 IDA1 candidates, 0 of 6
IDA2 candidates and 0 of 6 IDA3 candidates. Median MAE changes were
+€0.007, +€0.008 and +€0.046/MWh respectively. Validation covered 88 shared
IDA1 days and 86 shared days for each of IDA2 and IDA3.

These are exploratory results on previously inspected validation dates. They
do not establish that residual-demand information is unhelpful for other
models, features, horizons or periods. Forecast publication clocks are
assumptions. Per-candidate differences and day-block bootstrap intervals are
in `outputs/energy_price_forecasts/nord_da_2026_residual_demand_ablation_v1/`
and `outputs/energy_price_forecasts/nord_ida_2026_residual_demand_ablation_v1/`.
