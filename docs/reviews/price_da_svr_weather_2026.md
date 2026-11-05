# DA support-vector forecast search

The CPU grid tested RBF support-vector regressors with standardized inputs and
targets, a 60-day training window, two regularization strengths and two epsilon
values. Features contain prior DA curves, prior IDA prices and fundamentals,
with or without regional weather. Interaction ridge and Elastic Net references
use full training history. Training ends in January 2026; February–April is the
previously inspected validation period. Exact settings are in
`price_model_search_da_svr_regional_weather.yaml` and its manifest.

The best support-vector candidate used weather and scored 21.238 EUR/MWh MAE
and 16.568% WAPE on 7,868 products. The best interaction Elastic Net with
weather scored 11.860 EUR/MWh and 9.252% on the same evaluation set. The
support-vector candidates were not promoted to a later replay. All 12 fits
completed without recorded failures or convergence warnings. The grid differs
in training window and model family from the references, so it does not isolate
either factor. Weather vintage and publication clocks remain unverified or
assumed. Unit tests cover finite predictions for the adapter.
