# Day-ahead price curves with regional forecast weather

This CPU search compares native quarter-hourly DA price-lag, daily curve and
calendar features with and without temperature and shortwave forecasts for
Milan, Turin, Verona and Venice. It excludes load, renewable and other market
fundamentals, permitting a later comparison where those series are missing.
The weather archive is classified as operational-or-hindcast-unverified and
the cycle-plus-six-hours release clock is assumed.

Training ends in January 2026. Seven linear, Extra Trees and XGBoost candidates
were fitted to each feature arm and ranked by MAE on February–April 2026, a
period already used throughout model development. All 14 fits completed with
no reported failures or convergence warnings. Both arms use the same 10,128
training rows and 8,252 evaluation products across 86 complete days; three
days were excluded because required price-curve inputs were incomplete.

| Early-selected model | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Elastic Net, price and weather | 12.797 | 10.062% |
| Same Elastic Net, price only | 13.459 | 10.583% |

The gain is 0.662 EUR/MWh on identical products. A paired 5,000-resample
independent-delivery-day bootstrap with seed 173 gives a 95% percentile interval
of 0.348–0.982 EUR/MWh. It excludes serial dependence and model-selection and
repeated-inspection uncertainty. This selection-period result does not establish
out-of-sample performance or the 4% target. Full candidate results, source
snapshots and forecasts are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_early_v1/`.

The later weekly replay is configured in
`configs/research/price_model_rolling_da_price_weather_sep_oct.yaml`; its
September–October period has already been examined in other studies. Its
separately assembled catalog must pass source-lineage and shared-day preflight
before results can be reported.
