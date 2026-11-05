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

## September–October rolling replay

The earlier-selected Elastic Net was refitted weekly with a one-day gap. The
source-lineage and shared-day checks passed against the separately assembled
catalog, which retains the original snapshots and adds the later weather runs.
Both fits completed without reported failures or convergence warnings. The
models scored the same 4,992 products over 52 complete days; 23–31 October
were excluded because DA targets are missing.

| Features | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Price only | 16.845 | 7.850% |
| Price and regional weather | 16.630 | 7.749% |

The paired MAE improvement was 0.216 EUR/MWh. A 5,000-resample independent-
delivery-day bootstrap with seed 173 gives a 95% percentile interval of
-0.089 to 0.525 EUR/MWh. It does not account for serial dependence or earlier
model selection and does not establish a reliable improvement. September and
October have already been examined in other studies; weather vintage and
publication timing remain unverified assumptions. Full results and refit
records are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_sep_oct_rolling_v1/`.
