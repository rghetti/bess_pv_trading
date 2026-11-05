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

The following descriptive local-time breakdown uses the same 52 days, with
1,248 quarter-hour products per block. These blocks were inspected after the
replay and are not independent model-selection tests.

| Local time | Price-only MAE EUR/MWh | Price-only WAPE | With-weather MAE EUR/MWh | With-weather WAPE |
| --- | ---: | ---: | ---: | ---: |
| 00:00–05:59 | 11.985 | 5.950% | 11.969 | 5.942% |
| 06:00–11:59 | 19.427 | 8.634% | 19.139 | 8.506% |
| 12:00–17:59 | 20.884 | 10.998% | 20.243 | 10.660% |
| 18:00–23:59 | 15.085 | 6.233% | 15.168 | 6.267% |

Each block's WAPE uses its own sum of absolute realized prices as the
denominator. The midday period has the largest observed error in this sample;
these descriptive figures alone do not identify its cause.

## Delivery-hour interaction search

Following the time-of-day diagnostic, a CPU search compared the earlier plain
Elastic Net with ridge, Lasso and Elastic Net models that interact input signals
with delivery-hour sine and cosine features. The same price-only and weather
arms were scored on the 8,252 shared February–April selection products.

| Candidate | Feature arm | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: |
| Interaction Elastic Net, alpha 0.1 | Price and weather | 12.594 | 9.903% |
| Plain Elastic Net, alpha 0.1 | Price and weather | 12.797 | 10.062% |
| Interaction Elastic Net, alpha 0.1 | Price only | 13.459 | 10.583% |
| Plain Elastic Net, alpha 0.1 | Price only | 13.459 | 10.583% |

All ten fits completed without reported failures. Both alpha 0.01 interaction
Elastic Net fits reported convergence warnings; selection excludes warned fits.
This early period was repeatedly examined and the interaction search followed
the later time-of-day inspection, so the comparison is exploratory. A matched
September–October replay is configured, with the plain Elastic Net as a
reference; that later period was also previously examined. Input weather
vintages and historical release clocks retain the assumptions above.
