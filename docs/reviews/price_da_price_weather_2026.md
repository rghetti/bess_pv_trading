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

## May–August rolling replay

The early-selected Elastic Net was refitted weekly with a one-day gap for
3 May–31 August 2026. Both feature arms used 11,520 identical quarter-hour
products across 120 complete days and finished without recorded failures or
convergence warnings.

| Features | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Price only | 16.564 | 11.171% |
| Price and regional weather | 16.840 | 11.357% |

The weather model's paired MAE gain over price-only was -0.277 EUR/MWh. A
5,000-resample independent-delivery-day bootstrap with seed 173 gives a 95%
percentile interval of -0.748 to 0.201 EUR/MWh. This comparison is exploratory:
some days were examined in other studies, and the interval excludes serial
dependence and selection uncertainty. Source weather vintage and release timing
remain unverified assumptions. Predictions and refits are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_may_aug_rolling_v1/`.

### Online bias correction

An exploratory post-processing grid tuned a residual correction on May–June
and held its settings fixed for July–August. It considers no correction and
3/7/14/28-day windows, 1/4/24-hour local blocks and shrinkage 0/24/96. Each
daily correction uses only earlier products whose configured publication and
product-end clocks precede the decision. The chosen setting was a 28-day
window, one-hour blocks and shrinkage 24.

On the same 5,952 products across 62 July–August days, the corrected model
scored 15.993 EUR/MWh and 9.596% WAPE, compared with 16.433 EUR/MWh and
9.860% before correction. The paired MAE gain was 0.440 EUR/MWh; a 5,000-
resample independent-day bootstrap with seed 173 gives a 95% percentile
interval of -0.106 to 0.984 EUR/MWh. Both periods have been examined and the
interval omits serial dependence and selection uncertainty. Historical clocks
and weather vintage remain assumptions, so the result does not establish a
reliable improvement or the 4% target. Scores, aligned predictions and the
daily-error plot are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_online_bias_v1/`.

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

The subsequent weekly September–October replay completed all four selected and
reference fits with no reported failures or convergence warnings. Each scored
the same 4,992 products over 52 complete days.

| Candidate | Feature arm | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: |
| Interaction Elastic Net | Price and weather | 16.272 | 7.583% |
| Plain Elastic Net | Price and weather | 16.630 | 7.749% |
| Interaction Elastic Net | Price only | 16.657 | 7.762% |
| Plain Elastic Net | Price only | 16.845 | 7.850% |

On matched products the weather interaction model's MAE gain over the plain
weather model was 0.358 EUR/MWh. A 5,000-resample independent-delivery-day
bootstrap with seed 173 gives a 95% percentile interval of -0.001 to 0.740
EUR/MWh. This does not account for serial dependence, selection or repeated
inspection, and does not establish a reliable gain or the 4% target. The last
nine October days lack DA targets. Full forecasts and refits are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_interactions_sep_oct_v1/`.

The same selected interaction and plain models were also refitted weekly over
3 May–31 August on 11,520 matched products across 120 complete days. All four
fits completed without recorded failures or convergence warnings.

| Candidate | Feature arm | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: |
| Interaction Elastic Net | Price and weather | 16.219 | 10.938% |
| Interaction Elastic Net | Price only | 16.352 | 11.028% |
| Plain Elastic Net | Price and weather | 16.840 | 11.357% |
| Plain Elastic Net | Price only | 16.564 | 11.171% |

For the weather interaction model, the paired MAE gain was 0.621 EUR/MWh
against the plain weather model and 0.133 against the price-only interaction
model. Independent-day bootstrap 95% percentile intervals, 5,000 resamples
with seed 173, are 0.372–0.871 and -0.411–0.681 EUR/MWh respectively. They
omit serial dependence, selection and repeated inspection of these dates.
The first comparison favors the interaction specification in this sample;
the second does not establish an incremental weather gain. Historical
publication clocks and weather vintage remain assumptions. Predictions and
refits are in
`outputs/energy_price_forecasts/nord_da_2026_price_weather_interactions_may_aug_v1/`.

### September-tuned interaction bias correction

An exploratory online correction grid tuned on September MAE and evaluated
the fixed choice on available October products. It tested no correction and
3/7/14/28-day residual windows, 1/4/24-hour local blocks and shrinkage
0/24/96. Residuals contribute only when prior products have ended and their
assumed publication clock precedes the decision. The selected correction used
a seven-day window, one-hour blocks and shrinkage 24.

On 2,112 products across 22 October days, uncorrected MAE was 16.298 EUR/MWh
and WAPE was 7.192%; after correction they were 16.235 EUR/MWh and 7.164%.
The paired MAE gain of 0.063 EUR/MWh has an independent-day bootstrap 95%
percentile interval of -0.539 to 0.639 EUR/MWh, using 5,000 resamples and
seed 173. The interval includes zero and omits selection effects; both
months were inspected previously, the last nine October days lack targets,
and historical weather and publication assumptions remain. The output at
`outputs/energy_price_forecasts/nord_da_2026_interaction_weather_sep_oct_bias_v1/`
contains the full grid, predictions and daily-error plot.
