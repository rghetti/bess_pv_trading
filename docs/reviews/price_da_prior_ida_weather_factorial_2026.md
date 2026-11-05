# DA price curves, prior IDAs and regional weather

This CPU selection compares four matched feature arms: DA price curves alone,
the same curves with regional temperature and shortwave forecasts, with
24/48-hour IDA1 and IDA2 price lags, and with both additions. All arms use
9,644 identical training products through January 2026 and 7,868 matched
February–April selection products across 82 complete days. Weather is from
Milan, Turin, Verona and Venice; its operational-or-hindcast status is
unverified and cycle-plus-six-hours publication is assumed. Historical
auction release clocks are also partly assumed.

Five candidates were fitted per arm: Elastic Net, interaction ridge,
interaction Elastic Net, Extra Trees and CPU XGBoost. All 20 fits completed
without recorded failures or convergence warnings. The table uses the same
interaction Elastic Net in every arm to make the feature comparison explicit.

| Feature arm | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| DA curves | 13.223 | 10.315% |
| DA curves and weather | 12.278 | 9.578% |
| DA curves and prior IDAs | 13.128 | 10.242% |
| DA curves, prior IDAs and weather | 12.133 | 9.465% |

For the matched interaction model, the MAE gains from prior IDA inputs were
0.095 EUR/MWh without weather and 0.145 with weather. Adding weather to the
prior-IDA arm reduced MAE by 0.995 EUR/MWh. Independent-day bootstrap 95%
percentile intervals, with 5,000 resamples and seed 173, are respectively
-0.053–0.246, 0.019–0.278 and 0.574–1.408 EUR/MWh. These intervals ignore
serial dependence and selection uncertainty. The same period has been
repeatedly examined; neither the model nor feature rankings constitute an
untouched evaluation or establish the 4% target. Full scores, input snapshots
and forecasts are in
`outputs/energy_price_forecasts/nord_da_2026_price_ida_weather_early_v1/`.

Matched weekly May–August and September–October replays are configured as
separate later evaluations. Their manifests determine completion; those dates
have also been examined in related studies.

## May–August rolling evaluation

The earlier-selected interaction Elastic Net was refitted weekly with a one-day
gap from 3 May through 31 August. All four matched feature arms completed
without reported failures or convergence warnings on 10,272 products across
107 shared complete days. Missing input days are excluded.

| Feature arm | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| DA curves | 16.417 | 11.302% |
| DA curves and weather | 16.226 | 11.171% |
| DA curves and prior IDAs | 16.070 | 11.063% |
| DA curves, prior IDAs and weather | 15.949 | 10.980% |

The paired MAE gains from prior IDAs were 0.347 EUR/MWh without weather and
0.276 with weather; the weather gain with prior IDAs was 0.121 EUR/MWh.
Independent-day bootstrap 95% percentile intervals, 5,000 resamples with seed
173, were respectively 0.203–0.484, 0.160–0.387 and -0.430–0.697 EUR/MWh.
These omit serial dependence and selection uncertainty. Some later days were
examined in other studies, and the feature clocks and weather vintage remain
assumptions. The paired sample does not establish prospective performance or
the 4% target. Predictions and refits are in
`outputs/energy_price_forecasts/nord_da_2026_price_ida_weather_may_aug_rolling_v1/`.
