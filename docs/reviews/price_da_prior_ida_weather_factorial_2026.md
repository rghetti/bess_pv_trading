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
