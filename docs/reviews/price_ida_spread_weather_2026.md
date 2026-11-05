# IDA latest-price spreads with regional weather

This CPU search compares latest-cleared-price spread models with and without
temperature and shortwave forecast points for Milan, Turin, Verona and Venice.
The anchor is DA for IDA1, IDA1 for IDA2, and IDA2 for IDA3; the zero-spread
model is therefore the latest-cleared-price baseline. It uses price and calendar
inputs and excludes other fundamentals. Weather vintage is operational-or-
hindcast-unverified and the cycle-plus-six-hours publication clock is assumed.

Six candidates were fitted per feature arm and auction, using training data
through January and the already examined February–April 2026 selection period.
All 36 fits completed without recorded failures. Both feature arms use the same
products within each auction. IDA3 covers its native 48 quarter-hour products
per complete ordinary delivery day, rather than a full-day 96-product target.

| Auction | Days | Products | Best selection model | Model MAE EUR/MWh | Model WAPE | Latest-price MAE EUR/MWh | Latest-price WAPE |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| IDA1 | 88 | 8,444 | Elastic Net with weather | 5.910 | 4.641% | 5.986 | 4.701% |
| IDA2 | 86 | 8,252 | Extra Trees with weather | 6.884 | 5.366% | 6.910 | 5.386% |
| IDA3 | 86 | 4,128 | Ridge with weather | 11.449 | 9.109% | 11.592 | 9.223% |

A paired delivery-day bootstrap of the MAE gain over the latest-price baseline,
using 5,000 independent-day resamples and seed 173, yields intervals of
-0.060 to 0.210 EUR/MWh for IDA1, -0.057 to 0.111 for IDA2 and -0.123 to
0.420 for IDA3. These intervals ignore serial dependence, repeated selection and
the fact the models were selected on this same period. They do not demonstrate
a reliable gain or the 4% target. Results and source evidence are in
`outputs/energy_price_forecasts/nord_ida_2026_spread_price_weather_early_v1/`.

The selected candidate and zero-spread benchmark are configured for a matched
September–October rolling comparison. The later period was already examined in
other studies and the historical auction publication clocks are partly assumed.
