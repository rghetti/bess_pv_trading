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

## September–October rolling replay

All 12 selected and benchmark fits completed without reported failures or
convergence warnings. The models were refitted weekly with a one-day gap; each
row compares the same products and realized prices. The last nine or ten days
of October without targets and other auction-specific gaps remain excluded.

| Auction | Days | Products | Selected weather model MAE EUR/MWh | Model WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 50 | 4,800 | 8.769 | 4.110% | 8.885 | 4.164% |
| IDA2 | 49 | 4,704 | 10.353 | 4.833% | 9.939 | 4.640% |
| IDA3 | 50 | 2,400 | 15.025 | 7.025% | 14.998 | 7.013% |

The selected models are respectively Elastic Net, Extra Trees and ridge. Their
paired MAE gains over the baseline are 0.116, -0.415 and -0.027 EUR/MWh.
Independent-day bootstrap percentile intervals, 5,000 resamples with seed 173,
are -0.092 to 0.328, -0.806 to -0.051, and -0.745 to 0.723 EUR/MWh. These
intervals omit temporal dependence and selection uncertainty. The weather
candidate does not establish an improvement over the stronger baseline, and
the selected IDA2 tree performs worse on this sample. This previously examined
period is exploratory and its availability assumptions remain as above.
Predictions and refits are in
`outputs/energy_price_forecasts/nord_ida_2026_spread_price_weather_sep_oct_rolling_v1/`.

## May–August rolling replay

The same early-selected candidates and latest-cleared references were refitted
weekly with a one-day gap from 3 May to 31 August 2026. All 12 fits completed
without recorded failures or convergence warnings. Comparisons use identical
products within each auction; missing days are excluded.

| Auction | Days | Products | Weather model MAE EUR/MWh | Model WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 115 | 11,040 | 6.674 | 4.554% | 6.677 | 4.557% |
| IDA2 | 109 | 10,464 | 7.922 | 5.420% | 7.859 | 5.376% |
| IDA3 | 113 | 5,424 | 13.183 | 9.081% | 13.053 | 8.992% |

The models are respectively Elastic Net, Extra Trees and ridge. Their paired
MAE gains over the baseline are 0.004, -0.063 and -0.130 EUR/MWh. The 95%
percentile intervals from 5,000 independent-day bootstrap resamples with seed
173 are -0.135 to 0.138, -0.245 to 0.123 and -0.539 to 0.285 EUR/MWh.
All include zero and omit serial dependence and selection uncertainty. These
previously examined dates and the assumed historical clocks do not establish
an operational gain or the 4% target. Saved predictions and refits are in
`outputs/energy_price_forecasts/nord_ida_2026_spread_price_weather_may_aug_rolling_v1/`.

### Online bias correction

A residual-correction grid was tuned on May–June and evaluated on July–August
using only earlier completed products with assumed publication by the decision.
It included no correction and 3/7/14/28-day windows, 1/4/24-hour local blocks
and shrinkage 0/24/96. IDA1 selected a three-day, 24-hour-block correction with
shrinkage 96; IDA2 selected three days, four-hour blocks and shrinkage 96;
IDA3 selected no correction.

| Auction | Test days | Products | Corrected MAE EUR/MWh | Corrected WAPE | Latest-cleared MAE EUR/MWh | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 57 | 5,472 | 7.408 | 4.505% | 7.438 | 4.523% |
| IDA2 | 52 | 4,992 | 8.625 | 5.247% | 8.425 | 5.125% |
| IDA3 | 55 | 2,640 | 13.668 | 8.332% | 13.610 | 8.296% |

The paired MAE gains over the baseline are 0.030, -0.200 and -0.059 EUR/MWh.
Independent-day bootstrap 95% percentile intervals, using 5,000 resamples and
seed 173, are -0.238 to 0.277, -0.486 to 0.085 and -0.662 to 0.560 EUR/MWh.
These omit serial dependence and selection uncertainty. Both periods were
previously examined, and publication and weather-vintage assumptions remain;
the results do not demonstrate a reliable gain or the 4% target. The full grid,
aligned predictions and daily-error plots are in
`outputs/energy_price_forecasts/nord_ida_2026_spread_price_weather_online_bias_v1/`.
