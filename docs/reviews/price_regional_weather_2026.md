# Regional forecast weather in 2026 price searches

The day-ahead study adds forecast temperature and shortwave radiation at Milan,
Turin, Verona and Venice to the existing market fundamentals, with and without
past IDA prices. These are individual regional weather points, not a
generation-weighted regional aggregate or farm PV forecasts. The source is the
Open-Meteo archive classified as operational-or-hindcast-unverified; the
configured forecast availability is cycle plus six hours, an assumption rather
than verified historical publication. Shortwave is an hourly mean ending at its
valid timestamp and is held constant within that source interval.

All 36 day-ahead fits completed without reported failures or convergence
warnings on 82 shared February–April validation days. The best observed model
with prior IDA and weather features scored 9.25% WAPE and €11.86/MWh MAE;
the best without added weather scored 9.46% and €12.12/MWh. Both were chosen
on that same period.

The candidate-paired comparison is less uniform: weather improved 2 of 9
models without prior IDA features and 1 of 9 with them. Median MAE changes
were +€0.190/MWh and +€0.163/MWh respectively, where positive means worse.
The validation-leading weather candidate was the exception, so the difference
between best scores should not be interpreted as a general weather benefit.
The full paired results and day-block intervals are in
`outputs/energy_price_forecasts/nord_da_2026_regional_weather_ablation_v1/`.

A matched later rolling comparison followed the validation-selected candidates
and their no-weather counterparts over 3 May–31 August 2026. All eight fits
completed without reported failures or convergence warnings, with 107 shared
days and 10,272 quarter-hour products. The preflight verified the original
selection-period weather rows and source snapshots.

| Model and feature arm | Later MAE €/MWh | Later WAPE |
|---|---:|---:|
| Selected elastic net, prior IDA prices, with regional weather | 15.189 | 10.46% |
| Same estimator and prior IDA prices, without regional weather | 15.473 | 10.65% |
| Elastic net without prior IDA prices, with regional weather | 15.431 | 10.62% |
| Same estimator without prior IDA or regional weather | 15.712 | 10.82% |

Ridge reference pairs also improved slightly with the weather inputs. These
are matched model comparisons, but some later dates were inspected in previous
work and the source weather vintage is unverified; the evaluation is not a
globally untouched or operationally certified test. Forecasts and refit
records are in `outputs/energy_price_forecasts/nord_da_2026_regional_weather_selected_rolling_v1/`.

The corresponding IDA validation completed 36 fits without reported failures
or convergence warnings. Adding regional weather improved 5 of 6 IDA1 models,
3 of 6 IDA2 models and all 6 IDA3 models, but the median MAE changes were only
-€0.011, +€0.001 and -€0.030/MWh respectively. These paired differences are
small on the shared validation products and do not establish a durable effect.
The detailed paired report is in
`outputs/energy_price_forecasts/nord_ida_2026_regional_weather_ablation_v1/`.

## Daypart blending diagnostic

A later diagnostic combined the earlier-selected `interaction_elastic_0.1`
forecasts with prior IDA inputs, with and without regional weather. Four
six-hour local-time blend weights were selected from 0, 0.25, 0.5, 0.75 and 1
on May–June MAE, then fixed for July–August. The selected weather weights were
1.0, 1.0, 0.5 and 0.0 for the four successive blocks. This is post-processing
of saved forecasts on previously inspected periods, not a new holdout.

| Forecast on July–August | Products | MAE EUR/MWh | WAPE | 10:00–16:59 MAE EUR/MWh |
| --- | ---: | ---: | ---: | ---: |
| No weather | 4,896 | 15.265 | 9.331% | 22.081 |
| Weather | 4,896 | 14.896 | 9.106% | 21.015 |
| Daypart blend | 4,896 | 15.022 | 9.183% | 21.435 |

The predictions use identical timestamps and targets across 51 shared days.
A paired 5,000-resample independent-day bootstrap with seed 173 gives a 95%
percentile interval of +0.137 to +0.604 EUR/MWh for the weather model's MAE
improvement over its no-weather counterpart. This excludes serial dependence,
selection and repeated-inspection uncertainty. The daypart blend did not improve
on the weather model on this period. Weather source vintage remains unverified
and its configured availability is assumed, as described above.

## Online bias-correction follow-up

The earlier-selected weather model was passed through the existing online
residual-correction grid. Settings were selected on May–June MAE and held fixed
for July–August; the selected setting used a seven-day window, four-hour local
blocks and shrinkage 96. Only earlier completed products whose configured
publication clocks precede each decision contribute residuals. On the same
4,896 products across 51 days, MAE fell from 14.896 to 14.392 EUR/MWh and WAPE
from 9.106% to approximately 8.80%.

A paired 5,000-resample independent-day bootstrap with seed 173 gives a 95%
percentile interval of +0.079 to +0.926 EUR/MWh for MAE improvement. It omits
serial dependence, model and correction selection, and repeated-inspection
uncertainty. The weather vintage is unverified, publication clocks are partly
assumed, and both periods have been examined; this is exploratory and does not
establish 4% DA accuracy. Settings, aligned predictions and a daily-error plot
are saved in `nord_da_2026_regional_weather_online_bias_v1` under
`outputs/energy_price_forecasts`.
