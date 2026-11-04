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

A matched later rolling comparison is configured using the validation-selected
candidates and their no-weather counterparts. Its source snapshot verification
and complete-day coverage should be checked before interpreting the result.
Some later dates were inspected in previous work, so that evaluation is not a
globally untouched test.
