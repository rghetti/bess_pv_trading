# Italian national holidays in electricity-price forecasts

A binary feature marks delivery products whose Europe/Rome local date is an
Italian national holiday. It includes January 1 and 6, Easter Sunday and
Easter Monday, April 25, May 1, June 2, August 15, November 1, December 8,
25 and 26. It deliberately excludes municipal patron-saint festivals, which
vary by location. The [European Medicines Agency's 2026 Italian public-holiday
list](https://www.ema.europa.eu/en/documents/other/european-union-member-state-public-holidays-recorded-ctis-year-2026_en.pdf)
independently lists April 25, May 1 and Easter Monday; its Rome patron-saint
entry is local. The feature is known from the calendar at forecast time and
is generated independently of measured load and price outcomes.

The first matched tests fit each existing model with and without the binary
feature, using identical training and validation rows. The DA model permits
the holiday feature to interact with delivery time; intraday models predict
spreads to the latest cleared auction and validate the anchor against the
previously saved zero-spread forecasts. Models train through January 2026 and
are evaluated on previously inspected February–April delivery days. WAPE is
100 times absolute error divided by absolute actual price. Weather archive
vintages and historical publication clocks retain the assumptions described
in the main price-search report.

| Model and features | Holiday dates | Products | Baseline MAE EUR/MWh | Baseline WAPE | With holiday MAE | With holiday WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA interaction, weather | 2 | 7,868 | 12.278 | 9.578% | 12.191 | 9.510% |
| DA interaction, weather and prior IDAs | 2 | 7,868 | 12.133 | 9.465% | 12.055 | 9.404% |
| IDA1 Elastic Net spread, weather | 3 | 8,444 | 5.929 | 4.656% | 5.923 | 4.651% |
| IDA2 Extra Trees spread, weather | 3 | 8,348 | 6.885 | 5.377% | 6.897 | 5.387% |
| IDA3 ridge spread, weather | 3 | 4,128 | 11.442 | 9.103% | 11.373 | 9.049% |

No convergence warnings were recorded for these fits. The DA holiday dates
are April 6 and April 25; Easter Sunday is missing from the shared DA sample.
For the DA model with prior IDAs, MAE improved by 3.51 EUR/MWh on April 6 and
1.84 on April 25, but these are only two dates and their selection period was
already inspected. This does not establish why prices changed, a prospective
benefit, or the 4% target. Weekly May–August and September–October DA and IDA
replays are in progress; holiday-specific metrics will
be reported separately from full-period averages. In particular May 1 is
outside the May–August replay's May 3 start. Later periods have been inspected
in previous searches and are not untouched final holdouts.

The IDA1 and IDA3 rolling replays have completed without warnings. In
May–August, IDA1 baseline and holiday-feature WAPE were 4.557% and 4.558%
over 115 days; IDA3 scored 9.073% and 9.100% over 113 days. The two shared
holiday dates were June 2 and August 15. Their respective baseline-minus-
holiday MAE differences were -0.285 and +0.353 EUR/MWh for IDA1, and +0.657
and -1.104 for IDA3. In September–October, IDA1 scored 4.107% with or
without the flag and IDA3 scored 6.961% versus 6.946%, on 50 days each. No
national holiday fell in that later evaluation sample, though training-period
holidays can still affect the fitted coefficients. These few holiday dates do
not establish an effect or rule one out.

The reusable implementation is `src/forecasting/italian_holidays.py`; the
matched experiment is `scripts/evaluate_italian_holiday_feature.py`. It records
the saved matrix hash, model parameters, dates and convergence warnings.
The first saved comparisons are in
`outputs/energy_price_forecasts/italian_holiday_early_2026_v1/`.
