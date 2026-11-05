# Exact median DA forecast search with regional weather

This CPU grid compared exact standardized median regressions with interaction
ridge and Elastic Net, using prior DA curves, prior IDA prices and fundamentals,
with and without four regional weather locations. Training ends in January 2026;
ranking uses the already examined February–April validation period. Weather
vintage may be hindcast and its publication clock is assumed. Exact parameters,
source snapshots and coverage are in
`price_model_search_da_median_regional_weather.yaml` and its manifest.

On 7,868 matched products, the best median candidate with weather scored
12.133 EUR/MWh MAE and 9.465% WAPE. The interaction Elastic Net with weather
scored 11.860 and 9.252%; interaction ridge without weather scored 12.121 and
9.456%. All ten fits completed without recorded failures or convergence
warnings. The HiGHS one-thread option produces a SciPy warning that it is
forwarded to the underlying solver. Early scores are selection scores, not an
untouched test. The top three candidates, including the median model and their
matched weather arms, are configured for a later weekly replay in
`price_model_rolling_da_median_regional_weather.yaml`; its manifest is
authoritative for completion and later scores.

## May–August rolling replay

The configured replay completed all six fits without reported failures or
convergence warnings. Each used weekly refits with a one-day gap and the same
10,272 products across 107 complete days between 3 May and 31 August 2026.

| Model | Features | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: |
| Median, penalty 0.01 | Fundamentals and previous IDAs | 15.567 | 10.717% |
| Median, penalty 0.01 | Same plus regional weather | 15.556 | 10.710% |
| Interaction Elastic Net | Fundamentals and previous IDAs | 15.473 | 10.652% |
| Interaction Elastic Net | Same plus regional weather | 15.189 | 10.457% |
| Interaction ridge | Fundamentals and previous IDAs | 16.021 | 11.029% |
| Interaction ridge | Same plus regional weather | 15.769 | 10.856% |

The median and Elastic Net weather predictions have identical timestamps and
actual values. A paired delivery-day bootstrap of their MAE difference, using
5,000 resamples and seed 173, gives an Elastic Net improvement of 0.367 EUR/MWh
with a 95% percentile interval of -0.333 to 1.085 EUR/MWh. This interval does
not capture dependence between days, model selection, or repeated inspection of
the period. The results do not establish a reliable ranking or the 4% target.
The source weather's operational vintage and the publication clocks remain
unverified assumptions. Predictions, refits, manifests and plots are in
`outputs/energy_price_forecasts/nord_da_2026_median_regional_weather_rolling_v1/`.
