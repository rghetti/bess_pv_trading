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
