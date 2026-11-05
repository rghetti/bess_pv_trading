# Previous-day Swiss balance in September DA forecasts

This study adds the Swissgrid national consumption-minus-production balance at
the corresponding interval on the previous day to a price-curve DA model. It
is a national balance proxy, not a measured cross-border flow to Italy. The
configured publication clock is assumed; actual historical availability was not
verified. Candidates were selected by February–April 2026 MAE, and the selected
Elastic Net was refitted weekly for September with and without the Swiss input.
Both feature arms use shared training and evaluation rows. September has been
examined in related price-only studies, so this is exploratory.

| Features | Complete days | Products | MAE EUR/MWh | WAPE |
| --- | ---: | ---: | ---: | ---: |
| DA price curves | 30 | 2,880 | 16.722 | 8.126% |
| DA price curves and previous-day Swiss balance | 30 | 2,880 | 16.611 | 8.072% |

Saved predictions have identical product timestamps and actual prices. A paired
5,000-resample independent-day bootstrap with seed 173 gives a 95% percentile
interval of +0.005 to +0.228 EUR/MWh for MAE improvement. It does not account
for serial dependence, model and feature selection, or publication uncertainty,
and one month does not establish a durable effect. All early and later fits
completed without recorded failures; the two later fits had no recorded
convergence warnings. Exact feature, selection and catalog snapshot details are
in `price_model_search_da_swiss_prev_price.yaml`,
`price_model_rolling_da_swiss_prev_price.yaml` and their manifests.
