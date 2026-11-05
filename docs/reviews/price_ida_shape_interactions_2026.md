# IDA time-of-day interaction search

The early search tested ridge and Elastic Net models whose lagged-price and
cleared-auction effects vary smoothly with delivery hour, alongside non-interaction
ridge, Lasso and Elastic Net references. IDA1 and IDA2 use their configured
11-lag layout, while IDA3 uses its five-lag layout; this matters because the
interaction transformer identifies calendar columns by position. Feature sets
contain previous prices and earlier cleared auction prices, predicting either
the price or its spread to DA. Candidates were ranked on February–April 2026
MAE and the top candidate per auction was refitted weekly on the available
September–October products. Exact model parameters and source lineage are in
the auction-specific `price_model_search_ida*_shape.yaml` and
`price_model_rolling_ida*_shape.yaml` configs and manifests.

| Auction | Early-selected candidate | Later products | Later MAE EUR/MWh | Later WAPE |
| --- | --- | ---: | ---: | ---: |
| IDA1 | Interaction Elastic Net, spread to DA | 4,800 | 8.782 | 4.116% |
| IDA2 | Lasso, direct price | 4,704 | 10.127 | 4.728% |
| IDA3 | Ridge, spread to DA | 2,400 | 14.964 | 6.996% |

On the same IDA1 products and actuals, the earlier selected non-interaction
Elastic Net scored 8.781 EUR/MWh, within 0.001 EUR/MWh of the interaction
model. IDA2 and IDA3 selected non-interaction candidates during early ranking;
the later IDA2 365-day Lasso comparison in
`price_ida_training_windows_sep_oct_2026.md` scored lower MAE. All searches and
selected later fits completed without recorded failures or convergence
warnings. Both periods were examined in related studies, input publication
clocks are partly assumed, and missing-day coverage differs between markets.
These results do not establish that other interaction designs cannot improve
forecasts.
