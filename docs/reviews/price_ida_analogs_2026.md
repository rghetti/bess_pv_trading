# IDA nearest-neighbor price analogues

This CPU search standardized historical feature values using training data and
predicted either the price or its spread to cleared DA from 25, 100 or 400
nearest historical products, with distance weighting or uniform weights. Inputs
are prior same-auction prices and already cleared auction prices. The reference
grid contains ridge, Lasso and Elastic Net. Models train through January 2026
and are ranked by MAE on February–April products. Exact settings are in
`configs/research/price_model_search_ida_price_only_analog.yaml` and the output
manifest.

| Auction | Products | Best analogue MAE EUR/MWh | Best analogue WAPE | Best linear MAE EUR/MWh | Best linear WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 8,444 | 6.006 | 4.716% | 5.912 | 4.643% |
| IDA2 | 8,252 | 9.155 | 7.135% | 7.047 | 5.492% |
| IDA3 | 4,128 | 13.333 | 10.608% | 11.551 | 9.190% |

All 42 fits completed without recorded failures or convergence warnings. Each
best configuration was selected on this already inspected validation period;
the results do not measure untouched prospective performance. Historical
publication clocks are partly assumed. The nearest-neighbor factory is
thread-limited and its unit test verifies finite predictions. This grid does
not rule out different distance metrics, feature weighting or analogue designs.
