# Selected IDA Extra Trees in later rolling evaluation

One Extra Trees model per quarter-hourly IDA was selected on February–April
2026 validation and refitted weekly for the later May–August evaluation. The
linear comparisons are earlier selected models from the existing weekly rolling
study. For each auction, saved forecasts have identical timestamps and actual
prices across model families; coverage differs between auctions.

| Auction | Products | Extra Trees MAE EUR/MWh | Extra Trees WAPE | Linear MAE EUR/MWh | Linear WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 11,136 | 6.577 | 4.499% | 6.638 | 4.540% |
| IDA2 | 10,560 | 7.879 | 5.401% | 8.185 | 5.611% |
| IDA3 | 5,472 | 13.470 | 9.307% | 13.054 | 9.019% |

The three tree models completed without recorded failures or convergence
warnings. The older linear manifest lacks per-fit warning records. These dates
have been inspected in related analyses and some input availability timestamps
are assumptions; the results are exploratory and should not be interpreted as
an untouched test. The tree configuration and selection provenance are in
`configs/research/price_model_rolling_ida_extra_trees.yaml` and its output
manifest. The linear configurations and cadence are documented in
`price_ida_refit_cadence_2026.md`.
