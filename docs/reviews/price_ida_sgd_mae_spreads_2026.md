# Absolute-error linear IDA spread search

The CPU search compared standardized stochastic regressors trained with zero-
epsilon insensitive loss (absolute error) against ridge, Lasso and Elastic Net.
Inputs were prior prices and cleared-auction prices; targets were spreads to DA
or the latest cleared auction. Four stochastic regularization settings and three
linear references were ranked by February–April 2026 MAE. The best candidate per
auction was refitted weekly on September–October products, with a matched anchor
variant. Exact settings and selection lineage are in the
`price_model_*_ida_sgd_mae_spreads.yaml` configs and manifests.

| Auction | Early-selected model | Later anchor | Products | MAE EUR/MWh | WAPE |
| --- | --- | --- | ---: | ---: | ---: |
| IDA2 | Elastic Net | IDA1 | 4,704 | 9.927 | 4.635% |
| IDA2 | Elastic Net | DA | 4,704 | 11.559 | 5.396% |
| IDA3 | Stochastic absolute-error Elastic Net | IDA2 | 2,400 | 14.945 | 6.988% |
| IDA3 | Stochastic absolute-error Elastic Net | DA | 2,400 | 15.055 | 7.039% |

The prior full-history ridge with the same IDA3 latest anchor scored 14.964
EUR/MWh and 6.997% WAPE on identical timestamps and actual prices. A paired
5,000-resample independent-day bootstrap with seed 173 gives -0.161 to +0.194
EUR/MWh for the MAE improvement of the stochastic model, so the small observed
gain is not conclusive. The bootstrap excludes serial dependence and selection
uncertainty. All 28 early fits and four later fits completed without recorded
failures or convergence warnings. Both periods were previously examined,
publication clocks are partly assumed and missing days are excluded; this is
exploratory evidence, not an untouched prospective validation.
