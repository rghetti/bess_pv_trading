# IDA training-window search and later evaluation

The early search compared 90-, 180- and 365-day windows with full history for
ridge, Lasso, Elastic Net and two Extra Trees settings. It used price history,
previously cleared-auction prices and direct-price or spread-to-DA targets.
Candidates were ranked by February–April 2026 MAE. The top two per market and
full-history counterparts of selected windowed models were refitted weekly for
available September–October products. Both periods have been inspected in
related work, so the later comparison is exploratory. Publication times are
partly assumed. Configurations, source snapshots and missing-day details are
recorded in the two `price_model_*_ida_price_only_train_windows.yaml` configs
and their manifests.

| Market | Model and training window | Products | MAE EUR/MWh | WAPE |
| --- | --- | ---: | ---: | ---: |
| IDA1 | Elastic Net, full history | 4,800 | 8.781 | 4.116% |
| IDA1 | Elastic Net, 365 days | 4,800 | 8.827 | 4.137% |
| IDA2 | Lasso, 365 days | 4,704 | 10.059 | 4.696% |
| IDA2 | Lasso, full history | 4,704 | 10.127 | 4.728% |
| IDA2 | Extra Trees, full history | 4,704 | 10.319 | 4.817% |
| IDA3 | Ridge, spread to DA, full history | 2,400 | 14.964 | 6.996% |
| IDA3 | Ridge, direct price, full history | 2,400 | 14.971 | 7.000% |

The matched full/365-day forecasts have identical timestamps and actual prices
within each auction. A paired independent-delivery-day bootstrap with 5,000
resamples and seed 173 gives 95% percentile intervals for MAE gain from the
365-day window of -0.115 to +0.021 EUR/MWh for IDA1 and +0.013 to +0.124
EUR/MWh for IDA2. These intervals omit dependence between days and model/window
selection uncertainty; they do not establish a reliable general improvement.
All early and later fits completed without recorded failures, and the listed
later fits have no recorded convergence warnings. Evaluation coverage and
lag fallbacks are described in `price_price_only_sep_oct_2026.md`.
