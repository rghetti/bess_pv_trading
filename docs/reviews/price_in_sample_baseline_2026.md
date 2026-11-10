# Preliminary in-sample 2026 price proxies

This standalone diagnostic fits one Ridge model for each of DA, IDA1, IDA2
and IDA3 on all complete native-quarter-hour training rows in the configured
2025–2026 interval. It scores the same fitted models on available 2026 rows.
The predictions and scores are deliberately **in sample**: training includes
future labels relative to many predicted delivery days. They are an optimistic
descriptive baseline, not a measurement of forward predictive performance.

The source is the immutable `outputs/price_feature_search_full` snapshot,
not the live DuckDB. Its selected input period ends on 1 September 2026; it
cannot supply September–December 2026. DA's native 15-minute history begins in
October 2025; quarter-hour IDA history begins in January 2025. Missing data and
incomplete base features remove rows and are not replaced by zero. The manifest
records available and scored row counts, dates, input checksums and features.

Each Ridge fit uses standardized mandatory own-price lags, past price curves,
time/calendar and holiday indicators, previous auction prices where available,
and 27 general holiday-hour/adjacent-day indicators. IDA models predict a spread
over the last cleared auction and use its actual clearing price as the anchor.
Thus an IDA estimate is conditional on earlier auction outcomes and cannot be
used unchanged as an IDA forecast at the DA gate. This baseline does not use the
optional France, Switzerland, gas or NORD forecast blocks. Ridge uses squared
error plus an L2 penalty with configured alpha 1000; this is a simple fixed
reference, not the winner of the unfinished full model search.

From the project root, run after reviewing the plan:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_price_in_sample_baseline
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_price_in_sample_baseline --run
```

The second command performs four serial single-thread CPU fits and writes one
model and Parquet file per auction, an auditable manifest, monthly error plots
and best/worst observed-day plots. It refuses to overwrite an existing output
directory. Change `output_directory` in the study config for an independent run.
MAE, RMSE and WAPE are pooled over scored 2026 products and are in-sample.

No simulator code is changed. The files are standalone gate-level predictions,
not yet a drop-in multi-gate forecast provider for a continuous battery replay.
In particular, they must not be presented as if their models were fitted before
each decision. The separate realtime-MPC checkout has an existing forecast
provider for its own fitted artifact format; these search-model joblib files
are not that format. Monthly expanding-window forecasts can be produced as a
separate follow-on study when appropriate.
