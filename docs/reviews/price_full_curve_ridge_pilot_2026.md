# Full-curve price ridge pilot

Run: `outputs/price_all_auctions_curve_ridge_pilot_final`. The report contains
the pinned prepared-manifest hash, features, inner metrics, exclusions and
October metrics. Saved models and October predictions are in the same directory.
The script is `scripts/pilot_intraday_curve_ridge.py`; its `--run` option fits
models and requires a new output directory.

Each auction is modeled separately, using one daily sample and a fixed-length
vector of 96 outputs (48 for IDA3). The outputs are prices minus the configured
anchor (zero for DA; the latest available cleared auction for IDAs). Daily
inputs consist of the previous 1-, 2- and 7-day hourly own-auction curves,
calendar and cancellation flags, and complete quarter-hour vectors of the
available cleared-auction prices and 24-, 48- and 168-hour own-price lags.
Additional features are quarter-hour vectors for NORD load/generation/solar/wind
forecasts, previous-day and previous-week PSV gas, and previous-day French
nuclear, hydro, France-Italy commercial exchange and balance quantities.
Swiss quantities are omitted because the frozen snapshot lacks them for nearly
all October days. The past quantities follow the study's assumed availability;
no same-day realized quantity or target-derived spread is used.

Ridge minimizes squared error plus L2 penalty; multi-output ridge learns a
separate set of coefficients for each product, with shared inputs and one
regularization strength. It is not a recurrent or joint probabilistic model.
Each variant independently selects alpha from 100, 1000 and 10000 using the
existing chronological inner boundary, scored with MAE, then refits using all
eligible annual days. Inputs are standardized; missing input cells use medians
fitted on training data plus missingness indicators. Labels and anchors are
never imputed. Labels must have been published by the first October auction gate.
No November observations are used. October is a development evaluation because
it has already been inspected; it is not a final untouched test.

| Auction | Days | Core MAE | Core WAPE | With external MAE | With external WAPE | Reference MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | 30 | 18.24 | 7.74% | 17.92 | 7.60% | 21.36 |
| IDA1 | 29 | 8.92 | 3.78% | 8.96 | 3.80% | 8.82 |
| IDA2 | 30 | 10.17 | 4.27% | 10.32 | 4.34% | 10.20 |
| IDA3 | 31 | 16.35 | 6.72% | 16.38 | 6.73% | 14.59 |

MAE values are EUR/MWh. Models and reference are scored on identical rows per
auction and variant. DA's reference is the existing previous-price reference;
the IDA reference is the latest available cleared price. The fixed 96-product
format excludes October 25 for DA, IDA1 and IDA2, because the clock change
produces 100 products. IDA1 October 8 is absent because the GME snapshot records
a cancellation. IDA3 covers the afternoon, so retains October 25. The frozen
matrices predate the DST curve repair; the ridge input imputer handles the two
missing historical curve values on October 26 and 27, so those predictions
should not be interpreted as validating the DST correction.

The saved pointwise IDA2 model has MAE 10.10 on 2,688 overlapping products,
compared with 10.14 for this core curve pilot; its two missing October days and
the pilot's October 25 exclusion make its full-month score incomparable without
matching. For IDA3 both cover 1,488 products: the pointwise model has MAE 14.97
against 16.35 for the core curve pilot. DA and IDA1 selected pointwise models
currently have Swiss features on only one October day, too little for comparison.
