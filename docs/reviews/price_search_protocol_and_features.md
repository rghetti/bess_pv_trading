# Price search: exact features, training and evaluation

This describes `configs/research/price_feature_selection.yaml` and the prepared
`outputs/price_feature_search_full` search. The running search evaluates point
prices in the Italian NORD zone. One row corresponds to an auction and one
native delivery product. A feature's presence in the matrix does not imply it
was chosen by a particular candidate: each candidate receives the mandatory
price/calendar features and one selected subset of the external blocks below.

**Historical IDA warning:** the pinned input snapshot labels 2024 data as
`ida1_prices`, `ida2_prices` and `ida3_prices`, including January through
12 June. Pan-European IDAs in Italy began on 13 June 2024. The provenance and
auction identity of the earlier records therefore need independent verification;
the configured modern IDA gate and publication assumptions must not be treated
as established for those records. The old hourly IDA fits and previews are
exploratory until this issue is resolved. They used native 60-minute records,
not aggregated 15-minute target labels.

The same snapshot has hourly DA throughout 2024 and January–September 2025;
its IDA observations switch from hourly to quarter-hourly on 1 January 2025,
while DA switches on 1 October 2025. Thus January–September 2025 has hourly DA
and quarter-hourly IDA. For 15-minute IDA inputs the hourly DA clearing price
is carried across its four covered quarters; IDA target prices remain native.

## When and what is predicted

The decision clocks below are study assumptions, in Europe/Rome local time;
when recorded publication times are missing, the listed clearing clocks are
also assumptions, not historical evidence of availability. D is the delivery
day. IDA3 covers the afternoon products only.

| Auction | Decision | Assumed clearing if missing | Same-day earlier auctions used |
| --- | --- | --- | --- |
| DA | D-1 11:45 | D-1 12:57 | none |
| IDA1 | D-1 14:45 | D-1 15:30 | DA |
| IDA2 | D-1 21:45 | D-1 22:30 | DA, IDA1 |
| IDA3 | D 09:45 | D 10:30 | DA, IDA1, IDA2 |

DA predicts its absolute price. Each IDA model predicts the difference between
its target price and the latest available earlier auction's price for that
delivery product. The anchor is added back before scoring and plotting. These
are separate models for hourly and quarter-hourly native product regimes.

## Mandatory price and calendar features

For each row, the following features are included before optional external
blocks. Prices are in EUR/MWh. A price lookup is usable only when its declared
publication time is no later than the row's decision clock.

1. **Own-product elapsed-hour lags:** the same auction's price for products
   starting 24, 23, 25, 48, 47, 49, 72, 168, 167, 169 and 336 hours before
   the target product. IDA3 uses only 24, 48, 72, 168 and 336 hours because
   it does not cover the overnight hours. If a requested lag is unavailable,
   the most recent published price at the chosen local clock slot is used,
   within the configured 336-hour search window. Daily multiples can use the
   target's local clock slot to handle daylight-saving changes. Unresolved
   values are missing, not filled with zero.
2. **Time of day:** sine and cosine of the target product's local clock time.
3. **Day of week:** sine and cosine of weekday and a binary weekend indicator.
4. **Time of year:** sine and cosine of local day of year.
5. **Italian public holiday:** binary flag for 1 and 6 January, 25 April,
   1 May, 2 June, 15 August, 1 November, 8, 25 and 26 December, Easter
   Sunday and Easter Monday. Weekends have their own separate flag.
6. **Own-auction price curves:** the mean price for each local clock hour on
   delivery days D-1, D-2 and D-7. For 15-minute prices this is the mean of
   the four quarter-hours. Models for DA, IDA1 and IDA2 receive 24 hourly
   values per day; IDA3 receives the 12 values from 12:00 to 23:00. An hour
   with a nonstandard number of products or an unpublished value is missing.
7. **Earlier cleared auction prices:** IDA1 uses DA; IDA2 uses DA and IDA1;
   IDA3 uses DA, IDA1 and IDA2. For each earlier auction the model receives
   its price for the target delivery product plus the prices at offsets -4,
   -2, -1, +1, +2 and +4 native products. On a 15-minute grid those are
   -60, -30, -15, +15, +30 and +60 minutes. On an hourly grid they are hours.
   A neighbor crossing the local-day boundary uses the target product instead.
8. **Past prices of other auctions:** DA uses IDA1 and IDA2; IDA1 uses DA
   and IDA2; IDA2 uses DA and IDA1; IDA3 uses DA, IDA1 and IDA2. For each,
   the target-time-minus-24h and minus-48h price is supplied if published.

Earlier-auction features are usable at the corresponding IDA gate. They do
not make an actual DA clearing price available at the earlier DA decision.

## Optional external features: every configured column

Previous-day forecasts and lagged observations use the study's explicit
availability assumption; the catalog does not establish historical issue
times for all rows. Lagged values below refer to the same delivery-time slot
unless the field is daily or weekly. Source intervals are carried across their
own interval, then quarter-hours are averaged for an hourly target only when
all constituent quarters are present.

### NORD forecasts (`nord_forecasts`)

- `nord_load_forecast`: ENTSO-E NORD demand forecast for the target interval.
- `nord_generation_forecast`: ENTSO-E NORD generation forecast for that interval.
- `nord_solar_forecast`: ENTSO-E NORD solar-generation forecast.
- `nord_wind_forecast`: ENTSO-E NORD onshore-wind-generation forecast.

These are zone-wide series, not the small farms' weather or PV forecasts.

### Gas (`gas`)

- `psv_previous_day`: GME PSV Italian gas daily index for D-1.
- `psv_previous_week`: the same daily gas index for D-7.

### France (`france`)

- `fr_nuclear_previous_day`: RTE French nuclear generation, 24 hours earlier.
- `fr_hydro_previous_day`: RTE French hydro generation, 24 hours earlier.
- `fr_italy_commercial_previous_day`: RTE reported France–Italy commercial
  exchange, 24 hours earlier, retaining the source sign convention.
- `fr_balance_previous_day`: the selected RTE electricity-balance dataset's
  `value` field, 24 hours earlier; the search does not derive this quantity.

### Switzerland (`switzerland`)

- `ch_generation_previous_day`: Swissgrid Swiss generation, 24 hours earlier.
- `ch_consumption_previous_day`: Swissgrid Swiss consumption, 24 hours earlier.
- `ch_it_physical_previous_day`: Swiss net physical exports to Italy,
  24 hours earlier, using the catalog's net-export convention.
- `ch_fr_physical_previous_day`: Swiss net physical exports to France,
  24 hours earlier.
- `ch_reservoir_previous_week`: the BFE total Swiss reservoir-energy level,
  aligned with an explicit 168-hour lag; this is a state in GWh, not production.

### Same-delivery-day observed data (`realized_flows`)

These seven features are **retrospective**; they can use values realized after
an auction decision. They are not in the proposed previous-day preview.

- `ch_it_physical_same_day`: Swiss net physical exports to Italy.
- `ch_fr_physical_same_day`: Swiss net physical exports to France.
- `ch_generation_same_day`: Swiss generation.
- `ch_consumption_same_day`: Swiss consumption.
- `fr_physical_all_borders_same_day`: RTE physical exchanges across French borders.
- `fr_it_commercial_same_day`: RTE France–Italy commercial exchange.
- `fr_ch_commercial_same_day`: RTE France–Switzerland commercial exchange.

### Swiss commercial data (`swiss_commercial`)

These eight features are also **retrospective**, with source coverage currently
limited to 2026. They are excluded from the proposed previous-day preview.

- `ch_it_commercial_same_day`: commercial schedule, Switzerland–Italy border.
- `ch_fr_commercial_same_day`: commercial schedule, Switzerland–France border.
- `it_ch_da_spread_same_day`: Italian NORD minus Swiss DA price spread. For a
  DA prediction it directly contains the target Italian DA price: outcome leakage.
- `fr_ch_da_spread_same_day`: France–Switzerland DA price spread.
- `ch_it_ntc_same_day`: NTC Switzerland to Italy.
- `it_ch_ntc_same_day`: NTC Italy to Switzerland.
- `ch_fr_ntc_same_day`: NTC Switzerland to France.
- `fr_ch_ntc_same_day`: NTC France to Switzerland.

The full-history cohort evaluates 32 subsets of the first five blocks on the
same complete-case rows. A separate 2026 cohort evaluates 64 subsets of all
six blocks on its own common rows. Missing source fields are not silently
zero-filled. A complete-case cohort can discard a row even when a particular
candidate does not use the missing block, so subsets are compared on matched
observations. The two cohorts need not contain the same dates.

## Model families and searched hyperparameters

Each feature subset receives 20 reproducibly sampled settings for each family;
the distributions below are sampled, not exhaustively enumerated. Linear and
MLP input scaling is fitted using training rows only; MLP also standardizes its
training target. Trees use one CPU thread per fit.

| Family | Training objective | Searched settings |
| --- | --- | --- |
| Ridge | squared error + L2 | `alpha` 0.01–10,000, log-uniform |
| Elastic Net | squared error + L1/L2 | `alpha` 0.001–10 log-uniform; L1 ratio 0.1, 0.3, 0.5, 0.7, 0.9, 0.99; max iterations 10,000 |
| Random forest | squared error | 200/400/600 trees; minimum leaf 2/5/10/20/40; maximum depth unlimited/8/15/25; features 0.4/0.7/1.0 |
| Extra Trees | squared or absolute error | same tree counts, leaf sizes, depths and feature fractions as random forest |
| Histogram gradient boosting | absolute error | 150/300/600 iterations; 7/15/31/63 leaves; depth unlimited/3/5/8; learning rate 0.015–0.15 log-uniform; minimum leaf 20/50/100; L2 0/0.1/1/10/50 |
| LightGBM | absolute error | 250/500/900/1200 trees; 7/15/31/63 leaves; depth unlimited/4/6/8; learning rate 0.01–0.12 log-uniform; minimum child 20/50/100/150; subsample 0.65–1; feature fraction 0.55–1; L1 0/0.1/1/10; L2 0.1/1/10/100 |
| CatBoost | absolute error | 300/600/1000 iterations; depth 4/5/6/7/8; learning rate 0.01–0.12 log-uniform; L2 0.5–50 log-uniform |
| XGBoost | squared or absolute error | 250/500/900/1200 trees; depth 2/3/4/6/8; learning rate 0.01–0.12 log-uniform; minimum child weight 1/5/10/30; subsample 0.6–1; feature fraction 0.55–1; L1 0/0.1/1/10; L2 0.1/1/10/100 |
| MLP | squared error on standardized target + L2 | hidden layers (64), (128), (64,64), (128,64), (128,128); ReLU/tanh; L2 0.0001–1 log-uniform; learning rate 0.0002–0.01 log-uniform; batch 128/256; 300/600 maximum iterations; no early stopping |

## Training, refitting, embargo and scoring

Training uses **all eligible rows** from the configured start until the
training boundary, not a rolling last-N-days window. Its latest 20% of
eligible days (at least seven days) is the inner validation tail for choosing
hyperparameters. The full main search selects one inner winner for every
feature subset, refits it on all eligible training rows and evaluates the
separate development-validation period. It freezes winners by product, cohort
and availability class using validation WAPE, then refits each frozen winner
on eligible training plus validation history before scoring final test.

The **one-day embargo** removes one local delivery day immediately before each
outer validation/test boundary from the corresponding training labels. For
example, with validation starting 1 February, training includes eligible days
through 30 January and excludes 31 January. It is a precautionary gap between
fitting labels and evaluation, not a rolling-window length and not a rule that
prohibits using a legitimately available previous-day price as an input. The
inner train/tail split has no additional explicit one-day embargo. Publication
checks can further remove labels that were unavailable at evaluation time.

| Group | Training period before embargo | Development validation | Final test |
| --- | --- | --- | --- |
| DA hourly | 2024 | Jan–Jun 2025 | Jul–Sep 2025 |
| IDAs hourly | Jan–Jun 2024 | Jul–Sep 2024 | Oct–Dec 2024 |
| DA quarter-hour | Oct 2025–Jan 2026 | Feb–Apr 2026 | May–Aug 2026 |
| IDAs quarter-hour | Jan 2025–Jan 2026 | Feb–Apr 2026 | May–Aug 2026 |

The preview takes one best *saved inner-tail* candidate per family and product,
refits those candidates on all eligible training rows and chooses among them
on development validation. This is cheaper than the full search's one-refit-
per-subset protocol and need not find its final winner. Selecting and plotting
on validation makes the result exploratory; test is not touched. WAPE is
`100 * sum(abs(prediction - actual)) / sum(abs(actual))`. MAE is in EUR/MWh.
The reference is the latest cleared same-product price for IDAs and the first
own-price lag for DA; relative MAE divides model MAE by reference MAE.
