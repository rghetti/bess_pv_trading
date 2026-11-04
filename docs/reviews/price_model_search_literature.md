# Expanded electricity-price model search

The initial January/August study used one pooled ridge specification with a
fixed penalty and was a feature ablation, not an exhaustive model search.
The open benchmark by [Lago et al. (2021)](https://doi.org/10.1016/j.apenergy.2021.116983)
motivates a seasonal-naive reference, sparse autoregressive models, neural
networks, different estimation windows, model combinations and long test
periods. Its LEAR uses whole daily lagged price and exogenous curves; a pooled
Lasso on this library's existing interval-level features is a useful candidate
but is not a reproduction of LEAR. The benchmark also documents why individual
percentage errors are unstable near zero electricity prices.

For this dataset, search ridge, Lasso, elastic net, feed-forward neural nets,
and gradient-boosted trees with several depths, learning rates and regularization
settings, along with persistence and forest references. Vary training windows
where coverage permits and compare inputs in groups. In response to the product
resolution split, train and validate hourly day-ahead models using native hourly
prices, and quarter-hour models using only native quarter-hour prices. The
current initial search selects on January–June 2025 for hourly products and
February–April 2026 for quarter-hour products; later periods remain held out.
Their different forecast eras make the two resolution models separate tasks,
not directly comparable estimates of a treatment effect. Intraday and settlement
models use their own native-resolution data and folds in
`configs/research/price_model_search_intraday.yaml`. More history may improve
the hourly task, but cannot provide quarter-hour labels before those products.
The available 2026 span is shorter than the year-long test recommended in the
benchmark and cannot establish year-round generalization.

Report MAE and pooled WAPE (sum absolute errors divided by sum absolute
observations), the error ratio to a seasonal-naive forecast on identical
intervals, RMSE, per-month results and forecast coverage. Negative prices are
preserved; WAPE is undefined when all observed prices are zero. Probabilistic
scores and trading outcomes are separate from point-model selection. Compare
models on shared target days, disclose training-sample differences, and account
for multiple comparisons before claiming statistical superiority. The user's
requested previous-day realized features are retrospective; their exact
historical publication times are not a prerequisite at this stage.

`scripts/search_energy_price_models.py` prepares native-resolution data and
shared complete validation days for the candidate estimators. Its current
configurations are first validation searches, not a complete study over
every auction, training window, feature set, hyperparameter or held-out period.
These estimators are not yet integrated as portable MPC forecast artifacts.
Do not label them LEAR or claim a selected model or improved score until the
chronological search and held-out evaluation have been completed.

`configs/research/price_model_holdout.yaml` selects the three lowest pooled
validation-MAE combinations per resolution plus a ridge reference for each
feature set, using the completed validation manifest and its checksum. It
refits only after the selection period and evaluates hourly products from
2 July through 30 September 2025 and quarter-hour products from 2 May through
31 August 2026. The intervening day ensures the last validation day has ended
before the model-selection cutoff. Matching source data, feature definitions,
lags, seed, candidate settings and native resolution are required. August 2026
was inspected in an earlier ridge feature study, so this is held out from this
particular candidate selection, not a globally untouched research test. Scores
remain retrospective and apply only to complete common delivery days.

`configs/research/price_model_search_cleared_auctions.yaml` expands the IDA
search to 27 parameterized candidates and six feature groups per auction.
Besides the same auction's 24, 48, 72, 168 and 336 hour price lags and adjacent
hours around the daily and weekly lags, the cleared
features include the DA price for IDA1, DA and IDA1 for IDA2, and DA, IDA1 and
IDA2 for IDA3. They include the same product and the earlier cleared product at
15- and 60-minute offsets. The offset describes the source delivery time
relative to the target; its original publication clock is retained. A target
cannot use itself or a later auction as an earlier-auction feature. Exact
publication clocks are configured assumptions and missing prices remain
missing. Hourly DA products expanded to quarter-hours are valid coarse inputs,
not independent quarter-hour training targets. Additional feature groups add
NORD fundamentals and the retrospective previous-day European context. All
feature groups are scored on shared complete days, with training counts reported.
Two groups fit the IDA-minus-cleared-DA spread and add the DA price back before
scoring; they otherwise use the same features as their corresponding direct
price models. This tests whether learning the cross-auction spread is easier
than learning the absolute price without treating the DA value as an IDA target.
The 4% WAPE target is an objective for evaluation, not a guaranteed result;
selection and subsequent held-out scoring remain separate.
IDA3 uses only same-product daily and weekly historical lags: an adjacent-hour
IDA3 lag can cross its half-day delivery window and point to a nonexistent
product, as a full validation-coverage check demonstrated. Its neighboring
features from the earlier full-day auctions remain available when published.
`configs/research/price_model_search_nearest_spread.yaml` additionally compares
IDA2-minus-IDA1 and IDA3-minus-IDA2 targets against direct-price and
IDA-minus-DA targets on the same native delivery products. The latest earlier
market is selected from a validated chronological ordering, and its configured
publication clock still governs whether the anchor is usable.

`configs/research/price_model_search_swiss_border.yaml` tests the previous-day
Swiss--Italy commercial schedule, directional intraday NTC and Italian-NORD/
Swiss spot spread, using only the current 2026 border archive. The spread is
shifted by 24 hours and is not the contemporaneous target spread. It is a
retrospective feature with an assumed publication clock; exact historical
release and revisions remain unverified. Baseline and border feature groups
are restricted to the same training rows and common complete validation days,
so shorter border coverage does not itself change their training sample. The
January--February 2026 training window is short and the March--April validation
is not an untouched full-year test.

`configs/research/price_model_search_ida_hourly_2024.yaml` prepares a separate
search on 60-minute 2024 IDA source exports. It starts on 14 June, the first
delivery date following the June 2024 European IDA go-live, and evaluates
October--December after an earlier training cutoff. Earlier records are not
silently treated as the same market regime. The source is an hourly export;
whether each exported value is an original traded hourly product or a
retrospective aggregation must be verified from its source documentation.
The configured later-market publication clocks are retrospective assumptions,
not evidence of 2024 gate or release times. Price-only and earlier-cleared
feature groups use identical training and evaluation support.

`configs/research/price_model_search_cleared_2025.yaml` declares an additional
native-quarter-hour IDA validation period in June--August 2025, with training
strictly before June. Its same-delivery day-ahead price is sourced from the
then-hourly DA export and repeated as a coarse input over four quarter-hours;
the IDA target remains the original quarter-hour product. This gives an
additional historical regime without treating 2024 hourly IDA exports as
quarter-hour training labels. It is still retrospective and uses the stated
publication-clock assumptions.

`configs/research/price_model_search_da_curves.yaml` adds the full 24-hour
shape of the same market's previous 1, 2 and 7 local delivery days. Within a
native quarter-hour task, each hourly input is the average of its four native
prices, only when all four products are present and released; the prediction
target remains native quarter-hourly. Hours missing or repeated at daylight
saving transitions are not synthesized, so the affected candidate dates are
excluded and all candidates share the resulting validation days. This is a
pooled model with explicit daily-curve inputs inspired by LEAR, not a literal
per-hour LEAR replication. The price archive retains its native products and
these features never use a price after its configured publication time.
The same search includes regularized linear candidates with lag and exogenous
features interacted with sine and cosine of delivery hour, allowing a signal's
effect to vary smoothly over the daily price shape. Feature scaling is fitted
inside each training split. This is a structured interaction basis, not an
unrestricted per-product model.

`scripts/evaluate_price_ensembles.py` evaluates simple equally weighted
forecast combinations after the IDA search completes. For each market, it
ranks candidates by point MAE on 1 February--15 March 2026, chooses the number
of top-ranked models on 16--31 March from 1, 3, 5 and 10, and scores the fixed
choice on 1--30 April. All candidates must have exactly aligned products,
actual prices and seasonal references. The source search used fixed models
trained before February; no model is retrained during this combination test.
The source validation period has already been inspected in development, so
April is a chronological later slice for this combination selection, not a
globally untouched research test. The reported model selection, later-period
predictions, daily error plots and source-manifest hash enable independent audit.

`configs/research/price_model_rolling.yaml` tests weekly refits of the best
three daily-curve candidates per native resolution plus ridge references. The
selection uses the completed earlier curve-search manifest. The test spans
July--September 2025 for native hourly prices and May--August 2026 for native
quarter-hour prices; these dates have been inspected by earlier studies and are
not globally untouched. At each weekly refit, the initial eligible training
set is supplemented only by earlier common-day rows whose labels were published
and whose products had ended by the first decision cutoff, with a one-local-day
gap. This conservative implementation does not add excluded common-day rows or
the day between the original training cutoff and the test period. Model weights
remain fixed within each weekly block, while input lags update as configured.
Each prediction file has a refit sidecar listing block dates and training counts.
`configs/research/price_model_rolling_ida.yaml` applies the same weekly rule to
the validation-selected IDA1--3 candidates with earlier-cleared-auction features
and May--August 2026 evaluation. It selects from the completed February--April
IDA search before each later-period refit and retains the half-day IDA3 lags.
