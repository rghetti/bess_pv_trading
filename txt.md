# Task: Implement a causal, literature-informed benchmark for NORD DA prices and IDA1 spreads

## 1. Objective and working appråoach

Extend the existing `bess_pv_trading_features` repository with a reproducible forecasting benchmark for:

1. Day-ahead electricity prices, predicted before the DA submission cutoff.
2. The IDA1-minus-DA spread, predicted before the IDA1 submission cutoff.
3. The same spread anticipated before DA submission, for eventual use by the forward-looking trading optimizer.
4. Direct IDA1 price forecasting, as a challenger to the spread-based approach.

Implement a range of statistical and neural models, evaluate them chronologically, and produce forecast artifacts suitable for subsequent integration into the PV+BESS stochastic MILP.

The expected repository location is:

`/home/rghetti/bess_pv_trading_features`

Verify this and follow the repository’s `AGENTS.md` and existing conventions.

**Inspect the implementation first. Do not follow the design suggestions below mechanically.**

Understand the existing data catalog, price-forecasting workflows, feature builders, temporal availability rules, model interfaces, configurations, tests and reporting before deciding how to extend them.

Reuse existing functionality, especially the current ridge, lasso, elastic-net, Huber, chronological evaluation and scenario-generation components where applicable.

The mathematical requirements, information boundaries and accounting units below are mandatory. Proposed class structures, model settings and implementation order are guidance: choose cleaner alternatives when justified by the actual codebase.

Do not change the physical MILP, market constraints or settlement logic merely to implement forecasting experiments.

---

## 2. Available dataset

The reported study window is 1 January 2025–31 August 2026.

The DuckDB contains:

- NORD DA prices for all 608 delivery days.
- IDA1 prices for 596 delivery days.
- December 2024 price history for initial lag features.
- Daily Italian gas prices.
- ENTSO-E NORD load, solar, wind and total-generation forecasts.
- Farm weather forecasts.
- Hourly ECMWF forecast runs for nine northern Italian locations, including Aosta.
- Regional weather coverage of 1,211 out of 1,216 planned runs per location, with five documented provider gaps.
- Source vintages, native price intervals, DST handling and availability provenance.

Treat these as reported coverage figures to verify against the catalog, not assumptions to hard-code.

The current DA/IDA1 feature package assumes next-day gas prices are available before both auctions. That assumption requires a separate strict-availability treatment.

Start with a concise audit of the actual schemas, available columns, units, timestamps, missingness and coverage.

Do not regenerate the database, overwrite raw data, or initiate new provider/API downloads for this task.

Do not treat multiple weather locations or successive forecast runs as probabilistic ensemble members unless the source explicitly identifies them as such.

Although pooled models can use interval-level rows, the dataset contains only 608 DA delivery-day outcomes. Quarter-hour rows and weather locations are not independent additional years of price history.

---

## 3. Inspect the existing architecture before implementation

Identify:

- The authoritative DuckDB catalog and existing data readers.
- Native-product mappings and delivery-interval representations.
- Auction decision cutoffs, publication events and submission buffers.
- Existing feature selection by forecast vintage and availability time.
- Existing statistical forecasting models and their configuration.
- Current training, validation, refitting and prediction persistence.
- Existing scenario-generation and forecast-provider interfaces.
- Output conventions, experiment manifests and independent validators.
- Existing tests and dependency-management conventions.
- Any current LaTeX or methods documentation.

Preserve unrelated working-tree changes. Avoid broad refactoring.

After inspection, choose the smallest coherent implementation strategy and proceed. Resolve ordinary engineering choices autonomously.

Where missing provenance prevents a scientifically correct experiment, implement explicit validation/status handling and document the limitation rather than silently making assumptions.

---

## 4. Forecasting targets and information sets

Use EUR/MWh throughout.

For delivery day d and interval t, define:

`spread[d,t] = ida1_price[d,t] - da_price[d,t]`

Keep this sign convention consistent in training targets, outputs, metrics and future optimizer integration.

### Task A: `da_pre_da`

Predict the DA price curve using only information available before the buffered DA submission cutoff.

Permitted inputs include eligible historical prices, calendar information, published gas observations and eligible forecasts for the delivery horizon.

Do not use the realized DA curve for the target delivery day, later weather runs or later ENTSO-E publications.

### Task B: `ida1_spread_pre_ida1`

Predict the IDA1-minus-DA spread before the buffered IDA1 submission cutoff.

The published DA curve for the target delivery day may be used once it is actually available.

Reconstruct:

`predicted_ida1_price = observed_da_price + predicted_spread`

Preserve the forecast statistic: mean plus observed DA yields a mean IDA1 forecast; a quantile plus observed DA yields the corresponding IDA1 quantile.

The reconstructed IDA1 error and spread error are mathematically identical for this task. Do not present them as independent evidence.

### Task C: `ida1_spread_pre_da`

Predict the IDA1-minus-DA spread using only the pre-DA information set.

This task is needed because the forward-looking optimizer anticipates later opportunities before DA clears.

Do not use:

- The target day’s realized DA prices.
- Forecast revisions published after the DA cutoff.
- Information available only at IDA1 submission.

A separate model is the preferred initial implementation.

Predicted DA prices may be used, but historical training features must come from chronological out-of-fold DA predictions, not in-sample fitted values.

### Task D: `ida1_direct_pre_ida1`

Predict the IDA1 price directly, using the same eligible inputs as Task B, including the published DA curve.

Compare this with spread prediction. Do not assume either representation must be better.

### Implementation requirement

Represent the task, forecast origin and information cutoff explicitly. Avoid one ambiguous model that silently changes its information set depending on the caller.

Reuse the project’s market calendar. Do not hard-code nominal auction times where authoritative calendar and publication logic already exist.

---

## 5. Availability and leakage controls

Implement or extend an auditable as-of feature assembly process.

Track the existing equivalents of:

- Delivery interval.
- Forecast issue time.
- Publication or effective availability time.
- Retrieval time.
- Forecast decision cutoff.
- Source vintage and provenance.

Retrieval time is not proof of historical availability.

### Gas

Identify the exact gas series in the database and its publication convention.

Implement IF POSSIBLE:

- A strict mode using the latest observation demonstrably available at the decision cutoff.
- A separately labelled sensitivity preserving the existing assumed next-day-gas availability.

Include observation age where useful.

Do not silently apply the publication timing of one gas index to a different gas product.

### ENTSO-E forecasts


The name “day-ahead forecast” does not establish availability before DA submission. Total-generation forecasts may represent scheduled generation and require particular care.

Use the archived version available at the cutoff, not a subsequently revised historical value.

Distinguish:

- Verified available.
- Verified unavailable.
- Availability uncertain.

FOR THE FIRST EXPERIMENTS, TREAT ALL DAY AHEAD FEATURES AND FORECASTS AS 
AVAILABLE AT DA AND IDA1 TIME, WE WILL DEAL LATER WITH ACTUAL AVAILABILITY FOR NOW IGNORE IT, IN THIS FEATURE AND ALL OTHERS WHERE THE SAME PROBLEM OCCURS.

### Weather

Choose the latest eligible run that covers the requested delivery horizon.

For documented gaps, a preceding eligible run may be used if it still covers the horizon. Preserve forecast age and missingness indicators.

Do not fill gaps using a later run.

Differences between DA-time and IDA1-time forecasts must be constructed from the respective eligible vintages.

When no new run is available, the revision is zero and an indicator should identify the absence of an update.

### Training labels

INSTEAD FOR THE PRICES AND WEATHER FORECASTS WE HAVE, CAREFULLY LOOK AT THE TIME AVAILABILITY OF THE FORECASTS, USUALLY WITH OUR LAG LOGIC.

AT THE END OF YOUR IMPLEMENTAITON AND EXPERIMENTS, PRESENT TO ME THE EXZACT ASSUMTPIONS THE RESULTS RELY ON.
A label becomes usable when it is available, not automatically at the end of its delivery period.

DA and IDA1 prices can be published before delivery. Apply the correct publication logic when constructing the training set.

All preprocessing, calibration, feature selection, ensemble weights and residual models must use only eligible historical information.

---

## 6. Native resolution, DST and target construction

The study spans the DA transition to 15-minute products for delivery from 1 October 2025.

Preserve the distinction between:

- Hourly DA prices before the transition.
- Native quarter-hour DA prices afterward.
- Native IDA1 products throughout the available dataset.

Verify the stored product definitions.

### Pre-transition spread

For a quarter-hour IDA1 product inside an hourly DA product:

`spread[d,h,q] = ida1_price[d,h,q] - da_price[d,h]`

Repeat the applicable hourly DA price across its delivery quarters.

Do not linearly interpolate hourly DA prices into artificial quarter-hour prices.

Do not count the four repeated DA labels as four independent hourly auction observations.

### Evaluation views

Support:

1. A common hourly representation for architecture comparisons across the full sample.
2. Native-product evaluation for operationally relevant forecasts.

Use duration-consistent hourly aggregation.

Keep scores on the common hourly representation distinct from native-product scores.

### Optional hierarchical DA model

If it fits the existing interfaces, implement an hourly-level plus within-hour-shape variant:

`quarter_hour_price = hourly_mean_price + within_hour_deviation`

Within-hour deviations should sum to zero within the corresponding hour.

Train the hourly component using the longer history. Train native within-hour deviations using actual quarter-hour DA observations.

Compare this with a post-transition-only quarter-hour model. Do not assume the hierarchical version is superior.

### DST

Use UTC timestamps together with the relevant local delivery labels and timezone.

Preserve 23/25-hour and 92/100-quarter-hour days, including repeated clock hours.

Sequence models must use a documented masking or calendar adapter. Do not silently drop, merge or fabricate delivery products to force every day into a 24/96-element array.

---

## 7. Feature packages

Implement configurable feature packages so that the contribution of each data source can be measured.

Avoid giving every model all available weather columns by default.

### F0: Price and calendar

Examples:

- Eligible previous-day and previous-week price/spread profiles.
- Additional lags such as 2 and 3 days where justified.
- Same-period lags and compact daily-profile summaries.
- Weekday, weekend, holidays and seasonality.
- Delivery time and within-hour position.
- Product-resolution regime and relevant DST indicators.

For LEAR-style models, include full historical price-profile information where appropriate rather than only the same delivery-period lag.

### F1: Aggregate fundamentals

Add eligible forecasts of:

- NORD load.
- Solar generation.
- Wind generation.
- Total generation, only where availability supports its use.

Derive residual load consistently, for example:

`forecast_load - forecast_solar - forecast_wind`

Avoid double-counting renewable generation or creating an unsupported balance identity.

Include ramps and daily summaries where useful.

### F2: Gas

Add:

- Latest available gas observation.
- Eligible changes or rolling summaries.
- Observation age.

Keep strict and assumed-availability experiments separately identifiable.

### F3: Regional weather

Use only variables actually present in the archive.

Compare:

- Simple regional summaries.
- Location-specific predictors, including Aosta.
- Training-only dimensionality reduction where useful.

Candidate summaries include spatial means, ranges and dispersion.

Equal geographic weighting is a baseline, not a claim about electricity-demand or installed-capacity weights. Do not invent regional capacity weights.

Spatial dispersion is not automatically probabilistic forecast uncertainty.

Farm-local weather can be an additional ablation, but should not be treated as a substitute for regional market fundamentals.

### F4: Forecast revisions and IDA1 context

For Task B and Task D, include:

- Eligible DA-to-IDA1 forecast revisions.
- Forecast ages and update indicators.
- Published DA level and curve shape.
- Neighboring DA periods, ramps and deviation from daily mean.
- Historical spreads.

Add an optional “DA surprise”:

`observed_da_price - pre_da_predicted_da_price`

Generate historical surprise features from genuine chronological out-of-fold DA forecasts.

Do not use retrospectively refitted DA predictions.

### Feature engineering rules

Fit scalers, imputers, transformations, PCA and feature-selection procedures inside the chronological training workflow.

Do not clip negative prices or spreads to zero.

Do not apply PV-specific plateau or nighttime-cleaning rules to prices.

---

## 8. Model families

Use a shared experiment interface while allowing model-specific representations.

Implement the following families. Essential models should run with the core environment where feasible; heavier neural and foundation models may use explicit optional dependencies.

Never silently substitute one model for another.

### M0: Simple baselines

DA:

- Previous-day persistence.
- Previous-week persistence.
- Similar-day/calendar persistence.

Spread:

- Zero spread.
- Trailing mean or median by delivery period and day type.
- Previous-week spread persistence.

Zero spread corresponds to using observed DA as the IDA1 forecast in the post-DA task.

These baselines are mandatory.

### M1: Regularized linear models and LEAR-style models

Reuse existing ridge, lasso, elastic-net and Huber implementations.

Add or formalize a LEAR-style feature specification with price-profile lags, calendar effects and eligible exogenous predictors.

Compare:

- Delivery-slot-specific models.
- Pooled models with time-of-day interactions or other parameter-sharing mechanisms.

Candidate training windows include 56, 84, 168 and 365 days, plus expanding history, subject to available data.

Use a restrained search rather than all combinations.

Distinguish an exact literature reproduction from an extended LEAR-inspired model.

### M2: Generalized additive model

Implement a penalized GAM as an interpretable nonlinear statistical benchmark.

Candidate smooth terms:

- Delivery time.
- Residual load.
- Gas.
- Temperature.
- Selected price or spread lags.

Use only a few justified interactions.

Start with a point-forecast model. Distributional GAM extensions are secondary.

Do not impose monotonicity merely because a relationship sounds economically plausible.

### M3: LightGBM and CatBoost

Implement pooled tabular models for each relevant task.

Support:

- Conditional-mean training.
- Median training.
- Quantile regression.

Suggested initial complexity is modest: shallow trees, regularization and substantial minimum leaf sizes.

Tune with chronological validation and early stopping that never uses the final test set.

For IDA1, compare direct-price and spread targets using the same information set.

Use the actual installed library capabilities rather than assuming particular APIs exist.

### M4: Compact MLP and linear–nonlinear hybrid

Implement a compact feedforward neural baseline.

Suggested initial search:

- Two or three hidden layers.
- Widths around 32–128.
- Weight decay.
- Chronological early stopping.
- Reproducible seeds.

Start with a pooled interval-level representation. Add daily multi-output forecasting if the calendar adapter and sample size justify it.

Implement a hybrid:

`prediction = linear_component + neural_correction`

A jointly trained linear skip connection is acceptable.

For a separately trained residual corrector, use appropriate chronological out-of-fold base-model predictions.

### M5: Distributional neural networks

Implement compact conditional:

- Gaussian-output networks.
- Johnson-SU-output networks.

Predict valid distribution parameters and train with a numerically stable likelihood.

Use positive scale/concentration parameterizations, numerical safeguards and tests for finite losses and predictions.

Support a small ensemble of independently initialized networks.

Return the actual conditional mean, median and quantiles. A distribution’s location parameter is not necessarily its mean.

Evaluate predictions in original EUR/MWh units, with correct transformation handling.

These are marginal probabilistic models; do not describe them as joint trajectory generators without an additional dependence model.

### M6: NBEATSx

Implement an optional NBEATSx experiment using historical target sequences and cutoff-available future exogenous trajectories.

Suggested initial lookbacks are 7 and 14 days.

Use compact architectures and start on the hourly representation before adding native quarter-hour variants.

Handle DST and missing products explicitly.

Do not silently replace NBEATSx with ordinary N-BEATS or another sequence model while retaining the name.

### M7: TabPFN

Implement an optional TabPFN regression challenger using a documented, compatible package and checkpoint.

Compare bounded recent pooled context with delivery-slot-specific contexts where practical.

Control context size and computational cost.

Record package/checkpoint version and available training-data provenance.

Do not trigger large downloads or paid/API-backed inference implicitly. Missing weights, credentials or dependencies must produce a clear unavailable status.

Separate a retrospective methodological benchmark using a recently released model from a claim of historical deployability.

Use a separately identified postprocessing procedure when the selected implementation does not provide the desired predictive distribution directly.

### M8: Forecast combinations and probabilistic postprocessing

Implement:

- Equal-weight combinations of complementary point models.
- Regularized nonnegative combination weights learned from chronological out-of-sample predictions.
- Quantile regression averaging where appropriate.
- Rolling probabilistic recalibration/conformal-style adjustments where supported.

Isotonic distributional regression is an optional additional challenger if it can be implemented correctly with available dependencies.

Avoid very large ensembles of nearly identical models.

Keep ensemble training, calibration and final evaluation strictly separated.

Do not claim universal conditional coverage or protection against arbitrary distribution shifts.

---

## 9. Literature grounding

Use the literature motivating the preceding design to document model provenance:

- Lago et al. (2021): LEAR/DNN electricity-price forecasting benchmarks.
- Kath and Ziel (2018): elastic-net forecasting of quarter-hour electricity markets.
- Serinaldi (2011): generalized additive distributional modeling for electricity prices.
- Michalakopoulos et al.: recent boosted-tree comparisons for DA prices.
- Marcjasz et al. (2023): distributional neural networks.
- Olivares et al. (2023): NBEATSx.
- Lipiecki, Uniejewski and Weron (2024): probabilistic postprocessing.
- Brusaferri et al. (2025): online conformalized neural ensembles.
- Lipiecki and Weron (2026 preprint): foundation models, including TabPFN.
- El Mahtout and Ziel (2026 preprint): linear–neural hybrids and adaptive forecasting.

Verify bibliographic details before adding formal references.

Document which implementations reproduce a published specification and which are dataset-specific adaptations.

Do not invent benchmark results or assume published rankings transfer to NORD.

If a source is inaccessible, state the limitation and describe the implemented model directly rather than claiming faithful reproduction.

---

## 10. Forecast statistics and probabilistic outputs

Keep conditional means, medians and quantiles distinct.

The expected-profit optimizer generally requires conditional means or scenarios. A median model must not silently be presented as a mean forecaster.

For nonlinear target transformations, inverse-transforming a point prediction does not automatically recover the conditional mean. Handle back-transformation explicitly and test it.

An initial quantile grid can be:

`[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`

Use consistent quantiles across compared models.

Correct or constrain quantile crossing and report the procedure.

Do not infer extreme-tail accuracy from good central-interval coverage.

Store chronological out-of-sample residuals and, where available, PIT values. These will support later scenario generation.

Do not automatically sample periods, DA prices and spreads independently and label the result a validated joint scenario model.

---

## 11. Chronological evaluation protocol

Use the following primary experimental partition unless inspection reveals a documented reason to adapt it:

- January–December 2025: initial development/training history.
- January–March 2026: rolling validation and model selection.
- April–August 2026: locked final test.

December 2024 supplies initial price lags. Do not assume complete exogenous features exist for that month.

Use earlier rolling folds within development where needed.

### Refitting

Allow predeclared expanding or rolling refitting during validation and test.

Newly available labels can enter later fits. Hyperparameter selection, feature-package selection and adaptation rules must be frozen before final-test evaluation.

Support documented refit schedules appropriate to model cost. Compare retraining frequency separately when it materially affects results.

### Missing IDA1 days

Do not create zero-spread labels for missing IDA1 prices.

Distinguish actual absent/cancelled auctions from missing data where provenance allows.

Use common eligible evaluation cases for paired model comparisons and report exclusions.

### Avoid an uncontrolled experiment grid

Do not execute the full Cartesian product of models, feature packages, windows, seeds and availability assumptions.

A sensible staged design is:

1. Baselines, regularized linear models and boosted trees.
2. Compact neural, distributional and TabPFN challengers.
3. Additional architectures and combinations.
4. Focused feature-source and availability ablations.

Document tuning budgets and use comparable effort where practical.

---

## 12. Evaluation metrics

### Point forecasting

Report:

- MAE.
- RMSE.
- Bias.
- Daily curve losses.
- Scores by month, delivery period and product regime.
- Improvement over appropriate simple baselines.

Do not use MAPE for prices or spreads near zero.

For spread prediction, report skill relative to zero spread and a historical-spread baseline.

Direction accuracy and economically relevant threshold events are supplementary diagnostics, not replacements for price-error metrics.

Select any thresholds using development/validation data, not the final test.

### Probabilistic forecasting

Report:

- Pinball loss.
- CRPS where available or correctly approximated.
- Interval coverage.
- Interval width.
- Calibration by delivery period and regime.

Do not call an arbitrary average over a sparse, nonuniform quantile grid “exact CRPS.”

### Statistical comparison

Use paired losses on the same days.

Provide block-based uncertainty estimates where feasible, accounting for serial dependence.

State the aggregation convention and duration weighting. Do not treat every quarter-hour as an independent daily market experiment.

### Computational comparison

Record:

- Training/refit time.
- Inference time.
- Model size or memory where available.
- Hyperparameter search budget.
- Failures and unavailable optional models.

---

## 13. Interfaces, artifacts and later optimizer integration

Extend existing artifacts rather than introducing redundant output systems.

Forecast records should retain the project equivalents of:

- Run and model identifiers.
- Task and forecast origin.
- Decision cutoff.
- Delivery interval start/end and duration.
- Native market product/resolution.
- Predicted mean, median or quantile.
- Quantile level where applicable.
- Realized target when available for evaluation.
- Training cutoff and training window.
- Feature package and availability mode.
- Source-vintage provenance.
- Configuration, seed and model version.

For spread models, retain the spread forecast and reconstructed IDA1 price where reconstruction is valid.

Keep pre-DA and pre-IDA1 predictions separate.

Produce a comparable results table and suitable plots using existing reporting conventions.

The immediate goal is a sound forecasting benchmark, not a simultaneous rewrite of the trading simulator.

Provide a clean adapter for later forecast-provider integration if straightforward, but do not silently replace existing oracle experiments.

Missing model outputs must never fall back to realized prices.

Later economic evaluation should vary the DA predictor while holding the IDA1 predictor and trading policy fixed, then reverse that comparison, before testing joint replacements.

---

## 14. Tests

Add focused tests following repository conventions.

At minimum verify:

1. Spread sign and direct-price reconstruction.
2. Correct mapping of hourly DA prices to quarter-hour IDA1 products.
3. No artificial DA interpolation or quadruple-counting of hourly training labels.
4. DST handling, repeated clock hours and actual interval durations.
5. Exclusion of post-cutoff features and later forecast vintages.
6. Strict versus assumed gas-availability behavior.
7. Independent availability checks for ENTSO-E series.
8. Causal weather fallback with forecast age recorded.
9. Missing IDA1 observations are not converted to zero spread.
10. Task B can use published DA prices; Task C cannot use realized target-day DA.
11. DA-surprise features use chronological out-of-fold predictions.
12. Scalers, PCA, imputers and model selection do not fit on held-out data.
13. Mean/median/quantile outputs are correctly identified.
14. Distributional parameters, losses and predictions remain numerically valid.
15. Quantile-ordering and calibration logic.
16. Consistent forecast schemas across model families.
17. Optional dependencies fail or skip explicitly, without silent substitutions.
18. Reproducibility within documented numerical tolerances.
19. Model save/load and prediction round trips where applicable.
20. Compatibility with existing forecasting and replay regression tests.

Add an information-invariance test:

Changing future target values or post-cutoff data must not change an earlier prediction.

Test the entire feature-to-prediction pipeline, not only individual estimator classes.

---

## 15. Execution scope and deliverables

Proceed in stages, implementing executable support for the essential models first and then the optional challengers.

Run:

- Unit and integration tests.
- Synthetic smoke tests for the model interfaces.
- A small historical development-period pilot covering DA and IDA1-spread forecasting.
- A representative check across the product-resolution transition and DST handling.

Do not launch an uncontrolled full benchmark, large hyperparameter search or repeated final-test evaluation by default.

Do not use the locked final test as a smoke-test or debugging dataset. Prepare reproducible commands/configurations for the full run.

Do not present unrun models as validated.

Update forecasting documentation and the existing LaTeX implementation/methods report if one is maintained.

At completion, report:

- Architecture and integration points chosen.
- Data/provenance issues found.
- Implemented models and their availability status.
- Exact target and information-set definitions.
- Native-resolution and DST treatment.
- Configuration examples and commands.
- Tests and development-pilot results.
- Runtime and dependency considerations.
- Remaining limitations.
- Modified files.

The final implementation should make it possible to answer three questions separately:

1. Which model forecasts DA prices and IDA1 spreads best out of sample?
2. Which data sources and forecast revisions provide incremental predictive value?
3. Do those improvements subsequently improve PV+BESS decisions under a fixed, causal evaluation policy?

Prioritize trustworthy information timing, reproducibility and comparable experiments over the number of architectures implemented.