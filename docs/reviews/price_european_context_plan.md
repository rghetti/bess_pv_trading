# European market-context price ablations

`configs/research/price_european_context.yaml` declares a retrospective
comparison of standardized ridge forecasts for NORD DA, IDA1–3 and imbalance
settlement. It uses the same underlying target sources and January/August 2026
evaluation folds as the preceding price study. Ridge minimizes squared error
with coefficient penalty 20; a chronological 14-day tail supplies empirical
residual trajectories. Evaluation reports point MAE, empirical CRPS and coverage
on common complete days. No model is chosen by trading profit.

The price-only baseline uses price lags and calendar terms. The existing NORD
fundamentals add ENTSO-E load, total generation, solar and onshore wind. Separate
declared groups add PSV gas, Milan and Turin forecast temperature, French
nuclear and hydro generation plus France–Italy commercial schedules, or Swiss
national balance. A combined group includes those sets. A separate August-only
group adds Swiss–Italy commercial schedules, directional intraday NTC and the
Italy-NORD/Swiss spot spread from the Swissgrid 2026 file. These are feature
group comparisons, not claims about physical border flows or causal impacts.

The DA target uses 2024 price history. Intraday and settlement training begins
in 2025 under this configuration; the separate extended-2024 study preserves
its coarser hourly intraday training targets. Swiss national balance exists from
2025, and the currently linked Swissgrid border CSV begins in 2026, so those
models have shorter complete-feature training histories. The Swiss-border model
is not declared for the January fold. Differences can reflect both inputs and
training coverage; report retained training rows and common evaluation days.

French and Swiss realized system records are shifted by 48 hours with an assumed
two-hour post-interval release lag. PSV is shifted 48 hours and uses its
configured scheduled-publication assumption. Swissgrid border features use a
nine-day shift and a seven-day assumed publication lag; the exact historical
weekly publication clock is unknown. Regional Open-Meteo temperatures are
unweighted points, with an assumed six-hour cycle release and explicitly allowed
historical hindcasts. RTE and Swissgrid contain retrospective revised values;
these clocks do not prove historical vintages. No individual farm PV forecast is
used as an Italian zonal-price feature.

The catalog preflight completed without fitting. It found the French components,
Swiss national balance, PSV, both temperature points and the Swissgrid border
series. It does not establish that every model will fit or score every day.
French nuclear and hydro each had two missing quarter-hour features in the
selected source range; other missing values and the Swissgrid file gaps remain
explicit. The Swiss-border group uses a currently available annual 2026 source,
not an assumed 2024–2025 history. Monthly results, source snapshot hashes,
per-model exclusions and the shared-day evaluation must be inspected after a
run. The historical publication assumptions make results retrospective.

Once these fits finish, compare each added feature group to the fundamentals
baseline on identical target days and separately report training coverage,
per-market errors and distributional scores. A weekly Swiss reservoir level
is a slow seasonal state and is more naturally tested alongside seasonal
calendar terms than interpreted as an hourly cross-border signal. Its value
can be tested in a subsequent matched ablation using the latest assumed
released observation, a finite maximum age and the same training rows for both
models. The current run does not include this dataset. Do not interpret a
ridge coefficient as causal or select a model from one month alone; use the
held-out folds and inspect degradation across markets.

The catalog reader now supports `kind: released_state` for this follow-up.
Configure `availability_column: scheduled_available_at`,
`selectors: {variable: total_ch_percent}`, `max_age_hours: 240` and
`source_lookback_hours: 480`. The age limit is measured at delivery and the
lookback must cover both the publication delay and the allowed age. Expansions
for consecutive releases overlap deliberately: the forecast cutoff chooses
the latest eligible vintage, so a subsequent release cannot erase the older
value from a decision made before that release. Missing values are not filled.
The source snapshot's Friday release is still only an assumed clock, and this
reader preflight is not a fitted forecast or an ablation result.

`configs/research/price_european_previous_day.yaml` is a separate retrospective
75-fit comparison using realized French generation and exchange, Swiss balance
and borders, and PSV gas shifted by 24 hours instead of the earlier 48-hour or
nine-day shifts. A source-day midnight publication clock is a deliberate
placeholder so the previous day's full series can be used at any next-day
decision; it does not assert actual historical publication or operational
availability. A fixed 24-hour shift differs from matching local clock hours
across daylight-saving transitions. January and August evaluation periods do
not contain those transitions; training history can. The same price targets,
feature groups, 2024 DA training, residual calibration and scoring are used as
in the earlier study. Regional weather forecast handling is unchanged.
