# January 2026: forecast-price auction participation comparison

Status: preparation only. No model fitting, historical optimization, or monthly
results have been performed for this plan. The run requires a complete forecast
pipeline, software checks, preflight and explicit approval of the final plan.

## Scope and comparison

Use local delivery dates 1–31 January 2026 for farms 1, 345, 901 and 910.
For each farm, replay the eight subsets containing DA and any subset of IDA1,
IDA2 and IDA3. Each profile has a separate, continuous physical and financial
trajectory. Compare two planning information sets: forecast auction prices
and the same actual auction clearing curves known perfectly in planning. Both
arms use actual clearing prices at publication and actual imbalance settlement
for realized cash flow. Both use the same grid-tracking controller, assets,
initial state, PV forecasts and data assumptions. This is a price-information
benchmark, not perfect foresight of PV or imbalance prices.

Include the existing all-auction sequential-planning reference separately in
each arm. This gives 64 paired profile trajectories plus eight reference
trajectories. With four available auctions on every day, these amount to at
most 5,952 nominal auction solves, excluding preparation and any retries.
Use independent trajectories and carry battery SOC, degradation memory,
commitments and fees across days. Report terminal inventory because cash
profit does not value leftover energy. Preserve final-study restrictions and
report any execution shortfall rather than forcing a reset at midnight. Use the
existing farm-specific battery specifications and a 50% initial and final-study
SOC target, with no additional ordinary-midnight floor.

Use reviewed-GHI selected PV models, reviewed curated physical targets and the
configured local monthly night-zero policy. The reviewed targets include
explicitly marked synthetic or historical-GHI replacements where needed;
these are counterfactual and not metered. PV models were selected after the
study period, and their retrospective availability must be disclosed. Use
eligible updating deterministic forecasts, subject to the assumed six-hour
weather publication lag, and record forecast vintage and source counts.
No PV training or GPU work is part of this plan.

## Price models and retrospective selection

The early-selected quarter-hourly research designs currently documented are:

* DA: time-interaction Elastic Net, using lagged DA curves, lagged IDA prices,
  local delivery-time interactions and regional weather; squared-error plus
  regularization loss. The studied configuration uses alpha 0.1 and L1 ratio
  0.5. The candidate was selected after January using February–April data.
* IDA1: Elastic Net, alpha 0.1 and L1 ratio 0.5, predicting a spread to the
  latest cleared auction, with neighboring auction products and regional
  weather.
* IDA2: Extra Trees, 150 trees, minimum leaf size 15 and maximum feature
  fraction 0.6, predicting the same type of spread.
* IDA3: ridge regression, alpha 20, predicting the same type of spread.

See `price_da_prior_ida_weather_factorial_2026.md` and
`price_ida_latest_bidirectional_2026.md` for candidate evidence and details.
Other price-model searches are active, so the final plan must identify the
exact completed selection evidence and freeze the choices before fitting.
The requested retrospective selection is not historically causal. Even with
training labels and features restricted to each historical decision, the
model-design selection used later data and must be labeled accordingly.

Proposed fitting uses the selected architectures and chronological pre-January
training, with weekly refits and a full-day gap; each fit may use only labels
whose product end and assumed publication precede its cutoff. Scaling and
feature transformations fit only on the eligible training rows. Training
losses are the models' own losses; evaluate forecasts separately with matched
MAE, WAPE and per-day errors. There is no forecast-probability calibration in
this point-forecast comparison. The regularized models and tree are CPU-only;
use one native thread per fit and two independent farm workers for replays,
as requested by the user. Seed and exact estimator versions must be pinned.

Regional weather uses archived forecast data whose operational-versus-hindcast
status has not been verified. The archive's assumed publication time is cycle
plus six hours, and market publication clocks are also partly assumptions.
Input snapshots, checksums, exclusions, convergence warnings and any fallback
must be captured. Model fits and forecasts must never consume future auction
prices merely because these are in the stored data catalog.

## Information timing and implementation prerequisites

The latest IDA models predict spreads relative to preceding cleared auctions.
A model's native-gate feature table therefore cannot safely be used at an
*earlier* auction gate before those anchors have been published. The comparison
needs cutoff-specific forecasts or a separately identified causal fallback for
each unpublished auction, with an explicit provenance record for every product.
Published prices replace forecasts only after publication. Any anchor estimates
or fallback create a distribution shift relative to native-gate research scores.
The existing continuous runner passes one point price curve per auction; it
does not currently model a distribution of auction-price scenarios. Adding
stochastic price scenarios is outside the point-forecast comparison described
here and would require a separate explicit scenario-generation and dependence
plan.

The current saved research feature matrices contain 30 complete January days;
31 January is incomplete for DA/IDA1/IDA2 and absent for IDA3. They cannot be
used as the whole-month evaluation inputs without rebuilding or repairing the
missing features. Do not fill missing market prices with zero. Check-only
preflight must verify all dates, gates, source publication clocks, product
coverage, feature availability, the fitting cutoffs and fallback counts before
any monthly optimization. If coverage fails, report the gap and repair the
inputs or revise the plan before proceeding.

Implementation must connect the selected model designs to the price-provider
contract, prevent future leakage at each gate, validate mixed predicted and
published anchors, and produce a shared four-farm report. Add focused
software tests for feature and publication cutoffs, price provenance, forecast
changes without realized-price leakage, price settlement accounting, paired
profiles and checkpoint identity. No source-model retraining, evaluation run or
controller experiment is authorized merely by this preparation document.

## Outputs, compute and approval

Write one shared study directory containing the frozen configuration, code and
input checksums, forecast and publication provenance, per-profile interval and
daily ledgers, solver diagnostics, paired forecast-minus-oracle profit and
component differences, participation rankings, battery and inventory metrics,
price forecast scores, exclusions, and a single Markdown summary. Include
study-level plots and limit daily plots to at most three dates. Compare profits
as realized auction revenue plus signed imbalance settlement, less executed
degradation, imports and trading fees. Model objective values and forecast
penalties are not cash charges.

The solver uses Gurobi, one thread per independent farm worker, relative MIP
gap 1e-4 and no absolute gap; the existing 120-second per-auction cap yields
an upper limit near 198.4 aggregate solver-hours for 5,952 nominal solves, not a
runtime estimate. Limit the main run to two farms in parallel. The machine is
shared and currently has active GPU jobs and a CPU price-model search; check
resources and coordinate database use before training or replay.

Project instructions require the concrete completed plan, including actual
coverage and exact final model selection, to be presented for explicit approval
before any historical fitting or optimization. This document is preparation,
not evidence that the study is ready to run.
