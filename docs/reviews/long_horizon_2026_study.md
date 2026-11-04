# Prepared 2026 continuous controller studies

`configs/research/annual_2026_deterministic_matched.yaml` requests the entire
1 January–31 December 2026 as one continuous trajectory for each farm and
case, so SOC, commitments and degradation state carry across month boundaries.
It compares tracking with deterministic imbalance-penalty and forecast-economic
MPC, settlement-price oracle MPC and its 1% gap-qualified fallback. Each case is paired across pvlib
and the final neural forecasts. Oracle settlement and perfect auction prices
are retrospective information benchmarks; they are not operational forecasts.

`configs/research/jan_aug_2026_ensemble_matched.yaml` requests all of January
and all of August, each as a continuous monthly trajectory. Each month starts
from its configured initial state because the seven-month gap is not simulated.
For each farm and forecast generator, cases compare grid tracking against
deterministic penalty, ensemble penalty, ensemble oracle, oracle with 1% gap
fallback, economic ensemble MPC, and adaptive raw-ensemble economic MPC.
The two forecast generators use matching dates, settings and latest-available
cycle policies. All comparisons use live commitments and continuous replay.

The annual plan contains 32 jobs, 1,121,280 nominal MPC decisions and 93,440
auction submission solves. The January/August plan contains 96 jobs, 285,696
nominal MPC decisions and 23,808 auction submission solves, before retries.
Plans use eight independent
workers with one solver and native thread each; worker count can be changed at
launch after checking host load. The optimizer's standard relative gap is 1e-4;
the optional fallback threshold is 1%. Oracle stage one and stage two each have
their own ten-second budget. No model search, optimizer solve or replay was
started in preparing these configurations.

## Adaptive ensemble controller

At each physical step the model evaluates many future PV paths. The current
charge/discharge and PV-use command is common to every path. Future control
variables are shared among scenarios whose disclosed PV histories are in the
same node, and can differ after histories split. Here the configured branches
occur every four quarter-hour intervals, with one-interval observation delay and
0.1 MW PV-history quantization. The only executed command is the current one;
the controller then observes the updated state and solves again. This is a
restricted recourse policy based on PV histories, not perfect future knowledge.
The economic version optimizes weighted expected physical cashflow, battery
wear and the configured terminal term; its monetary CVaR weight is zero unless
explicitly changed. It is distinct from the oracle's two-stage minimum-imbalance
then economic-refinement formulation.

## Readiness limitations

The reviewed PV target and the existing measurement archive end in August 2026;
the annual study requires reviewed targets and forecast inputs through December.
Cached NORD settlement prices currently end in September and auction prices in
October, so later market data also need acquisition and coverage checks. These
are observations of the current catalog, not proof that every earlier day is
complete. The currently selected neural bundles are placeholders until the
user's larger training finishes and its final models are supplied.
The economic cases currently point to the existing January-fitted settlement
model; the underway price-model search and its held-out evaluation must inform
the final forecast artifacts and retraining schedule before this annual study.

The currently available January ensemble covers January, but the existing
August archive starts with the 1 August cycle. It lacks the preceding cycle
needed at the first 1 August day-ahead decision; do not silently omit that day
or substitute an unavailable cycle. A 31 July cycle or another explicitly
agreed forecast source is required for the full-month ensemble comparison.
More ensemble archives should be added as actual paths with explicit overlap
precedence and validated member/cycle/lead coverage.

`research_plan` checks required files and counts nominal jobs; a zero count of
missing paths is **not** full temporal or data-quality coverage. Run the ordinary
`--check-only` replay preflight after all inputs and models are available and
inspect every day and exclusion before running. For the annual calendar the
configuration extends current gate validity through 2026-12-31 as an explicit
unchanged-calendar assumption; recheck any later market-rule changes.

Report per-farm and portfolio daily profit, imbalance, wear, runtime, solver
status, safety overrides and terminal energy, with common-day comparisons and
plots. The tracking controller uses interval PV feedback; the MPC's current PV
input is a forecast. Terminal penalty settings and final battery energy differ,
so cash profit alone does not isolate a controller effect. Reviewed replacement
targets and all timing assumptions must remain visible in the interpretation.
