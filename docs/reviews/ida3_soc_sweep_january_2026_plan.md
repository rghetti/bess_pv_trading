# January 2026 extended IDA3 midnight SOC sweep

## Objective and comparison

For each of farms 1, 345, 901 and 910, run the same chronological greedy
grid-tracking replay for each hard auction-planning midnight SOC target in
`configs/research/ida3_next_day_soc_sweep_january_2026.yaml`: 7%, 10% through
90% in five-point increments, and 93%. This is 19 targets per farm, 76
trajectories total. All trajectories begin at 50% SOC; the planning target
does not enforce actual tracker SOC. The IDA3 optimization includes the rest
of the current day and the following day, with the same hard target at each
future modeled midnight. The fixed SOC at the connecting midnight limits how
much the next day can affect current-day choices.

Score January 1–31 delivery products in Italian local time, with February 1 as
unscored planning and auction context. The first event is the December 31 DA
auction for January 1; there is no December 31 IDA3 solve. Retain the existing
auction chronology and price-taking full-execution rules. If an auction is
infeasible, retain an existing valid planned order where available; otherwise
record the failed trajectory. Report failures without hiding completed runs.

## Information and limitations

Use the same reviewed AC PV dataset, monthly night policy and labeled
synthetic/reconstructed replacements as the January comparison. Latest
eligible deterministic weather forecasts have an assumed six-hour release
lag and 48-hour context. Use the existing four-cycle BiGRU models for farms 1
and 345 and TCN models for 901 and 910; no new training, calibration, tuning,
scenario generation or downloading. Model loss functions, hyperparameters and
training/validation split are documented in
[the original January plan](ida3_next_day_january_2026_plan.md). January 2026
was used for model selection, so these are retrospective comparisons with
selection leakage. The tracker observes realized interval-average PV
retrospectively. The planner uses actual future auction clearing prices as an
oracle, including next-day prices at IDA3, and realized settlement prices for
accounting. These are not causal live-market results.

All targets within each farm must use matching realized PV and market prices;
check shared January inputs against the existing historical run at preflight
and reconcile per-product inputs across completed trajectories. Forecast and
market preparation is shared once per farm within the batch, while each
trajectory has its own target-specific physical state, configuration and
provenance fingerprint.

## Compute and outputs

Preflight all four farms before starting any trajectory, then run up to seven
independent trajectories in parallel, refilling each worker when it finishes,
with two Gurobi threads per worker. This is at most 14 Gurobi threads on a
machine reporting 10 logical CPUs. The upper bound is 9,652 auction submissions
(76 times 127); timeouts, unavailable auctions and fallbacks can change actual
work. No GPU or package installation is required. Release the shared batch
preflight cache on successful completion; retain it on failure for diagnosis.

Produce one combined Markdown index and a separate Markdown table and compact
plot per farm, plus machine-readable CSV and input-matching evidence. Report
realized operating profit, auction revenue, imbalance settlement, modeled
degradation, absolute imbalance, curtailment, percentages of gross PV, battery
use, initial and actual end-of-replay SOC, actual midnight SOC and fallback
counts. Profit excludes final inventory valuation; targets may leave different
amounts of stored energy, so raw profit differences do not account for that
residual value. No model-selection winner is determined automatically.
