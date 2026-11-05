# One-week current-interval MPC priority comparison

Status: explicitly approved and completed. See [results](first_interval_priority_one_week_results.md) and the [independent audit](first_interval_priority_one_week_audit.md).

## Comparison and objectives

Use 15–21 January 2026, Europe/Rome, farms 1, 345, 901 and 910. For each farm,
run three separate continuous closed-loop all-auction trajectories, with the
existing grid tracker recorded as a reference for each job:

1. `original`: the existing deterministic MPC objective, EUR 500/MWh times
   total absolute imbalance plus degradation, imports and terminal cost.
2. `weighted_k10`: the same objective, except the current interval's absolute
   imbalance receives ten times its ordinary weight (EUR 5,000/MWh). Future
   interval weight and all costs remain unchanged.
3. `strict_first`: solve first for minimum current-interval absolute imbalance
   over the physical horizon and binding terminal constraints. Subject to that
   first-stage incumbent plus 1e-7 MWh, minimize the original horizon objective.
   The first-stage minimum is certified only to the reported Gurobi MIP gap and
   numerical tolerances. Do not refine an uncertified time-limited incumbent;
   use a validated stage-one action and label it as such. If refinement fails,
   retain the validated first-stage action. If no valid action is available,
   use and report the existing grid-tracking fallback.

Both strict stages share the ten-second total per-decision model-build/solve
budget. Record both stage statuses, objective bounds, gaps, cap, current
imbalance and numerical tolerance. A 1e-7 MWh cap tolerance is a planning
constraint tolerance, not an absolute MIP gap; the solver uses relative MIP
gap 1e-4 and no absolute MIP gap. Tests and audit check physical residuals,
objective reconstruction and terminal feasibility. The strict-first outcome
is not automatically identical to greedy tracking because the MPC enforces the
entire remaining horizon's physical and hard terminal restrictions.

Keep the planned IDA3 target from the most recent completed IDA2 snapshot until
IDA3 is submitted, then use the binding submitted position. The current
interval always uses the actual submitted position. Curtailment recovery is
**disabled in all three cases** to isolate priority; its separately tested
week remains available as context. Other settings match the recent seven-day
recovery study: 50% initial SOC, soft 50% ordinary-midnight target, separate
binding 50% final-study planning target, no additional ordinary-midnight
floor, configured per-farm batteries, power limits and throughput wear.
Actual SOC, degradation, commitments and fees carry chronologically without
resets. Controllers can change later auction orders, so these are complete
closed-loop policies rather than a frozen-commitment control comparison.

## Inputs and information

Use the old frozen reviewed-GHI selected per-farm PV models: BiGRU (1),
Transformer (345), BiLSTM (901), and TCN (910). No fitting, training loss,
model selection, calibration or GPU inference is part of this run. Their
training used historical GHI and selection used later data, so these are
retrospective models; do not generalize from this development week. At each
cutoff, deterministic PV uses the latest eligible weather cycle, with an
assumed six-hour release lag, configured month-specific nighttime zero policy,
and the reviewed curated physical targets. Flag synthetic and historical-GHI
fills where they occur; these are counterfactual, not metered.

Both MPC and grid tracking use the disclosed current-interval average PV in
this retrospective study; future realized PV remains hidden. The physical MPC
has no future settlement-price information. Actual NORD imbalance prices are
used for realized settlement only. Upper auction reoptimization retains the
existing actual auction-price oracle. Prices, commitments, fees, model/data
fingerprints and source checksums are recorded. Current-PV interval averages
are an idealized assumption and do not establish real-time sensor availability.

## Execution and outcomes

The matrix has 12 jobs (four farms times three MPC cases), with 8,064 nominal
MPC decisions and up to 672 auction solves including tracker references. Use
two workers, one CPU/Gurobi thread per worker, no GPUs, and the configured
10-second MPC / 120-second auction limits. This is at most about 44.8 aggregate
solver-hours under the nominal caps, not a runtime estimate; preparation,
retries and reports add time. The machine is shared. No full-year, extra farm,
or additional sensitivity experiment is authorized by this plan.

Report per farm and pooled: actual current and total absolute imbalance,
signed settlement and exposure, auction quantities and revenue, degradation,
imports and fees, realized profit, curtailment, throughput, actual and final
SOC, terminal outcome, noon/midnight projection errors, first-stage
certification, stage-2 use, numerical rejection, solver time, fallbacks and
safety protection. Profit is actual revenue plus signed settlement minus
executed wear, imports and trading fees; planning penalties and terminal value
are not actual charges. Include inventory beside profit comparisons. The
first-interval sums across decisions do not sum overlapping horizon forecasts.

One shared output folder holds the configuration, source and input checksums,
full plans, per-interval controls, paired daily and whole-run results and plots;
per-day plots are capped at two dates. Independently reconstruct objective,
SOC, power/POI, commitment and financial values and verify the strict cap.
One week and old models provide an illustrative sample, not a final ranking or
annual generalization. Prepare future new-model reruns separately.

Configuration: `configs/research/first_interval_priority_one_week.yaml`.

```bash
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/first_interval_priority_one_week.yaml --check-only --workers 2
```

The user approved the run after the plan and preflight were presented. Check-only preflight and unit/integration software verification did not start historical controller experiments.
