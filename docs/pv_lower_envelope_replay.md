# Pointwise PV minimum: formulation and replay comparison

This study compares farm 1's 51-member exact-delivery model with a model using
one synthetic trajectory: the minimum across the 51 PV forecasts at each
interval. The latter is an envelope, not necessarily a trajectory belonging to
one weather member. The study covers 2–11 August 2026, with continuously carried
SOC, degradation, fees and commitments. The 1 August delivery day is omitted
because its preceding-day ensemble cycle is absent.

## Why the planning formulations are equivalent here

At a fixed auction cutoff, all members share auction orders and battery commands.
Exact delivery makes grid exchange equal to the same committed position for
every member. The battery contribution is also the same, so each member must
use the same PV injection `x[t]`. Scenario-specific PV curtailment is free and
can take any value between zero and that member's available PV, so feasibility
requires `0 <= x[t] <= min_s PV[s,t]`. Conversely, any feasible envelope plan
can be lifted to all 51 members by setting each member's curtailment to
`PV[s,t] - x[t]`. The lifted values are identical for equal observed PV.
Shared state, positions and battery actions give the same revenue, physical
costs and objective in every scenario, so the feasible projected decisions and
optimal objective coincide. This includes negative prices, battery constraints,
fixed commitments and identical transaction fees. It assumes there is no
curtailment cost, scenario-specific price or battery recourse, or planned
imbalance. Changing these assumptions requires a new argument.

This equivalence concerns the planning MILP at the **same state and cutoff**.
An optimizer may choose different schedules among equally valuable optima; the
full model saves one member's planned curtailment while the envelope saves its
own. The feedback controller sees realized PV and now recovers curtailed PV by
reducing discharge or increasing charge within its power and SOC limits. Thus
independent replays can follow different physical states, commitments and
realized profits even when the planning problems are mathematically equivalent.
The controller is not represented as a scenario-dependent battery recourse
policy inside the planning MILP.

## Procedure and outputs

The full-model replay runs as in the existing August study. At each of its
available auction gates, an additional envelope solve uses that very same
full-model state, forecast, fixed commitments and prices; the shadow plan is
not submitted. The second continuous replay submits the envelope model's own
orders and carries its own realized state forward. The study compares daily
and period realized profit, imbalance, curtailment, recovery, battery usage and
fees, as well as interval-level dispatch and commitments. The matched-state
table reports objective differences, incumbent validation, feasibility for all
members, solver bounds, gaps and statuses; objective differences and overlap of
finite-gap bounds are numerical evidence, not an exact proof. Solver failures,
fallbacks and time limits remain visible.

The production optimizer now uses the envelope by default for eligible models.
The pilot scripts explicitly disable that reduction when building their full
reference cases, so a repeated comparison still solves the unreduced model.
The controller-policy version changed with the production reduction; the
previous study's checkpoints cannot be resumed with the new version.

The 50 members use the prior delivery day's 00 UTC ensemble cycle, available
under the configured six-hour latency assumption; the deterministic member
updates at each cutoff. Both use the farm 1 saved direct TCN and the configured
monthly nighttime clipping. The artifact was made available in October 2026,
so this August evaluation is retrospective. Measured PV receives the configured
nighttime clipping; the model-training episode exclusion list is not applied to
replay observations. Actual GME auction prices are assumed known in planning,
and actual imbalance prices are used for settlement. No retraining, calibration
or forecast post-processing is performed.

The experiment uses up to 111 solves (37 gates for each continuous replay plus
37 matched-state shadow solves), each with 10 threads, a 120-second solver
limit and a 0.01% relative MIP gap; model build, input preparation and plotting
are additional time. Outputs and checkpoints go under
`outputs/continuous_optimization/comparisons/farm_1/envelope_replay_2026_08_02_to_11`.
Input fingerprints, the policy version and the study-code hash are checked on
resume. Do not change the study code or configuration while resuming.

Preflight without running an optimization:

```bash
PYTHONPATH=src python scripts/compare_pv_envelope_replay.py \
  --config configs/full_experiments/envelope_replay_2026_08_02_to_11.yaml --check-only
```

After approval, run the same command without `--check-only`. The comparison does
not use or overwrite the existing January or August monthly experiment results.
