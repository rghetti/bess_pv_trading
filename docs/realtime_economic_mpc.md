# Real-time economic MPC

This layer controls physical PV+BESS operation inside the existing chronological
auction simulator. Gurobi optimizes one remaining-local-delivery-day horizon at
each quarter-hour start. Only its first action executes. No fitting, probability
calibration, auction strategy, CVaR term, price forecast, or ancillary service is
introduced.

## Information and controller selection

`execution.dispatch_policy` retains `track_committed_grid` as the shared default
and `follow_battery_plan` as a legacy option. Four new options are available:

| Policy | PV representation | Future imbalance prices |
|---|---|---|
| `deterministic_mpc` | Latest eligible deterministic forecast | None |
| `deterministic_mpc_price_oracle` | Same deterministic forecast | Explicit actual-price oracle |
| `ensemble_mpc` | Frozen weather members and updating deterministic member | None |
| `ensemble_mpc_price_oracle` | Same ensemble | Explicit actual-price oracle |

The optional `causal_track_committed_grid` diagnostic computes tracking commands
from the interval-start deterministic forecast, freezes them, and applies only
protection during execution. It is excluded from the approved historical smoke.

In the default causal mode, actual state and PV measurements are available only
from completed intervals. The optional
`realtime_mpc.current_pv_information: realized_interval_average` explicitly gives
all MPC variants the realized average of the interval being controlled. It
replaces the first PV value in every scenario, leaving future PV as forecasts.
This is a retrospective current-PV information experiment; it does not expose
any later interval's production. The default remains `forecast`, and that mode
rejects supplied current realized PV. Forecast issue and publication clocks must precede the cutoff; the
current standard assumes publication six hours after cycle initialization.
Immediate completed-interval telemetry remains required. These publication
times are historical assumptions where source publication records are absent.

Weather members use the preceding delivery day's 00 UTC cycle, frozen for the
delivery day. The deterministic forecast updates at each eligible cutoff.
Fifty weather members plus the deterministic trajectory receive weights 1/51 by
default; the weather-only sensitivity uses 1/50. An explicit `scenario_weights`
mapping must identify all represented scenarios and sum to one. Raw ensemble
weights are not calibrated probabilities. Scenario identity never becomes
observable information.
The weight mapping applies to ensemble MPC; deterministic MPC always has weight one.
Current PV disclosure does not identify an ensemble member or reweight future
trajectories; future members retain their explicit raw weights.

Preflight still checks every distinct UTC cutoff, but opens the Zarr store and
reads farm metadata once for the cutoff scan. It caches raw weather values and
aligned PV conversions by selected forecast cycle, then slices the remaining
horizon for each cutoff. Converted PV is cached only when the physical
parameters are fixed; a cutoff-varying parameter registry bypasses this cache.
Eligibility and provenance are rebuilt for every request. The caches live for
one preparation call, not across separate commands; the full study launcher
shares one preparation between both comparison modes. Progress reports every
100 cutoffs and gives the raw-weather and PV cache hit counts.

The revised smoke uses the selected Farm 1 weather bundle `direct_tcn_02` for
deterministic forecasts, all 50 ensemble trajectories, and the common auction
source strategy. Its 12 frozen weather/calendar features feed a direct TCN
(width 16, depth 3, kernel size 7; inference dropout disabled). The network was
trained through 30 December 2025 and selected using January–April 2026 results.
The registry marks availability on 27 October 2026. Consequently January use
explicitly enables retrospective artifacts and includes hindsight model
selection; these dates are not an independent model-selection holdout. There
is no training, calibration, or hyperparameter search in this workflow.

Artifact and dependency checksums, declared training/selection periods and
availability are pinned. Torch inference shares the configured two CPU threads;
no GPU is used. The same bundle supplies initial frozen-forecast measurement
fallbacks, which remain flagged rather than treated as raw observations. No
additional conversion loss is applied to measured AC PV. The operational
pvlib path and default-disabled replacement remain available. Loading ensemble
data for MPC does not enable ensemble auction planning. Earlier pvlib smoke
results are preserved under the original run ID.

## Formulation

### Optional shared-first-action scenario schedules

`execution.dispatch_policy: ensemble_mpc_adaptive` with
`realtime_mpc.recourse_mode: first_action_only` gives each forecast trajectory
its own battery charge/discharge, operating mode, utilization and energy
schedule after the current interval. Only the first physical command is
shared across all members. The initial energy and energy immediately after
the shared first command are therefore common. Future per-scenario states
then follow their own schedules, with all physical and configured hard
terminal constraints enforced separately. The default `observed_history`
mode retains the existing delayed-observation scenario tree.

First-action-only recourse assumes full scenario-path revelation after the
first command inside the planning approximation, even when observed PV
prefixes are identical. Its continuation values can be optimistic. Actual
future PV is never supplied and only the common first command executes;
the next interval replans from the observed physical state. The deterministic
member must be retained first for this mode's auction shadow projection.
If settlement-price forecasts are also enabled, scenario-owned future commands
can similarly depend on the corresponding complete forecast price paths. The
pure-imbalance comparison uses no settlement-price forecast or oracle.
That path is rolled from actual observed state through the executor with
eligible deterministic PV, retaining outstanding commands and protection.
No scenario-dependent future command is treated as an executed promise, and
no MPC is solved during that boundary rollout.

`realtime_mpc.planning_objective: imbalance_only` minimizes the expected
sum of absolute energy deviations in MWh, without the EUR 500 multiplier,
degradation cost, import cost or inventory preferences in the objective.
It supports ordinary price-blind and adaptive MPC; it rejects settlement
price objectives and monetary CVaR. Physical efficiencies, power/SOC bounds,
degradation memory and explicit hard terminal constraints still apply.
Executed wear, imports and signed settlement remain measured in financial
reporting. There is no implicit wear tie-break among equal-imbalance plans.
The default `imbalance_and_costs` retains the existing EUR planning objective.
The horizon remains the rest of the current local delivery day.

The prepared comparison is `configs/research/first_action_only_targeted.yaml`.
It includes the provisional IDA3 planning targets described below, while
financial settlement continues to use actual submitted quantities.

Let `Q_submitted[t]` be the sum of quantities actually submitted by the cutoff,
including unpriced submissions. The default `realtime_mpc.provisional_ida3_mode:
ida2_plan` adds the IDA3 order forecast saved in the completed IDA2 auction
snapshot for IDA3-eligible afternoon products, until IDA3 is submitted. Thus
`Q[t] = Q_submitted[t] + Q_provisional_IDA3[t]` is the planning target. The
provisional component is zero on the executable current interval and stops
immediately at actual IDA3 submission, before price publication. Submission
uses the configured decision buffer; no 10:00 timestamp is hard coded. Signed
planned purchases and sales are both supported. Missing IDA2 plans, unavailable
IDA3 products and excluded auctions do not generate provisional quantities.
`submitted_only` reproduces the earlier planning behavior. Other unsubmitted
auction orders remain excluded. There are no trade variables or auction revenues in the MPC.
The submitted ledger, trading fees and realized imbalance settlement retain
only actual submitted quantities. Full MPC plans and scenarios record the
submitted, provisional and total planning positions separately. Revised plans
can change later actual auction submissions in closed-loop comparisons.
Positive grid power denotes export; positive imbalance denotes excess export.

For every represented PV scenario `s`, with nonnegative weights `w[s]`:

```text
0 <= a[t] <= 1
PV_injected[s,t] = a[t] * PV_available[s,t]
g[s,t] = PV_injected[s,t] + discharge[t] - charge[t]
delta[s,t] = g[s,t] * dt[t] - Q[t]
e[t+1] = (1-self_discharge_per_hour)^dt[t] * e[t]
         + eta_charge*charge[t]*dt[t] - discharge[t]*dt[t]/eta_discharge
e[0] = observed energy
L = sum_s w[s] * sum_t abs(delta[s,t])
```

Battery charge/discharge, binary operating mode, utilization fraction and energy
are common over the **entire horizon**, not independently optimized by member.
Power, SOC and directional POI bounds remain hard in every scenario.
Shared battery-transition and degradation helpers are used by auction planning
and MPC; execution uses the same transition. The throughput convention costs
both directional AC throughput once. Xu mode reuses canonical segment priorities
and carried segment memory, and continues rejecting nonzero self-discharge.

Price-blind controllers minimize:

```text
kappa * L + executed-plan degradation + expected import cost + terminal cost
```

`kappa` defaults to 500 EUR/MWh. It is a configurable planning coefficient,
independent of future realized prices. No separate PV-curtailment penalty is
included. Curtailment can reduce expected excess-export imbalance. Storing PV
instead is selected when its future benefit exceeds wear and inventory costs.
If all forecast members have zero PV in an interval, the otherwise unidentified
utilization setpoint is fixed to one, passing unexpected PV through protection.

Terminal modes are `none`, `soft_target`, and `linear_value`. The default soft
target is 50% nominal SOC with 50 EUR per MWh of absolute internal-energy
deviation. Linear value subtracts the configured EUR/MWh times closing stored
energy from minimization cost. These values are planning terms, not cash flows.
The configured final-study equality applies only on the final delivery day;
an explicitly configured terminal minimum remains a horizon-end restriction.
No hard 50% equality is added to intermediate midnights. Actual terminal misses
after protection or fallback remain visible rather than resetting actual SOC.

Oracle controllers first minimize `L` with the same physical and hard terminal
constraints. If this stage reaches Gurobi optimal status within configured
tolerances, its incumbent value defines `L_min`. The second stage maximizes:

```text
sum_s w[s] * sum_t settlement[s,t]
    - degradation - expected import cost - terminal cost

subject to L <= L_min * (1 + relative_budget) + absolute_budget_mwh
```

Single-price settlement is exactly `lambda[t] * delta[s,t]`. Supported directional
prices use `lambda_plus[t]*delta_plus - lambda_minus[t]*delta_minus`. Negative
prices reverse receipts/payments; signed settlement is never replaced by
`-|lambda|*|delta|`. Budgets default to zero. The supplied 5% examples are
sensitivity configurations and are not automatically executed.

Nonnegative imbalance splits are exact under the absolute-imbalance objective.
Single-price economic stages use their sum as a conservative budget bound;
physical absolute imbalance is independently reconstructed for reporting and
validation. Distinct directional curves additionally enforce sign exclusivity.
The battery's binary exclusivity is always enforced.

One ten-second wall budget covers building and both Gurobi stages. Stage 1 may
consume the remaining budget before economic refinement. An uncertified Stage 1
incumbent is usable as a price-blind plan but does not establish a certified
minimum or permit Stage 2. When Stage 2 has no usable solution, the validated
Stage 1 action is retained and labeled. No valid action triggers explicit
tracking fallback; its interval-average PV feedback advantage is disclosed.
Solver objectives, bounds, gaps, stage runtimes and budget use are retained.
Relative MIP gap defaults to 1e-4; no absolute MIP gap is configured.

## Execution and auction boundary projection

Each first action fixes battery AC MW and the utilization fraction. In causal mode
it precedes disclosure of current PV; the revised experiment deliberately reveals
the current interval average before solving. The physical environment applies the fraction to realized PV and
delegates to `follow_battery_plan` protection. It does not invoke grid tracking
to economically replace MPC dispatch. Protection can reduce charging/discharging
and curtail PV to enforce physical limits. Commands, execution, feedback and
safety overrides are separately recorded. Historical interval-average production
is an idealized fast-protection input, not a pre-decision measurement.

Auction timing and guaranteed execution remain unchanged. Publication books
quantity times actual auction price once; artificial planning costs never enter
real settlement. Auction and MPC plans occupy separate stores. An auction inside
an unfinished quarter cannot replace its outstanding physical action.

At every auction cutoff, the boundary shadow rollout starts from the most recent
actual state and degradation memory. It preserves the unfinished interval action,
then follows the last completed MPC's common future controls with the **latest
eligible deterministic PV forecast at that auction cutoff**, including for
ensemble MPC. It stops exactly at delivery midnight or IDA3 noon, without solving
any hypothetical future MPCs or choosing a weather member. Uncovered intervals
use explicitly logged forecast-only tracking/protection. The first study day's
supplied midnight state is retained without inventing preceding-day operation.
In current-PV-known mode, the locked unfinished interval uses its already
disclosed PV scalar; all subsequent intervals use deterministic forecasts. The
rollout never retrieves future actual production, and logs each PV input's role.

The shadow state is named `projected_soc_mpc_deterministic`. It is not an expected
ensemble SOC. Pure battery-power commands can make it insensitive to PV until
protection overrides a command; this is expected. Projection never overwrites
actual state. Actual state carries across noon/midnight, and projection errors
are resolved only when the relevant real boundary is reached. MPC is not forced
to achieve tomorrow's auction optimizer's earlier assumed energy.

## Comparison workflows and outputs

`experiment-realtime-mpc` uses the existing continuous runner in two modes:

* `frozen_commitments`: generate a deterministic integrated auction/tracker source
  once, or load a checksummed `frozen_commitments.json` produced by this workflow.
  Replay submissions at their original timestamps. Preserve the source auction
  plan snapshots for baseline execution. Every controller must reconcile identical
  quantities, prices, revenues and fees. Only physical control differs.
* `continuous`: allow actual executed SOC and deterministic MPC projections to
  affect later auction solves. Report resulting auction-revenue and fee differences;
  the comparison includes both physical control and changed future trading.

The upper auction-price oracle is unchanged and recorded independently of the
lower imbalance-price oracle. A price-blind physical controller running under
oracle-generated auction commitments is not a fully causal trading strategy.
The existing tracking baseline sees realized interval-average PV. Its current-PV
information advantage is flagged in causal MPC comparisons; the revised
current-PV-known comparison supplies that same interval-average value to MPC.
The battery action is still fixed for the interval, followed by protection.

Outputs preserve existing continuous CSV/JSON files and add `mpc_plans`,
`mpc_actions`, `mpc_scenarios`, `mpc_solves`, `controller_intervals`, `controller_daily_kpis`,
`controller_kpis`, paired daily metrics, oracle values, and controller comparisons.
Full open-loop plans, first issued commands, scenario weights/forecasts, forecast
clocks, failure reasons, safety events, stage objectives and boundary errors remain
available. Realized operating profit is auction revenue plus signed settlement,
less executed degradation, import costs and trading fees; it excludes planning
penalties and terminal inventory values.

Daily expected imbalance sums only first-action predictions, avoiding overlapping
horizons. Fallback intervals without a prediction are counted separately from
represented expected intervals. Maximum cumulative shortage is the largest
deficit in cumulative signed imbalance since local midnight. Zero-denominator
prices and ratios are null. SOC averages use interval durations and both energy
boundaries. Final inventory differences are reported without adding salvage to
realized profit.

Plots cover cumulative absolute imbalance, daily signed settlement, daily profit
differences, actual SOC, planned/executed power, imbalance versus price, oracle
value, midnight projection errors, profit versus volume, and solver runtimes.
Detailed per-day figures cover at most three delivery days, including source,
controller and comparison folders. `representative_days` may request fewer;
larger requests are capped at three. Reporting removes obsolete generated daily
figures beyond the selected days. CSV metrics and aggregate plots cover the run.
Actual Gurobi time and Python wall/build time remain distinct. MPC checkpoints
retain plans and outstanding actions under a separate policy identity. Individual
continuous MPC replay can resume; comparison orchestration currently requires a
fresh run and does not resume a partially completed comparison.
Rerunning a frozen comparison reuses its verified ledger and preserves the
original auction-source diagnostics; controller trajectories restart from the
same initial state rather than regenerating the common auction decisions.

## Reproduction and approved smoke

Use the existing `bess` environment without changing or installing packages.
The machine-bound Gurobi license requires the host execution context; sandbox
HostID 0 fails license matching.

```bash
PYTHONPATH=src python -m pytest -q tests/unit/test_realtime_mpc.py tests/integration/test_realtime_continuous.py
PYTHONPATH=src python -m bess_pv_trading experiment-realtime-mpc --config configs/full_experiments/realtime_frozen_comparison.yaml --check-only
PYTHONPATH=src python -m bess_pv_trading experiment-realtime-mpc --config configs/full_experiments/realtime_frozen_comparison.yaml
PYTHONPATH=src python -m bess_pv_trading experiment-realtime-mpc --config configs/full_experiments/realtime_continuous_smoke.yaml
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/mathematical_model.tex
```

The revised smoke covers Farm 1, 15–16 January 2026: all five policies with one
frozen ledger, then deterministic MPC in continuous operation with tracking as
reference. It uses a 0.75 MWh / 0.5 MW battery, 90% initial SOC, 50% final-study
target, no additional midnight floor, two solver/inference threads, existing preprocessing,
selected TCN weather conversion, current-interval realized PV disclosure to MPC,
and existing pinned NORD auction/settlement snapshots. Its run ID is
`farm1_2026_01_15_16_mpc_tcn_current_pv`. There is no training,
validation split, hyperparameter search or model selection. Report paired
imbalance, signed cash, profit, inventory, projection and runtime results without
claiming a two-day sample validates a market strategy. Do not automatically run
month-long, additional sensitivity, or other-farm studies.

Completed smoke results and numerical evidence are in
[the two-day results](results/realtime_mpc_2026_01_15_16/README.md). The separate
[full-January study plan](realtime_mpc_january_2026_study_plan.md) specifies the
all-auction deterministic-MPC versus feedback comparison. Its launcher,
`scripts/run_realtime_mpc_comparison.py`, prints the plan by default and starts
optimization only with `--run`. The completed January input-only preflight and
its input fingerprint are in
[the preflight record](realtime_mpc_january_2026_preflight.json).

Missing data is not downloaded or filled by this workflow. Existing nighttime
priors and frozen-forecast measurement fallbacks remain part of preprocessing and
provenance. Source measurements can contain hidden curtailment/faults; available
generation is not an independently measured unconstrained PV potential. Physical
protection is idealized interval-average feedback; intrainterval dynamics, ramps,
forecast calibration, uncertain initial SOC, and coordinated next-day reserve
targets remain outside this implementation.

### Optional grid-neutral curtailment recovery

`execution.mpc_curtailment_recovery: true` applies a physical feedback step
only to issued MPC actions. After the normal power, SOC and POI protection, it
reduces curtailment and decreases net battery discharge by the same AC power.
It reduces discharge first, then charges if needed, while respecting available
SOC headroom, the interval duration, charging efficiency and battery power.
The protected grid exchange is unchanged by recovery. The simulator records
`curtailment_recovered_mw`, executed controls, feedback adjustment and actual
SOC; the next decision sees the resulting state. Fallback grid tracking retains
its existing behavior. Auction boundary projections replay this rule with
eligible deterministic forecast PV and preserve outstanding actions without
additional MPC solves. The MPC optimizer itself is unchanged.

Recovery can increase wear, consume charging headroom or cause the final
executed SOC to differ from a planning terminal target, especially in the last
interval; report those outcomes. As with the existing tracker, use of interval
average PV for execution is an idealized retrospective physical-feedback
assumption, not proof of real-time availability of the interval average.

### Optional first-interval imbalance priority

The deterministic price-blind `imbalance_and_costs` policy can set
`realtime_mpc.first_interval_priority` to `weighted` or `lexicographic`.
`weighted` multiplies only the first interval's imbalance penalty by
`first_interval_weight`; the default mode `none` preserves previous behavior.
`lexicographic` first minimizes current expected absolute imbalance and then
minimizes the ordinary horizon objective with the first-stage incumbent plus
`first_interval_priority_tolerance_mwh` as a cap. Both stages share the total
MPC time budget. Only a first stage with solver status optimal is treated as
certified within its reported relative MIP gap and tolerances. A validated
uncertified incumbent can supply an action without horizon refinement; a failed
refinement retains the validated first-stage action. The realized first-interval
PV disclosure is controlled independently and remains a retrospective
assumption when interval-average realized PV is selected.

The price-blind `ensemble_mpc_adaptive` policy with `first_action_only`
recourse also supports `weighted`: the first-interval term is averaged over
the scenario weights, and only the first action is shared. Future paths then
have separate schedules, an optimistic path-revelation assumption. Realized
current-interval PV, when selected, is identical in all scenarios.
Lexicographic ensemble priority, oracle, economic and pure-imbalance variants
remain unsupported. Planning penalties are not realized settlement charges.
