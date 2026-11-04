# Continuous PV–BESS energy-market research extensions

The implementation compares physical MPC controllers with grid tracking inside
the existing continuous auction replay. Every controller carries its own actual
battery state and submits its own binding auction quantities. The new study
launcher supports **continuous optimization only**; it does not run frozen-ledger
experiments. The existing standalone frozen-ledger workflow remains separate.

The user has authorized extensive retrospective MPC-versus-tracking comparisons.
Loading this configuration still does not launch the entire matrix: the command
prints a concrete plan by default, and `--run` executes the selected cases.
This prevents an ordinary inspection command from starting a long batch. It is
an explicit execution selector, not another approval requirement.

## Implemented research directions

| Direction | Implementation and controlled comparison |
| --- | --- |
| Forecast economic MPC | `deterministic_mpc_economic` and `ensemble_mpc_economic` use explicit settlement forecasts, separately from the existing penalty and realized-price-oracle controllers. `causal_auctions` additionally replaces perfect-foresight auction prices with fitted forecasts. |
| Adaptive recourse and temporal dependence | `ensemble_mpc_adaptive` shares the current command and branches later controls only on disclosed PV histories. Raw trajectories, independently shuffled trajectories and ECC are explicit options. |
| Information versus auction access | A four-cell design combines latest/frozen-initial PV and price information with all four auctions/DA only. The reporting module estimates the interaction using matched assets and dates. |
| Forecast quality versus physical flexibility | Fresh per-farm configurations scale energy, power or the POI independently. Forecast scores remain separate from realized policy profit. |
| Conditional continuation value | A nominal forecast dynamic program supplies a piecewise-linear value of terminal stored energy. The main MPC represents that curve exactly; its construction remains an approximation. |
| Decision-aware scenario reduction | Weighted medoid selection uses configurable PV, ramp, cumulative deficit/balance, saturation, POI and monetary-exposure features, retaining original paths and transporting probability mass. |

Monetary CVaR is an additional controlled risk contrast. It differs from the
existing auction model's CVaR of absolute imbalance **volume**.

The principal implementation files are
`src/optimization/adaptive_mpc.py`, `src/scenario_research.py`,
`src/forecasting/energy_prices.py`, `src/forecasting/continuation.py`,
`src/simulation/energy_price_inputs.py`,
`src/simulation/energy_research_study.py` and
`src/reporting/research_study.py`.

## Formulation and interpretation

Let $q_t$ be the already committed MWh, $p_{s,t}$ the PV scenario,
$c_{n,t},d_{n,t}$ the AC charge/discharge command owned by history node $n$,
and $u_{n,t}\in[0,1]$ the PV utilization fraction. Each path satisfies

\[
g_{s,t}=p_{s,t}u_{n(s,t),t}+d_{n(s,t),t}-c_{n(s,t),t},
\quad b_{s,t}=\Delta t\,g_{s,t}-q_t.
\]

Battery energy follows the shared canonical efficiency/self-discharge equations.
All paths satisfy the SOC box, POI bounds, power bounds, binary charge/discharge
exclusivity and any configured terminal target. Xu cycle-segment memory is carried
on every conditional path when that degradation model is selected. The current
node is shared across every scenario. Only its first command is issued.

At time index $t$, a recourse node may depend on PV observations through
$t-\ell$, with observation lag $\ell\ge1$. Splitting occurs only at the
configured relative interval indices. Quantization limits the number of distinct
observed histories; it never uses future observations or a scenario's label.
The current implementation does not reveal simulated settlement prices along a
recourse path. This is an explicitly restricted PV-information policy.

With forecast settlement cash $R_s$, wear $D_s$, import fees $I_s$,
terminal value $V(E_{s,T})$, weights $w_s$ and optional imbalance regularizer
$\kappa$, the economic objective minimizes

\[
\sum_s w_s\left[D_s+I_s-R_s-V(E_{s,T})
 +\kappa\sum_t |b_{s,t}|\right]
 +\lambda\operatorname{CVaR}_{\alpha}(D_s+I_s-R_s).
\]

The economic imbalance regularizer defaults to zero independently of the legacy
500 EUR/MWh tracking penalty. Monetary risk excludes artificial planning
penalties and terminal salvage. It covers future physical operating loss, not
uncertain receipts from unpriced auction commitments. Forecast objective values,
terminal values and risk terms are never booked as realized cash.

Scenario controls and SOC trajectories are retained explicitly. The existing
auction shadow projection uses one labelled representative path; averaging
conditional charge/discharge commands would not define a physically valid policy.
The representative path is not a forecast of which scenario will occur.

The legacy common-control, additive-objective MPC is invariant to independent
per-lead permutations that preserve each time's joint PV/price samples. A unit
test checks this useful negative control. Adaptive recourse and pathwise risk
provide mechanisms through which temporal dependence can affect decisions.

For the legacy settlement-oracle controller, the default remains to proceed
to economic refinement only when stage one meets the configured `1e-4` MIP gap.
The optional `oracle_fallback_max_relative_gap` enables a gap-qualified
incumbent. The study includes explicit 1% fallback cases, and gives the oracle
imbalance stage and economic stage independent limits of 10 seconds each.
Outside those studies, `oracle_stage2_time_limit_seconds` opts into a separate
second-stage limit; otherwise the stages share `solver_time_limit_seconds`,
and `oracle_fallback_time_reserve_fraction` reserves part of that shared budget.
Stage one continues if the incumbent does not meet
the threshold. Refinement must respect the feasible incumbent's reconstructed
absolute-imbalance cap plus the configured budget; a failed or invalid
refinement retains that incumbent. Its lower bound and reconstructed relative
gap are recorded, and an approximate incumbent is never labeled a proven
minimum. For incumbent U and lower bound L, the possible excess over the true
minimum is bounded by U - L plus any declared extra imbalance budget (subject
to solver numerical tolerances). The gap convention is `(U - L) / U`.

An optional numerical recovery is configured with
`oracle_retry_numeric_focus` (1, 2 or 3) and optionally
`oracle_retry_feasibility_tolerance`. It is disabled when the focus is null.
If an economic-stage candidate fails the physical or imbalance-cap checks,
the controller discards that candidate and attempts one re-solve using the
remaining economic-stage budget. A retry tolerance can only tighten the active
solver tolerance. The retry records its trigger and stage timing, and it must
pass the same independent checks; if it fails or no time remains, the validated
stage-one plan is retained. This does not retry a missing incumbent or an
objective-component validation failure. The existing studies did not enable
this option, and selected numerical replays are not evidence of its broad effect
on closed-loop performance.
The `deterministic_oracle_numeric_retry` and `ensemble_oracle_numeric_retry`
study cases enable focus 3 and feasibility tolerance `1e-9` and compare against
their respective no-retry oracle cases. Each retains its own continuous
grid-tracking comparison. These cases are declarations for future runs; they
are not part of the already running 104-job January batch.

## Data, forecast training and retrospective scope

The default matrix uses the four configured farms and two short development
windows: 15–16 January 2026 and 2–3 August 2026. January was previously inspected;
neither window is presented as an untouched test set. August starts on the second
day because the available August ensemble archive starts on 1 August and the
auction-consistent convention uses the preceding day's 00 UTC cycle. The standard
configuration currently applies a six-hour weather availability assumption.

The physical execution trajectory comes from the reviewed
`data/farms/curated/reviewed_ghi_2025_to_2026_08` dataset. The loader preserves
user-reviewed targets, replaces retained daylight exclusions with the dataset's
historical-GHI pvlib reference, applies the configured night-zero policy, and
records each interval's source. These are retrospective counterfactuals where
synthetic, not independent AC measurements. The deterministic forecast weather
still comes from the farm Zarr and metadata Parquet inputs. January and August
use their own ensemble archives.
Actual auction and settlement prices continue to come from the DuckDB catalog.
No study command substitutes zero for a missing price or silently downloads new
data. Existing preprocessing, seasonal night policies, interpolation and flagged
fallbacks are inherited and retained in replay provenance. Asset ratings and POI
limits remain configured assumptions, not measured equipment specifications.

Retrospective weather/PV model inputs and explicitly disclosed oracle prices are
valid comparison inputs when labelled. The default auction planning uses the
same actual clearing curve for every PV member. Only `causal_auctions` changes
auction price information. Physical economic controllers use separately fitted
settlement forecasts. Existing grid tracking can react to actual interval PV,
while the default MPC current-PV input is the forecast; the outputs disclose this
information difference and any tracking fallback.

`configs/research/price_forecast_training_v3.yaml`,
`configs/research/price_forecast_extended_2024_training_v2.yaml`, and
`scripts/fit_energy_price_forecast.py` define the chronological forecasting
workflow. It compares seasonal persistence, standardized ridge regression on
calendar/price lags, and ridge with configured fundamentals. Ridge minimizes
squared error plus its coefficient penalty. A chronological calibration tail
supplies whole-day residual trajectories. Proper CRPS, point MAE and coverage
are evaluation quantities, not the training objective. The training workflow
reports all declared models rather than selecting the highest trading profit.
Its snapshot IDs, checksums, publication-clock assumptions and exclusions remain
part of the resulting artifacts.

The controller matrix names `ridge_price_only` artifacts by fold as an
explicit baseline choice and includes a `ridge_fundamentals` comparison.
Day-ahead training starts in 2024. The extended intraday models also use the
2024 hourly exports as coarse training targets, repeated on the quarter-hour
grid; these are not independent native quarter-hour products. Settlement
history starts in 2025. The October 2025, January 2026 and August 2026 folds
remain chronological, with a held-out calibration tail. Historical publication
clocks are assumptions, not proven vintage availability. These files must
actually exist. The launcher lists
missing artifacts and refuses to start the selected batch if any are absent.
Observed-price sidecars support later eligible lag updates; eligibility filtering
still occurs at prediction time. Exact historical publication evidence is not
inferred from download timestamps. January/August fits must not train on their
own evaluation targets, even for a retrospective study.

ECC requires a real calibrated-marginal NPZ with the schema validated by
`energy_price_inputs.py`; the matrix names a separate file for each farm and
window. It neither invents a calibration model nor treats sorted raw members as
evidence of calibration. ECC rearranges marginal samples according to raw
ensemble ranks, following [Schefzik et al.](https://arxiv.org/abs/1302.7149).
The fitted affine bias and spread parameters use January 1–14 reviewed targets and are
frozen for January 15–16 and August 2–3 evaluation. At replay time they apply
to the currently eligible raw member forecasts, so a new cycle or member count
is supported. Calibrated ECC and calibrated shuffle retain identical empirical
marginals, allowing a dependence comparison. Reviewed target creation is
retrospective; treating the target as available at interval completion is a
simulation assumption, not historical availability evidence. Results from the
prior raw-target calibration remain separately identified and should not be
reused as the reviewed-target calibration.

The continuation calculation conditions on nominal future PV, settlement prices
and known commitments after the current MPC horizon. It uses an explicit SOC
grid and throughput-wear approximation. Its terminal curve is not a learned
battery lifetime model or a value of unmodeled future auctions. Piecewise-linear
value-to-go functions are also used in storage arbitrage research, including
[Zheng, Jaworski and Xu](https://arxiv.org/abs/2108.06000); this implementation
does not reproduce that paper's Markov-price algorithm or claim its performance.

## Concrete study matrix and compute

The current default YAML declares **28 cases**, **4 farms** and **2 windows**: **224 jobs**.
Each job runs grid tracking plus one target controller for two delivery days.
This gives **448 controller trajectories**, **43,008 MPC decisions** and
**3,392 nominal auction-submission solves**. The DA-only cases have fewer auction
solves. Counts are computed from the configured market calendar, including
92/100-product DST days, rather than assuming 96 products per date.

The cases cover deterministic/ensemble penalties, deterministic/ensemble
settlement-price oracles, deterministic/ensemble forecast economics, adaptive raw
scenarios, price-only and fundamental causal auction prices, shuffled/ECC/reduced scenarios, continuation,
monetary risk, the information/access factorial, and independent energy/power/POI
contrasts. Calibrated forecasts are also crossed with each asset contrast, and
calibrated ECC is compared with a shuffled version with identical marginals.
The two additional cases compare budgeted numerical retry with its matched
deterministic or ensemble oracle baseline. Previously launched batches retain
their pinned 26-case configuration and their own manifest counts.
This inventory is not a hyperparameter search. Broader January/August
evaluation requires editing the explicit windows and reviewing the resulting
counts and data coverage.

Execution uses spawn processes with the requested **42 workers**, reduced to the
number of selected jobs when that is smaller. Each worker has one native-library
thread and one Gurobi thread by default. `--workers` and `--solver-threads` permit
smaller or differently allocated batches within the 48-core machine budget.
There is no GPU work in this matrix. The default physical MPC solve limit is
10 seconds and the auction limit 120 seconds, with relative MIP gap `1e-4` and
no absolute gap. These are per-solve budgets; the whole matrix can take many
hours and should be launched in deliberate subsets. Before broad runs, inspect
machine usage because these compute resources are shared.

`configs/research/deterministic_continuous_validation.yaml` extends the legacy
deterministic penalty/oracle comparisons to 1–7 October 2025 and the full month
of May 2026. Its 16 jobs cover four farms and 38 delivery days per controller:
29,184 physical MPC decisions and 2,432 nominal auction solves. These cases need
no ensemble archive and no fitted settlement model. Earlier 2025 windows are
not enabled by silently extending the current calendar's validated start date.
2024 data are intended for price-model training; this configuration does not
invent a 2024 PV/BESS evaluation dataset.

Workers report ordinary failures per job. A crashed worker produces an executor
failure rather than a silently lost task. The project's Python 3.14 supports
immediate termination of running workers on launcher interruption; older Python
versions can cancel queued tasks but may leave already running tasks completing.
The optional `--license-file` selects an existing Gurobi license for workers.
Host-bound licensing may require execution outside a sandbox that masks host
identity. The launcher never changes a license or the conda environment.

```bash
# Display one bounded comparison without reading historical arrays or solving.
python scripts/run_energy_research_study.py --case ensemble_penalty --farm 1 --window january_development

# Validate cached inputs and all forecast requests; fit nothing and solve nothing.
python scripts/run_energy_research_study.py --case ensemble_penalty --farm 1 --window january_development --check-only

# Execute the selected continuous comparison under the existing authorization.
python scripts/run_energy_research_study.py --case ensemble_penalty --farm 1 --window january_development --run-id farm1_ensemble_validation_v1 --run

# Select several controllers/farms explicitly in the same new batch.
python scripts/run_energy_research_study.py --case deterministic_penalty --case ensemble_penalty --farm 1 --farm 345 --window january_development --run-id two_farm_penalties_v1 --workers 2 --run
```

Use `--run-id` or a new `energy_research_study.run_id` for a different selection or rerun.
The launcher atomically creates the batch directory, holds a process lease,
disables underlying overwrite behavior and saves each resolved case config.
Preflight creates no replay outputs. All enabled model inputs are mandatory;
the launcher never falls back to an oracle to make a forecast case run.

## Outputs and scientific comparisons

The batch manifest records exact case counts, source hashes, configuration hash,
artifact requirements and each job's status. Every replay retains its input
fingerprint/provenance, actual auction ledger, interval dispatch, conditional MPC
plans, realized cash, solver status, runtime and fallback evidence. The research
configs write completed days of horizon diagnostics to checksummed Parquet
partitions, keeping current controller state in memory. Use
`bess_pv_trading.reporting.mpc_history.iter_mpc_history` to stream those records;
the default outside these studies remains the existing in-memory output mode.

Auction forecast cases currently supply point price curves to the auction
optimizer. Settlement-price uncertainty uses residual trajectories. The library
does not yet learn a joint multi-auction spread distribution, and this study
therefore cannot establish the value of that unimplemented extension.

`daily_metrics.csv` joins completed jobs. Each case receives a Markdown/CSV/JSON
comparison and plots against **its own** continuous grid-tracking baseline.
Declared cross-case baselines receive separate contrasts, and complete matched
information/access cells receive a factorial report. When all four relevant
cases are present, separate energy, power and POI reports quantify how the
calibration effect changes with physical flexibility, after verifying that the
only physical differences are the declared ratings. These comparisons include
changes to later auction quantities caused by a different controller's state;
they are full continuous-policy effects, not fixed-commitment effects.

Report realized operating profit, signed settlement, absolute imbalance, wear,
throughput, runtime and fallback rate alongside proper forecast scores. Resample
whole calendar-date blocks jointly across farms; four farms exposed to shared
prices and weather are not independent repetitions. Missing dates remain explicit.
The two-day development windows cannot support precise tail-risk or confidence
claims: reporting flags insufficient calendar blocks and sparse empirical tails.
No result should be described as an economic improvement merely because its
forecast score improves, or as a better forecast merely because one trading
policy earns more profit.

Continuous **intraday order-book trading** remains outside this implementation.
Here, “continuous” means the existing chronological DA/IDA auction and physical
control replay. Order-book trading would require timestamped quotes or trades
and an explicit execution/liquidity model; auction clearing prices do not supply
those observations.
