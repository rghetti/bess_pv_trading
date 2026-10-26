# PV and battery trading in the Italian electricity auctions

This repository studies one photovoltaic farm and an AC-coupled battery sharing a point of interconnection (POI). It prepares farm and market data, converts rolling weather forecasts into PV power with pvlib, optimizes energy positions in MGP and MI-A1/2/3 with Gurobi, executes the plans against realized PV through a feedback controller, and settles delivery deviations afterwards.

The historical workflows use **one-day planning horizons in the NORD zone, with perfect auction-price information and one deterministic weather-forecast member**. Single-day and independent-day participation studies refresh the PV forecast before each active auction and make four auction decisions when all auctions take place. A separate continuous-operation replay runs consecutive delivery days in chronological order: each solve plans one delivery day, while actual battery state, submitted commitments, degradation history and monthly fee volume carry between days. It does not combine multiple delivery days into one MILP. Separate hindsight optimizations use realized PV to compare the feedback trajectory with a perfect-information schedule and to measure the incremental operating value of a battery against a PV-only plant. Price-forecast experiments and their optional stochastic optimizer-input adapter are archived; new causal price forecasting is future work.

The full equations, information structure, execution policy, and economic comparisons are in **[the mathematical model](docs/mathematical_model.pdf)**, with editable **[LaTeX source](docs/mathematical_model.tex)**. This document replaces the retired implementation-status LaTeX/PDF. Numerical results belong to each run's saved configuration and KPIs, rather than a fixed result in this README.

## Start here

Run commands from the repository root in the existing `bess` environment:

```bash
conda activate bess
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m bess_pv_trading --help
```

On the current workstation the interpreter is `/home/rghetti/miniforge3/envs/bess/bin/python`. The project runs directly from the checkout; its root compatibility package also exposes the original audit tools. There is no package-install step in this workflow. Runtime dependencies include NumPy, pandas, SciPy, xarray/Zarr, PyArrow, PyYAML, DuckDB, pvlib, Matplotlib, and, for optimization, `gurobipy` with a working Gurobi license. LaTeX tools are optional for rebuilding the model document. Do not change the shared environment or its license to run a study.

For the current historical workflow, first check that the private farm stores and imported market snapshots are present. Validation does not download data:

```bash
python -m bess_pv_trading validate-config --config configs/full_experiments/gme_oracle_2025-01-01_farm910_perfect_pv_comparison_terminal_soc_50.yaml
```

Use the reusable [`gme_day.yaml`](configs/full_experiments/gme_day.yaml), or save a new study under `configs/full_experiments/` importing the standard directly:

```yaml
include: ../standard.yaml
farm_id: 345

gme_oracle_experiment:
  study_id: my_january_study
  delivery_day: '2025-01-01'

market:
  valid_from: '2025-01-01'
  valid_to: '2025-01-01'

pv:
  direct_exceeds_ghi: clip_and_flag

settlement:
  enabled: true

solver:
  feasibility_tolerance: 1.0e-08
  integrality_tolerance: 1.0e-08

experiment_outputs:
  run_id: my_january_study_farm${farm_id}
```

All remaining battery, SOC, market, execution, degradation, comparison and output settings come from [`configs/standard.yaml`](configs/standard.yaml). To inspect the complete effective YAML or switch farms without editing the study:

```bash
python -m bess_pv_trading resolve-config --config configs/full_experiments/gme_day.yaml --farm-id 345
```

Farm IDs `1`, `345`, `901` and `910` select their own weather, measured PV, metadata and hypothetical BESS sizing. `plant.farm_id` and `forecast.farm_id` follow the root `farm_id`. Automatic POI ratings use each farm's declared AC nameplate capacity as an explicit proxy, with per-farm measured ratings configurable. Automatic battery sizes come from `farms.<id>.battery`: farm 1 uses 0.75 MWh / 0.5 MW, farm 345 uses 1 MWh / 0.5 MW, farm 901 uses 2 MWh / 1 MW, and farm 910 uses 5 MWh / 2.5 MW. Maximum charge/discharge power is symmetric. All inherit 90% AC-to-AC round-trip efficiency and throughput degradation. Numeric experiment-level battery fields override automatic sizing. See [the configuration guide](configs/README.md) for selection rules and the integrated fitted-model versus PVLib study configs.

Experiments default to replacing previous outputs for the same ID and run scope (`experiment_outputs.overwrite_existing: true`). Keep `resume: false` for a fresh rerun after changing parameters; explicit resume preserves and validates checkpoints. Output roots remain configurable, allowing separate comparisons without changing experiment IDs. Single-day reruns preserve other farm/day scopes.

Then run it offline:

```bash
python -m bess_pv_trading experiment-gme-oracle --config configs/full_experiments/my_january_study.yaml
```

Existing run directories cannot be overwritten. Change `experiment_outputs.run_id` for a new run. Configuration validation checks declared settings; the experiment additionally checks actual data coverage, product resolution, forecast eligibility, settlement alignment, and solver feasibility. It never silently substitutes synthetic market prices.

## What happens in a full experiment

1. Read the selected day's actual NORD auction prices from `data/api/market.duckdb`, with source snapshot/checksum provenance. Require complete native product curves for active auctions. Explicitly recorded or configured cancellations remove the corresponding auction and its trade variables; an unexplained gap is an error.
2. Select the latest weather forecast available at the first decision, convert it into AC PV power, and prepare realized AC production for execution. Declared missing-value policies and every fallback remain recorded.
3. Immediately before each active auction, refresh the remaining PV forecast, carry the current battery state and all prior commitments, and optimize the remaining horizon. Only that auction's quantity is submitted. Later-auction quantities in the solve are revisable plans.
4. At price publication, book each submitted quantity times the actual clearing price once. The current auction-only GME replay performs no additional optimization at publication or at each physical interval.
5. At delivery, the controller tries to track the cumulative committed grid position using realized interval-average PV. It adjusts charging/discharging and PV curtailment within SOC, battery-power, and POI limits. Residual mismatches are retained.
6. Apply the actual Terna NORD quarter-hour settlement curve after physical execution. Compute realized profit from auction cash flow, signed imbalance cash flow, executed battery wear, and any configured import fees.
7. Solve a full-day perfect-information PV+BESS problem using the same effective realized PV and actual auction prices. If enabled, solve the same asset counterfactual with zero battery capacity and power. Save schedules, comparisons, KPIs, model diagnostics, and plots by file type.
8. If enabled, simulate the remaining day with each pre-final auction plan frozen, using its future orders and the same physical feedback. Save full-day profit differences, terminal-energy differences and per-baseline ledgers without additional optimization solves.

IDA3 closes at 10:00 Italian time and plans only the 48 quarters from 12:00 to midnight. Its forecast is selected at the configured submission cutoff (09:55 with the standard five-minute buffer). Previously submitted morning quantities and the earlier dispatch commands remain in force until noon. The planner projects noon SOC and degradation from the latest observed state using those retained commands, the configured controller and cutoff-available PV forecasts. Actual execution continues with realized PV; its noon SOC can differ from this projection. Each solve records the delivery boundary and the projected locked prefix.

The shared default is `transaction_fees.mode: gme_real`: the progressive GME MPE variable tariff applies to gross purchases plus sales, using the submission month. The first 20,000 MWh/month are exempt; subsequent marginal rates are €0.04/MWh through 1,000,000 MWh, €0.03 through 10,000,000 MWh and €0.02 thereafter. The configured December 2024–December 2026 window is marked `assumed_unchanged`, with the [GME source](https://gme.mercatoelettrico.org/it-it/Home/Accesso-ai-Mercati/Elettricita/MercatiElettrici/CorrispettiviSettlement/CorrispettiviElettricita) retained. The default represents a standalone simulated operator with no earlier/external volume, starting on 1 December 2024; continuous runs carry submitted monthly volume. For a larger operator portfolio, set `operator_scope: operator_with_exogenous_volume` and supply its known `initial_state`. Fixed membership/annual/ACER charges remain outside this variable tariff. Set `mode: none` explicitly for a fee-free comparison. Saved runs retain their recorded fee configuration; use a new run ID when changing fee assumptions.

With all four auctions active, this means **four auction-decision solves, one perfect-information PV+BESS solve, and one optional PV-only solve**. The replay's `optimizer_solves`/`reoptimization_solves` counts describe the auction replay; counterfactual diagnostics are saved separately. This is different from synthetic configurations that optionally replan physical controls every interval.

Every experiment saves `json/optimization_timing.json`. It records Gurobi's native `Runtime` in seconds for each solve, with count, total, mean, minimum, maximum and population standard deviation (`std_ddof: 0`). Unsuccessful solves are included; a repeated Gurobi optimize call contributes its own runtime. `by_category.replay` covers the auction/control solves, while perfect-information PV+BESS and PV-only models have separate categories. Python solve wall time and model-build time have their own statistics. The aggregate is also included in `json/kpis.json` for full GME experiments. Frozen-plan simulations contribute no optimization time.

Participation studies save the same report at the study's `json/optimization_timing.json`, with pooled solve statistics, `experiment_total_runtime_seconds` across daily profile cases, and `experiment_replay_runtime_seconds` for comparing trading strategies without their benchmark costs. `by_profile` reports the same experiment statistics for every integrated profile and the sequential baseline. Each case appears once after resuming; existing raw timing diagnostics can supply older cases. Missing timing data remain explicit and are excluded from statistics. Partial studies report their completed cases.

To refresh timing reports for an existing study (including a process started before timing aggregation was added), run `python -m bess_pv_trading report-optimization-timing --config configs/full_experiments/gme_participation_2025-01.yaml` from the repository root. It reads saved diagnostics and writes JSON without running optimizations or downloading data. A running study reports its completed checkpoint; rerun the reporting command after the study finishes for the full aggregate. A single GME experiment config is also supported when it has an explicit run ID.

## Auction participation and average performance

Choose any non-empty participation subset in a single-study YAML:

```yaml
gme_oracle_experiment:
  participating_auctions: [IDA1, IDA2]
```

Omitting this field (or using `null`) selects all configured auction datasets. Chronological gate order is preserved. Unselected auctions have no price-data requirement, order variables, submission solve or transaction fees. A selected cancelled auction is skipped under the existing cancellation rules. The single-run example is `configs/full_experiments/gme_oracle_2025-01-01_farm910_ida1_ida2.yaml`. Single-auction cases, including IDA3 alone, are supported.

If every selected auction is documented or configured as cancelled on a day, the profile still contributes that full day to the comparison. There are no trades, auction/control solves or auction fees; zero-position grid feedback charges available PV within the SOC box and curtails the remainder. Executed wear and actual imbalance settlement still enter profit. No initial auction objective exists, so expected-profit KPIs are null. A configured terminal target is not enforced by this controller. Missing or partial price data without cancellation evidence remain errors.

If participation starts after delivery begins, the prefix through the first auction's delivery start uses zero contracted delivery and grid feedback with zero planned battery commands: PV can charge the battery within its SOC box, and remaining surplus is curtailed. For IDA3 alone, this continues until noon. It earns no uncontracted export revenue in this prefix. Its executed/projected SOC and wear carry into the first auction solve. No additional auction optimization is inserted before the first selected gate. The initial expected daily profit includes the already executed/projected prefix costs; the first solve's residual objective is retained separately in the strategy summary.

For a matched-day comparison, configure a `participation_study` and run:

```bash
python -m bess_pv_trading experiment-participation --config configs/full_experiments/gme_participation_2025-01-01_03.yaml
```

For all 31 independent delivery days of January 2025, use `configs/full_experiments/gme_participation_2025-01.yaml` instead. It runs 465 integrated profile/day cases and 31 sequential all-auction cases, with the current shared settings and a fresh output ID. Initial and terminal planning SOC are 50%; documented cancelled auctions remain in the common-date comparison.

`gme_oracle_experiment.auction_planning_mode` selects `integrated` (default) or `current_auction_only`. The latter creates trades only for the auction currently closing; earlier submissions remain fixed, and future auctions contribute no decision variables or revenue. It still optimizes battery dispatch and SOC through the end of delivery, refreshes PV at every selected gate, and executes the same feedback policy. Current-auction actual clearing prices are used under the study's price-oracle convention. When a later sequential solve is infeasible under `previous_plan`, it retains the previous dispatch and submits zero new trades.

`participation_study.sequential_all_auctions_baseline: true` adds one sequential baseline on all configured available auctions; the integrated all-auction profile must also be selected. The flag defaults to false and is enabled in the monthly YAML. The orange **Sequential (all auctions)** row sits directly below its integrated counterpart in the comparison plot. Its matched-day KPIs and integrated-minus-sequential profit differences are in `csv/sequential_daily_results.csv` and `json/sequential_baseline.json`. `csv/participation_comparison.csv` contains the plotted rows. These baseline results do not alter the ranking of the 15 participation subsets.

That monthly configuration enables `execution.infeasible_auction_policy: previous_plan`. An updated forecast can make exact delivery infeasible for already committed products before the next auction's first tradable interval. On a solver-confirmed infeasible auction decision, this policy submits that auction's quantities from the last feasible deterministic snapshot and retains its dispatch commands. Earlier commitments stay fixed; the ordinary physical controller executes against realized PV and settles unavoidable imbalances. The failed forecast attempt and reused plan source are recorded separately, without claiming a new feasible objective. It adds no recovery optimization. The default policy is `error`; missing previous plans, ambiguous stochastic snapshots and other solver failures still stop. `execution.save_infeasibility_diagnostics: true` saves IIS conflict reports. Per-run KPIs retain decision failures and attempt counts; the comparison tables include fallback-day and failed-decision counts.

`participation_study.resume: true` (enabled in the monthly YAML) verifies the same saved configuration, auction dispatch scope, completed child manifests, input fingerprints and recorded profits before reusing profile/day checkpoints. Results from before the IDA3 noon-scope correction require a fresh run ID; the supplied January YAML selects a new directory. Incomplete child directories are retained under `diagnostics/incomplete_runs/` and rerun; changed configurations are rejected. Successful completed cases are preserved after interruption. The numerical execution handoff floors solver-tolerated negative command roundoff to zero only after independent solution validation.

```yaml
participation_study:
  candidate_auctions: [DA, IDA1, IDA2, IDA3]
  required_auctions: []
  profiles: null
  delivery_days: ['2025-01-01', '2025-01-02', '2025-01-03']
  output_root: outputs/participation_studies
  run_id: my_participation_comparison
  plots: true
  perfect_foresight_reference: true
  sequential_all_auctions_baseline: false
  per_profile_plots: false
  save_model_files: false
  profit_tolerance_eur: 0.000001
```

Four optional auctions give **15 non-empty profiles**. Set `required_auctions: [DA]` for the **eight** profiles with DA mandatory and each IDA independently optional. Or give explicit `profiles`, for example `[[IDA1, IDA2], [IDA3], [DA, IDA3]]`. An empty participation profile is not modeled. The study dates, asset, execution, degradation, settlement, fees and solver settings remain YAML-controlled. The supplied three-day study disables PV-only/frozen-plan comparisons and per-profile plots/models to limit repeated artifacts; those features can be enabled through their normal configs.

Every profile starts each delivery day with the same configured battery and operator fee state. Days are independent experiments; SOC and monthly traded-volume history are not automatically carried from one study day to another. The initialization forecast cutoff is shared before the earliest candidate gate/day start. Delivery grids, effective realized PV, initial PV forecasts, shared actual auction-price curves, available per-auction PV input snapshots, settlement data and common asset/settings are fingerprinted and checked for equality across profiles and the sequential baseline. Input snapshots include failed attempts, independently of any fallback plan retained for execution. The batch resolves native auction-product durations separately for each date through the market adapter. Later forecast updates occur only at selected auction decisions. All profiles must complete the same requested dates with actual settlement; missing/invalid data fail the study instead of producing averages over different available dates.

Results live under `outputs/participation_studies/<run_id>/`: `csv/daily_results.csv`, `csv/profile_ranking.csv`, `csv/daily_best_profiles.csv`, `json/manifest.json`, the effective config, and `plots/participation_profiles.png`. Individual replay artifacts remain under `runs/<profile>/farm_<id>/<day>/`. Ranking uses **mean realized full-day net profit** after settlement, executed wear, import and transaction fees. It also records total/minimum/maximum profit, median, 10th/90th percentiles, daily standard deviation, negative-profit and zero-active-auction day counts, wins including tolerance ties, paired differences to the largest selected profile, and mean final SOC/energy. Chart whiskers are descriptive daily standard deviations. The mean of daily winners is not a fixed participation strategy, and the best mean on these dates is not an established optimum on unseen dates. Ending inventory remains unvalued and must be examined alongside cash profit.

The comparison adds a gold **Perfect foresight (all auctions)** reference above the ranked profiles by default. It averages full-day perfect-information `ObjVal` values on the identical dates and effective realized PV, participating in all configured auctions that remain available, with the same assets, costs and configured terminal SOC. The existing all-auction profile's benchmark is reused; if that profile is absent, one additional perfect-information solve per day supplies the reference without another auction replay. `participation_study.perfect_foresight_reference: false` disables it. Its daily objectives and summary are saved in `csv/perfect_foresight_daily.csv` and `json/perfect_foresight_reference.json`; profile tables also report objective-minus-realized-profit differences. The reference is displayed separately from profile ranks: the feedback profits include actual imbalance settlement and may finish at different SOC values, so the reference objective is not a guaranteed ceiling on those cash profits.

## Continuous optimization across delivery days

Run the separate January workflow from the repository root:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-continuous --config configs/full_experiments/continuous_optimization_2025-01.yaml
```

`continuous_optimization.start_date` and `end_date` are inclusive local delivery
dates. The January config covers 1–31 January 2025, supplies **90% SOC at
1 January midnight**, and begins with the DA/IDA1/IDA2 auctions on 31 December
for that first delivery day. December realized production is unnecessary:
the first midnight state is an initial condition. The study stops after the
last quarter on 31 January, at local midnight on 1 February, and submits no
orders for delivery outside its configured dates.

The horizon of each optimization remains one delivery day. During delivery day
`d`, the event replay can submit IDA3 orders for `d` and next-day DA, IDA1 and
IDA2 orders for `d+1`. At each next-day gate, the planner takes the latest
observed battery state and projects
the retained current-day commands through midnight using the forecast available
at that submission cutoff and the same grid-tracking controller. This projected
midnight state initializes the next-day solve; future realized SOC is never an
input. Submitted orders remain fixed. Current-day IDA3 only replaces dispatch
from noon onward and cannot change the next day's submitted DA position. Actual
SOC, executed degradation history and operator transaction-volume state carry
continuously across midnight. Fresh forecasts remain subject to the configured
UTC availability lag, currently six hours.

There are **three independent continuous trajectories** by default: integrated
all-auction planning, sequential `current_auction_only` planning, and an
integrated perfect-PV reference using the same daily horizon and actual prices.
Select trading models with `continuous_optimization.planning_modes` and disable
the additional reference with `perfect_pv_reference: false`. By default,
`perfect_pv_reoptimize_at_each_auction: false` solves the reference only at DA
for each delivery day (or the first available auction if DA is cancelled).
It retains that day's dispatch and submits the saved intraday orders at their
actual gates, with ordinary publication, fees and settlement. Set the flag to
`true` to solve the perfect-PV reference again at each active auction.
Each trajectory
starts from the same configured initial state and thereafter uses its own
execution state. `state.terminal_soc_percent: 50` constrains planned SOC only
at the final study boundary; intermediate days have no terminal target or
inventory salvage. `null` disables that final constraint. Realized final SOC
is reported separately, since feedback execution can differ from the plan.

Results are saved under `outputs/continuous_optimization/<run_id>/` with per
strategy `csv/`, `json/` and `plots/` folders. They include interval execution,
daily/monthly settled profit, accepted trades, forecasts and dispatch plans,
projected-versus-realized starting states, degradation, imbalances, final
inventory and paired strategy differences. Study-level comparison tables and
plots keep perfect-PV continuous profit separate from the objectives of
overlapping auction solves. Native Gurobi timing reports include individual
attempts, mean/min/max/std, strategy/day summaries and separate reference costs;
objective values are never summed across overlapping horizons.

If a later solve is confirmed infeasible and
`execution.infeasible_auction_policy: previous_plan` is enabled, integrated
planning retains the last feasible delivery-day plan and submits its saved
quantity for the current auction. `current_auction_only` retains its physical
commands and submits zero new quantity. Earlier submissions remain fixed in
both modes. The failed solve and fallback source are recorded; no recovery
optimization is added. With the default `error` policy, the experiment stops.

Study-level SOC prediction figures are separated by solved strategy:
`plots/midnight_soc_prediction_errors_<strategy>.png` and
`plots/ida3_noon_soc_prediction_errors_<strategy>.png`. Only strategies present
in that run are plotted. Each strategy's own `plots/` folder retains the same
figures without the filename suffix.
In daily figures, "Scheduled battery (active plan)" is the command actually
in force in each interval: normally IDA2 before noon and IDA3 afterwards for
reoptimized trajectories, or the retained DA plan for the default perfect-PV
reference. Battery panels use the configured symmetric power rating as their
axis range so numerical roundoff around zero remains visually negligible.
`plots/daily_imbalances_by_strategy.png` compares integrated and current-auction-only
planning in separate panels, with daily surplus and shortfall energy in MWh
aggregated from the quarter-hour realized imbalances on the same vertical scale.

The January YAML enables `continuous_optimization.resume`. A resumed run checks
its saved configuration and input fingerprints, restores a completed midnight
checkpoint and replays any work after that boundary. Restored state includes
submitted and pending commitments, physical/aging history and fee state.
Discarded solver attempts remain in `json/optimization_attempts.json`, marked
as rolled back, and remain included in actual runtime statistics. Change the
run ID for a new study. All historical inputs are local; this command makes no
API calls. Plot/model switches come from `experiment_outputs`, and farm and
asset parameters use the same shared configuration as other experiments.
The continuous controller currently requires interval-end telemetry
(`source.telemetry_latency_minutes: 0`); preflight rejects a nonzero telemetry
delay. This is separate from the forecast availability lag.

## Files and configuration

```text
src/
  data/             DuckDB catalog, providers, workbook imports, farm adapters, data policy
  forecasting/      pvlib conversion and PV-output calibration
  optimization/     shared MILP builder, prepared-input adapters, degradation, validation
  simulation/       auction events, immutable commitments, feedback execution, settlement
  reporting/        figures, output organization, legacy evidence reporting
configs/
  api_experiments/   acquisition, provider diagnostics, workbook imports
  price_experiments/ retained market-import example; prior price studies are archived
  full_experiments/  complete historical/synthetic studies and prepared-input examples
  standard.yaml    common parameters and feature flags for every workflow
  data_schema.yaml  data-schema entry point importing the standard
  farms_nord.yaml   selected-farm PV preparation overrides
bess_pv_trading/     preserved audit tools and checkout import compatibility
data/               private inputs, DuckDB snapshots, reusable derived data; ignored by Git
outputs/            run-specific results; ignored by Git
docs/               mathematical model and focused technical notes
tests/              unit/integration tests and explicitly synthetic fixtures
```

YAML supports one `include`, resolved relative to the including YAML file. Nested mappings merge recursively; lists and scalar values replace inherited values. Data/output paths are generally repository-relative, so use the repository root as the working directory. Every shipped configuration imports `configs/standard.yaml` directly and overrides only its differences. References such as `${farm_id}` resolve after the final overrides; explicit `null` is preserved. Use `resolve-config` to inspect the effective YAML. Runs save that effective configuration. See [the configuration guide](configs/README.md).

Useful full-study entry points:

| Configuration | Purpose |
| --- | --- |
| `gme_day.yaml` | Reusable single-day study for any selected farm; dates and parameters are YAML-controlled |
| `continuous_optimization_2025-01.yaml` | Consecutive January delivery with 90% initial SOC, continuous state/commitments, and a 50% planned final SOC |
| `gme_oracle_2025-01-01_farm910_perfect_pv_comparison_terminal_soc_50.yaml` | Current January study with a 50% planned terminal SOC, perfect-PV comparison, and optional PV-only baseline |
| `gme_oracle_2025-01-01_farm910_reoptimization_value.yaml` | Isolated January 1 study of full-day profit with plans frozen after each pre-final auction |
| `gme_oracle_2025-01-01_farm910_perfect_pv_comparison.yaml` | Same comparisons with the inherited free terminal SOC |
| `gme_oracle_2025-01-01_farm910_quarter_hour_ida.yaml` | Native hourly DA / quarter-hour IDA historical replay |
| `synthetic_small.yaml` | Reproducible synthetic replay comparing four trading strategies |
| `synthetic_xu.yaml` | Synthetic segmented depth-dependent aging example |
| `synthetic_dst_spring.yaml`, `synthetic_dst_fall.yaml` | Synthetic 23-hour and 25-hour delivery days |
| `prepared_optimization.example.yaml` | Initial point-price planning from complete prepared CSV forecasts |

These files are under `configs/full_experiments/`. The prepared examples deliberately contain `null` inputs that must be supplied. Earlier experimental YAMLs remain reproducibility settings; their filenames do not establish which data or assumptions were used in an old saved run.

### Battery, POI, and initial/terminal state

For the GME study, configure the following in your study YAML:

```yaml
farm_id: 910
plant:
  poi_import_mw: auto
  poi_export_mw: auto

battery:
  capacity_mwh: 5.0
  max_power_mw: 2.5
  soc_min_percent: 10.0
  soc_max_percent: 90.0
  efficiency_convention: combined
  combined_charge_efficiency: 0.97
  combined_discharge_efficiency: 0.97
  self_discharge_per_hour: 0.0

state:
  initial_soc_percent: 50.0
  terminal_soc_percent: 50.0  # null omits the terminal equality

solver:
  threads: 10
  time_limit_seconds: 120.0
  mip_gap: 0.0001
  feasibility_tolerance: 0.00000001
  integrality_tolerance: 0.00000001
  seed: 17
  output_flag: false
```

These are the current farm-910 study assumptions, not verified equipment ratings. The selected farm has 6.638 MW AC nameplate. `max_power_mw` is one symmetric AC charging/discharging rating. SOC percentages refer to nominal internal energy capacity, so changing capacity automatically rescales the SOC limits and GME initial/terminal energy. Absolute MWh SOC limits and separate charge/discharge or battery-inverter power limits are rejected. Internal solver state and prepared-input/synthetic boundary states still use explicitly named MWh fields.

Battery charging and discharging can contribute to grid imports/exports as well as absorb PV. Only PV can be curtailed. The battery energy transition applies charging and discharging efficiencies exactly once. Alternatively choose `converter_and_cell` and provide its separate converter/cell efficiency fields; do not mix the two conventions. There is no additional independent battery inverter power limit.

Every auction trade, cumulative contracted position after each auction, and physical grid power uses the configured directional POI limits. There is no separate 1 MW commercial cap. Individual trades can offset earlier trades, so trading between auctions can generate spread revenue in both the PV-only and PV+BESS cases.

The terminal equality constrains optimization plans. The realized feedback controller enforces physical limits and grid tracking, rather than a separate terminal-SOC objective; it may finish away from the target after forecast errors. The perfect-information schedule obeys its terminal equality. Compare terminal energy as well as profit when interpreting results.

Solver optimality uses a **relative MIP gap of `1e-4` (0.01%)** by default. Do not configure an absolute MIP gap. Status, objective, achieved relative gap, runtime, and model size are saved with each solve.

`feasibility_tolerance` and `integrality_tolerance` map to Gurobi's numerical `FeasibilityTol` and `IntFeasTol`; the historical study sets both to `1e-8` so tiny binary-mode leakage does not violate the independent physical checks. These numerical tolerances are distinct from the relative optimality gap.

### Trading, execution, and comparisons

```yaml
forecast:
  reoptimization_pv_policy: latest_available_forecast_at_each_reoptimization

trading:
  optimize_imbalances: false
  imbalance_penalty_eur_per_mwh: 0.0
  grid_import_fee_eur_per_mwh: 0.0

execution:
  dispatch_policy: track_committed_grid

experiment_outputs:
  plots: true
  save_model_files: true
  pv_only_baseline: true
```

`optimize_imbalances: false` omits optimized imbalance variables and requires forecast grid energy to equal the contracted position in every planning interval. Actual deviations are then an execution/settlement outcome. The general formulation also has an optional optimized-deviation mode, which is not used by the current farm-910 experiment. Curtailment has no explicit cost in the current objective.

The alternative PV policy is `freeze_initial_forecast_for_unobserved_delivery_intervals`. The alternative execution policy is `follow_battery_plan`, with the same physical protection. The GME workflow always disables additional physical-control optimizations, retaining one solve per active auction.

Set `experiment_outputs.pv_only_baseline: false` to skip the extra PV-only solve, its files and comparison figure. Its objective and incremental-value KPI fields become `null`, with `pv_only_baseline_enabled: false`. The perfect-information PV+BESS-versus-feedback comparison still runs. The switch defaults to `true`; the comparison YAML declares it explicitly.

### Value of later reoptimization

The optional frozen-plan experiment measures how much later auction replanning adds relative to keeping an earlier plan. Configure it independently of the PV-only comparison:

```yaml
reoptimization_baseline:
  enabled: true
  freeze_after_auctions: null  # All active auctions except the last.
  plots: true
```

Set `enabled: false` to skip it. To select particular freeze points, use a list such as `freeze_after_auctions: [DA, IDA1]`. The last active auction is always excluded, and cancelled auctions generate no baseline. An empty selection generates no baseline. The chart also respects the master `experiment_outputs.plots` switch. This feature defaults to disabled when the section is absent; the comparison configs enable it explicitly.

For each freeze point, earlier submitted quantities and the executed prefix stay identical to the normal replay. All remaining auction quantities and requested battery/curtailment commands come from that snapshot. The frozen future orders are still submitted at their scheduled gates and priced at actual clearing prices. No later forecast or optimization changes them. The same configured controller (grid tracking in the current study) executes against the same realized PV, starting from the actual carried SOC and degradation memory at the simulation boundary. Financial accounting includes the counterfactual's own auction revenue, gross transaction fees, executed wear, import fees and actual imbalance settlement.

`reoptimization_gain_after_<auction>_eur` is **fully reoptimized realized profit minus frozen-plan realized profit**, using the full delivery day in both cases. `frozen_plan_profit_after_<auction>_eur` records the baseline profit itself. These are cumulative benefits of all later replanning after that freeze point: they are signed, need not be positive or monotonic, and must not be added together. The comparison includes changes to orders, battery plans and state feedback; it does not isolate a causal forecast-quality effect. Final SOC/energy differences are reported without an assumed salvage value. Missing actual settlement keeps the settled-profit comparison unknown. Current snapshot replay requires deterministic plans; it does not select a lucky contingent scenario.

These simulations add **zero optimization solves**. The complete planned trade vector is retained in `expected_dispatch_plans.csv` through `planned_trade_<auction>_mwh` columns. Outputs are `csv/reoptimization_frozen_plan_summary.csv`, `csv/reoptimization_frozen_plan_dispatch.csv`, `csv/reoptimization_frozen_plan_trades.csv`, `json/reoptimization_frozen_plan_baselines.json`, and, for priced comparisons, `plots/reoptimization_value.png`. The JSON summary and individual scalar gains are also included in `json/kpis.json`/`csv/kpis.csv` as appropriate. The older open-loop imbalance diagnostic remains a separate check that follows battery commands without grid feedback and is not this profit comparison.

An isolated January 1 study with a 50% planned terminal SOC is ready at:

```bash
python -m bess_pv_trading experiment-gme-oracle --config configs/full_experiments/gme_oracle_2025-01-01_farm910_reoptimization_value.yaml
```

As with all fixed-ID experiments, choose a new `experiment_outputs.run_id` before rerunning an already completed study.

### Degradation and investment interpretation

The current farm-910 study uses `degradation.model: throughput`. It charges the sum of AC charging and discharging energy. Its coefficient is derived from the YAML replacement-pack price, currency conversion, reference cycle life and depth of discharge, normalized for the configured battery efficiencies. The resolved EUR/MWh coefficient and replacement cost are saved with the run.

The inherited example uses a historical generic pack-price proxy and 2,500 cycles at 80% depth; these are editable study assumptions, not a battery-specific installed cost or lifetime calibration. End-of-life capacity metadata does not dynamically reduce usable capacity. `xu_segmented` is a separate optional model with cycle-depth memory, marginal discharge costs and additional priority binaries; see the mathematical document and [Xu implementation notes](docs/xu_degradation.md).

The battery-value comparison is

`battery_incremental_operating_value_eur = PV+BESS perfect-information objective - PV-only perfect-information objective`.

Both cases use the same realized PV, actual prices, auction eligibility and POI. The PV-only solve reuses the same builder with battery capacity, power, stored energy and wear set to zero; battery variables are fixed at zero rather than handled by a different formulation. The PV+BESS case retains its configured initial/terminal energy policy. Its objective difference is after modeled degradation and before capital expenditure, fixed O&M, financing, taxes or residual inventory valuation. It is distinct from feedback-controller realized profit. An investment study needs a representative longer horizon and an explicit cost/cash-flow treatment; a single hindsight day is an operational comparison.

### Changing the study day

Keep `gme_oracle_experiment.delivery_day`, `forecast_run.delivery_day`, the initial forecast cutoff, and `market.valid_from`/`valid_to` consistent. Ensure all selected price datasets, required weather vintages, realized PV and enabled settlement prices cover that day. Set `auction_datasets.<auction>.native_interval_minutes` to the actual native product resolution. Choose a fresh run ID. Calendar and availability settings are explicit study assumptions and must match the period studied.

## Data acquisition and local imports

Private farm inputs are the Zarr and Parquet paths in `configs/data_schema.yaml`, currently under `data/farms/`. The shared acquisition/import catalog is **`data/api/market.duckdb`**. It stores normalized products and typed/raw snapshots with request identity, checksums, retrieval metadata and revision history. Farm Zarr data is independent of this database.

Use one writer at a time. Stop a running downloader before opening the database from another analysis process. Offline experiments read completed snapshots; incomplete acquisition does not prevent studying complete covered days. Do not treat retrieval time or an unverified provider creation timestamp as historical publication evidence.

### Provider APIs

Plan requests without network/authentication, inspect cache coverage, then download explicitly when needed:

```bash
python -m bess_pv_trading download-data --config configs/api_experiments/api_history.yaml --dry-run
python -m bess_pv_trading api-cache-status --config configs/api_experiments/api_history.yaml
python -m bess_pv_trading download-data --config configs/api_experiments/api_history.yaml --interactive
```

The history config currently requests 1 January 2025 through 31 August 2026 (`api.end` is exclusive) and completes requested datasets month by month using daily request keys. Rerunning resumes cached completed chunks; transient failures have bounded retry/backoff. `--refresh-cache` requests new snapshots while retaining prior revisions. No data is downloaded by an offline experiment command.

If Terna returns HTTP 403, the default `api.terna_http_403_policy: continue_other_providers` pauses further Terna network requests for that run and continues ENTSO-E across all remaining months. Valid cached Terna chunks remain usable. The final manifest reports `status: partial`, the failed/deferred Terna chunks and per-provider completion totals; the CLI exits with code 2 after finishing the remaining providers. HTTP 403 may indicate quota exhaustion or another access restriction. Rerun the same command later without `--refresh-cache` to retry pending Terna chunks and reuse completed ENTSO-E data. Set the policy to `error` for immediate failure instead. A running process must be restarted to pick up this change.

| Provider | Configuration examples | Credentials / scope |
| --- | --- | --- |
| ENTSO-E | `entsoe_api.yaml`, `api_history.yaml` | `ENTSOE_API_TOKEN`; day-ahead prices and regional forecasts |
| Terna | `terna_api.yaml`, `terna_wind_check.yaml` | `TERNA_CLIENT_ID`, `TERNA_CLIENT_SECRET`; configured load, forecast and renewable datasets |
| GME | `gme_api.yaml`, `gme_history.yaml` | `GME_USERNAME`, `GME_PASSWORD`; separate GME API account, MGP and MI-A prices |

Provider configs live under `configs/api_experiments/`. Use the hidden `--interactive` prompts for missing credentials, or set the environment variables named by the chosen config. Never put secrets into YAML or source control. See [API storage/workflow](docs/api_pipeline.md), [ENTSO-E](docs/entsoe_api.md), [Terna](docs/terna_api.md), and [GME](docs/gme_api.md). The legacy file-cache migration command is available when needed:

```bash
python -m bess_pv_trading migrate-api-cache --config configs/api_experiments/api_history.yaml
```

### GME and Terna workbooks

The manual historical study uses local GME auction-price and Terna imbalance workbooks:

```bash
python -m bess_pv_trading import-gme-workbooks --config configs/api_experiments/gme_workbook_import.yaml
python -m bess_pv_trading import-terna-workbooks --config configs/api_experiments/terna_workbook_import.yaml
```

The import YAMLs declare paths, sheet/column mappings, native product resolutions, selected zone and date ranges. The current GME config covers hourly MGP January–August 2025 and native quarter-hour MI-A1/2/3 from 1 January through 1 April 2025 inclusive. It archives and retires the superseded hourly IDA snapshots while preserving DA and unrelated providers. The Terna config selects NORD macrozone quarter-hour settlement prices for January–August 2025. Actual local coverage is established by importer manifests and catalog checks, not by these configured ranges alone.

Native `Periodo` indices and local delivery dates identify MI products, including DST and the IDA3 noon-to-midnight window. The importer requires supported missing-price markers/cancellation evidence; missing files or partial curves are not inferred to mean cancellation. Workbook settlement remains a signed energy-accounting implementation under the study's selected price convention, not a complete reproduction of every participant-specific imbalance charge.

## Time, data quality, and PV forecasting

All internal event/product timestamps are timezone-aware and use UTC. Market days, gates and figures use `Europe/Rome`. Local midnight-to-midnight grids produce 92, 96 or 100 quarter-hours on spring-DST, ordinary and autumn-DST days. Repeated local hours retain their UTC offsets/folds.

The shared Terna workbook parser decodes `02:01/16/31/46` as the second occurrence of `02:00/15/30/45` on the autumn transition day, retains the original labels and prices, and orders both hours by UTC. Spring skips the nonexistent 02:00 hour without creating prices. See the [manual workbook workflow](docs/api_pipeline.md#stage-manual-gme-and-terna-excel-exports-while-acquisition-continues) for staging and post-download checks.

| Quantity | Representation |
| --- | --- |
| Historical farm `power` | Interval-average AC kW, start-labelled, scaled to MW once |
| Farm installed nameplate | AC kW in metadata; reported as MW AC |
| Historical satellite GHI | Observed irradiance, not an available future weather forecast |
| Weather radiation | End-labelled hourly means; redistributed over their contained quarters |
| Other weather fields | Interpolated only under their declared semantics and coverage |
| Forecast vintages | Separate cycle and availability times; default availability lag is six hours |
| Optimization powers / stored energy | AC MW / internal MWh |
| Auction quantities / prices | Signed interval MWh / EUR per MWh |

Weather cycles occur about every six hours; overlapping cycles are successive vintages, not independent ensemble members. At each auction the rolling policy selects the latest eligible vintage, including IDA3's remaining delivery horizon. More than one auction can legitimately use the same vintage when no newer cycle is available yet. Realized production is used for feedback and settled observations, not substituted for unknown future PV in an ordinary auction plan. At a cutoff inside an unfinished interval, the next boundary state is projected with forecast PV through the controller; the final interval-average measurement is not known yet.

The data policy preserves raw values and records flags, adjustments and evidence times. Current policies include nonzero plateau treatment, a configured local night window, allowlisted nonnegative forecast fields, and two-sided interpolation of eligible historical power/GHI gaps. Those are explicit analysis assumptions: some corrections become knowable only after a later sample. As-of views retain that distinction. Forecast gaps are not automatically filled; the historical GME study permits its configured night-zero forecast handling and a bounded, recorded realized-power fallback.

For **PV model training and benchmark scoring**, `pv_data_quality.sustained_zero_outage` in `configs/standard.yaml` now masks a contiguous zero-AC daylight run when it contains at least four native quarters with satellite GHI at or above 100 W/m². The full daylight zero run is masked, including quarters with missing satellite GHI. Its name is historical: the flag establishes anomalous zero output for prediction fitting, **not the cause**. Economic curtailment, a plant outage and bad telemetry are distinct possibilities, especially on days with negative market prices. Genuine nighttime and shorter daylight zeros remain. Raw AC, GHI and market data are unchanged; eligible modeling views retain exclusion flags and their original values. Historical research artifacts made before this rule use their original masks and must be labeled as pre-curation results. Existing global 2025 farm parameters were fitted before this rule; changing the evaluation mask alone does not retrain them.

Run the data-quality audit before choosing additional exclusions:

```bash
PYTHONPATH=src python -m bess_pv_trading audit-pv-data-quality \
  --config configs/pv_experiments/audit_data_quality_2025_2026.yaml
```

It groups overlapping auction views into physical 15-minute intervals, writes per-farm evidence and episode tables under `outputs/pv_data_quality/full_history_2025_to_2026_08/`, and creates `csv/review_queue.csv`. The queue proposes low output under measured high GHI, short partial recoveries between zero runs, flat nonzero plateaus and AC-rating excess; weather disagreement and missing satellite GHI are reported separately. These candidates do **not** alter eligibility. Daily plots have three aligned panels: selected neural AC (`prediction_selected_mw`) versus measured AC, saved pvlib AC (`prediction_pvlib_mw`) versus measured AC, and forecast GHI versus satellite GHI. Black is measured AC or satellite GHI; matching colors identify the DA/IDA auction forecast in every panel. The dashed AC line is the farm rating, and both AC panels use the same vertical scale. Plot prediction columns and labels are configurable under `pv_data_quality_review`; each saved prediction row must match the raw AC corpus before plotting. Review these plots and the source evidence, copy the queue to a stable CSV, fill `decision` with `keep` or `exclude`, and set `review.exclusions_csv` to that path. An `exclude` decision masks only the farm and inclusive UTC interval in research fits/scores; it never edits source measurements. The study owner identified farm 345's 1 May 2025 zero-production interval as emergency curtailment, not an equipment-failure diagnosis; the source measurements remain unchanged. To materialize the current automatic mask, run `curate-pv-outages --config configs/pv_experiments/curate_sustained_zero.yaml`. To rescore frozen validation predictions without refitting, run `audit-pv-validation --config configs/pv_experiments/audit_validation_2026_jan_apr.yaml`.

The PV model uses forecast GHI and horizontal direct irradiance, solar position, fixed or single-axis orientation, selectable isotropic/Hay–Davies/Perez–Driesse plane-of-array irradiance, Faiman temperature from air temperature and wind speed, and PVWatts DC power. Farm metadata capacity is AC. The DC reference rating is AC nameplate divided by the independent `pv.dc.reference_conversion_efficiency` assumption (currently 0.96). Explicit mode converts DC to AC with `pv.output.pv_output_efficiency`; calibrated mode uses `pv.output.calibrated_coefficient`. Both modes then apply the transformer efficiency and cap the forecast at the farm AC rating; a lower YAML AC cap can be selected. Measured AC is not converted again. The current location is an acknowledged Milan proxy; tracker geometry and temperature coefficients are configured assumptions. Farm 901 defaults to a front-side-only approximation; `pv.bifacial.enabled` enables explicit infinite-sheds front/rear irradiance with declared row geometry and bifaciality. Clouds and dew point may be retained in the input store but do not directly enter this conversion. Known farm-specific PV settings can override shared assumptions under `farms.<id>.pv`. See [data contract](docs/dataset_contract.md) and [PV notes](docs/data_pv_notes.md).

To fit a static farm coefficient from local weather vintages and measured AC production, run `python -m bess_pv_trading calibrate-pv-output --config configs/pv_calibration_2025-01_farm910.yaml`. The command selects the newest weather cycle available at each quarter-hour start, runs pvlib with the fixed DC reference, excludes invalid/low-DC and configured reviewed or adjusted measurements, and fits a zero-intercept least-squares coefficient. Its chronological holdout compares the training-only fit with the explicit coefficient; `fit.json` reports that comparison and the all-pairs refit. The resolved configuration, aligned pairs and plot are saved under `outputs/pv_calibration/<run_id>/{json,csv,plots}`. To use a different fitted number later, override `farms.<id>.pv.output.calibrated_coefficient` and use `method: calibrated_scale` in that same output mapping. Farm-specific values take precedence over common PV values. The fitted coefficient is an aggregate predictive scale affected by weather error and plant operation, not a hardware efficiency measurement.

This legacy single-coefficient command still uses its original observation-adjustment policy and does not apply the new sustained-zero screen. Use `study-pv-calibration` for a fit on the versioned curated auction-view population; do not compare the two commands' scores without aligning their masks and forecast cutoffs.

To measure PV prediction errors on every weather-GHI disagreement episode for one farm, run `python scripts/analyze_pv_weather_disagreement.py --config configs/pv_experiments/audit_data_quality_2025_2026.yaml --farm-id 1 --output outputs/pv_weather_disagreement/farm_1`. It reads frozen saved predictions and the existing audit, without training or optimization. Group, auction and individual-episode CSVs report MAE, RMSE, signed bias and maximum error normalized by installed AC power. Group results include all daylight, weather-disagreement quarters, quarters outside weather disagreement, and quarters outside the union of all audited episode types; the exclusions are evaluation sensitivities, not operational filters or evidence that the measurements are wrong. Both valid-raw and curated target populations are reported, with training/validation/test periods kept separate; `all_dates` is descriptive and includes fitted dates. `--prediction-column` and `--baseline-column` select other saved forecast columns. The summary plot compares disagreement versus other daylight errors (blue selected model, orange calibrated pvlib). The example-day plot shows measured/predicted AC, satellite/forecast GHI and signed normalized AC error; orange shading marks the audited episode and colored lines identify the auction forecast. Source checksums and resolved settings are saved alongside the results.

The reproducible **physical PV calibration study** uses auction-time vintages, rather than selecting new weather at delivery. Run it with:

```bash
PYTHONPATH=src python -m bess_pv_trading study-pv-calibration \
  --config configs/pv_experiments/calibration_2025.yaml
```

It independently fits farms 910, 1, 345 and 901. January is warmup; February–December are scored on held-out delivery months. A monthly model is issued at local midnight on the day before that month's first delivery, using only completed, available targets. The last seven complete historical days select expanding versus 90-day history, transposition, and regularization; the chosen specification is refitted before forecasting the next month. Every family also retains its own predictions. `selected` chooses a family using that earlier validation only, so it need not be the best family in hindsight. The bundled study overrides each farm's `pv` mapping to `null`, retaining the original unfitted reference even after calibrated global parameters are promoted to the standard config.

The comparison includes the fixed forecast, clipped scale-only MSE/Huber fits, scale plus Faiman thermal fits, tracker fits that also adjust GCR/maximum rotation, and farm 901 rear-side thermal fits with the declared row geometry held fixed. The DC reference, temperature coefficient, transformer factor, known metadata and proxy location remain fixed. Numerical bounds, search budgets, worker count and all other controls inherit documented defaults from `pv_calibration` in `configs/standard.yaml`. The model uses identical day/auction weights and the same physical forward calculation as the operational forecaster. Raw finite AC targets remain available; the current sustained-zero screen changes model-fit and evaluation eligibility while retaining other genuine zeros. Satellite GHI is used for that screen and diagnostics, never as a future forecast input. No target is filled from a prediction. The completed 2025 fit described below predates this screen.

Outputs under `outputs/pv_calibration/<run_id>/` contain study comparisons and per-farm `json`, `csv`, `parquet`, and `plots` folders. `predictions.parquet` records the cutoff, vintage, target availability, frozen eligibility mask, warmup/held-out phase and every method's forecast. CSVs include auction/month/season/hour/lead/weather error summaries, daily energy bias, seven-day paired block intervals, coverage, and parameter histories. Energy summaries stay separate by auction because their delivery horizons overlap. `retrospective_*` files use a final all-year refit and are explicitly in-sample. Completed method fits are resumed only when configuration and source fingerprints match.

To retrain as new observations arrive, change `pv_calibration.as_of_utc` in the example YAML and run:

```bash
PYTHONPATH=src python -m bess_pv_trading retrain-pv-calibration \
  --config configs/pv_experiments/retrain_example.yaml
```

Retraining uses complete historical days before that timestamp and the same selection/fitting code. Change `pv_calibration.farm_ids` to train any subset; the example selects farm 910. Each timestamp creates a separate run. For an issue-time research replay, set `pv.parameter_registry` to the run's `json/parameter_registry.json` and use the same physical/input assumptions as its calibration config. The loader selects the latest matching farm artifact issued by the decision cutoff, checks its checksum and fixed assumptions, and records the artifact in forecast provenance. Before the first artifact it uses the configured baseline. This optional registry defaults to `null`.

**Ordinary experiments automatically use the final global farm parameters in `configs/standard.yaml`, under `farms.<id>.pv`.** The family with the lowest February–December held-out daylight RMSE is selected for each farm, and its final all-2025 refit supplies the static parameters. Their full provenance and in-sample fit scores are versioned in `configs/pv_parameters/2025/`. As requested, these are treated as prior farm knowledge at every experiment date, including January 2025; they are separate from the chronological monthly benchmark. Edit a farm's `pv` mapping to change its parameters, or set it to `null` to use the shared unfitted reference. Farm-specific PV values take precedence over the common `pv` assumptions. Fitted thermal/tracker values remain effective predictive parameters because location, weather error, outages and unmeasured module temperature can confound their physical interpretation.

The completed run `outputs/pv_calibration/physical_pv_all_farms_2025/` includes all four farms. Monthly held-out family RMSE improves by approximately 7.7% for 910, 59.5% for 1, 56.5% for 345, and 9.6% for 901. These scores describe the chronological family trajectories, not the in-sample final global fit. Thermal Huber was selected for farms 1 and 345; tracker MSE for 901 and 910. Several thermal parameters reach their configured bounds; farm 901's final tracker fit retains the best finite iterate after a numerical budget stop, flagged in its artifact. These are the best parameters found within the configured search, without a guarantee of a global optimum. The LaTeX reference records the full selection procedure and installed numbers.

The final static fit's **all-2025 in-sample daylight absolute errors**, compared on the same raw measurements and weighted equally by day/auction view, are:

| Farm | Mean before → after (MW) | Minimum before → after (MW) | Maximum before → after (MW) |
| --- | --- | --- | --- |
| 910 | 0.717 → 0.682 | 0 → 0 | 5.816 → 5.383 |
| 1 | 0.145 → 0.055 | 0 → 0 | 0.726 → 0.484 |
| 345 | 0.248 → 0.108 | 0 → 0 | 1.203 → 0.922 |
| 901 | 0.443 → 0.391 | 0 → 0 | 3.090 → 3.107 |

The maximum is the worst absolute quarter-hour error; MAE is an average. Farm 901 shows that reducing average error need not reduce the worst error. `csv/before_after_global_parameters.csv` and its JSON companion also report capacity-normalized errors, RMSE, signed bias, per-auction results and a separate chronological held-out comparison. `plots/before_after_global_parameters.png` visualizes the table. Held-out figures remain in each farm's `plots/`; retrospective fit figures are kept in `plots/retrospective/` so they cannot overwrite held-out figures.

The additional `plots/trajectory_examples/farm_<id>_three_days.png` figures compare raw measured AC with all four operational forecasts under those final global settings, including the next-midnight sample and installed AC threshold. The common dates represent low, medium and high portfolio-normalized daily production; they were chosen independently of forecast residuals. The associated CSVs and JSONs preserve values, resolved configurations, forecast provenance and capacity checks. They illustrate the final static model on its fitting year, rather than held-out forecast performance. To regenerate them without refitting or trading:

```bash
PYTHONPATH=src python -c 'from bess_pv_trading.forecasting.pv_global_plots import plot_global_trajectories; plot_global_trajectories("outputs/pv_calibration/physical_pv_all_farms_2025", "configs/standard.yaml")'
```

Before 1 October 2025, **DA energy orders are hourly**, represented as equal quarter-hour shares. **IDA orders and physical battery controls remain quarter-hourly**, including in January 2025. Repeating a native hourly DA price into four objective coefficients does not create four independent DA prices or require hourly-flat battery dispatch. IDA3's configured gate is 10:00 local, its decision cutoff is five minutes earlier, and its eligible deliveries begin at 12:00 local. Other gates/results and their historical validity are YAML settings. Figures include the final delivery boundary when available.

## CPU neural PV benchmark

The `feature/neural-pv-benchmark` branch adds a separate model-comparison workflow:

```bash
PYTHONPATH=src python -m bess_pv_trading study-pv-neural \
  --config configs/pv_experiments/neural_2025_2026.yaml
```

It uses all available delivery dates from January 2025 through August 2026. Training uses 2025, validation uses January–April 2026, and the untouched test uses May–August 2026. A configured one-day embargo excludes late training/validation targets at split boundaries so all fitted observations precede the next split's first DA cutoff. Imputation and standardization use training observations only. Early stopping, architecture selection and ridge regularization use validation only. The test is used to report every frozen candidate and the previously selected winner; its scores never choose the deployed artifact. The test period also measures seasonal generalization beyond the winter/spring validation months.

Each farm independently trains six candidates: dense MLP, residual MLP, and temporal convolution, each with core and extended weather inputs. Core inputs are forecast GHI/horizontal-direct radiation, air temperature, 10 m wind components/speed, deterministic solar position, cyclic delivery hour/year/month and forecast cycle hour, and forecast lead time. The extension adds dew point, cloud layers, precipitation, snow, visibility and 100 m wind. Month and auction embeddings are learned. Missing optional features use persisted training medians and missingness indicators. Separate per-farm weights learn the plant response; fitted DC reference/output coefficients, GCR, maximum rotation, cell temperature and POA are **not neural inputs**. Measured AC and satellite GHI are never input features. Actual source tracking/orientation metadata is retained in the model contract; constant farm attributes need not be estimated from redundant columns within a farm-specific model.

The output is a sigmoid AC-nameplate fraction, zeroed outside the deterministic daylight/night policy. The target is unmodified measured quarter-hour AC divided by known installed AC for normalization; no efficiency loss or forecast substitution is applied. All candidates and the calibrated pvlib reference share one eligibility mask, including genuine zero production, operating anomalies and coverage gaps. Models are fitted with configurable MSE/Huber and AdamW; each day/auction view has equal aggregate loss weight. IDA3 uses only its local-noon delivery suffix. Every input view uses the newest weather cycle available at that auction's buffered cutoff, with the six-hour UTC lag and DST-aware native grid.

The temporal convolution uses neighboring values of the **same already available forecast curve**, including its later leads. Its symmetric context is valid forecast information; it does not use later measured power, another weather cycle, or an adjacent delivery day's targets. Sequences never cross date/auction/vintage boundaries. The calibrated pvlib baseline remains exactly the global 2025 settings in `standard.yaml`, without further tuning on 2026. A validation-selected ridge weather model is an additional sanity baseline.

All controls inherit `pv_neural` in `configs/standard.yaml`: split dates, embargo, farm selection, feature/architecture candidates, widths/depths, embeddings, loss, learning rate, training budget, seeds, CPU workers/threads and outputs. PyTorch is optional for other commands and this workflow explicitly uses CPU. The benchmark uses four independent farm workers with two CPU threads each by default; it does not allocate GPU memory. Add dependencies through the `bess` environment only after permission; prefer conda (`conda install -n bess -c conda-forge pytorch-cpu`).

Outputs live in `outputs/pv_neural/<run_id>/`, with study-level comparisons and per-farm `csv`, `json`, `parquet`, `plots`, and `models` folders. They include raw aligned targets and forecasts, separate train/validation/test metrics, mean/minimum/maximum absolute errors, RMSE, signed bias, per-auction/month/hour and daily energy errors, paired seven-day block uncertainty intervals, learning curves, representative test trajectories with the installed AC threshold, input/source fingerprints and resolved configurations. Overlapping auction views are kept separate in energy reporting. `models/selected.pt` and `selected.json` identify each farm's validation-selected neural model, including preprocessing, metadata, architecture, training history and a weight checksum. Other candidates remain reusable. Completed fits resume only with matching configuration, source and training-code fingerprints.

`forecasting.pv_neural_predict.predict_neural_frame(frame, artifact, farm_id=...)` loads a saved model on CPU and predicts without requiring measured targets. It checks farm identity, feature order and the weights checksum. The standard trading forecast remains pvlib; choosing a neural model for trading is a subsequent experiment. For a short four-farm execution check use `configs/pv_experiments/neural_validation.yaml`.

### Neural PV hyperparameter search

Run a separate, resumable CPU-parallel search with:

```bash
PYTHONPATH=src python -m bess_pv_trading tune-pv-neural \
  --config configs/pv_experiments/neural_tuning_2025_2026.yaml
```

The search keeps the benchmark's 2025 training period, January–April 2026 validation period, one-day availability embargo, and May–August 2026 untouched test period. It samples eight settings for each of the MLP, residual, and temporal convolution architectures with both core and extended weather inputs, including the existing default in each group. Width, depth, dropout, learning rate, weight decay, batch size, loss, embedding sizes, and temporal kernel are configurable. Independent farm/model trials run in separate CPU processes with one thread each. The standard setting leaves six logical CPUs idle for other cluster users; an experiment may set a lower worker cap. This study caps at 42 workers on the 48-CPU cluster. It also compares validation-selected weighted OLS, Ridge, Lasso, and Elastic Net models using both feature sets. No additional Python package is needed.

Each candidate is fit using measured AC power normalized by the farm rating. The predictions are clipped to the farm's AC range for validation and test scoring, matching the neural output constraint and calibrated pvlib's AC cap; measured AC targets remain unmodified. Hyperparameters and model families are selected only on the earlier validation period. The test period reports the resulting selected neural models, best neural model per architecture, best setting in each regression family, and the unchanged calibrated pvlib baseline. Outputs include every validation trial, selected model coefficients and weights, paired per-farm test metrics, plots, resolved configuration, source fingerprints, and a completion manifest under `outputs/pv_neural_tuning/<run_id>/`.

The completed `neural_pv_2025_to_2026_08` run fits 24 neural candidates across the four farms and evaluates all 123 test delivery days. There are 204,276 aligned auction/quarter-hour rows per farm over 608 days. With equal day/auction daylight weighting, validation-selected neural performance is:

| Farm | Selected neural model | pvlib test RMSE (MW) | Neural test RMSE (MW) | RMSE reduction |
| --- | --- | --- | --- | --- |
| 1 | Temporal convolution, core | 0.06998 | 0.06119 | 12.6% |
| 345 | Dense MLP, core | 0.27046 | 0.25209 | 6.8% |
| 901 | Temporal convolution, core | 0.58712 | 0.54140 | 7.8% |
| 910 | Residual MLP, core | 1.22336 | 1.29883 | −6.2% |

The neural model does not improve every farm or metric: farm 910 worsens, and farm 345's MAE increases despite its smaller RMSE. Ridge achieves 0.20050 MW RMSE on farm 345. The candidate chosen by earlier validation need not have the lowest test error; no test ranking was used to change that choice. Core configurations were selected for all four farms, but widths also differ between core and extended candidates, so this comparison does not isolate a causal effect of individual weather features. The test months and validation months cover different seasons; one seed and one chronological split limit broader conclusions. No neural model is promoted automatically into trading defaults.

`plots/farm_model_comparison.png` shows every frozen candidate and baseline. Each farm has learning curves and three representative test-day plots, with separate DA/IDA panels, measured AC and the rating threshold. The independent `json/validation_audit.json` verifies all 24 model checksums/CPU states, unchanged targets and pvlib curves, split embargo, shared evaluation masks and validation selection, and target-free selected-artifact replay on first/middle/last test days. A completed-run resume also checks that weights and full prediction tables remain unchanged.

## Outputs and how to read them

Historical studies are organized under:

```text
outputs/full_experiments/gme_oracle/<run_id>/farm_<id>/<delivery_day>/
  json/
    manifest.json                     artifact inventory and input provenance
    study_config.json                 effective configuration
    kpis.json                         aggregate metrics and assumptions
    pv_forecast_updates.json          forecast cycles used at auction decisions
    perfect_pv_counterfactual.json     full-information PV+BESS result and validation
    perfect_pv_only_counterfactual.json  optional PV-only result
    oracle/                           auction solve/publication/ledger summaries
  csv/
    kpis.csv                          scalar KPI table
    auction_prices_expanded.csv       objective coefficients on physical intervals
    pv_forecast_and_realized.csv      forecast/measurement comparison and vintages
    pv_power_allocation.csv           PV-to-grid, PV-to-battery, curtailed PV
    perfect_pv_counterfactual_schedule.csv
    perfect_pv_only_counterfactual_schedule.csv
    reoptimization_imbalance_*.csv    snapshot-plan ex-post settlement diagnostics
    oracle/                           executed dispatch, auction plans and trade ledger
  plots/                              study figures and oracle/ dispatch figures
  models/                             optional LP exports, including oracle/ residual models
```

Availability of settlement/counterfactual files follows enabled settings and actual data. The saved manifest inventories files actually present. Important figures include DA/IDA1/IDA2 prices together, separate IDA3 prices, four auction-specific forecast/dispatch figures, all-used PV vintages, executed SOC and tracking, PV allocation, settlement/profit breakdown, the perfect-PV-versus-feedback comparison, and the optional storage-value comparison. The removed combined `expected_dispatch_by_auction` plot is not generated.

`json/kpis.json` includes study date/farm, installed capacity, complete asset/solver/degradation settings, auction cash flows, realized gain, signed and absolute imbalance, curtailment, charging/discharging, throughput and executed wear, SOC metrics, model sizes/statuses, forecast errors and vintages, source snapshot provenance, and expected-versus-realized revenue/profit differences. Detailed runtimes and validation residuals are in the solve diagnostics; dispatch CSVs hold the complete state trajectory. Differences use **realized minus initial expected**; percentages divide by the absolute initial expectation and are `null` for a zero baseline. If actual settlement is unavailable, net settled profit is unknown rather than silently priced at zero.

The current comparison fields are `perfect_information_objective_eur`, `pv_only_perfect_information_objective_eur`, and `battery_incremental_operating_value_eur`. Those objectives are solver `ObjVal` values. The storage-value difference compares the two hindsight models, while `total_gain_eur` is the executed feedback strategy's settled profit. Degradation in these distinct trajectories is not interchangeable: executed wear is computed from executed powers; each counterfactual reports its own component costs.

Single `optimize` runs save `json/result.json` and `arrays/solution.npz`. Synthetic replays use the same file-type organization under `outputs/trading/<run>/farm_<id>/<day>/`, with one subdirectory per strategy.

## Other supported workflows

| Task | Command and example configuration |
| --- | --- |
| Inspect farm schemas | `inspect-data --config configs/farms_nord.yaml` |
| Prepare policy-treated data | `prepare-analysis-data --config configs/farms_nord.yaml` |
| Convert a selected weather vintage to PV | `forecast-pv --config configs/farms_nord_radiation_sensitivity.yaml` |
| Single synthetic optimization | `optimize --config configs/full_experiments/synthetic_small.yaml` |
| Synthetic strategy replay | `run-synthetic --config configs/full_experiments/synthetic_small.yaml` |
| Canonical CSV/XML market imports | `import-market-data --config configs/price_experiments/italy_nord.example.yaml` after supplying files |

Prefix each command with `python -m bess_pv_trading`. `backtest` currently routes synthetic configurations through their replay and rejects an unsupported generic historical backtest; use `experiment-gme-oracle` for the implemented manual-data historical experiment. `report-implementation` is a retained legacy evidence renderer, not the source of the current mathematical document; the old JSON evidence and bibliography are historical records.

Earlier price-prediction research code, dedicated tests, derived training tables, and study configs have been moved to `/home/rghetti/price_prediction_archive` so a new forecasting model can start from a clean slate. The active historical market-import example remains in `configs/price_experiments/italy_nord.example.yaml`; raw market data in DuckDB is untouched. Archived model-training commands are no longer part of the current CLI.

Historical price-prediction research and its derived training tables are preserved outside the repository at `/home/rghetti/price_prediction_archive`. Current GME experiments use actual auction curves as an oracle benchmark; they do not claim a causal price forecast. Raw market observations remain in DuckDB.

The original farm audit and cloud comparison tools remain available through `python -m bess_pv_trading.audit --config configs/audit.yaml` and `python -m bess_pv_trading.cloud_check --help`. Audit outputs are review candidates and raw-data diagnostics; they are distinct from the policy-treated optimization data.

## Validation and maintaining the documentation

The optimizer independently validates extracted energy/power balances, SOC/power/POI limits, product equalities, information sharing, degradation and objective components before a schedule is accepted. The executor also enforces physical limits and carries realized stored energy and wear. Tests cover farm data semantics, provider/workbook parsing, DuckDB provenance, market calendars, optimization, feedback, settlement and output organization.

To run the suite explicitly in the existing environment:

```bash
python -m pytest -q
```

Solver-dependent tests require the existing Gurobi license. Successful software checks do not establish hardware calibration or historical publication validity. Continuous intraday trading, bid-price/acceptance models, market impact, portfolio coupling, ancillary services, battery calendar aging, dynamic capacity fade, and a complete investment valuation are outside the current model.

Rebuild the model document with installed LaTeX tools:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs/generated/latex docs/mathematical_model.tex
cp docs/generated/latex/mathematical_model.pdf docs/mathematical_model.pdf
```

The LaTeX is maintained from the actual implementations in `optimization/model.py`, `optimization/degradation.py`, `simulation/replay.py`, `simulation/synthetic.py`, `simulation/gme_oracle.py`, and `simulation/settlement.py`. Update both the equations and practical config descriptions when behavior changes. Further details are in [battery physics](docs/battery_physical_model.md), [DuckDB/API workflow](docs/api_pipeline.md), [source notes](docs/sources.md), and the new [model document](docs/mathematical_model.tex).
