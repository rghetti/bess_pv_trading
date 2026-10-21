# PV and battery trading in the Italian electricity auctions

This repository studies one photovoltaic farm and an AC-coupled battery sharing a point of interconnection (POI). It prepares farm and market data, converts rolling weather forecasts into PV power with pvlib, optimizes energy positions in MGP and MI-A1/2/3 with Gurobi, executes the plans against realized PV through a feedback controller, and settles delivery deviations afterwards.

The current historical experiment is **one delivery day in the NORD zone, with perfect auction-price information and one deterministic weather-forecast member**. It refreshes the PV forecast before each active auction and makes four auction decisions when all auctions take place. Separate hindsight optimizations use realized PV to compare the feedback trajectory with a perfect-information schedule and to measure the incremental operating value of a battery against a PV-only plant. The general optimizer also supports scenario trees and probabilistic prepared forecasts; the current historical study does not use learned uncertain price forecasts.

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

Farm IDs `1`, `345`, `901` and `910` select their own weather, measured PV and metadata. `plant.farm_id` and `forecast.farm_id` follow the root `farm_id`. Automatic POI ratings use each farm's declared nameplate capacity as an explicit proxy, with per-farm measured ratings configurable. The BESS retains its configured size. See [the configuration guide](configs/README.md) for all selection and override rules.

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

With all four auctions active, this means **four auction-decision solves, one perfect-information PV+BESS solve, and one optional PV-only solve**. The replay's `optimizer_solves`/`reoptimization_solves` counts describe the auction replay; counterfactual diagnostics are saved separately. This is different from synthetic configurations that optionally replan physical controls every interval.

## Auction participation and average performance

Choose any non-empty participation subset in a single-study YAML:

```yaml
gme_oracle_experiment:
  participating_auctions: [IDA1, IDA2]
```

Omitting this field (or using `null`) selects all configured auction datasets. Chronological gate order is preserved. Unselected auctions have no price-data requirement, order variables, submission solve or transaction fees. A selected cancelled auction is skipped under the existing cancellation rules. The single-run example is `configs/full_experiments/gme_oracle_2025-01-01_farm910_ida1_ida2.yaml`. Single-auction cases, including IDA3 alone, are supported.

If every selected auction is documented or configured as cancelled on a day, the profile still contributes that full day to the comparison. There are no trades, auction/control solves or auction fees; zero-position grid feedback charges available PV within the SOC box and curtails the remainder. Executed wear and actual imbalance settlement still enter profit. No initial auction objective exists, so expected-profit KPIs are null. A configured terminal target is not enforced by this controller. Missing or partial price data without cancellation evidence remain errors.

If participation starts after delivery begins, the prefix uses zero contracted delivery and grid feedback with zero planned battery commands: PV can charge the battery within its SOC box, and remaining surplus is curtailed. It earns no uncontracted export revenue in this prefix. Its executed SOC and wear carry into the first auction solve. No additional auction optimization is inserted before the first selected gate. The initial expected daily profit includes the already executed/projected prefix costs; the first solve's residual objective is retained separately in the strategy summary.

For a matched-day comparison, configure a `participation_study` and run:

```bash
python -m bess_pv_trading experiment-participation --config configs/full_experiments/gme_participation_2025-01-01_03.yaml
```

For all 31 independent delivery days of January 2025 (465 profile/day runs), use `configs/full_experiments/gme_participation_2025-01.yaml` instead. It retains the same battery, initial/terminal planning SOC and other assumptions, including the two cancelled IDA3-only days in the averaging sample.

That monthly configuration enables `execution.infeasible_auction_policy: previous_plan`. An updated forecast can make exact delivery infeasible for already committed products before the next auction's first tradable interval. On a solver-confirmed infeasible auction decision, this policy submits that auction's quantities from the last feasible deterministic snapshot and retains its dispatch commands. Earlier commitments stay fixed; the ordinary physical controller executes against realized PV and settles unavoidable imbalances. The failed forecast attempt and reused plan source are recorded separately, without claiming a new feasible objective. It adds no recovery optimization. The default policy is `error`; missing previous plans, ambiguous stochastic snapshots and other solver failures still stop. `execution.save_infeasibility_diagnostics: true` saves IIS conflict reports. Per-run KPIs retain decision failures and attempt counts; the comparison tables include fallback-day and failed-decision counts.

`participation_study.resume: true` (enabled in the monthly YAML) verifies the same saved configuration, completed child manifests, input fingerprints and recorded profits before reusing profile/day checkpoints. Incomplete child directories are retained under `diagnostics/incomplete_runs/` and rerun; changed configurations are rejected. Successful completed cases are preserved after interruption. The numerical execution handoff floors solver-tolerated negative command roundoff to zero only after independent solution validation.

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
  per_profile_plots: false
  save_model_files: false
  profit_tolerance_eur: 0.000001
```

Four optional auctions give **15 non-empty profiles**. Set `required_auctions: [DA]` for the **eight** profiles with DA mandatory and each IDA independently optional. Or give explicit `profiles`, for example `[[IDA1, IDA2], [IDA3], [DA, IDA3]]`. An empty participation profile is not modeled. The study dates, asset, execution, degradation, settlement, fees and solver settings remain YAML-controlled. The supplied three-day study disables PV-only/frozen-plan comparisons and per-profile plots/models to limit repeated artifacts; those features can be enabled through their normal configs.

Every profile starts each delivery day with the same configured battery and operator fee state. Days are independent experiments; SOC and monthly traded-volume history are not automatically carried from one study day to another. The initialization forecast cutoff is shared before the earliest candidate gate/day start. Delivery grids, effective realized PV, initial PV forecasts and shared actual auction-price curves are fingerprinted and checked for equality across profiles. The batch resolves native auction-product durations separately for each date through the market adapter. Later forecast updates occur only at selected auction decisions. All profiles must complete the same requested dates with actual settlement; missing/invalid data fail the study instead of producing averages over different available dates.

Results live under `outputs/participation_studies/<run_id>/`: `csv/daily_results.csv`, `csv/profile_ranking.csv`, `csv/daily_best_profiles.csv`, `json/manifest.json`, the effective config, and `plots/participation_profiles.png`. Individual replay artifacts remain under `runs/<profile>/farm_<id>/<day>/`. Ranking uses **mean realized full-day net profit** after settlement, executed wear, import and transaction fees. It also records total/minimum/maximum profit, median, 10th/90th percentiles, daily standard deviation, negative-profit and zero-active-auction day counts, wins including tolerance ties, paired differences to the largest selected profile, and mean final SOC/energy. Chart whiskers are descriptive daily standard deviations. The mean of daily winners is not a fixed participation strategy, and the best mean on these dates is not an established optimum on unseen dates. Ending inventory remains unvalued and must be examined alongside cash profit.

The comparison adds a gold **Perfect foresight (all auctions)** reference above the ranked profiles by default. It averages full-day perfect-information `ObjVal` values on the identical dates and effective realized PV, participating in all configured auctions that remain available, with the same assets, costs and configured terminal SOC. The existing all-auction profile's benchmark is reused; if that profile is absent, one additional perfect-information solve per day supplies the reference without another auction replay. `participation_study.perfect_foresight_reference: false` disables it. Its daily objectives and summary are saved in `csv/perfect_foresight_daily.csv` and `json/perfect_foresight_reference.json`; profile tables also report objective-minus-realized-profit differences. The reference is displayed separately from profile ranks: the feedback profits include actual imbalance settlement and may finish at different SOC values, so the reference objective is not a guaranteed ceiling on those cash profits.

## Files and configuration

```text
src/
  data/             DuckDB catalog, providers, workbook imports, farm adapters, data policy
  forecasting/      pvlib conversion, linear/conditional price models, Gaussian calibration
  optimization/     shared MILP builder, prepared-input adapters, degradation, validation
  simulation/       auction events, immutable commitments, feedback execution, settlement
  reporting/        figures, output organization, legacy evidence reporting
configs/
  api_experiments/   acquisition, provider diagnostics, workbook imports
  price_experiments/ forecasting, calibration, and chronological evaluation studies
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
| `gme_oracle_2025-01-01_farm910_perfect_pv_comparison_terminal_soc_50.yaml` | Current January study with a 50% planned terminal SOC, perfect-PV comparison, and optional PV-only baseline |
| `gme_oracle_2025-01-01_farm910_reoptimization_value.yaml` | Isolated January 1 study of full-day profit with plans frozen after each pre-final auction |
| `gme_oracle_2025-01-01_farm910_perfect_pv_comparison.yaml` | Same comparisons with the inherited free terminal SOC |
| `gme_oracle_2025-01-01_farm910_quarter_hour_ida.yaml` | Native hourly DA / quarter-hour IDA historical replay |
| `synthetic_small.yaml` | Reproducible synthetic replay comparing four trading strategies |
| `synthetic_xu.yaml` | Synthetic segmented depth-dependent aging example |
| `synthetic_dst_spring.yaml`, `synthetic_dst_fall.yaml` | Synthetic 23-hour and 25-hour delivery days |
| `prepared_optimization.example.yaml` | Initial point-price planning from complete prepared CSV forecasts |
| `stochastic_prepared.example.yaml` | Initial/residual planning from PV members, saved price models, calibrations, and commitments |

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

| Quantity | Representation |
| --- | --- |
| Historical farm `power` | Interval-average AC kW, start-labelled, scaled to MW once |
| Farm installed nameplate | AC kW in metadata; reported as MW AC |
| Historical satellite GHI | Observed irradiance, not an available future weather forecast |
| Weather radiation | End-labelled hourly means; redistributed over their contained quarters |
| Other weather fields | Interpolated only under their declared semantics and coverage |
| Forecast vintages | Separate cycle and availability times; default publication lag is four hours |
| Optimization powers / stored energy | AC MW / internal MWh |
| Auction quantities / prices | Signed interval MWh / EUR per MWh |

Weather cycles occur about every six hours; overlapping cycles are successive vintages, not independent ensemble members. At each auction the rolling policy selects the latest eligible vintage, including IDA3's remaining delivery horizon. More than one auction can legitimately use the same vintage when no newer cycle is available yet. Realized production is used for feedback and settled observations, not substituted for unknown future PV in an ordinary auction plan. At a cutoff inside an unfinished interval, the next boundary state is projected with forecast PV through the controller; the final interval-average measurement is not known yet.

The data policy preserves raw values and records flags, adjustments and evidence times. Current policies include nonzero plateau treatment, a configured local night window, allowlisted nonnegative forecast fields, and two-sided interpolation of eligible historical power/GHI gaps. Those are explicit analysis assumptions: some corrections become knowable only after a later sample. As-of views retain that distinction. Forecast gaps are not automatically filled; the historical GME study permits its configured night-zero forecast handling and a bounded, recorded realized-power fallback.

The PV model uses forecast GHI and horizontal direct irradiance, solar position, fixed or single-axis orientation, isotropic plane-of-array irradiance, Faiman temperature from air temperature and wind speed, and PVWatts DC power. Farm metadata capacity is AC. The DC reference rating is AC nameplate divided by the independent `pv.dc.reference_conversion_efficiency` assumption (currently 0.96). Explicit mode converts DC to AC with `pv.output.pv_output_efficiency`; calibrated mode uses `pv.output.calibrated_coefficient`. Both modes then apply the transformer efficiency and cap the forecast at the farm AC rating; a lower YAML AC cap can be selected. Measured AC is not converted again. The current location is an acknowledged Milan proxy; tracker geometry and temperature coefficients are configured assumptions. Bifacial tracking is currently a front-side-only approximation. Clouds and dew point may be retained in the input store but do not directly enter this conversion. See [data contract](docs/dataset_contract.md) and [PV notes](docs/data_pv_notes.md).

To fit a static farm coefficient from local weather vintages and measured AC production, run `python -m bess_pv_trading calibrate-pv-output --config configs/pv_calibration_2025-01_farm910.yaml`. The command selects the newest weather cycle available at each quarter-hour start, runs pvlib with the fixed DC reference, excludes invalid/low-DC and configured reviewed or adjusted measurements, and fits a zero-intercept least-squares coefficient. Its chronological holdout compares the training-only fit with the explicit coefficient; `fit.json` reports that comparison and the all-pairs refit. The resolved configuration, aligned pairs and plot are saved under `outputs/pv_calibration/<run_id>/{json,csv,plots}`. To use the fitted number later, set `pv.output.method: calibrated_scale` and `pv.output.calibrated_coefficient` in an experiment override. The fitted coefficient is an aggregate predictive scale affected by weather error and plant operation, not a hardware efficiency measurement.

Before 1 October 2025, **DA energy orders are hourly**, represented as equal quarter-hour shares. **IDA orders and physical battery controls remain quarter-hourly**, including in January 2025. Repeating a native hourly DA price into four objective coefficients does not create four independent DA prices or require hourly-flat battery dispatch. IDA3's configured gate is 10:00 local, its decision cutoff is five minutes earlier, and its eligible deliveries begin at 12:00 local. Other gates/results and their historical validity are YAML settings. Figures include the final delivery boundary when available.

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
| Initial/residual prepared optimization | `optimize --config configs/full_experiments/stochastic_prepared.example.yaml` after filling its inputs |
| Single synthetic optimization | `optimize --config configs/full_experiments/synthetic_small.yaml` |
| Synthetic strategy replay | `run-synthetic --config configs/full_experiments/synthetic_small.yaml` |
| Baseline DA price experiment | `experiment-api-prices --config configs/price_experiments/price_experiment.yaml` |
| Load/wind/solar feature study | `experiment-price-fundamentals --config configs/price_experiments/price_fundamentals_january.yaml` |
| Rolling point-price model selection | `study-linear-prices --config configs/price_experiments/price_linear_methods.yaml` |
| Gaussian calibration study | `calibrate-price-scenarios --config configs/price_experiments/price_gaussian_adaptive.yaml` |
| Frozen later-period evaluation | `check-frozen-prices --config configs/price_experiments/price_forward_check.yaml` |
| Canonical CSV/XML market imports | `import-market-data --config configs/price_experiments/italy_nord.example.yaml` after supplying files |
| Portable level/spread model training | `train-price-models --config configs/price_experiments/synthetic_prices.yaml` for the synthetic fixture |

Prefix each command with `python -m bess_pv_trading`. `backtest` currently routes synthetic configurations through their replay and rejects an unsupported generic historical backtest; use `experiment-gme-oracle` for the implemented manual-data historical experiment. `report-implementation` is a retained legacy evidence renderer, not the source of the current mathematical document; the old JSON evidence and bibliography are historical records.

Price studies use chronological training, earlier validation for model selection, separate out-of-sample errors for calibration, and later evaluation periods. Features include lagged prices, calendar/seasonal terms, daily profiles, and configured regional load/wind/solar forecasts. Available estimators include OLS, ridge, recency-weighted ridge, lasso, elastic net and Huber. Gaussian scenarios add separately estimated bias and zero-mean temporally correlated innovations to a point prediction; covariance settings are saved rather than an unexplained fixed noise level. Native hourly/quarter-hour and DST grids remain distinct.

In the prepared multiauction route, `branches` is the number at each unknown auction result: two branches at four auctions give up to 16 price paths per PV member, capped by the configured joint-leaf limit. Sampled/known predecessor prices enter conditional IDA means; innovations are independent between auction equations and correlated over delivery products within an equation. Shared-history decisions remain nonanticipative. The actual-price benchmark repeats the same clearing curve across PV members and adds no noise. Retrieved historical revisions with unknown publication vintages remain explicitly retrospective; a timestamped download alone does not establish causal forecast eligibility. See [prepared forecast contracts](docs/forecast_optimization_inputs.md) and [price study notes](docs/market_price_notes.md).

The original farm audit and cloud comparison tools remain available through `python -m bess_pv_trading.audit --config configs/audit.yaml` and `python -m bess_pv_trading.cloud_check --help`. Audit outputs are review candidates and raw-data diagnostics; they are distinct from the policy-treated optimization data.

## Validation and maintaining the documentation

The optimizer independently validates extracted energy/power balances, SOC/power/POI limits, product equalities, information sharing, degradation and objective components before a schedule is accepted. The executor also enforces physical limits and carries realized stored energy and wear. Tests cover farm data semantics, provider/workbook parsing, DuckDB provenance, price forecasts, market calendars, optimization, feedback, settlement and output organization.

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
