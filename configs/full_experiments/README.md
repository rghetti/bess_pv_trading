# Full experiment configurations

Every YAML imports `../standard.yaml` directly and contains only its study-specific overrides. Shared feature flags and asset/model parameters live in that one standard; explicit overrides preserve earlier examples' settings. See [the configuration guide](../README.md) for merge rules, farm selection and the effective-YAML command.

Start with `gme_day.yaml` for a single-day study on any farm. Set root `farm_id`, or pass `--farm-id 1`, `345`, `901` or `910`; selected PV/weather and metadata, automatic POI ratings and active output labels follow that selection. Battery size remains independently configurable. Set measured POI ratings under `farms.<id>` or override `plant` limits explicitly. Set study dates, data selection and output IDs in YAML and inspect with `resolve-config` before running.

`synthetic_fees_none.yaml`, `synthetic_fees_flat.yaml` and
`synthetic_fees_gme.yaml` compare transaction-fee modes on the same market.
Use `optimize` for the initial plan or `run-synthetic` for the executed ledger.
The GME example declares a synthetic starting operator volume and explicitly
assumes the published tariff applied historically. See
[fee formulas, state and reporting](../../docs/formulation.md#market-transaction-fees).
The shared default now enables the GME variable tariff; the `none` example
explicitly disables it. Its standalone zero-volume default retains the first
20,000 MWh monthly exemption. Set known operator portfolio volume in
`transaction_fees.initial_state` when that assumption does not apply.

Small synthetic solver examples remain here for software validation. Add real-data studies as separate files with names that describe the study and period.

## Continuous optimization

`continuous_optimization_2025-01.yaml` is a separate consecutive-delivery
experiment. From the repository root, run:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-continuous --config configs/full_experiments/continuous_optimization_2025-01.yaml
```

Change only `continuous_optimization.start_date` and `end_date` to choose the
inclusive delivery-date range. The January example assumes 90% SOC at the
first midnight and starts trading its first delivery day in the preceding
31 December DA/IDA1/IDA2 auctions. It requires those prices and eligible
weather forecasts, but no 31 December realized PV. Actual operation ends at
midnight after the configured final date; no outside-range orders are created.

This runs integrated and `current_auction_only` planning as separate continuous
trajectories, plus a perfect-PV reference with the same daily planning horizon.
`planning_modes` selects either or both trading modes;
`perfect_pv_reference: false` skips the reference. Each trajectory carries its
own realized SOC, degradation, fees and commitments. DA/IDA1/IDA2 use the
current state and forecast/controller projection of retained current-day
commands to initialize next-day plans. They never use future realized midnight
SOC. Current-day IDA3 stays restricted to noon–midnight delivery. Forecast
selection uses each gate's submission cutoff and the shared six-hour UTC
availability lag.

The shared `state.terminal_soc_percent` applies only to the final optimization
boundary, with 50% in this example. Intermediate days have no terminal target
and no inventory salvage; realized terminal SOC may differ through feedback.
All battery, POI, conversion, solver and settlement settings retain their usual
shared YAML sections. `experiment_outputs.plots` and `save_model_files` control
plots and model exports. Farm selection works with root `farm_id` or
`--farm-id` as in the single-day workflow.

Outputs under `outputs/continuous_optimization/<run_id>/` include per-strategy
interval/daily KPI CSVs, submitted trade ledgers, state projections and raw solve
diagnostics, with JSON and figures in their own folders. Comparison tables
report paired daily and total settled-profit differences, inventory and
operating costs. Perfect-PV profit comes from its continuous execution;
overlapping per-gate objective values remain diagnostics and are not summed.
Midnight and IDA3-noon projection plots are separate for each solved strategy:
`midnight_soc_prediction_errors_<strategy>.png` and
`ida3_noon_soc_prediction_errors_<strategy>.png`.
`json/optimization_timing.json` records native Gurobi runtimes, mean/min/max/std
and strategy/day summaries, keeping reference solves separately identifiable.

The example enables `continuous_optimization.resume: true`. Resume verifies
configuration and input fingerprints, restores the latest completed midnight
checkpoint and replays work after it, including existing next-day orders and
fee/aging history. Rolled-back solver attempts remain saved in
`json/optimization_attempts.json` and counted in actual runtime statistics;
business decisions are saved separately in `json/solves.json`. Use a fresh
`run_id` when changing study settings. The workflow uses local data without
API requests and does not change the independent participation study.

The corrected January studies use native hourly DA prices/orders and native
15-minute IDA1/2/3 prices/orders. `decision_resolution_minutes: 15` permits battery
commands to change each quarter-hour at all four optimizations. An hourly DA
order is allocated equally across its four quarters without tying battery
commands or IDA adjustments together. Old saved January results used hourly
IDA summaries and hourly battery controls and must not be used as corrected
economic benchmarks. Reimport using `configs/api_experiments/gme_workbook_import.yaml`
before a new real-data run; the importer validates all files, archives obsolete
hourly IDA snapshots outside the catalog, and replaces them transactionally.
The shared catalog now contains 21,640 native IDA prices through 1 April 2025
inclusive. MI-A2 is documented as cancelled on 4 February; MI-A3 on 17 and
22 January. Native IDA data after 1 April are unavailable until additional
workbooks are imported. Replacement and verification reports are under
`outputs/data_imports/gme_native_prices/`; retired snapshots are under
`data/archive/gme/hourly_ida/`.

`gme_oracle_2025-01-01_farm910_quarter_hour_ida.yaml` reproduces the corrected
native-price January study in a fresh `native_ida_pt15` output folder. It
inherits the latest rolling-PV, grid-feedback, POI-bounded trade, throughput
degradation and reoptimization settlement settings.

`gme_oracle_2025-01-01_farm910_perfect_pv_comparison.yaml` adds one
deterministic, ex-post MILP using the effective realized AC PV series and all
actual native auction prices. It keeps the same battery, SOC box, POI, DA-hour
equality, quarter-hour IDA products, exact delivery and throughput wear. Its
four-panel plot compares feedback and full-information battery dispatch, PV
curtailment and SOC, and shows the dispatch difference through the next
midnight boundary. KPIs and solver validation are saved separately. The model
feasible objective (`ObjVal`) is reported as the perfect-information benchmark
and compared with feedback profit. Positive ex-post imbalance receipts are
reported separately because they are outside that optimization objective.

`gme_oracle_2025-01-01_farm910_throughput.yaml` selects the largest farm (6.638 MWp
DC), a 5 MWh BESS with a symmetric 2.5 MW AC power limit, percentage SOC bounds,
and a free terminal SOC. It sets `trading.optimize_imbalances: false`: forecast
grid energy must equal the contracted position in every optimization interval,
and imbalance decision variables are omitted. Executed deviations are measured
afterward; its `settlement` section loads actual Terna NORD prices for each
quarter-hour after replay. Missing or misaligned prices fail explicitly. Import
the monthly exports first with `configs/api_experiments/terna_workbook_import.yaml`.
It uses a source-based replacement cost to
calculate linear throughput wear. See [the physical model](../../docs/battery_physical_model.md)
for equations, conversion conventions, references and the four auction decisions.

The GME YAML selects
`forecast.reoptimization_pv_policy: latest_available_forecast_at_each_reoptimization`.
Every auction uses the latest weather cycle published by its buffered cutoff,
with the `source.availability_lag_hours` assumption or explicit publication
metadata. IDA3 updates the remaining physical horizon, keeping past execution
and earlier commitments fixed. PV plots show every distinct cycle, labeling
shared DA/IDA1 forecasts once; source cycles are logged per solve and interval.
The standalone update trace is `json/pv_forecast_updates.json`, and forecast
curves by auction are in `csv/pv_forecasts_by_auction.csv`.

`gme_oracle_2025-01-01_farm910_rolling_pv.yaml` reruns that day with the same
assets and grid-feedback settings under a separate rolling-PV run ID. Its
completed run retains three weather cycles and four optimal auction solves.
The run includes an independent execution/accounting audit and JSON/CSV
comparisons against the retained fixed-forecast grid-feedback result.

`gme_oracle_2025-01-01_farm910_poi_only.yaml` reruns the January study with
individual trade and cumulative position limits derived from the POI, under a
fresh output ID. Earlier PV runs retain their original commercial cap settings.

`gme_oracle_2025-01-01_farm910_reoptimization_imbalance.yaml` adds an ex-post
open-loop settlement audit for every auction decision snapshot. It holds each
snapshot's planned battery commands and contracted grid target fixed, substitutes
the measured PV series, applies physical SOC/POI protection, and settles the
resulting deviations at the realized Terna imbalance prices. The summary CSV
reports both each snapshot's full remaining horizon and the common remaining
horizon across reoptimization steps. It reports settlements against both the
snapshot's own planned target and the final realized contract, so the common
horizon comparison holds the contracted position fixed. The interval CSV
supports drill-down. This open-loop comparison isolates plan quality from the
normal closed-loop execution controller, which actively tracks the committed
grid position.

Future runs automatically save expected-versus-realized mismatches for gross
auction revenue and final net profit in the replay summary and GME KPI JSON/CSV.
The baseline is the initial auction solve; signed differences are realized minus
expected, with percentages divided by the absolute expectation. A zero baseline
has a null percentage, and unavailable imbalance prices leave the final net
profit mismatch null. No additional YAML option is needed.

Completed GME and synthetic trading runs group files by type within each
`<run>/farm_<id>/<study_date>/` directory:

```text
plots/          # Figures; strategy-specific figures under plots/<strategy>/
csv/            # KPI tables, price curves and PV data
csv/<strategy>/ # Executed dispatch, auction plans and the trade ledger
json/           # Manifest, effective configuration, KPIs and audit records
json/<strategy>/# Summary, solves, publication records and event trace
models/<strategy>/ # Exported solver models
```

The manifest is `json/manifest.json` and inventories the existing artifacts
relative to the study directory. Stored model references follow the grouped
paths. Single `optimize` runs use `json/result.json` and `arrays/solution.npz`.
Only the latest grid-feedback study was retained during output cleanup; older
experiment/QA outputs were removed, while input data and acquisition/import
provenance remain available.

GME runs save `plots/pv_power_allocation.png` and its source data in
`csv/pv_power_allocation.csv`. The three stacked components (PV exported,
PV charging the battery, and PV curtailed) sum to realized AC PV in every
interval, with the production total overlaid. Uncurtailed PV supplies charging
first; grid-supplied charging and battery discharge are excluded from the PV
components. This replaces the combined `expected_dispatch_by_auction.png`;
the four individual auction reoptimization plots remain available.

Use only `battery.max_power_mw`, `soc_min_percent` and `soc_max_percent` for power
and SOC limits. Separate battery inverter limits and absolute MWh bounds are
rejected. Efficiencies remain configurable independently of power ratings.

Trading has no independent commercial volume cap. Configure `plant.poi_import_mw`
and `plant.poi_export_mw`: each signed auction trade and the cumulative contracted
position after each auction are bounded by these ratings times the delivery
interval duration. Physical grid power uses the same POI ratings. A reversal
trade remains within the directional POI limit. Earlier retained results used the
removed 1 MW commercial ceiling and are historical results under those settings.

Delivery-time plots include both interval boundaries, through next-day local
00:00 when the full day is available. Power and price are held over their actual
hourly or quarter-hour interval; SOC uses its recorded end state, and cumulative
settlement includes the last interval's cash flow at its end. Missing intervals
remain gaps. Boundary drawing does not add observations or energy to the dataset.

The `gme_oracle_2025-01-01_farm910_perfect_pv_comparison.yaml` study adds an
ex-post full-day optimization using the effective realized AC PV series and
all actual auction clearing prices, subject to the same configured assets,
POI, market products and degradation model. This is a clairvoyant benchmark,
not an executable strategy. Its feasible objective (`ObjVal`) is used as the
perfect-information benchmark KPI and compared with feedback profit. Any PV
measurement gaps use the configured execution fallback and are recorded in
the diagnostics.

Every perfect-information comparison also solves a PV-only case with zero
BESS capacity and power, using the same realized PV, clearing prices, POI,
auction products and terminal SOC policy. The reported battery incremental
operating value is the PV+BESS objective minus the PV-only objective, after
modeled battery wear and before battery capital, fixed operating and financing
costs. A one-day ex-post value helps isolate operational contribution but does
not by itself establish investment payback. The config
`gme_oracle_2025-01-01_farm910_perfect_pv_comparison_terminal_soc_50.yaml`
sets the terminal SOC target to 50%.

Set `experiment_outputs.pv_only_baseline: false` in a study YAML to skip the
PV-only optimization, its schedule and plot, and the incremental-value KPI. It
defaults to `true` in the perfect-information comparison study and adds one
optimization solve when enabled.

`gme_oracle_2025-01-01_farm910_reoptimization_value.yaml` measures the value of
later reoptimization using the terminal-50% study. Configure independently:

```yaml
reoptimization_baseline:
  enabled: true
  freeze_after_auctions: null  # Or [DA, IDA1] for selected freeze points.
  plots: true
```

The last active auction and cancelled auctions are always excluded. Setting
`enabled: false` skips the simulations; `plots: false` suppresses their chart.
The feature defaults to disabled if absent. Each baseline keeps already
submitted orders and executed history, freezes the snapshot's remaining orders
and physical commands, and executes with the same feedback against measured PV.
Future frozen quantities are submitted at their original gates. Actual gross
fees, executed wear, import fees and settlement enter the full-day profit.
Reference-minus-frozen profit is a signed cumulative value of all later
reoptimizations; gains across freeze points are not additive. Final energy
differences are reported without salvage valuation. Baselines require
deterministic snapshots and add no optimizer solves. CSV/JSON outputs and the
optional chart follow the standard file-type folders.

Auction selection is `gme_oracle_experiment.participating_auctions`, e.g.
`[IDA1, IDA2]` or `[IDA3]`; absent/null selects every configured dataset. Only
selected active auctions create orders and submission solves. Before an
intraday-only first decision, zero-position grid feedback runs without an
auction optimization and carries actual energy and wear into that decision.

`gme_participation_2025-01-01_03.yaml` drives the `experiment-participation`
command. `participation_study.candidate_auctions` lists optional candidates;
`required_auctions: []` with all four generates 15 non-empty profiles, while
`required_auctions: [DA]` generates eight. `profiles` can instead specify exact
auction lists. Configure common `delivery_days`, output root/run ID, summary
plots, per-profile plots and model exports. Assets and economics use the same
inherited YAML fields as individual experiments. Every profile must complete
the identical dates and effective realized PV series. Day starts independently
reuse configured SOC/fee state. Averages rank full-day realized net profit,
report daily variation and final inventory, and describe the selected dates
only. Output CSVs, JSON and summary plots are separate from each profile's
replay artifacts. No download is triggered by either participation workflow.

`gme_participation_2025-01.yaml` extends the same experiment to all 31 January
2025 delivery days (465 integrated profile/day cases plus 31 sequential all-auction cases). Days independently reset configured
initial SOC and fee state; this is not a rolling multi-day simulation. A fully
cancelled selected profile keeps its day with zero trades/auction solves and the
zero-position physical feedback policy, including executed costs. Its initial
auction expectation is null. Missing data without cancellation evidence fail.
Ranking tables also include medians, 10th/90th percentiles and negative-profit/
zero-active-auction day counts; chart whiskers remain daily standard deviations.
The monthly YAML also enables `execution.infeasible_auction_policy: previous_plan`:
a confirmed infeasible auction solve reuses that auction's orders and physical
commands from the last feasible deterministic plan, then the usual controller
and actual imbalance accounting execute it. No recovery solve or constraint
relaxation is inserted. The default is `error`. Failed decisions, source plans,
fallback counts and optionally IIS reports (`save_infeasibility_diagnostics`)
are retained, including those cases in each profile's common-date average.
Monthly `resume: true` verifies the unchanged configuration and saved input/profit
checkpoints, reuses completed cases, and retains incomplete children under
`diagnostics/incomplete_runs/` before rerunning them.

The monthly configuration enables `participation_study.sequential_all_auctions_baseline`.
The default remains false. This baseline uses `gme_oracle_experiment.auction_planning_mode:
current_auction_only` rather than `integrated`: only the closing auction has free
trade variables, with earlier commitments fixed and no later auction revenue.
Battery dispatch is still planned through delivery end. A later infeasibility
retains prior dispatch commands and submits zero new trades. Sequential results
and paired profit differences are saved separately, and the comparison chart
adds an orange row immediately after integrated all-auction planning. The refreshed
monthly output ID keeps earlier study results intact.

`participation_study.perfect_foresight_reference` defaults to `true` and adds a
gold reference row above the ranked participation profiles. It uses the mean
perfect-information objective with effective realized PV and every configured
available auction. Existing all-auction benchmark schedules are reused; studies
that omit that profile add one perfect-information solve per day, with no extra
auction replay. Daily objectives and summary are saved separately in
`csv/perfect_foresight_daily.csv` and `json/perfect_foresight_reference.json`.
The objective keeps configured costs and terminal SOC; the other rows are actual
settled feedback profits and can have different ending inventory.
