# Full experiment configurations

Store one complete, reproducible end-to-end study configuration per YAML file in this directory. These configurations should explicitly provide the dataset and snapshot selection, study dates, PV forecast selection, market calendar and product resolution, battery and POI parameters, inverter efficiencies, starting and terminal state, imbalance policy, degradation model, Gurobi settings, and output locations. Keep illustrative values clearly labeled; the runner should validate required fields rather than silently supplying plant-specific defaults.

`synthetic_fees_none.yaml`, `synthetic_fees_flat.yaml` and
`synthetic_fees_gme.yaml` compare transaction-fee modes on the same market.
Use `optimize` for the initial plan or `run-synthetic` for the executed ledger.
The GME example declares a synthetic starting operator volume and explicitly
assumes the published tariff applied historically. See
[fee formulas, state and reporting](../../docs/formulation.md#market-transaction-fees).

Small synthetic solver examples remain here for software validation. Add real-data studies as separate files with names that describe the study and period.

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
