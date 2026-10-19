# Full experiment configurations

Store one complete, reproducible end-to-end study configuration per YAML file in this directory. These configurations should explicitly provide the dataset and snapshot selection, study dates, PV forecast selection, market calendar and product resolution, battery and POI parameters, inverter efficiencies, starting and terminal state, imbalance policy, degradation model, Gurobi settings, and output locations. Keep illustrative values clearly labeled; the runner should validate required fields rather than silently supplying plant-specific defaults.

Small synthetic solver examples remain here for software validation. Add real-data studies as separate files with names that describe the study and period.

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
the corrected POI-derived cumulative position limits and a fresh output ID.
Earlier fixed and rolling PV runs retain their original commercial cap settings.

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

Use only `battery.max_power_mw`, `soc_min_percent` and `soc_max_percent` for power
and SOC limits. Separate battery inverter limits and absolute MWh bounds are
rejected. Efficiencies remain configurable independently of power ratings.

Trading has no independent commercial volume cap. Configure `plant.poi_import_mw`
and `plant.poi_export_mw`: cumulative contracted positions after each auction
are bounded by these ratings times the delivery interval duration. Physical grid
power uses the same POI ratings. An auction adjustment may close and reverse
a previous position; it has no separate cap. Earlier retained results used the
removed 1 MW commercial ceiling and are historical results under those settings.

Delivery-time plots include both interval boundaries, through next-day local
00:00 when the full day is available. Power and price are held over their actual
hourly or quarter-hour interval; SOC uses its recorded end state, and cumulative
settlement includes the last interval's cash flow at its end. Missing intervals
remain gaps. Boundary drawing does not add observations or energy to the dataset.
