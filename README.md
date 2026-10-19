# PV+BESS auction trading and farm audit

The new application under `src/bess_pv_trading/` implements a node-based stochastic Gurobi MILP, causal data policy and pvlib AC forecasts, price regression/scenario interfaces, and an event-driven synthetic auction replay. The original audit below remains available. The implementation plan is `CODEX_PV_BESS_IMPLEMENTATION_PROMPT_DATASET_BOUND.md`; actual status and evidence are in [docs/implementation_status.pdf](docs/implementation_status.pdf), with [LaTeX](docs/implementation_status.tex) and [JSON](docs/implementation_status.json).

The default relative MIP gap is `1e-4` (0.01%); no absolute MIP gap is configured. This convention is also recorded in `AGENTS.md`.

Every submitted auction quantity is assumed to execute in full at its published
clearing price. Quantities stay fixed while their price is unpublished. Each
publication books the actual cash flow once and refreshes conditional forecasts
and the residual plan; forecast objectives and realized profit are reported
separately.

**API acquisition:** run the verified provider pipeline yourself with hidden credential entry:

```bash
conda activate bess
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/api_download.yaml --interactive
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/api_history.yaml --interactive
```

The first command checks one day; the second covers the farm-record window, 1 January 2025–31 August 2026. The provider configurations now store raw responses, exact typed snapshots, revision history and queryable records together in `data/api/market.duckdb`. Requests still complete month by month and resume at daily request boundaries after an outage. Acquisition plots are grouped by dataset/month under `outputs/api/<run>/`; CSV exports are no longer duplicated for each request. `--dry-run` needs no credentials or network, while `--refresh-cache` retains previous database snapshots. See [API commands and storage contract](docs/api_pipeline.md). IDA prices, verified imbalance settlement and unrecorded historical publication remain separate gaps.

Import the existing verified file cache once, entirely offline, before resuming acquisition or running experiments:

```bash
PYTHONPATH=src python -m bess_pv_trading migrate-api-cache --config configs/api_experiments/api_history.yaml
```

Migration is resumable and idempotent. It verifies raw and normalized checksums, compares imported values and dtypes, and writes its report under `outputs/api/migrations/`. Original files remain a backup. The Python dependency is supplied by conda's `python-duckdb` package in the existing `bess` environment. Farm Zarr data is independent of this migration.

**GME intraday prices:** after obtaining a separate GME API username/password,
run the one-day check and then the missing IDA history:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/gme_api.yaml --interactive
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/gme_history.yaml --interactive
```

Both use `data/api/market.duckdb`. The second command requests actual MI-A1,
MI-A2 and MI-A3 NORD prices, month by month, without repeating the ENTSO-E DA
download. Run one writer at a time. [GME access and storage details](docs/gme_api.md)
distinguish documented/offline-tested parsing from pending authenticated checks.

For the downloaded GME XLSX files, import hourly MGP prices for January–August
2025 and native quarter-hour MI-A1/2/3 prices for 1 January–1 April 2025 inclusive,
then run the configured one-day oracle benchmark:

```bash
PYTHONPATH=src python -m bess_pv_trading import-gme-workbooks --config configs/api_experiments/gme_workbook_import.yaml
PYTHONPATH=src python -m bess_pv_trading experiment-gme-oracle --config configs/full_experiments/gme_oracle_2025-01-01.yaml
```

Before October 2025, DA orders remain constant across the four quarters of each
hour; IDA orders and battery controls can vary every 15 minutes at all four
optimization steps. The importer validates chronological `Periodo` indices,
daylight-saving transitions and complete daily curves. It archives and removes
the obsolete hourly IDA snapshots while preserving MGP and other providers.
Only documented or explicitly configured cancellations are skipped; missing
or partial curves fail. Earlier saved January results require a new run with
the corrected data and product resolutions.

Results include auction prices, four dispatch plans, PV allocation, SOC, KPIs
and solver/model diagnostics, grouped into subfolders by file type. The
perfect-price result is an oracle benchmark. The farm910 throughput study
loads actual Terna NORD quarter-hour settlement prices for post-processing.
BESS and POI values remain configurable in the full-experiment YAML files.

Completed database snapshots can be used immediately, even when the configured download window is incomplete:

```bash
PYTHONPATH=src python -m bess_pv_trading api-cache-status --config configs/api_experiments/api_history.yaml
PYTHONPATH=src python -m bess_pv_trading experiment-api-prices --config configs/price_experiments/price_experiment.yaml
```

Experiments are offline and read only completed snapshots. Stop the downloader before opening the same database for analysis; embedded DuckDB uses a process-level write lock. Several offline experiments can share the database read-only. All connections are short-lived so provider backoff does not hold a database lock.

The offline price experiment compares calendar/lag ridge regression with day/week seasonal baselines on a chronological holdout, separately for hourly and quarter-hour regimes. Monthly reusable tables go under `data/analysis/api_prices/`, with metrics, predictions, models and plots under `outputs/price_experiments/<run>/`. Retrieved revisions support a retrospective study; unknown historical publication prevents claiming a validated trading backtest.

The first regional-forecast comparison uses the complete January 2025 ENTSO-E
price/load/solar/wind caches and holds out 25–31 January:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-price-fundamentals --config configs/price_experiments/price_fundamentals_january.yaml
```

It compares calendar/price lags, added load, and added load/solar/wind against
day/week baselines on identical products. Monthly joined tables live under
`data/analysis/price_fundamentals/`; frozen inputs, provenance, models, predictions
and plots live under `outputs/price_fundamentals/<run>/`. This small retrospective
study uses 17 training days and a seven-day holdout; historical forecast vintages
remain unverified. Long history downloads now retry transient failures up to
eight times with visible bounded backoff; rerun the same command after an outage
to reuse completed caches.

The rolling linear-price pipeline compares richer lag/calendar/profile models,
including a model for the change from yesterday's price, against yesterday,
last week and their fixed average:

```bash
PYTHONPATH=src python -m bess_pv_trading study-linear-prices --config configs/price_experiments/price_linear_study.yaml
PYTHONPATH=src python -m bess_pv_trading study-linear-prices --config configs/price_experiments/price_linear_fundamentals.yaml
PYTHONPATH=src python -m bess_pv_trading study-linear-prices --config configs/price_experiments/price_linear_hourly.yaml
PYTHONPATH=src python -m bess_pv_trading study-linear-prices --config configs/price_experiments/price_linear_methods.yaml
```

Model settings are selected on an earlier validation period. A separate period
estimates forecast bias and Gaussian error covariance; the final chronological
holdout measures accuracy and interval coverage. Coefficients refit on a configured
schedule using earlier days only. Hourly and quarter-hour prices stay separate.
The fundamentals configuration uses the common completed forecast history as it
arrives. Each run freezes inputs and source manifests under
`outputs/linear_price_studies/<run>/`, with reusable monthly features under
`data/analysis/linear_prices/`. Reloadable Gaussian artifacts, held-out scores,
sample price paths and plots live in that run's `uncertainty/` subdirectory.

The hourly-structure configuration also compares separate linear coefficients
for each delivery hour. The fitting-method comparison reuses that run's frozen
DuckDB-derived inputs through August 2026: ordinary least squares, ridge,
lasso, elastic net, Huber and recency-weighted ridge. All methods retain
training-only scaling, chronological selection and portable coefficients.
Method-specific penalties and iterative convergence settings are saved with
each candidate; failed convergence cannot silently become a different fit.
A separate command selects rolling Gaussian covariance
settings using the calibration period alone, then updates uncertainty from
completed earlier forecast errors:

```bash
PYTHONPATH=src python -m bess_pv_trading calibrate-price-scenarios --config configs/price_experiments/price_gaussian_adaptive.yaml
```

On the existing July–August quarter-hour holdout, the selected point model's
MAE is 16.44 EUR/MWh, versus 17.50 for the fixed day/week average. Adaptive
Gaussian 95% intervals cover 92.3% of prices; their estimated bias correction
also increases MAE to 17.20. These results do not establish calibrated uncertainty.
The saved September–October protocol freezes these settings before inspecting
the additional prices. Download that price-only extension, then evaluate it
after the downloader finishes:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/entsoe_price_extension.yaml --interactive
PYTHONPATH=src python -m bess_pv_trading check-frozen-prices --config configs/price_experiments/price_forward_check.yaml
```

The check reads DuckDB snapshots and saves a new run under
`outputs/price_forward_checks/`. Incomplete data produces a waiting status rather
than a partial scored holdout. Weekly coefficient refits and Gaussian updates
use only earlier eligible observations; model and covariance settings are not
reselected from the new period.

The completed September 1–October 16 check scores 4,416 prices. The frozen ridge
model has MAE 15.56 EUR/MWh versus 18.29 for the day/week average; the weighted
model has MAE 16.16. Their adaptive nominal 95% coverage is 94.25% and 95.24%,
respectively. The weighted intervals are wider, and its CRPS is worse. Both
checks reproduce all 8,160 earlier point predictions exactly. These are
previously uninspected historical revisions, with unverified publication
vintages; the results do not establish operational trading performance.

The Gaussian mean is the linear prediction plus a separately estimated bias;
noise has zero mean and covariance estimated from out-of-sample prediction
errors. `forecasting.gaussian_prices.gaussian_price_tree` connects these paths to
the MILP's existing scenario/tree contracts. It rejects unverified vintages by
default; retrospective studies require an explicit opt-in. A DA-only schedule
is supported while actual IDA labels remain absent. See
[price study protocol and limitations](docs/market_price_notes.md).

`build_future_linear_price_features` assembles tomorrow's predictors without
requiring its realized prices. `LinearPriceForecast.load(...).predict(...)`
reuses a selected point model, and `GaussianPriceCalibration.load(...)` reuses
its uncertainty calibration. Feature settings and model cutoffs must match the
saved study; the most recent refit cannot be used to predict earlier test days.

`forecasting.price_oracle.build_perfect_foresight_scenarios` supplies the separate
perfect-foresight benchmark. It repeats actual price curves across the supplied
PV members, currently one, with no price noise. It retains member probabilities
and shared decisions until completed PV observations distinguish members. Its
outputs explicitly identify a non-executable benchmark. Forecast and oracle
comparisons must use the same assets, settlement assumptions and PV observation
convention.

**Depth-dependent aging:** `configs/full_experiments/synthetic_xu.yaml` enables the implemented Xu segmented cycle-depth equations, with discharge-only costs, carried segment memory and explicitly illustrative lifetime/cost parameters. It can be used with either `optimize` or `run-synthetic`. The throughput baseline remains available. See [Xu equations, assumptions and executed tests](docs/xu_degradation.md).

Use the existing `bess` environment without installing anything:

```bash
conda activate bess
PYTHONPATH=src python -m bess_pv_trading validate-config --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading inspect-data --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading prepare-analysis-data --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading forecast-pv --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading optimize --config configs/full_experiments/synthetic_small.yaml
PYTHONPATH=src python -m bess_pv_trading run-synthetic --config configs/full_experiments/synthetic_small.yaml
PYTHONPATH=src python -m bess_pv_trading run-synthetic --config configs/full_experiments/synthetic_dst_fall.yaml
PYTHONPATH=src python -m bess_pv_trading run-synthetic --config configs/full_experiments/synthetic_dst_spring.yaml
PYTHONPATH=src python -m bess_pv_trading import-market-data --config configs/price_experiments/synthetic_prices.yaml
PYTHONPATH=src python -m bess_pv_trading train-price-models --config configs/price_experiments/synthetic_prices.yaml
PYTHONPATH=src python -m pytest -q --junitxml=docs/generated/tests.xml
PYTHONPATH=src python -m bess_pv_trading report-implementation --config configs/report.yaml
```

`backtest` accepts the synthetic configuration and runs its complete event replay. Historical market replay rejects missing real auction prices, unresolved settlement and unspecified hardware. `configs/price_experiments/italy_nord.example.yaml` describes the canonical market import/training boundary; it requires actual files. `configs/price_experiments/synthetic_prices.yaml` uses a small, explicitly synthetic sparse fixture, preserves gaps and demonstrates the import/cache/regression commands. It does not supply historical market labels. `optimize` also accepts prepared complete PV/point-price CSV forecasts and explicit equipment through `configs/full_experiments/prepared_optimization.example.yaml`; its null inputs must be supplied first. That route validates provenance and supports initial pre-DA planning.

The broader prepared-input route in
`configs/full_experiments/stochastic_prepared.example.yaml` accepts probabilistic PV members,
saved point/conditional-spread models and Gaussian calibrations, known auction
prices and committed positions. It supports both initial and residual solves
through the same `optimize` command. See the
[canonical input contract](docs/forecast_optimization_inputs.md) for the table
schemas, timing checks and explicit study assumptions. Its integration evidence
uses synthetic fixtures; actual intraday history is still required for empirical
trading evaluation.

Trading outputs are grouped under `outputs/trading/<run>/farm_<id>/<delivery day>/` by file type: `plots/<strategy>/`, `csv/<strategy>/`, `json/<strategy>/` and `models/<strategy>/`. These contain the immutable accepted-trade ledger, dispatch, event trace, solver/independent validation records, cash flow summaries and plots. The run manifest is `json/manifest.json`; GME full experiments use the same layout, with study-wide figures and KPI files directly inside the corresponding type folders. Single `optimize` runs save `json/result.json` and `arrays/solution.npz`. Generated run IDs are unique; existing run directories cannot be overwritten. Equipment values in `synthetic_small.yaml` are illustrative. Ordinary, spring and autumn DST days use 96, 92 and 100 products. The four separate benchmarks use the same realization and constraints; the oracle explicitly receives future truth. Pending submitted quantities remain fixed until publication, when contractual cash enters the accepted ledger once.

`configs/farms_nord.yaml` binds the actual schema. Its conservative real-data PV example flags four BHI/GHI contradictions and cannot provide a complete optimizer input. `configs/farms_nord_radiation_sensitivity.yaml` explicitly clips those contradictions as an exploratory sensitivity. Both retain source values, flags, exact-grid AC evaluation and assumptions. No farm or market series is silently made complete.

Measured production is already **AC**. Installed capacity is **DC nameplate**. The assumed **0.97 efficiency is the BESS converter's efficiency in each direction**; it is not applied to measured PV or direct PV exports. The fallback PV DC-to-AC forecast factor is separately configured and unfitted. Milan geometry, tracker parameters and farm-901 front-side-only treatment remain declared approximations. Available aging models are constant AC charge-plus-discharge throughput and Xu segmented cycle-depth aging; their coefficients and replacement cost remain explicit assumptions.

See [dataset contract](docs/dataset_contract.md), [assumptions](docs/assumptions.md), [formulation](docs/formulation.md), [data/PV notes](docs/data_pv_notes.md), [market/price notes](docs/market_price_notes.md) and [consulted sources](docs/sources.md). No dependencies, licenses, commits or remotes are changed by these commands. If an execution sandbox masks the host identifier, solver commands need the ordinary host execution context for the existing license.

## Preserved farm data audit

The first implementation audits four anonymous PV plants before any PV model is fitted. It reads the full Zarr time series and Parquet farm metadata, keeps all raw values, and writes reviewable CSV tables and PNG plots. There is no automatic correction, exclusion mask, time shift, or fault verdict.

The source files live in `data/farms/`. The `data/` directory is also reserved for later price downloads; raw and downloaded data are ignored by Git.

## Run

Use the existing `bess` environment and run from this repository:

```bash
conda activate bess
python -m bess_pv_trading.audit --config configs/audit.yaml
```

The command creates a new `outputs/qa/<UTC run ID>/` directory. Set `outputs.run_id` in the YAML to give a run a fixed name; a run ID cannot overwrite an existing output directory. The CLI prints the destination at completion. Every run includes a copy of its effective YAML config.

The config chooses input paths, the inclusive UTC historical interval-start window, Milan proxy coordinates, the assumed forecast publication lag, diagnostic bins and thresholds, and plot limits. Paths in the config are resolved from this repository. The bins are *count sensitivities*: changing a bin does not change any source value. The episode thresholds only select review candidates and are also adjustable.

`plots.pinned_events` can name a farm, category, and UTC instant to keep specific observed episodes in the bounded plot set. The supplied config pins the four previously identified examples; pins outside a selected date window are skipped.

## Input conventions

| Coordinate | Meaning |
| --- | --- |
| `time` | Start of a historical 15-minute interval. `power` is average AC kW; satellite `ghi` is W/m² with the same start-label convention. The satellite within-interval measurement operator is not independently established. |
| `cycle_time` | Timestamp identifying one forecast issue or vintage, usually six hours after the preceding cycle. It is not a delivery timestamp. |
| `lead_time` | Offset from `cycle_time` to a forecast target, at hourly steps from zero to 48 hours in the current store. |
| `valid_time` | `cycle_time + lead_time`. For radiation, this marks the *end* of the forecast hour: an hourly value at 11:00 UTC represents `[10:00, 11:00)`. |

The current store has one forecast per cycle and **no ensemble-member dimension**. Later ensemble data can add members while retaining the vintage and valid-time structure. A cycle is treated as usable from `cycle_time + availability_lag_hours` (four hours in the default config). For each historical quarter-hour the audit attaches the latest forecast whose cycle was available before its forecast hour started. Forecast accuracy tables separately score every forecast vintage against the mean of its four matching historical GHI quarters, where all four quarters are present. No future-issued cycle is used as event context.

The Parquet file supplies the farm-level metadata by `farm_id`, including `total_power`, which is **installed DC capacity in kW**. Solar elevation is calculated at the midpoint of each historical interval with Milan as a proxy; farm coordinates are anonymous, so twilight classifications are approximate. All timestamps in outputs are UTC, including those on daylight-saving transition days.

The new Zarr labels `shortwave_radiation` and `direct_radiation` as hourly interval-average W/m² without identifying the source parameters. If they originated from ECMWF `ssrd` and `fdir`, respectively, their physical counterparts are total shortwave and direct-beam irradiance on a **horizontal** surface. ECMWF defines [ssrd](https://codes.ecmwf.int/grib/param-db/169) as direct plus diffuse surface radiation and [fdir](https://codes.ecmwf.int/grib/param-db/228021) as its direct component on a horizontal plane. That source mapping remains an inference, not a verified field-level provenance claim. The audit compares shortwave to historical GHI as a diagnostic; it does not call either forecast ground truth.

## Review files

| File | Use |
| --- | --- |
| `summary.md` | Coverage, grid and missing checks, event counts, reporting-bin sensitivity, forecast errors, lag correlations, and examples to inspect. |
| `events.csv` | One row per adjacent suspicious episode, with a stable ID, UTC interval, raw extrema, neighboring values, solar elevation, and available forecast context. A simultaneous daytime GHI-zero episode has a portfolio row as well as individual-farm rows. |
| `historical_context.csv` | One row per farm and 15-minute historical interval, with raw power/GHI and time-aligned forecast context; open or plot directly in VS Code or pandas. |
| `forecast_checks.csv` | Forecast-vs-GHI error summaries by farm and lead hour. |
| `review_template.csv` | Event IDs with blank `decision` and `notes` columns for a human review. |
| `plots/<farm_id>/<date>/` | PNG overview and bounded event-window plots. |

An event is a prompt to inspect a period, not a judgement that the underlying GHI or power is wrong. The candidate families include deep-night positive power, daylight zero GHI, unexpectedly low power under substantial GHI, abrupt power or GHI moves, constant positive power, missing measurements, and values outside configurable physical ranges. Event windows include surrounding points to show whether the series recovers. Shared GHI events help distinguish a common irradiance data issue from one plant's power issue.

The CSV uses short decimal representations of the source `float32` values. Reading those columns back as `float32` exactly reproduces the Zarr values; reading them as default `float64` can display tiny decimal differences because of the change in numeric type.

Reviewing and labeling these events remains a separate task. The new application applies only the explicitly requested derived policy, retaining original audit verdicts. Legacy audit column names such as `power_dc_kw` are preserved for compatibility and refer to the corrected measured AC series; they do not imply a second conversion.

## Cloud correlation study

Run the supplementary study against a completed audit:

```bash
python -m bess_pv_trading.cloud_check --config configs/audit.yaml \
  --audit-dir outputs/qa/<run_id>
```

It compares shared GHI zeros and sudden AC power drops with available total, low, middle and high cloud forecasts. The main comparisons select intervals with recorded/forecast irradiance or production evidence; they do not infer actual daylight from Milan coordinates. An irradiance-only comparison shows the effect of possible faulty power readings. Milan solar elevation is retained as a reference and selection sensitivity, and never determines a fault verdict. Instantaneous hourly cloud cover is interpolated within one forecast cycle to each measurement midpoint. Cloud tendencies over 15 and 60 minutes also use one cycle, which prevents forecast revisions from appearing as passing clouds. The study writes CSVs, a report, and plots below that audit's `cloud_cover_check/<study_run_id>/` directory.

The comparisons include cloud-bin event rates, month/UTC-hour matched controls for shared zeros, selection and hourly aggregation sensitivity checks, and power-drop rates conditioned on enough preceding production to allow the drop. Cloud cover is a forecast of coverage, and does not measure cloud optical thickness or verify what happened directly over the farm. `cloud_analysis` in the YAML sets the comparison thresholds and plot limits. These define reporting cohorts, not cleaning rules.

The supplementary study checks that the audit's date window, solar proxy, publication lag and event-selection thresholds still match the supplied YAML. Rerun the main audit first if those choices change.
