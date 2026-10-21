# Downloading reusable API data

## Stage manual GME and Terna Excel exports while acquisition continues

The manual workbook workflow prepares a separate DuckDB dataset and never opens
`data/api/market.duckdb` during staging:

```bash
PYTHONPATH=src python -m bess_pv_trading stage-market-workbooks \
  --config configs/api_experiments/manual_market_workbooks.yaml
```

The generated `data/staging/manual_market/<run>/` contains a self-contained
`market.duckdb`, `manifest.json`, `daily_coverage.csv`, `monthly_coverage.png`,
`duplicates.json`, `quarantine.json`, and an `import.yaml` for that exact staged package. Raw Excel
bytes and SHA-256 checksums remain in the snapshots. Every normalized record
retains its original filename, source row, textual price, and explicit UTC start
and exclusive end. The local calendar is Europe/Rome. No Excel dependency is
installed: the existing value-only XLSX reader is reused.

- MGP prices remain native hourly products before delivery on 1 October 2025,
  and native quarter-hour products thereafter. Headers and the market regime
  must agree.
  The transition is documented in [GME's go-live notice](https://www.mercatoelettrico.org/en-us/Home/NoticesandAnnouncements/NoticesandAnnouncementsME?id=7101).
- IDA1/2 retain their whole-day quarter-hour products; IDA3 starts at local noon.
  Chronological one-based GME periods map onto actual 23-, 24- or 25-hour days.
- Terna normalized rows select the NORD macrozone and the final imbalance-price
  column. Base/incentive components and auxiliary quantities retain their own
  fields and units. SUD rows stay in the raw source archive, outside the selected
  normalized price dataset.
- Exact overlapping products are compared by provider, dataset, zone, UTC start
  and UTC end. Equal values and quantities are deduplicated; conflicting values
  or overlapping intervals of different duration stop the merge. No averaging,
  interpolation, hourly splitting, zero filling or automatic revision overwrite
  occurs. Valid negative and zero prices are retained.
- Explicit cancelled-auction notes remain cancellation metadata; they do not
  become observations. Missing files and missing price cells remain gaps.
- Terna's user-confirmed autumn export convention uses `02:01/16/31/46` to
  identify the second occurrence of `02:00/15/30/45`. The shared
  `terna_workbook_import.ambiguous_time_policy: second_occurrence_minute_marker`
  decodes those markers only within the actual repeated autumn hour. Both
  occurrences retain their prices and are sorted by UTC: the four first-fold
  quarters precede the four second-fold quarters. `source_timestamp_text`,
  `source_row`, and `local_time_fold` retain the mapping evidence. The same policy
  applies to direct Terna imports and every workbook in manual staging, across
  years. Duplicate/missing quarters and markers outside the repeated hour fail
  validation. Optional `reject` and `quarantine_day` policies remain available.
- Spring days contain 92 quarter-hours: local `01:45` is followed by `03:00`.
  No synthetic prices or intervals fill the missing hour. Nonexistent source
  labels are rejected, as supported by [pandas timezone localization](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.tz_localize.html).
  Ordinary days contain 96 quarters and autumn days 100; native hourly MGP
  before October 2025 contains 23/24/25 products instead. IDA3 remains the 48
  quarters from local noon to midnight.

After the downloader has finished and released the live database, use the
generated configuration for an explicit transactional merge and full audit:

```bash
PYTHONPATH=src python -m bess_pv_trading merge-market-workbooks \
  --config data/staging/manual_market/<run>/import.yaml
PYTHONPATH=src python -m bess_pv_trading check-market-history \
  --config data/staging/manual_market/<run>/import.yaml
```

These commands are prepared for later use; staging never invokes them. The merge
verifies staged checksums/native SQL parity, compares every product against the
existing selected-provider history, adds only new products, and preserves raw
source evidence even for entirely duplicate workbooks. A price/boundary conflict
rolls back the transaction. Repeating the same merge is idempotent. ENTSO-E and
other unselected snapshot pointers must remain unchanged.

The audit covers every requested delivery day from January 2025, including the
January–March IDA history already present in the main database. It checks native
durations and aligned boundaries, NORD/EUR-per-MWh semantics, missing/duplicate
products, cancellation notes, stored checksums, SQL-vs-Parquet parity, and exact
staged-value/interval parity. It writes `postcheck.json` and daily coverage beside
the package. Gaps produce an explicit non-passing status and exit code 2. Focus
day summaries explicitly cover the Q1/April boundary, October resolution change,
and both spring and autumn DST dates. The requested window
is frozen in generated YAML; rerun staging when adding new exports. Available
day-ahead products beyond that window remain preserved and are reported by their
actual end date.

Historical publication remains unknown unless supplied by independent evidence;
import time is never substituted for historical availability.

The optimizer and executor use timezone-aware UTC boundaries and each interval's
elapsed duration for energy, SOC, revenue and settlement. Native product groups
retain chronological order across the two autumn folds. Regression tests cover
92/100-quarter solver horizons, battery efficiency/SOC accounting, all four
auctions, full-day dispatch and ex-post cash reconciliation. Forecast residual
vectors are a separate requirement: calibration needs enough complete days with
the requested product count and rejects insufficient matching DST history.
Price-grid comparisons normalize timestamp precision so identical cached
nanosecond timestamps and calendar/Python timestamps align without shifting any
interval boundary or price.

## Consolidated DuckDB workflow

The combined and provider-specific acquisition configurations use
`api.storage_backend: duckdb` and `api.database_path: data/api/market.duckdb`.
`api.cache_directory: data/api` identifies the old file cache for migration;
it is not the destination of new downloads. `api.database_threads` bounds each
connection's CPU use. The existing `bess` environment needs conda's
`python-duckdb` package; installing or changing environment packages requires
the user's permission.

Import downloaded data without credentials or network access:

```bash
PYTHONPATH=src python -m bess_pv_trading migrate-api-cache --config configs/api_experiments/api_history.yaml
PYTHONPATH=src python -m bess_pv_trading api-cache-status --config configs/api_experiments/api_history.yaml
```

Migration verifies complete current and archived snapshots against both original
checksums, imports exact normalized tables and raw responses, and checks database
round trips against the source values and dtypes. Repeating it does not duplicate
snapshots. Original files are retained as a backup. No-data receipt timestamps
and retained unparsed responses are preserved; partial requests are not treated
as completed data. Reports and coverage plots are grouped by migration run under
`outputs/api/migrations/`.

The database contains native queryable records together with immutable raw and
Parquet snapshot BLOBs. Those BLOBs retain exact provider evidence, original
checksums, pandas dtypes and metadata; the native records support convenient SQL.
Refreshing a request adds a snapshot and changes its current pointer, retaining
earlier revisions. Request identities include all existing provider options and
vintage fields. No rows are resampled, filled or collapsed by delivery timestamp.

Experiments use the database through the existing catalog API and do not fetch
missing data. Their source manifests record database snapshot identifiers and
original checksums. Retrospective studies still cannot infer historical
publication from retrieval time.

Stop acquisition before opening this same embedded database for an experiment;
separate read-only processes can share it when no writer is open. Connections
are closed between requests, including provider retry/backoff waits, and lock
conflicts fail clearly. This workflow does not introduce a database server.

Resume the original command after a provider outage, without `--refresh-cache`.
Completed migrated requests are reused from DuckDB without credentials. New
successful requests commit atomically, so a later network failure keeps earlier
days available for offline analysis. Database acquisition plots are grouped by
provider/dataset/zone/month under `outputs/api/<run>/plots/`; daily CSV/Parquet/PNG
directories are no longer generated.

The command is ready for the existing `bess` environment after DuckDB setup. Run it yourself in a terminal on the machine holding this repository. Credentials are entered invisibly, remembered only for the running process, and never written to configuration, logs, data manifests or the token cache. Authenticated downloads are run by the user; stop acquisition before opening the same database for offline analysis.

Start with the one-day configuration to check your account's subscriptions and actual response coverage:

```bash
cd /home/rghetti/bess_pv_trading
conda activate bess
PYTHONPATH=src python -m bess_pv_trading download-data \
  --config configs/api_experiments/api_download.yaml --interactive
```

Then download the full farm-record window, **1 January 2025 through 31 August 2026**:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data \
  --config configs/api_experiments/api_history.yaml --interactive
```

`api.end` is exclusive and date-only bounds use `Europe/Rome`. The historical configuration ends at `2026-09-01`. It plans 6,080 logical dataset/day chunks across ten datasets; actual HTTP calls can be fewer because Terna's observed and forecast total load share one response. Availability and provider access are checked from responses, not assumed from the requested window. An interrupted run resumes from verified daily caches when the command is rerun.

`api.download_order: month_first` is the default: retrieve all January days for prices, then January load, generation, solar, wind and the configured Terna datasets, before starting February. Daily cache keys are unchanged. Progress identifies the current month and dataset, with updates every 25 chunks and on dataset/month transitions. Set `dataset_first` to retain the older ordering.

A process that was already running before this change retains its original request order. To adopt monthly scheduling, press Ctrl-C in that terminal and rerun the same command above **without `--refresh-cache`**. Completed prices and load days will be reused. An interrupted in-flight request can be fetched again; completed daily caches remain available.

The history configuration uses eight attempts for transient 429/5xx/network
failures with two-second exponential backoff capped at 30 seconds per wait.
The CLI prints each retry. HTTP 503 is service unavailability before parsing;
it is not evidence of a schema mismatch. A failure after the bounded retry
budget retains all earlier daily caches. Terna has separate two-second request pacing.

`api.terna_http_403_policy: continue_other_providers` is the shared default.
The first HTTP 403 from a Terna data or token request pauses Terna network
requests for the remainder of that run. Cached complete Terna chunks and
cached no-data receipts can still be reused. All other configured providers,
including ENTSO-E requests in later months, continue with their existing pacing
and retry settings. This handles a quota-related 403 without claiming that
every 403 proves quota exhaustion; other access restrictions can return it too.

The denied chunk is recorded as `failed-http`, and remaining uncached Terna
chunks as `deferred-provider`. Neither status creates a no-data receipt or a
completed cache entry. The saved manifest reports `status: partial`, failed and
deferred counts, the sanitized provider pause reason and per-provider completion
totals. The CLI returns exit code 2 only after finishing the other providers.
Rerun the same command when access is available again: completed ENTSO-E chunks
are reused and pending Terna requests are attempted again. Pause state is local
to the process and does not change credentials or persist a provider block.
Set the policy to `error` to retain immediate failure on HTTP 403. Other errors
keep their existing behavior; this policy does not bypass ENTSO-E's own limits
or retries. Restart an already-running downloader to use the updated code.

All provider configurations now share `data/api/market.duckdb`. The legacy
migration source is `data/api`. The original five Terna smoke caches were copied
from `data/api/cache` into the canonical file cache after checking equivalent
requests, raw checksums and identical normalized values. Their original nested
snapshots remain untouched and are excluded from migration to avoid duplicate
imports; the reuse manifest is under `outputs/api/cache_reuse/`. Omitted and
null interval assumptions are equivalent only in that explicitly verified copy;
new smoke/history requests use identical options and cache identities.

Inspect monthly completeness offline after stopping the downloader:

```bash
PYTHONPATH=src python -m bess_pv_trading api-cache-status \
  --config configs/api_experiments/api_history.yaml
```

This command is offline. It compares the configured daily requests against completed snapshot metadata and distinguishes cached data, explicit no-data receipts and missing chunks; invalid cache entries raise an error. A no-data receipt is a known gap, not a usable observation. Status checks read database metadata (and file presence for legacy caches); experiment ingestion additionally verifies raw and Parquet checksums.

Inspect the plan without network access or credentials:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data \
  --config configs/api_experiments/api_history.yaml --dry-run
```

For unattended use, set `ENTSOE_API_TOKEN`, `TERNA_CLIENT_ID` and `TERNA_CLIENT_SECRET` securely in the process environment and omit `--interactive`. Environment-variable names can be changed under `api.credentials`. An uncached unattended request fails immediately when credentials are absent. Cached requests do not access credentials. Terna access tokens are renewed using the expiry returned by the provider; credentials are not repeatedly prompted during a long run.

Provider-only examples are `configs/api_experiments/entsoe_api.yaml` and `configs/api_experiments/terna_api.yaml`. The combined configurations use one common cache root.

## What is downloaded

| Provider | Dataset | Retained physical meaning |
| --- | --- | --- |
| ENTSO-E | `day_ahead_prices` | Native NORD DA clearing-price products, EUR/MWh |
| ENTSO-E | `load_forecast` | Day-ahead load forecast, MW |
| ENTSO-E | `generation_forecast` | Day-ahead generation forecast, MW; direction retained |
| ENTSO-E | `solar_forecast` | Day-ahead solar forecast, MW |
| ENTSO-E | `wind_onshore_forecast` | Day-ahead onshore-wind forecast, MW |
| Terna | `total_load` | NORD total-load observations, MW |
| Terna | `total_load_forecast` | NORD total-load forecast field, MW |
| Terna | `load_forecast` | NORD MGP-session load forecast, MW |
| Terna | `wind_forecast` | NORD MSD1-session wind forecast, MW; not relabeled DA |
| Terna | `renewable_generation` | National photovoltaic/wind observations, GWh; not relabeled forecasts |
| GME | `day_ahead_prices` | Native NORD MGP zonal prices, EUR/MWh |
| GME | `ida1_prices`, `ida2_prices`, `ida3_prices` | Actual MI-A1/2/3 zonal prices, EUR/MWh; IDA3 afternoon products only |

Terna's total-load query uses the documented provider zone name `North`; normalized rows use `NORD`. National renewable actuals are stored under `IT` and stay GWh. ENTSO-E's configured NORD EIC is `10Y1001A1001A73I`. Native hourly prices before the quarter-hour DA regime stay hourly. No negative prices are clipped, no missing market prices are filled, and no observed hourly price is presented as four observed quarter-hour prices.

The ENTSO-E/Terna requests do not provide IDA1/2/3 clearing prices. The separate
GME adapter now supports their documented API, using GME-issued credentials and
the same DuckDB storage. Start with `configs/api_experiments/gme_api.yaml`, then
`configs/api_experiments/gme_history.yaml`; see [GME commands, access and schema limits](gme_api.md).
Its actual authenticated response coverage remains to be checked by the user.
Terna imbalance-price downloads remain blocked because the official fee documentation gives conflicting routes; the code does not guess between them. An imported price or an API download does not establish the applicable historical settlement formula. See [ENTSO-E details](entsoe_api.md), [Terna details](terna_api.md) and [sources](sources.md).

## Legacy storage layout

The following directory format remains supported for migration and explicit
`api.storage_backend: files` configurations. Existing directories are retained;
the default DuckDB workflow above does not write new daily files:

```text
data/api/
  entsoe/day_ahead_prices/NORD/2025-10-01/<request-id>/
    response.xml
    records.parquet
    records.csv
    manifest.json
    values.png
    revisions/<previous-retrieval-UTC>/...
  terna/total_load/NORD/2025-10-01/<request-id>/
    response.json
    records.parquet
    records.csv
    manifest.json
    values.png
  terna/renewable_generation/IT/2025-10-01/<request-id>/...
outputs/api/<run-UTC>/manifest.json
```

Parquet is the reusable typed table; CSV supports direct inspection. Raw provider payloads remain beside the normalized data. Manifests identify the request, options, source conventions, retrieval time, dataset documentation, schema version and payload/table checksums. Cache reads verify both the raw and Parquet files. Request keys include provider, dataset, zone, interval, options and requested vintage/revision; authentication is transport-only. Normalized rows keep original source identifiers, units, timing and revisions where supplied.

`--refresh-cache` explicitly requests fresh responses, archives previous files in `revisions/`, and regenerates the plot. Identical source HTTP requests can supply several documented fields while preserving their common retrieval timestamp. Reuse never crosses different requested vintage/revision keys.

Explicit no-data responses are recorded as `no_data.json` receipts. They remain gaps, with no manufactured zero measurements or prices. The downloader continues with other days and reports `complete-with-gaps`. Absence receipts expire after the configured `no_data_cache_hours` (24 by default); use `--refresh-cache` to retry immediately. Authentication failures, unsupported schemas, ambiguous timestamps and invalid quantities fail clearly. Earlier successful daily caches remain available for the next run.

## Read the data again

```python
from bess_pv_trading.data.api_catalog import load_api_cache_records

prices, provenance = load_api_cache_records(
    "data/api/market.duckdb", provider="entsoe",
    dataset="day_ahead_prices", zone="NORD",
)
```

For direct SQL inspection, `current_api_records` contains the current snapshot
of each exact request, and `api_records` also retains archived revisions:

```python
import duckdb

with duckdb.connect("data/api/market.duckdb", read_only=True) as connection:
    coverage = connection.execute("""
        SELECT provider, dataset, zone, count(*) AS records
        FROM current_api_records
        GROUP BY provider, dataset, zone
        ORDER BY provider, dataset, zone
    """).df()
```

Use the catalog reader for experiment inputs: it verifies the retained source
and normalized snapshot checksums and restores their original DataFrame dtypes.

The reader selects current completed snapshots and verifies checksums by default.
Passing a legacy cache directory instead of a `.duckdb` path remains supported.
Different request IDs can still represent overlapping vintages; retain their
provenance and select the intended vintage rather than silently dropping duplicates.

Common columns include `provider`, `dataset`, `zone`, `value`, `unit`, `product_start`, `product_end`, `retrieved_at`, `source_document_time`, `historical_available_at`, `availability_provenance` and `historical_feature_eligible`. Terna also retains `reference_time` and source records. Unverified interval boundaries remain missing instead of being inferred from cadence; explicit configured interval assumptions are marked separately.

**Retrieval today is not historical publication.** ENTSO-E document creation and Terna forecast-reference labels do not, by themselves, establish availability before a historical auction. Such rows retain unknown historical availability and are ineligible for causal historical features. Exact publication records or an explicitly justified policy are needed before historical feature/backtest use. Timestamped snapshots collected now can provide availability evidence for subsequent operational decisions; they do not reconstruct absent old vintages.

## Start price experiments with an incomplete download window

The price-only experiment reads completed DA caches and needs no credentials or other datasets:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-api-prices \
  --config configs/price_experiments/price_experiment.yaml
```

It compares a ridge regression using calendar cycles and same-local-clock price lags of one and seven days against previous-day and previous-week baselines. The final configured delivery days are held out chronologically; feature scaling and coefficient fitting use the training period only. Hourly and quarter-hour delivery regimes are fitted and evaluated separately. Missing lag clocks, including unmatched DST folds, are excluded without filling prices.

The source may encode pre-October hourly prices as constant blocks on a 15-minute transport grid. The configured market-regime policy reconstructs hourly observations only from complete, equal within-hour source values; it retains every quarter-hour price from 1 October 2025. A03 compression block length alone does not identify the commercial product duration. [GME's go-live notice](https://www.mercatoelettrico.org/en-us/Home/NoticesandAnnouncements/NoticesandAnnouncementsME?id=7101) establishes the transition date.

Derived monthly price tables are stored under `data/analysis/api_prices/`; experiment predictions, metrics, models, figures, configuration and source-checksum provenance are grouped under `outputs/price_experiments/<run>/`. Raw API files are read only. These are **retrospective experiments using retrieved revisions**, not validated historical trading backtests: historical publication and revision availability are not reconstructed. IDA regressions still need their actual labels and timing evidence.

The regional-forecast prototype runs on the complete January ENTSO-E caches:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-price-fundamentals \
  --config configs/price_experiments/price_fundamentals_january.yaml
```

It compares calendar/price lags, added regional load, and added load/solar/wind
forecasts. Every model uses the same complete training and evaluation products.
Quarter-hour MW forecasts are duration-averaged inside native hourly price
products only when coverage is complete; gaps stay missing, overlaps fail and
coarser forecasts are not split into finer products. Hourly prices are not
interpolated. Train-only scaling/fitting uses 8–24 January (408 products), after
seven days of lag history; 25–31 January is held out (168 products). Earlier
holdout days may supply subsequent days' observed price lags. No noise is added.

The frozen source-checksum provenance and joined inputs are under
`outputs/price_fundamentals/<run>/`; convenient monthly CSV/Parquet views are under
`data/analysis/price_fundamentals/NORD/year=2025/month=01/`. Forecast publication
and revision availability remain unverified. This short study validates the
comparison pipeline; it cannot establish broad forecasting skill or trading
profitability. Forecast-source columns identify ENTSO-E day-ahead forecasts,
not Terna national actual generation or its later MSD-session wind forecasts.

## Offline verification

```bash
PYTHONPATH=src python -m pytest -q \
  tests/unit/test_api_download.py \
  tests/unit/test_entsoe_api.py tests/unit/test_terna_api.py
```

The tests cover official response shapes, native product durations, UTC/DST, no-data gaps, cache integrity and resume, duplicate-response reuse, authentication, credential redaction, retries, and refresh history. They do not prove account permissions or historical provider completeness.
