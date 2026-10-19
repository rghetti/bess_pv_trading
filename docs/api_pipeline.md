# Downloading reusable API data

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
budget retains all earlier daily caches. Authentication/permission failures
remain immediate errors. Terna has separate two-second request pacing.

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
