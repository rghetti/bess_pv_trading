# GME day-ahead and intraday auction prices

The downloader supports GME's documented `PublicMarketResults` /
`ME_ZonalPrices` API. Its separate GME-issued username and password are requested
invisibly after a cache miss. Register through the [official access page](https://www.mercatoelettrico.org/en-us/Home/APIService).
ENTSO-E tokens and Terna application credentials do not grant GME access.

After registration, first check one complete ordinary delivery day:

```bash
conda activate bess
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/gme_api.yaml --interactive
```

This requests NORD MGP, MI-A1, MI-A2 and MI-A3 for 1 October 2025. Then acquire
the missing intraday history for 1 January 2025 through 31 August 2026:

```bash
PYTHONPATH=src python -m bess_pv_trading download-data --config configs/api_experiments/gme_history.yaml --interactive
```

The history configuration requests three auctions across 608 delivery days,
or 1,824 daily cache requests, completing all datasets for a month before the
next month. Existing ENTSO-E DA data is retained; the GME history command does
not redownload it. Rerun without `--refresh-cache` to resume. Both configurations
write `data/api/market.duckdb`, with run manifests and monthly plots under
`outputs/api/gme/`. Use one acquisition process for this embedded database at a
time. Offline planning/status also needs the database to be free of another
process's write lock.

For unattended runs, set `GME_USERNAME` and `GME_PASSWORD` securely in the
process environment. Their names are configurable under `api.credentials`.
The API uses JSON POST authentication and subsequent Bearer requests; login
responses and tokens stay in memory. The configured 900-second local refresh
bound is shortened by a readable JWT expiry. Request pacing and bounded retries
are configurable; user-specific quotas are not assumed.

| Dataset | GME segment | Native observed products in the supported window |
|---|---|---|
| `day_ahead_prices` | MGP | Hourly before 1 October 2025; quarter-hour thereafter |
| `ida1_prices` | MI-A1 | Quarter-hour, full delivery day |
| `ida2_prices` | MI-A2 | Quarter-hour, full delivery day |
| `ida3_prices` | MI-A3 | Quarter-hour, local noon through midnight |

The parser preserves source `FlowDate`, `Hour`, `Period`, market, zone and notes.
Native chronological products map to `Europe/Rome`, including DST folds. IDA3
retains 48 afternoon products rather than invented morning rows. Negative
prices survive. Aggregated DA PT30/PT60 responses cannot substitute for native
quarter-hour observations. Missing prices remain missing; explicit empty or
cancelled auctions become bounded no-data receipts and remain coverage gaps.

The raw JSON envelope contains a base64 ZIP response. The downloader validates
and decodes it in memory, retaining the exact original envelope and normalized
typed snapshot inside DuckDB. Safe schema mismatches can be retained and
reparsed after a parser fix without another request. Credential echoes in the
envelope, decoded file or archive metadata prevent retention. Existing provider
request identities are unchanged.

```python
from bess_pv_trading.data.api_catalog import load_api_cache_records

ida1, snapshots = load_api_cache_records(
    "data/api/market.duckdb", provider="gme", dataset="ida1_prices", zone="NORD",
)
```

The source guide does not supply historical publication/revision timestamps for
each returned price record. Retrieval, flow dates and ordinary result schedules
do not replace that evidence: `historical_feature_eligible` remains false and
publication timestamps remain missing. Actual authenticated schema/coverage
still needs the user's first check; offline tests use explicitly synthetic
documented-format responses. The alternative is a user-exported XML archive
from [GME's public MI-A1 price download page](https://www.mercatoelettrico.org/it-it/Home/Esiti/Elettricita/MI/MI-A1/Download/Download?valore=Prezzi),
with corresponding MI-A2/MI-A3 pages. Its wide XML format requires its own
explicit normalization, rather than the JSON API decoder.

## Manual GME XLSX import and one-day experiment

The four user-downloaded NORD workbooks for 1 January–31 August 2025 can be
imported offline into the same DuckDB catalog. Paths, date bounds, column names,
decimal conventions, and native resolution are declared in
`configs/api_experiments/gme_workbook_import.yaml`:

```bash
PYTHONPATH=src python -m bess_pv_trading import-gme-workbooks \
  --config configs/api_experiments/gme_workbook_import.yaml
```

The importer retains each original XLSX as a checksummed DuckDB snapshot and
normalizes observed native products. Explicit cancellation notes become gaps;
they are not converted to zero prices. A missing complete auction is skipped
by the study, while a partial or misaligned curve stops the run.

The configured 1 January 2025 benchmark is in
`configs/full_experiments/gme_oracle_2025-01-01.yaml` and runs with:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-gme-oracle \
  --config configs/full_experiments/gme_oracle_2025-01-01.yaml
```

It selects the latest causally available PV weather vintage at every auction
reoptimization, measured AC output for
execution after each interval closes (with only explicitly configured missing
interval fallback), and actual GME clearing curves as an oracle. The oracle is a
perfect-price upper-bound comparison, not an executable strategy. Auction
timing assumptions, the 50% initial SOC, illustrative plant and BESS ratings,
all efficiencies, imbalance assumptions, and Gurobi settings are in YAML.
Artifacts include separate MGP/MI-A1/MI-A2 and MI-A3 price plots, each auction's
expected dispatch plan with its PV forecast, forecast-versus-realized PV,
percentage SOC, dispatch and ledger tables, per-solve
model sizes and exported LP files, and a KPI manifest under
`outputs/full_experiments/gme_oracle/`.

The larger-farm throughput study is configured separately:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-gme-oracle \
  --config configs/full_experiments/gme_oracle_2025-01-01_farm910_throughput.yaml
```

Choose a fresh `experiment_outputs.run_id` when rerunning an existing named
study. The runner protects completed outputs. Each active auction triggers one
optimization before submission. Price publication updates the fixed-trade
ledger and information only; its journal is `json/oracle/price_publications.json`.
This study enforces planned delivery per native hour with zero imbalance cost,
while recording realized forecast errors separately. See
[battery and delivery-balance equations](battery_physical_model.md).

For PV conversion, pvlib supplies DC power. The existing configurable DC-to-AC
output factor is preserved and multiplied by the separately configured
`pv.output.transformer_efficiency`; the current value `1.0` means no measured
transformer loss is assumed, not that the transformer is known lossless.

Sources: [GME API technical guide, pages 5–7 and 49–50](https://gme.mercatoelettrico.org/Portals/0/Documents/en-US/20251015Manuale_tecnico_API_En.pdf),
[GME January 2025 product rule](https://www.mercatoelettrico.org/Portals/0/Documents/it-IT/20250101DTF25MPE.pdf).
