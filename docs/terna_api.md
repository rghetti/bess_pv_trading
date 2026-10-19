# Terna acquisition adapter

The adapter implements authenticated retrieval and schema normalization for five
series below. The API routes and fields were checked against Terna's public
documentation on 2026-10-16. Tests use provider-shaped fixtures and an injected
transport. User-run responses have also been inspected, including the wind
schema correction below; the agent did not access the user's credentials.

Inspect the five planned requests without credentials or network access:

```bash
python -m bess_pv_trading download-data --config configs/api_experiments/terna_api.yaml --dry-run
```

Run the same command without `--dry-run` after setting the credential environment
variables, or use `--interactive` for hidden terminal entry. The top-level `end`
date is exclusive. Existing daily caches are read before credentials are needed;
`--refresh-cache` explicitly requests a new source snapshot. Raw responses,
normalized tables, retrieval clocks and provenance now live in
`data/api/market.duckdb`, selected by the provider configuration. Provider,
dataset, zone and local day remain request metadata keys. Run manifests and
dataset/month plots live beneath `outputs/api/terna/`. Existing daily files are
retained for migration and explicit legacy `storage_backend: files` runs; see
[the shared storage contract](api_pipeline.md).

| Dataset | Route | Response array / value | Interpretation |
| --- | --- | --- | --- |
| `total_load` | `/load/v2.0/total-load` | `total_load` / `total_load_MW` | Load observation, MW |
| `total_load_forecast` | same | `total_load` / `forecast_total_load_MW` | Forecast load, MW; original forecast availability unknown |
| `load_forecast` | `/market/v1.0/input/forecast-load` | `forecast_load` / `load_MW` | Explicit MGP or MSD session forecast, MW |
| `wind_forecast` | `/generation/v2.0/wind-production-forecast` | observed `wind_production_forecast` / `forecast`; documented `forecast_load` / `load_MW` also accepted | Wind forecast for MSD1 through MSD6; MW follows the endpoint documentation |
| `renewable_generation` | `/generation/v2.0/renewable-generation` | `renewable_generation` / `renewable_generation_GWh` | National actual energy by source, GWh |

Reference: [total load](https://developer.terna.it/docs/read/apis_catalog/load/Total_Load),
[load forecast](https://developer.terna.it/docs/read/apis_catalog/market/Load_Forecast),
[wind forecast](https://developer.terna.it/docs/read/apis_catalog/generation/Wind_Production_Forecast),
and [renewable generation](https://developer.terna.it/docs/read/apis_catalog/generation/Renewable_Generation).
Renewable actuals remain GWh; the parser does not assume a duration to convert
them into MW, nor describe national observations as NORD forecasts. No verified
solar-forecast or DA/IDA clearing-price route is implemented in this adapter.

The user-downloaded wind response for 1 July 2025, MSD1, contains 768 records
across eight zones under `wind_production_forecast`, with the value named
`forecast`. Filtering `North` gives 96 NORD records. This differs from the public
page's `forecast_load` / `load_MW` example. Both explicit variants are supported;
mixed arrays or missing value fields are rejected. Rows retain `source_array`,
`source_field` and `unit_provenance`. The observed `forecast` field has no unit
suffix; its MW interpretation follows the endpoint documentation and is recorded
as such. The response's explicit publication clock is preserved; its quarter-hour
reference cadence alone does not establish interval start/end semantics.

Authentication uses a form-encoded POST to
`https://api.terna.it/public-api/access-token`, with `client_id`, `client_secret`,
and `grant_type=client_credentials`. Terna documents Bearer tokens with an
`expires_in` of 300 seconds. The caller sends `Authorization: Bearer ...` only to
the fixed API host. There is no documented scope parameter; none is invented.
Credentials belong in `TERNA_CLIENT_ID` / `TERNA_CLIENT_SECRET` or the download
command's hidden interactive prompt. Tokens and token responses are not data
cache artifacts. See the official [token instructions](https://developer.terna.it/docs/read/Access_Token)
and [request examples](https://developer.terna.it/docs/read/How_to_make_a_request).

In the Terna application page, **Key** maps to `TERNA_CLIENT_ID` and **Secret**
maps to `TERNA_CLIENT_SECRET`; use both from the same application. The API/package
name is not a credential. Terna's [application setup instructions](https://developer.terna.it/docs/read/How_to_register_an_application)
say to retain the **Public** package, which grants access to its public APIs.
Existing environment values take precedence over interactive entry; unset the
two variables before restarting with `--interactive` to enter them afresh.

HTTP 401 means valid authentication was not accepted. HTTP 403 means access was
refused; the status alone does not establish whether credentials, application
permissions or another provider policy caused it. The downloader distinguishes
`Terna authentication/token request failed` from `Terna data request for ... failed
after obtaining an access token`. The latter establishes that token acquisition
succeeded before the data request was rejected. Errors omit response bodies,
query parameters, credentials and tokens. Authentication failures stop the run;
rerunning reuses completed daily caches.

When token acquisition succeeds but the dataset returns 403, the public examples
leave an authorization-format ambiguity: the general request guide shows
`Authorization: Bearer <token>`, while the Total Load cURL example shows
`Authorization: <token>`. Compare those documented formats explicitly:

```bash
PYTHONPATH=src python -m bess_pv_trading diagnose-terna --config configs/api_experiments/terna_api.yaml --interactive
```

This performs one token exchange and one GET per format on the first configured
Terna day/dataset. Each request is made once, without redirects; credentials,
tokens, authorization headers and response bodies are not printed or saved.
Successful bodies must pass the Terna parser. Nonsecret `business_id` values can
be supplied to Terna support to identify the rejected requests. The command does
not write caches or change the downloader. An explicit
`api.terna_authorization_scheme: raw` enables the raw-token example for ordinary
downloads if the diagnostic confirms it works; the default remains `bearer`.
Both failures leave other request requirements or provider restrictions unresolved.

The user-run check on 16 October 2026 returned 96 parsed NORD total-load records
for 1 July 2025 with `bearer`; `raw` returned HTTP 401. Keep `bearer`. The check
also included `businessID`, so it does not establish whether the earlier 403 was
temporary or related to that header. Ordinary Terna GETs now include the same
documented optional support header. Its generated UUID is transport-only and
does not alter daily cache keys; a rejected request includes it in the error for
Terna support. This is not evidence that the header caused or resolved the 403.

When a successful HTTP response fails schema validation, the downloader retains
the raw response and checksum-bearing schema-error metadata in DuckDB, after
checking that the response does not echo the active credentials or access token.
With the explicit legacy file backend these are `unparsed_response.json` and
`schema_error.json` beside the daily cache location. Retained unparsed responses
are neither normalized data nor no-data receipts. A normal restart reparses
that exact response without requesting
credentials or downloading it again; `--refresh-cache` explicitly fetches a new
response. Earlier successful daily caches remain intact. This allows correction
against the observed provider schema when it differs from the public examples.

`configs/api_experiments/terna_wind_check.yaml` isolates the configured wind request with two-second
minimum pacing and one attempt. It uses the same daily cache identity as the
provider smoke configuration. On HTTP failures, `provider_hint` is a fixed,
heuristic category derived in memory from the provider error (rate/quota,
invalid/expired token, insufficient scope/subscription, policy block, or
unspecified); no response text or arbitrary provider field is printed.
The combined and Terna-only configurations now use a configurable two-second
minimum interval for Terna operations, while ENTSO-E keeps its existing pacing.
This is a precaution during acquisition, not proof that rate limiting caused
the earlier intermittent 403 responses.

Queries use inclusive `dateFrom` / `dateTo` in `dd/mm/yyyy`, built from explicit
Italian calendar dates. The caller slices by day for cache reuse. `biddingZone`
and `type` are repeated parameters when multiple values are requested. The
documented total-load bidding-zone spelling is English (`North`, not `NORD`);
misspellings may otherwise return an empty body. `zone` in configuration is a
local response filter. No pagination parameters, maximum range, guaranteed
historical coverage, or forecast revision archive are documented for these
routes. The parser rejects an unexpected continuation hint and reports duplicate
source identities rather than silently treating a response as complete.

`reference_time` preserves Terna's reference label in UTC and retains the source
zone, offset, dimensions, and complete source record. Source offsets are checked
against the named timezone. Repeated autumn hours need an explicit offset;
nonexistent spring hours and inconsistent offsets fail validation. Where a
reference's interval boundary convention remains unverified, `product_start`
and `product_end` remain missing. Setting both `interval_minutes` and
`timestamp_label` (`start` or `end`) explicitly aligns them and records
`configured_interval_assumption`. The example leaves these unset; cadence alone
does not establish interval semantics.

`retrieved_at` is separate from `source_document_time` and
`historical_available_at`. Wind's `prevision_date` is retained as document time,
never inferred to be publication time. A date-only `publication_date` is
preserved without creating a midnight clock. Only an explicit publication clock
sets `historical_available_at` and `historical_feature_eligible=true`. Neither
retrieval today nor Terna's general schedule description proves the historical
availability of a returned revision. Unknown availability excludes the series
from causal historical feature selection until independently supported evidence
is supplied.

Automatic imbalance price API acquisition remains explicitly blocked. Terna's
[Daily Prices](https://developer.terna.it/docs/read/apis_catalog/fees/Daily_Prices)
and [Preliminary Prices](https://developer.terna.it/docs/read/apis_catalog/fees/Preliminary_Prices)
pages each disagree internally: their principal route uses `/fees/v1.0/...`,
whereas their cURL example uses `/ifees/v1.0/market/...`. The fields include
`unbalance_price_EURxMWh`, but the live authenticated endpoint and historical
settlement regime still require confirmation. The adapter fails with this
specific reason instead of trying multiple undocumented alternatives. Manual
imports with explicit product, revision, and publication provenance remain
available in `data/market_prices.py`. Terna's separate market-output UP/DOWN
prices are balancing-market aggregates; they are not IDA clearing prices.

## Manual quarter-hour settlement workbooks

Import the provided monthly `Riepilogo_Mensile_Quarto_Orario_YYYYMM.xlsx`
exports with:

```bash
PYTHONPATH=src python -m bess_pv_trading import-terna-workbooks --config configs/api_experiments/terna_workbook_import.yaml
```

The YAML selects `NORD` by its field value, not worksheet row order. Row 2
contains column headers; row 1 is a title. The normalized dataset is
`provider=terna, dataset=imbalance_prices, zone=NORD` in `data/api/market.duckdb`.
It retains the final price, base price, incentive, aggregate sign/volume,
exchange quantities, original row and file, and source checksum. The original
XLSX bytes and normalized table are immutable monthly snapshots. Publication
time is unknown, so these exports are not eligible historical forecast features.

The explicit label convention is local interval start in `Europe/Rome`, converted
to UTC. Every expected native quarter-hour must occur exactly once. The March
2025 file contains 2,972 NORD rows, including 92 on the spring DST-transition day.
Autumn labels without offsets are rejected rather than guessed. No prices are
interpolated or resampled, and zero prices remain valid. Base plus incentive is
checked against the final price under the YAML rounding tolerance; settlement
uses the final column directly.

The importer writes a report with monthly row counts and snapshot identifiers to
`outputs/data_import/terna_imbalance_prices/NORD/2025-01_2025-08/manifest.json`.
Repeated imports verify existing snapshots and reuse them when source bytes and
normalization settings match.

The full experiment's `settlement` YAML section enables exact interval matching
against the executed trajectory. The price table is loaded after physical replay,
never supplied to the optimizer. Executed surplus/shortfall is priced separately
in every quarter-hour with signed cash flow `price * imbalance_mwh`. The run saves
the aligned source table, interval cash flows, receipts/payments and corrected
profit, plus settlement and profit-breakdown plots.
