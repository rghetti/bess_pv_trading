# ENTSO-E input pipeline

The ENTSO-E adapter implements these documented data items. `configs/entsoe_api.yaml`
requests one local delivery day for NORD; `end` is exclusive. The shared downloader
splits requests by local day, checks each cache first, preserves raw responses,
and writes canonical records and coverage metadata. No packages are installed.

| Dataset | documentType | processType | Additional filter | Unit |
|---|---|---|---|---|
| `day_ahead_prices` | A44 | absent | identical `in_Domain` / `out_Domain` | EUR/MWh |
| `load_forecast` | A65 | A01 | `outBiddingZone_Domain` | MW |
| `generation_forecast` | A71 | A01 | `in_Domain` | MW |
| `solar_forecast` | A69 | A01 | `in_Domain`, `psrType=B16` | MW |
| `wind_onshore_forecast` | A69 | A01 | `in_Domain`, `psrType=B19` | MW |
| `wind_offshore_forecast` | A69 | A01 | `in_Domain`, `psrType=B18` | MW |

The production endpoint is `https://web-api.tp.entsoe.eu/api`, confirmed in the
[official endpoint reference](https://transparencyplatform.zendesk.com/hc/en-us/articles/15696677194644-Request-Endpoint).
The document/process/domain mapping follows the
[official extraction guide](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fweb+api%2FIG-for-TP-data-extraction-process.pdf),
and generation/load XML units and types follow the
[Generation and Load implementation guide](https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/resources/Transparency/MoP%20Ref05%20-%20EMFIP-1-gl-market-document_V3R0-2014-01-24.pdf).
Solar and wind PSR codes are checked against the
[official PSR list](https://transparencyplatform.zendesk.com/hc/en-us/articles/15856995130004).
The [current API reference](https://documenter.getpostman.com/view/7009892/2s93JtP3F6)
is linked by ENTSO-E's
[request-parameter documentation](https://transparencyplatform.zendesk.com/hc/en-us/articles/15696716612372-Request-Parameters).
Its interactive page did not expose readable request bodies during implementation;
the public collection export returned HTTP 403. No undocumented extra query filters
are inferred from third-party clients.

`10Y1001A1001A73I` is listed as BZN IT-North in the
[official EIC area list](https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC).
The same code appears for NORD in the
[June 2021 scheduling-area annex](https://eepublicdownloads.entsoe.eu/clean-documents/nc-tasks/SDAC%20costs%20coefficient%20%E2%80%93%2017.06.2021.pdf).
Those references support the zone identity, without proving a dataset's coverage on
each requested historical date. A coverage report must be inspected after download.
Country IT is a different scope and is never silently substituted for NORD.

## Credentials and cache

Use `ENTSOE_API_TOKEN` in the process environment, or the downloader's explicit
interactive credential option for hidden terminal input. Cached requests need no
token. Unattended calls never prompt. The token is added only by the HTTP transport;
the request builder, configs and manifests contain no secret. Provider rejection
text is suppressed because a server could reflect request parameters.

## Interval and information contracts

The adapter decodes namespaced `Publication_MarketDocument` and `GL_MarketDocument`.
It validates document, process, zone, units, PSR, revisions and positions. Negative
prices survive unchanged. A71 production and consumption series retain a `direction`
column; they are never silently summed or netted.

All products use explicit UTC start/end timestamps. PT15M, PT30M and PT60M retain
their source duration. A01 missing positions stay missing; A03 constant blocks are
decoded at the source resolution until the next position or Period end. This is
source-format decoding: hourly products never become observed quarter-hour prices.
Disjoint Periods remain gaps. Missing/depublished values stay NaN with a status;
duplicates and overlapping products fail explicitly. These rules follow the
[official curve-type specification](https://eepublicdownloads.entsoe.eu/clean-documents/EDI/Library/cim_based/Introduction_of_different_Timeseries_possibilities__curvetypes__with_ENTSO-E_electronic_document_v1.4.pdf).

The [official response documentation](https://transparencyplatform.zendesk.com/hc/en-us/articles/15727773247124-Query-Response)
confirms UTC responses, A03 curves, whole-day expansion for price requests, and
HTTP 200 acknowledgements for no matching data. Actual UTC interval bounds preserve
23/25-hour local days. An acknowledgement is never parsed as a successful dataset.

`source_document_time` records `createdDateTime`; `retrieved_at` records retrieval.
Neither proves historical publication. Downloaded rows therefore have an empty
`historical_available_at`, `historical_feature_eligible=false`, and explicit
availability provenance. API snapshots cannot enter a causal historical feature
set or the strict market importer until publication evidence is supplied. Returning
a forecast for a historical delivery day does not establish its original vintage.

IDA1/2/3 and Italian imbalance settlement remain outside this verified adapter.
A non-DA contract is rejected if returned under A44. No continuous-market index,
national price or balancing-energy price is substituted for an auction or imbalance
settlement. Live authenticated coverage is not claimed by the fixture tests.
