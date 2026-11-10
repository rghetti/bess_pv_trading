# Price forecasting data inventory

The independent checkout `/home/rghetti/bess_pv_trading_features` has a source-data catalog at `data/api/market.duckdb` and separate prepared feature packages under `outputs/price_features`. The catalog's current-record view was inspected on 2 November 2026. It contains market prices, gas prices, Italian electricity forecasts and observations, weather forecast archives, and French and Swiss realized electricity balances.

A record's presence in the database does not by itself establish that it was available at a historical auction gate. Most records have unknown historical publication times, and several prepared features use explicitly labelled timing assumptions. Date ranges below are observed extents, not claims of uninterrupted coverage. Counts are current catalog rows, which can include overlapping source snapshots and should not automatically be read as unique products.

## Electricity and gas prices

| Dataset | What is stored | Observed coverage and qualifications |
|---|---|---|
| GME NORD day-ahead, MGP | Zonal clearing prices in EUR/MWh, with source intervals and workbook provenance. | Records span local delivery dates in 2024 through October 2026. The 2024 audit found 8,784 distinct hourly products, complete for all 366 local days. DA was hourly before October 2025 and quarter-hourly afterward in the prepared workflow. |
| GME NORD intraday, MI-A1/2/3 | Zonal auction prices in EUR/MWh, including source cancellation information where supplied. | Records span 2024 through October 2026. The supplied 2024 exports are hourly; whether those values are native products or hourly summaries has not been established. Later data include quarter-hour products. MI-A3 covers only its applicable delivery horizon, so its smaller row count does not alone indicate a gap. |
| ENTSO-E NORD day-ahead | Another source of NORD DA prices, in EUR/MWh. | Current rows run from the start of 2025 through 16 October 2026, at quarter-hour resolution as returned and normalized in this catalog. This is a separate source from the GME series. |
| GME PUN | National PUN price, separate from NORD. | December 2024: 744 hourly rows. |
| GME IGI gas index | Daily PSV gas index, EUR/MWh, with the gas delivery date retained separately from electricity-product intervals. | 974 distinct gas dates, 1 January 2024–31 August 2026, all with values. The files do not supply an observed publication clock. |
| Terna NORD imbalance prices | Quarter-hour imbalance settlement prices, with additional settlement fields where provided. | Local delivery dates from 1 January 2025 through 30 September 2026; 61,244 rows. These are settlement observations, distinct from auction clearing prices. |

GME prices retain source files and snapshots, but the original historical publication timestamps were not supplied. The detailed 2024 source audit is in `bess_pv_trading_features/docs/history_2024.md`.

## Italian electricity fundamentals

ENTSO-E NORD forecasts contain four separate MW series: load, total generation, solar generation and onshore wind generation. The current catalog has approximately 67,000 rows per series, spanning local delivery dates from 2024 through August 2026. Source records include hourly and quarter-hour intervals, depending on period and response.

For 2024, the stored audit found that load and total generation covered every 15-minute audit slot. Solar and wind each lacked the local delivery dates 7 March, 9 October, 27 October and 12 December, totaling 97 missing hours per series. These checks do not certify the historical publication time or prove that the returned forecast revision was the one visible before an auction.

Terna records contain the following:

| Series | Type and geographic scope | Observed reference-time extent |
|---|---|---|
| MGP load forecast | Forecast, NORD, MW | January–October 2025 |
| Total-load forecast | Forecast, NORD, MW | January–October 2025 |
| MSD1 wind forecast | Forecast, NORD, MW | January–12 October 2025 |
| Total load | Realized load, NORD, MW | January–October 2025 |
| Renewable generation | Realized generation, national Italy, GWh | January–September 2025 |

The Terna forecast reference labels and, for the relevant load and wind series, interval boundaries remain unresolved in the feature workflow unless explicitly configured. The wind records carry a provider publication-date classification; that does not by itself resolve interval semantics or establish an auction-specific publication clock. The 2024 acquisition was deferred, so the existing Terna fundamentals are principally 2025 data.

The database does not currently show a GME preliminary-demand dataset in its current records, even though the public-feature workflow documents how such files could be imported.

## Weather forecast archives

There are two distinct weather sources. Both retain forecast initialization, valid time, lead, location, variable, units and source provenance.

| | Farm archive | Regional Open-Meteo archive |
|---|---|---|
| Locations | Four anonymous farms: 1, 345, 901, 910 | Nine named points: Milan, Turin, Genoa, Bologna, Verona, Venice, Bolzano, Trieste and Aosta |
| Variables | 2 m temperature, shortwave radiation, cloud cover, and east/west and north/south wind components at 10 m and 100 m | 2 m temperature, shortwave radiation, cloud cover, and wind speed at 10 m and 100 m |
| Forecast horizon | Stored leads 0–48 hours | Stored leads 0–48 hours; configured archive requests are for 00 and 06 UTC runs |
| Stored extent | Valid times from late December 2024 to early September 2026 | Valid times from March 2024 to early September 2026 |
| Volume | Approximately 3.35 million variable records | Approximately 3.63 million variable records |
| Availability | Underlying NWP source and actual historical publication are unverified; the feature config uses an inferred four-hour release lag | ECMWF IFS Single Runs archive; actual release is unverified and the feature config uses an inferred six-hour release lag |

Regional locations remain individual sampling points; the data are not labelled as a load-weighted or generation-weighted NORD average. Shortwave radiation is stored as an hourly mean ending at its timestamp; the other listed weather variables are treated as instantaneous.

The 2024 regional archive has substantial explicit gaps. The stored audit reports 438 of 732 requested 00/06 UTC runs present, with 294 missing. The acquisition documentation identifies the early IFS archive as hindcasts, so those records are not evidence of operational forecasts issued at the time. The same audit reports a missing lead-zero shortwave value for each available 2024 run/location group. The farm archive does not provide full-year 2024 weather.

The archive retains structural missing values and source values, including negative radiation values; those are not automatically replaced with zeros. The detailed weather and availability descriptions are in `bess_pv_trading_features/docs/price_features.md`.

## French and Swiss electricity balances

Both are realized national system observations, expressed as a balance proxy with positive values when reported consumption exceeds generation. They are not measured cross-border flows to Italy.

| Source | Data and resolution | Stored extent |
|---|---|---|
| RTE France | Half-hour actual consumption, generation components, pumping and a derived balance. Source nature, including provisional/consolidated status, is retained. | Local delivery dates from 2024 through August 2026; 46,746 rows. The 2024 audit found 17,566 of 17,568 expected half-hour intervals. |
| Swissgrid Switzerland | Quarter-hour control-block production and consumption, originally in kWh and normalized to average MW; balance is consumption minus production. | Local delivery dates from 2025 through September 2026; 61,244 rows. No 2024 Swiss annual workbook is in the current catalog. |

The two countries' consumption definitions differ, particularly around pumping, losses and own-use. Historical publication times are unknown. The source fields and calculations are described in `bess_pv_trading_features/docs/country_balance_features.md`.

## Prepared tables and provenance

The source catalog stores normalized records alongside immutable snapshots, raw payloads or source files, checksums, request identities, retrieval times, source metadata and explicit no-data receipts. It also retains superseded snapshot history; `current_api_records` identifies the current snapshot for each request. Product timestamps are stored with time-zone information, and the workflow preserves distinct repeated autumn DST hours.

Prepared packages contain native-product feature tables, feature lineage, source snapshot references, coverage reports and validation files. The inspected package `outputs/price_features/package-20261027T122925515399Z` in the features checkout is a 71-row, three-day validation package, not a complete training table. It contains DA target prices, matching-period price lags at 1, 2, 3, 7 and 14 days, a two-week same-weekday average, IGI, and the four ENTSO-E forecast series. Weather was excluded from that package. Its forecast and gas features use the documented `assume_day_ahead_known` policy; passing its timing checks confirms consistency with those assumptions, not historical publication.

Across the source catalog, almost all price, fundamental and weather records are marked historically ineligible because publication was not verified; the Terna wind publication-date classification is an exception in the stored flags, with the qualifications above. Actual target-day load, generation and national balances are observations, while auction prices used as targets are ex-post labels. The feature workflow records assumed, inferred and known availability separately.
