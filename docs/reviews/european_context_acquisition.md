# French and Swiss electricity context

The acquisition workflow `scripts/download_european_context.py` uses the study
configuration `configs/research/european_context.yaml` and the existing feature
checkout's `data/api/market.duckdb`. Run it only when no incompatible downloader
or writer is using that database. It retains raw responses, normalized rows,
snapshot IDs and hashes; retrieval time is not historical publication time.

The existing RTE monthly snapshots contain French national nuclear and hydro
production, and retain the source's France–Italy and France–Switzerland
commercial exchange fields in `source_record`. These are commercial schedules,
not physical border flows. The source data and original native half-hour
handling are described in the reused `country_balances` parser. Its source is
the [RTE consolidated éCO2mix export](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/information/).
At this inspection the database contained 32 monthly RTE snapshots covering
2024 through August 2026. Requests for September and October 2026 returned empty
JSON arrays. The original downloader labeled those requests complete; it has now
been corrected so an empty response remains a recorded snapshot but the month
is reported incomplete and can be retried. No missing interval is set to zero.

Swissgrid's [cross-border data](https://www.swissgrid.ch/en/home/operation/grid-data/cross-border-load-flows.html)
were downloaded on 4 November 2026. Snapshot ID:
`8a39b27971c4d3c1d4dce2fffb2d963653bc1d47246ef9bc528879295b39871b`.
Raw SHA-256:
`e65026c4ecc56367f29a333b7397b25c61ceda4a3926865e5da81a64b6af6007`.
The 2026 file contains 29,280 native quarter-hour timestamps through 1 November
and eight selected fields, for 234,240 rows. These fields are directional
intraday NTC, commercial net schedules and day-ahead spot spreads for the Swiss
borders with Italy and France. Signed values and missing fields are retained as
published. The fall repeated hour is preserved as two distinct UTC intervals.

The source states that its file is published weekly, but the exact release time
of each historical record is not in the downloaded file. Its current file is a
retrospective snapshot and does not prove a historical decision-time vintage.
It is not a source of physical border flows or day-ahead NTC. The official page
currently links only its 2026 annual CSV; earlier years must not be silently
assumed present. Missing-value counts and the exact provider URL are retained
in the snapshot metadata.

French nuclear and hydro data and a Swiss aggregate balance are already available
for the declared price ablations. A Swiss regional reservoir series is published
separately by the Swiss Federal Office of Energy; it is not represented by the
Swissgrid balance or the border CSV and has not yet been acquired here.
