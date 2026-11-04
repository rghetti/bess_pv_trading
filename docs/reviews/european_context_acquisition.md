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

The Swiss Federal Office of Energy separately publishes [weekly reservoir
levels](https://www.bfe.admin.ch/de/fuellungsgrad-speicherseen). The public CSV
was acquired on 4 November 2026 as snapshot
`4fbf941d15d7256605f9e92e0f184982adf4b9ca57cd91a3dbcbb224b47ff111`
(raw SHA-256 `520038460d8c73f821486ced6b2c292af3e4d43fd4d1dc89c775a0cc8122544c`).
It contains 1,401 weekly observations from January 2000 through November 2026,
with regional and national stored energy and national capacity percentage.
The reference is Sunday midnight in Zurich time. The energy dashboard reports
Thursday publication; the configured Friday 00:00 local availability is an
assumption and does not establish historical release times or revisions.
This is stored hydro energy, not hydro generation or cross-border flow. The
currently running price comparison was configured before this acquisition and
does not include reservoir levels. A future feature must carry forward only the
latest assumed released value and cap its age; otherwise an interval join would
incorrectly discard it or introduce unavailable values.
