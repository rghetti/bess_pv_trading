# Updating the price-study catalog

From the repository root, run this after any other database writer has finished:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.update_price_market_data \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --providers energy_charts entsoe france_final france_realtime \
    swissgrid_energy swissgrid_border bfe --run
```

The end date defaults to tomorrow in Europe/Rome, so available source data
through the current local date can be requested. The default 14-day lookback
supports routine daily runs. If the updater has not run for longer, add
`--since YYYY-MM-DD` to request an earlier backfill. This command excludes GME
API credentials because the GME workbooks are imported manually.
ENTSO-E requests within the requested window are refreshed so that a previously
partial daily response does not permanently mask later published records.
The ENTSO-E token is requested with hidden input only when absent from the
environment and the process has a terminal. It is kept in the process environment,
not saved in YAML or DuckDB. For unattended execution use a securely provisioned
`ENTSOE_API_TOKEN` and add `--non-interactive`; absent credentials are reported.

The updater processes providers sequentially and writes a dated JSON report in
`outputs/data_updates/price_market/`. It can update public Energy Charts DA,
ENTSO-E DA/load/generation/solar/wind when a token is supplied, GME DA/IDA when
GME-specific credentials are available, RTE consolidated and provisional real-
time observations, Swissgrid physical data and border data, and BFE reservoirs.
Source publication and historical availability remain distinct from retrieval.
RTE's real-time dataset is a separate provisional source and does not silently
replace consolidated values. To capture delayed RTE consolidated/final releases
and revisions, `france_final` refreshes a configured 550-day window by month;
the real-time source uses the normal request window. RTE can publish empty
consolidated months until its release, so inspect the result status and keep the
real-time fallback distinct. Swissgrid physical data are published monthly;
the updater discovers each current official workbook URL instead of using a
stale link, and also refreshes the preceding year during its configured revision
window. Swissgrid border files are discovered by year and the preceding year is
refreshed around New Year. If a new annual workbook is not yet published, the
report marks discovery as incomplete. Unchanged nonempty RTE files and
unchanged Swissgrid/BFE downloads reuse their existing snapshots rather than
reimporting the same payload every day. Provider failures or missing credentials
are reported individually; a completed request does not prove every product
has already been published.

RTE requests use stable full-month boundaries, including for the currently open
month. This does not require the future part of the month to exist: the report
separately counts expected and observed half-hours through the last completed
interval, using the source timezone and preserving DST. It reports both the
number of timestamped source rows and rows with a finite balance; it does not
assert that every generation or border feature is populated. The provider's
`status` indicates whether requests completed; `coverage_status` and each
month's `coverage` indicate whether elapsed intervals are present. A refresh
that omits previously stored intervals preserves the earlier snapshot and is
reported as partial. An upstream empty final response can reflect a delayed
release and is not proof that a real-time month is absent.

GME IDA is not supplied by the Energy Charts DA endpoint. If GME credentials are
unavailable, download native NORD 15-minute MI-A1, MI-A2 and MI-A3 workbooks to
`data/api/`, then stage them with:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.stage_ida_workbooks \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --start 2026-10-20 --end-exclusive 2026-11-11
```

The generated stage directory contains an `import.yaml` and `NEXT_STEPS.txt`.
The merge checks overlaps, native 15-minute periods, DST and checksums before
publishing. Check its report for missing or cancelled auctions.

The gas IG Index/IGI still needs a supported automatic source or manual monthly
workbooks. A safe manual import is staged with
`python -m scripts.import_gme_gas_workbooks stage --stage-database PATH`; after
review and after other writers stop, use `merge --stage-database PATH --database
DATABASE`. Add `--delete-verified` only when the original files should be removed
after the source bytes and imported values are verified. The configured feature
is the GME IGI in EUR/MWh, not an interchangeable TTF or other gas series.

No cron job is installed. Schedule the command according to the machine's own
policy if recurring automatic execution is desired; avoid concurrent DuckDB
writers and retain the generated reports.
