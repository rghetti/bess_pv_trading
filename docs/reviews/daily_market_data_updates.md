# Updating the price-study catalog

From the repository root, run this after any other database writer has finished:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.update_price_market_data \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --since 2026-09-01 --run
```

The end date defaults to tomorrow in Europe/Rome, so all source data available
through the current local date can be requested. Omit `--since` on later runs
to refresh the configured 14-day lookback, or provide an earlier date to backfill.
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
replace consolidated values. Swissgrid sources may be published monthly rather
than daily. Provider failures or missing credentials are reported individually.

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
