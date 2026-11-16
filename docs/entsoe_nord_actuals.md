# ENTSO-E NORD actual load and generation

The acquisition configuration requests actual total load (A65/A16) and actual generation by type (A75/A16) for solar (B16), onshore wind (B19), offshore wind (B18), hydro pumped storage (B10), hydro run-of-river (B11), hydro reservoir (B12) and fossil gas (B04). All requests are scoped to IT-North and preserve native intervals, original XML, retrieval time, type and production/consumption direction in the configured DuckDB catalog. Actual availability, returned production types, revision history and coverage for this particular zone and period have not been established. An explicit no-data response is recorded separately; it is never converted to zero.

The aggregate day-ahead generation forecast (A71/A01) currently used in the NORD price features covers total scheduled generation, not a separately identified hydro forecast. ENTSO-E's standard A69 day-ahead technology forecast covers wind and solar, and is not a hydro forecast. France hydro generation and Swiss reservoir state were included in earlier external feature ablations, but the current TimesFM NORD baseline does not have a dedicated NORD hydro feature.

For realized total hydro production, combine B10 production, B11 production and B12 production only when all required series are observed for the interval and have compatible units and scope. Pumped-storage consumption is a different direction and must not be summed as generation. Do not infer zero for an unavailable type; report completeness before constructing an aggregate. Publication times are not reconstructed from retrieval or document creation; previous-day availability for research requires an explicit assumption. This config prepares acquisition only and does not add actuals to any TimesFM input.

Run from the repository root, after coordinating with any other DuckDB writer. Dates in the YAML are editable and the end is exclusive:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -u -m src download-data --config configs/api_experiments/entsoe_nord_actuals.yaml --provider entsoe --dry-run
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -u -m src download-data --config configs/api_experiments/entsoe_nord_actuals.yaml --provider entsoe --interactive
```

The example requests eight datasets from 2025-09-01 through 2026-11-15; there is one request per dataset and local delivery day. The database path in the YAML is `data/api/market.duckdb` in this checkout. Ensure it is the catalog intended for the next study and do not run two writers on the same file. A token is prompted invisibly when missing.
