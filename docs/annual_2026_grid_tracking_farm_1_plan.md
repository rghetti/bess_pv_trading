# Farm 1: 2026 continuous grid tracking

This is a proposed retrospective replay, pending explicit approval to run.
The first study covers farm 1 from 1 January through
31 August 2026, the period indicated by the local farm-data store. It is one
continuous trajectory, not separate monthly runs. The endpoint does not prove
that every required interval exists. Later months require additional data.

The configuration is `configs/research/annual_2026_grid_tracking_farm_1.yaml`.
The plan-only command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/bess/bin/python scripts/run_realtime_mpc_comparison.py --config configs/research/annual_2026_grid_tracking_farm_1.yaml --mode continuous
```

After PyTorch is installed, add `--check-only` to validate data, model and
forecast inputs without running the optimization. Review the preflight output,
including snapshot IDs, coverage and forecast provenance, before requesting
approval. The corresponding `--run` is for the approved experiment only.

On 6 November 2026, the offline preflight passed for 243 days and 23,324
intervals, with 959 forecast cutoffs and no solves. Input fingerprint:
`d529c041d02e5e98856035d756a8c175a6ca3824ed3447b5c266355e459597ef`.
The farm 1 GRU loaded under PyTorch 2.2.2 with matching model and preprocessing
checksums. Some days have fewer than four available auctions under the catalog
and calendar policy; the configuration skips auctions without products and
fails on partial products. This preflight does not validate solve performance.

## Forecast and data assumptions

Use the farm 1 `bigru_extended_037` four-cycle model and the deterministic
weather store from the shared configuration. Select the latest cycle available
at each reoptimization, assuming a six-hour release lag for the 00, 06, 12 and
18 UTC cycles. This is an explicit historical availability assumption. The
model artifact is enabled retrospectively; its recorded availability in the
configuration is a conservative placeholder, not proof of its actual release.
Review the resolved forecast cutoffs and 48-hour issue context in preflight.

The model has three bidirectional GRU layers of width 128 and extended weather
and calendar features. Its existing training used 2025 historical GHI and a
Huber loss; training used a batch size of 16 and learning rate approximately
0.000181. Validation used January–April 2026 forecast GHI and selected the
lowest daylight day-balanced nRMSE; May–August was reported as a test period.
These periods have been inspected, so the replay is retrospective rather than
an untouched evaluation. No fitting, calibration or hyperparameter search is
part of this run. The fitted preprocessor and month-specific zero-night policy
come from the saved bundle and configuration.

The reviewed production dataset can contain explicitly identified synthetic
pvlib replacements; these are counterfactual, not measurements. Missing
daylight data and prices must not be silently filled. Preflight must verify
coverage, reviewed exclusions and the actual provenance of every target and
forecast, including a `pinned` neural model rather than the configured pvlib
fallback; the existence of a Zarr or DuckDB file is not evidence of complete
coverage. The data catalog is `data/api/market.duckdb` and must not be opened
concurrently with an active downloader. Model and input fingerprints must be
recorded with the output.

## Controller, economics and outputs

Run only `track_committed_grid` with integrated DA and IDA1–3 planning, 15-minute
products, one worker and nine Gurobi threads. Auction planning assumes the
actual clearing-price curves are known in advance, as in the archived tracking
comparison; this is a price-oracle benchmark, not a deployable price forecast.
All submitted quantities are committed and realized cash flows use actual
clearing prices. Initial and final SOC are both 50%; other plant and cost
assumptions come from `configs/standard.yaml`. The relative MIP gap is 1e-4,
and the per-auction solve limit is 120 seconds.

Report realized profit and its auction, settlement, fee and wear components,
absolute and signed imbalance, curtailment, SOC, physical overrides and solver
status; group outputs and selected daily plots by run and date. There is no
additional controller baseline or model selection in this first run. The
calendar-only plan counts 243 days, 23,324 quarter-hour intervals including the
spring clock change, and up to 972 auction events. It establishes neither
input completeness nor runtime. A full preflight may be expensive and must be
run only when no downloader is using the catalog.
