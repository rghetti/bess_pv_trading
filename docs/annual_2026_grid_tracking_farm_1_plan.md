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
products, one worker and nine Gurobi threads. This is a greedy, single-interval
physical controller: it reacts to current interval-average PV, current SOC and
the committed grid target, subject to battery and connection limits. It can use
the retained current-interval curtailment instruction. It does not solve an MPC
problem or look ahead when choosing real-time battery dispatch. The idealized
use of realized interval-average PV is a physical feedback assumption, not a
claim that the full interval average is known at its start.

The separate auction optimizer plans future products at each auction using PV
forecasts. Auction planning assumes the
actual clearing-price curves are known in advance, as in the archived tracking
comparison; this is a price-oracle benchmark, not a deployable price forecast.
All submitted quantities are committed and realized cash flows use actual
clearing prices. Initial SOC is 50% and auction planning targets 50% at the end
of the study; greedy feedback does not guarantee the realized final SOC. Other
plant and cost assumptions come from `configs/standard.yaml`. The relative MIP gap is 1e-4,
and the per-auction solve limit is 120 seconds.

Report realized profit and its auction, settlement, fee and wear components,
absolute and signed imbalance, curtailment, SOC, physical overrides and solver
status; group outputs and selected daily plots by run and date. There is no
additional controller baseline or model selection in this first run. The
calendar-only plan counts 243 days, 23,324 quarter-hour intervals including the
spring clock change, and up to 972 auction events. It establishes neither
input completeness nor runtime. A full preflight may be expensive and must be
run only when no downloader is using the catalog.

## Previous-model comparison

`configs/research/annual_2026_grid_tracking_farm_1_previous_model.yaml` uses the
same study settings and greedy tracker with the earlier farm 1 reviewed-GHI
`bigru_extended_021` bundle. Its separate run ID prevents collisions. The
previous model is a four-layer bidirectional GRU of width 32 with 21 features;
it was trained using historical GHI and Huber loss, and selected using 2026
validation. Its reported validation and test used the preceding delivery day's
00 UTC forecast; this experiment applies the latest eligible deterministic
cycle, so those historical scores do not establish updating-cycle performance.
Both model selections are retrospective, and neither run trains a model.

The local curated replay loader uses 14,828 metered intervals, 8,332 configured
night-zero intervals, 137 reviewed daylight pvlib replacements, and 27 retained
daylight exclusions reconstructed with historical-GHI pvlib over the requested
2026 window. These counts describe the current local input files. The synthetic
and night-zero values are counterfactual inputs to dispatch and settlement, not
independent measured production. Per-interval source labels and checksums are
retained in each run's input provenance. Compare realized operating profit,
settlement, wear, imbalance, curtailment and final SOC by day, and distinguish
forecast objective values from realized cash flow.

Run the previous model's `--check-only` preflight before approval and execution.
Avoid overlapping farm runs or opening the catalog during an active downloader;
each run uses nine solver threads. No previous-model full preflight or solve has
been run yet. Unlike the user's existing new-model configuration, this new
configuration disables overwriting an existing output directory.

## Selected-farm parallel launcher

`scripts/run_annual_grid_tracking_farms.py` prepares the same continuous greedy
tracker for any selection of farms 1, 345, 901 and 910, each with its own
four-cycle model. Without an action flag it prints a plan and does not run a
preflight or solve. Example:

```bash
/opt/anaconda3/envs/bess/bin/python scripts/run_annual_grid_tracking_farms.py --farms 1 345 901 910 --workers 2 --run-id annual_2026_example
```

Add `--check-only` for input preflight, or, after reviewing and approving the
plan, `--run` to preflight every selected farm before starting any solve. The
run creates a new batch directory with per-farm configurations, logs and run
outputs; it refuses an existing batch ID. Each farm gets a Markdown report at
`reports/farm_<id>.md`, including realized financial and physical metrics,
auction solver statuses, curated PV source counts and links to detailed outputs.
Failed runs and preflights are marked without presenting incomplete metrics as
completed results. Check-only uses temporary files and
does not reserve the batch ID. Choose a new batch ID for every execution.

Two workers each request nine Gurobi threads, for up to 18 solver threads on
this ten-core machine. Inference may also use the configured solver thread
count. Use `--workers 1` to avoid this oversubscription, particularly while
another experiment is running. Full input preflights for farms 345, 901 and 910
and concurrent license, memory and runtime behavior remain unverified. As in
the single-farm study, confirm that no market downloader is using the DuckDB
catalog before preflight or launch. The selected models and synthetic curated
labels make all these retrospective counterfactual replays.
