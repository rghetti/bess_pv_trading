# Repository review: real-time economic MPC

Reviewed 3 November 2026 against source commit `caa242c`. Work is isolated in
`/home/rghetti/bess_pv_trading_realtime_mpc`, branch
`feature/realtime-economoic-mpc` (spelling retained from the request).
This is a standalone local Git repository, not a linked worktree or a newly
published GitHub repository. The original checkout has not been modified.

The copy contains Git history, tracked source/configuration/reports, and independent
copies of local `data/` and `models/`. The original 132 GB `outputs/` tree and the
untracked `docs/price_forecasting_data_inventory.md` were excluded. Consequently,
workflows requiring prepared outputs must prepare them or explicitly select an
existing source. No training, model search, historical replay, or provider download
was launched for this review. Tests use fixtures; the performance check reads prices
only. The existing `bess` environment was used without installing packages.

**Findings and changes**

| Priority | Finding | Change and evidence |
| --- | --- | --- |
| High | Auction curves with correct start times but shortened/incorrect product end times passed `_load_auction_prices`; missing declared resolutions were dropped and fractional resolutions were truncated to integers. | Validate exact durations and exact finite resolution declarations before mapping prices. Regression cases cover malformed ends, fractional/missing declarations, both DST transitions, all four auctions and the DA resolution change. |
| High | Absolute-path MPC launchers resolved relative data/model paths from the caller's current directory. From the original checkout this could read the original database instead of the copy. | Both real-time comparison launchers set `_project_root` to their own repository. A test changes into an unrelated directory and checks the preparation boundary. |
| Medium | Each day of continuous preparation decoded and hashed the same large GME workbook snapshots again. | Add caller-owned `SnapshotReadCache`, bounded by entry count, keyed by absolute database path, immutable snapshot ID and verification policy. Reuse it across daily price loads; retain short database connections. Refresh, verification-policy, eviction and public-frame mutation tests protect correctness. |
| Medium | Fully cached downloads called `load_raw` again for every daily result; DuckDB raw-only reads also materialized normalized Parquet bytes. | Fetch raw data only when missing siblings need it, and select no normalized BLOB for raw-only reads. Full-cache tests for DuckDB and legacy files assert zero raw-only rereads and no HTTP/authentication. |
| Medium | Terna recovery scanned the entire plan and deserialized candidate frames to find one monthly sibling. Raw and parsed responses remained retained across completed batches. | Index candidates by full transport identity; use manifests and a pinned raw snapshot ID for recovery. Release response caches after their last planned consumer. Existing monthly, gap, DST, parser recovery, legacy and provider-pause tests remain applicable. |
| Medium | Four farm workers each limited Gurobi to two threads, but numerical/inference pools had no corresponding bound. | Add a reusable fresh-process initializer for BLAS/OpenMP and PyTorch. The May launcher exposes `--native-threads` (default 1), separate from solver YAML. A fresh-process test inspects actual pool sizes. |
| Medium | Online CRPS `--workers 0` silently selected the configured 42 workers; pools used `fork` after numerical imports. | Reject zero, use `spawn`, and apply the same initializer with workflow `threads_per_worker: 1`. The zero-worker CLI test runs in plan-only mode. |
| Low | May worker results caught `BaseException`, potentially treating interruption/system-exit as an ordinary failed job; docstrings still described eight jobs/two modes. | Catch ordinary `Exception`; describe the current four continuous-mode jobs. Executor shutdown still waits for running work; immediate process-tree cancellation was not implemented. |
| Low | Standard POI provenance said DC nameplate although the resolver uses AC capacity. Config documentation described archived price-model training as active. | Correct the provenance label to AC and describe the retained price import example accurately. Numerical asset assumptions are unchanged. |

These changes preserve provider quotas/retry policies, data gaps, observation
cleaning, model selection, auction objectives, solver tolerances and realization
accounting. HTTP acquisition remains sequential: the downloader has mutable token,
retry, provider-pause and cache state that is not a thread-safe work queue. Adding
concurrent writes to the shared database would violate its access model; parallel
read-only study workers are the appropriate current boundary. See
[DuckDB concurrency](https://duckdb.org/docs/stable/connect/concurrency).

**Formulation and execution traced in the code**

`src/domain.py`, `src/scenarios.py`, `src/optimization/model.py` and
`src/optimization/physical.py` define the deterministic equivalent of a
scenario-tree MILP. Auction trade quantities and battery controls belong to
information nodes; scenarios sharing an information node share its decisions.
Curtailed PV can have separate observation-dependent owners in supported policies.
PV injection, signed grid exchange and battery energy are constrained per scenario.

For interval duration Δt in hours, charge/discharge powers c,d in AC MW, internal
energy E in MWh and hourly self-discharge σ:

```
E[t+1] = E[t] (1-σ)^Δt + η_charge c[t] Δt - d[t] Δt / η_discharge
0 <= c[t] <= Pmax z[t]
0 <= d[t] <= Pmax (1-z[t]),  z[t] binary
G[s,t] = PV[s,t] - curtailment[s,t] + d[t] - c[t]
-POI_import <= G[s,t] <= POI_export
E_min <= E[s,t] <= E_max
```

The auction objective weights scenario revenue and enabled settlement terms,
less wear, import charges and transaction fees. Existing accepted and pending
positions enter cumulative commitments. Trades and cumulative positions have
POI-derived bounds. Native auction products have equality constraints across
represented dispatch intervals. This bounds each individual auction adjustment as
well as the cumulative position; it is a modeling restriction to preserve unless
explicitly changed, not an exchange-rule claim.

The standard study disables optimized imbalance and requires planned delivery
balance. Other study configurations enable imbalance penalties or stochastic
penalty/CVaR terms. The perfect-price historical workflows use the same actual
clearing curve across PV members. They are information benchmarks, not evidence
of achievable causal price forecasts. `src/scenarios.py` separately supports
synthetic forecast-price branching at publication events.

`src/simulation/continuous.py` distinguishes submissions, publications and physical
execution. Submission fixes quantities and books applicable submission fees;
publication books actual quantity × clearing price once. Pending/unpriced orders
already count as commitments. Each trajectory carries realized SOC, wear memory,
fees and commitments across days. Planning values are kept separate from realized
cash flow. `src/simulation/settlement.py` prices physical deviations ex post.

`src/optimization/mpc.py` introduces no new auction trades. It receives fixed
commitments, SOC, future PV and optional explicit imbalance-price oracle curves.
All scenario members share one open-loop charge/discharge trajectory and one PV
utilization fraction per interval; grid injection and imbalance differ by member.
The receding horizon applies the first control, then resolves. This is not a fully
adaptive within-horizon scenario-tree controller.

The price-blind objective minimizes expected absolute imbalance × planning penalty,
plus wear, import fees and terminal-inventory cost. The oracle first minimizes
expected absolute imbalance; if stage one is certified optimal under the solver
settings, stage two maximizes signed settlement value less costs within the
configured additional imbalance budget. Directional settlement uses sign binaries.
Stage-two rejection preserves a validated stage-one action. A failed/unusable solve
invokes the configured tracking fallback. Terminal modes include no cost, a soft
SOC target and a linear inventory value, alongside hard study targets/floors.

Wear can be throughput-linear or the Xu segmented formulation. Standard defaults
currently select throughput economics; the presence of Xu code does not mean every
study uses it. Relative MIP gap is `1e-4`; no absolute MIP gap was introduced.
Current May configurations retain two Gurobi threads and a 10-second MPC budget.

**Configuration and study map**

All 135 YAML files loaded through the actual configuration loader: 8 at the config
root, 10 acquisition, 67 full experiments, 49 PV studies and one price-import
example. Loading validates configuration, not input completeness or study success.
Includes resolve relative to YAML files; mappings merge recursively, lists replace,
explicit nulls survive, and typed references resolve after overrides. Selected-farm
PV mappings override shared PV settings. Most scripts still require the repository
working directory as documented; the two MPC launchers now explicitly anchor it.

| Study family | Current implementation/configuration | Interpretation |
| --- | --- | --- |
| Physical PV and data review | `src/data/policy.py`, `src/forecasting/data/`, physical PV models, `scripts/review_curated_pv.py`, reviewed-target/HPO scripts | Source AC and satellite GHI remain distinct from curated targets, exclusions and night rules. Configured review decisions are not an automatic fault diagnosis. |
| Deterministic learned PV | `src/forecasting/training/`, `inference/pv_forecast_bundle.py`, `models/pv/farm_*/`, forecast-search/timing-ablation scripts | Tracked pilot artifacts and selection files exist. Current standard replacement map points to TCN bundles for all four farms; enabling replacement is study-specific. Selection can reflect already-inspected 2026 report data. |
| Physical/temperature/geometry comparisons | `pv_realized_*`, `compare_pv_geometry_factorial.py`, realized-irradiance configs | Proxy locations, fitted output scales and tracker geometry must remain explicit; realized-weather comparisons are retrospective. |
| Probabilistic PV | `study_pv_probabilistic.py`, `pv_probabilistic_*`, January/August configs | Raw 50-member weather, deterministic augmentation and postprocessing are separate stages; first-day availability differs by archive. |
| Mayer postprocessing/neural | `study_pv_mayer.py`, `train_pv_mayer_neural.py`, isolated R runner, censored/quantile networks | Chronological study/training settings are explicit. Marginal predictive distributions need temporal coupling before use as coherent optimization paths. Existence of configs is not evidence training completed. |
| Online CRPS/ridge/bias | `study_pv_online_crps.py`, `src/forecasting/online/` | Chronological updating and forecast availability determine admissible information. Worker topology changes do not authorize rerunning the study. |
| Continuous auction comparisons | `continuous.py`, `continuous_inputs.py`, January/August farm launchers, participation scripts | Compare integrated/current-auction planning, PV scenarios/envelopes and perfect-information references while carrying state. |
| Real-time controller comparisons | `realtime_mpc_study.py`, `run_realtime_mpc_comparison.py`, May all-farm launcher | Frozen commitments isolate control changes; continuous comparisons let controllers change future auction commitments. May currently runs four farms × one mode × three controllers. |
| Price acquisition/import | `src/data/api.py`, ENTSO-E/Terna/GME parsers, workbook import and catalog modules | DuckDB is canonical; legacy file caches remain compatible. Historical publication eligibility is distinct from retrieval time. |
| Price-model research | Only one `configs/price_experiments/` import YAML remains | The README says prior price research was moved to an external archive. A fitted causal price forecasting pipeline must not be inferred from the active oracle workflows. |

Important current assumptions: weather availability lag is **6 hours** in
`standard.yaml`, rather than the earlier illustrative 4-hour project description.
Quarter-hour delivery calendars handle DST through timezone-aware timestamps.
Current May studies use `realized_interval_average` PV disclosure, so their
current-quarter information is stronger than ordinary start-of-quarter telemetry.
Model artifacts available in October are explicitly allowed retrospectively in
May. Their results cannot be labeled a fully causal live controller evaluation.

**Local data verified in the copy**

The original and copied market database have identical SHA-256:
`f20b01761aae234dd7d8afe222300d116247f11d2212be069c923d402b547a82`.
The copy contains 8,924 data files and 62 model files, with no symlinks back to the
source. Reading used a short read-only connection to the copied file; no
source database migration, mutation or network acquisition was performed.

| Archive | Dimensions | Coordinate coverage |
| --- | --- | --- |
| `data/farms/pv_farms_timeseries_20250101_20260831.zarr` | 4 farms; 58,368 measurement times; 2,440 cycles; 49 hourly leads | Measurements 2025-01-01 00:00 through 2026-08-31 23:45 UTC; weather cycles 2024-12-30 00 UTC through 2026-08-31 18 UTC. |
| `data/farms/nwp_ens_20260101_20260131.zarr` | 4 farms × 32 cycles × 49 leads × 50 members | 2025-12-31 through 2026-01-31, 00 UTC cycles; leads 0–48 hours. |
| `data/farms/nwp_ens_20260801_20260930.zarr` | 4 farms × 61 cycles × 49 leads × 50 members | 2026-08-01 through 2026-09-30, 00 UTC cycles; leads 0–48 hours. |

The January archive includes the prior-day cycle for January 1. The August archive
lacks July 31: preceding-day ensemble comparisons start August 2. Measurement
coverage stops in August even though that ensemble archive extends into September.
The May YAML names a May ensemble archive that is absent locally, but the ensemble
feature is disabled there; this is not a current deterministic-run failure.

| Farm | Metadata AC kW (rounded) | Tracking ID | Configured BESS MWh / MW |
| --- | ---: | ---: | ---: |
| 1 | 953.28 | 0 | 0.75 / 0.5 |
| 345 | 1,524.60 | 0 | 1.0 / 0.5 |
| 901 | 4,160 | 3 | 2.0 / 1.0 |
| 910 | 6,638 | 1 | 5.0 / 2.5 |

POI ratings are illustrative symmetric AC-nameplate proxies. Shared SOC bounds
are 7–93%; initial/final targets vary by study. Bifacial metadata does not imply
that bifacial gain is enabled in current shared PV settings.

The copied catalog's current snapshots have these counts. Bounds below are request
coverage envelopes in local-day terms, **not proofs of uninterrupted coverage**;
no detailed completeness or historical-availability audit was rerun.

| Provider / series | Current snapshots | Normalized rows | Request envelope |
| --- | ---: | ---: | --- |
| ENTSO-E day-ahead prices | 654 | 62,780 | 2025-01-01 through 2026-10-16 |
| ENTSO-E generation forecast | 605 | 58,076 | 2025-01-01 through 2026-08-31 |
| ENTSO-E load forecast | 608 | 58,364 | Same envelope |
| ENTSO-E solar forecast | 607 | 58,268 | Same envelope |
| ENTSO-E onshore wind forecast | 607 | 58,268 | Same envelope |
| GME day-ahead prices | 7 | 43,703 | 2025-01-01 through 2026-10-22 |
| GME IDA1 | 8 | 62,012 | 2025-01-01 through 2026-10-21 |
| GME IDA2 | 8 | 62,684 | Same envelope |
| GME IDA3 | 8 | 31,392 | Same envelope; IDA3 covers a shorter delivery window |
| Terna imbalance prices | 29 | 61,244 | 2025-01-01 through 2026-09-30 |
| Terna load forecast | 424 | 40,704 | 2025-01-01 through 2026-02-28 |
| Terna renewable generation, national IT | 398 | 76,416 | 2025-01-01 through 2026-02-04 |
| Terna total load | 422 | 40,512 | 2025-01-01 through 2026-02-28 |
| Terna total-load forecast | 422 | 40,512 | Same envelope |
| Terna wind forecast | 419 | 40,224 | Same envelope |

GME workbook snapshots span many days, explaining why per-day reads benefited
from decoded-snapshot reuse. Raw bytes, normalized bytes, checksums and immutable
snapshot identifiers remain authoritative. No gap is changed to a zero and no
retrieval clock is promoted to historical publication evidence.

**Performance and validation**

A deterministic read-only microbenchmark loaded all four auction curves for
May 1–4, 2026 from the copied catalog. It performed no PV inference, model fit,
optimization or replay. With memoization disabled, 16 snapshots were decoded in
2.978 seconds; with the bounded cache, 4 were decoded in 2.165 seconds. Arrays,
skipped-auction records and provenance were exactly equal. This is a single local
check with possible filesystem-cache effects, not a general 27% end-to-end speedup
claim. The structural result is 75% fewer snapshot decodes for those repeated reads.

The cache is bounded by snapshot count, not bytes; a single very large workbook can
still consume substantial memory. `continuous_optimization.price_snapshot_cache_entries`
controls the entry count (default 8). Farm processes own separate caches. The May
launcher supports `--workers` and `--native-threads`; solver threads stay in YAML.
PyTorch pools are initialized before inference as required by
[PyTorch's thread-setting contract](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html).
These limits cover BLAS/OpenMP and PyTorch, not a global CPU quota across every
library: DuckDB retains its own configured threads and Arrow can use separate pools.

Checks use existing libraries and synthetic fixtures. Initial sandboxed MPC tests could not
obtain solver plans; rerunning with Gurobi license access passed. No provider
credentials were requested and transport regressions use a fake monthly provider.

**Remaining limitations and concrete follow-up areas**

- This is a targeted code and configuration review with affected regression tests,
  not a formal proof of the MILP or exhaustive execution of every study. Test
  discovery was checked; only the relevant suites were executed.
- Parallel HTTP fetching is not implemented. Any future design needs isolated
  provider sessions, shared rate limits, coordinated token renewal/provider pause,
  bounded response queues and one database writer. Existing process workers must
  continue to use read-only snapshots during experiments.
- The all-farm launcher has a fixed summary filename and underlying runs use stable
  IDs with `resume: false` and overwrite enabled. Concurrent invocations targeting
  the same output root can race. Use distinct `--output-dir` values; a cross-process
  output lease is not implemented in this change.
- Immediate interruption of already-running executor tasks remains unresolved.
  Changing `BaseException` handling improves propagation but does not terminate
  child process trees or undo partially generated output.
- Some other HPO/review launchers retain separate worker-limit conventions and
  default multiprocessing contexts. This change centralizes the two reviewed
  parallel paths, not every historical study runner.
- Online CRPS worker settings participate in its study manifest. Changing these
  settings can reject an old run ID on resume; use a deliberately selected new
  study ID after experiment approval rather than bypassing manifest checks.
- The README's older pilot-score tables describe earlier selections; current
  executable defaults and tracked `selection.json` files should be used to identify
  active models. Reproducing historical scores requires their original artifacts,
  settings and splits, not the current default TCN map.
- Current-interval realized PV disclosure, retrospective model selection, proxy
  locations and perfect auction prices are material interpretation limits. The
  current May workflow is an explicitly retrospective comparison.

**Recorded verification results**

- 194 passed: optimization contracts/solver, MPC, native auction loading,
  continuous inputs/configuration, comparison launcher, and both continuous and
  real-time continuous integration suites (288.11 seconds; Gurobi license access).
- 70 passed: initial combined data/cache, continuous-input and parallel-launcher
  regression pass (133.83 seconds).
- 64 passed: launcher, online-CRPS and configuration-inheritance checks after
  introducing the shared process initializer (81.65 seconds).
- 43 passed: final cache-setting validation, both MPC launcher checks and source
  layout (68.67 seconds).
- 54 passed: final acquisition/cache regressions, including offline GME cross-zone
  reuse, Terna monthly batching/DST/gap recovery and provider pause/resume
  (117.44 seconds).
- All 135 YAML files loaded; 136 source, 46 script and 85 test Python files parsed.
  These counts describe the reviewed checkout, not completed research studies.
- Git whitespace checks passed. Test groups overlap; their counts must not be
  added as a unique test count.
