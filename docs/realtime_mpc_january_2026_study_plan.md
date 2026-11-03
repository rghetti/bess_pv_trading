# Full-January MPC versus feedback study plan

Status: prepared for review; **the full-month study has not been run**.

The 31 daily input/auction checks completed on 3 November 2026. An additional
full-month forecast-cutoff scan was interrupted before completion; the complete
month preflight is therefore not certified. The launcher completes that scan
before starting any historical optimization.

## Objective and exact comparison

Compare the current deterministic, price-blind economic MPC with the unchanged
`track_committed_grid` feedback controller for Farm 1 over **1–31 January 2026**,
inclusive delivery dates in Europe/Rome. Both strategies participate in every
available DA, IDA1, IDA2 and IDA3 session. The configured chronological submissions,
guaranteed execution, immutable commitments and publication accounting remain
unchanged. This is one all-auction profile, rather than a search over participation
subsets. No other farms, month, controllers, or sensitivity runs are included.

Run two separately interpreted comparisons:

1. **Frozen commitments, primary.** Generate one all-auction ledger with the
   existing integrated deterministic-TCN auction strategy and grid feedback.
   Freeze actual submissions, timestamps, prices, fees and baseline plan snapshots.
   Replay that same ledger with feedback and deterministic MPC, both starting at
   the same supplied energy. Auction revenue and fees must reconcile identically.
   This comparison isolates physical control conditional on those commitments.
2. **Full continuous, secondary.** Give each controller its own actual carried SOC,
   projections and subsequent auction solves. Later auction submissions and revenue
   can differ. Report those differences separately; this comparison covers the
   complete trading/control policy rather than physical control alone.

The launcher defaults to printing this plan. `--run` explicitly starts the study;
`--check-only` performs cached-data preflight without optimization. Both modes
share one prepared input set. A verified frozen ledger is reused on subsequent
attempts, preserving its original source diagnostics.

## Information, model and physical settings

Both controllers receive current-interval realized AC PV as an exogenous input.
For MPC this means every scenario's first PV value is fixed to that interval's
effective measured average. **Future realized PV remains hidden.** The study is a
retrospective information benchmark because the complete interval average would
normally become observable only at its end.

Use the existing Farm 1 `direct_tcn_02` weather bundle for initialization,
deterministic forecasts, auction forecasts, measurement fallbacks and shadow
projections. It has 12 frozen weather/calendar features, width 16, depth 3 and
kernel size 7. Its original training used MSE on 2025 data; no training loss is
optimized in this study. Inference uses frozen preprocessing/weights and disabled
dropout. No new calibration, model selection, splits, copulas or price forecasts
are introduced.

The model was selected using January–April 2026 results, and its registry
availability is 27 October 2026. January reuse explicitly enables retrospective
artifacts and includes hindsight selection. The month is therefore **not an
independent model-selection holdout**. Preserve manifest/dependency checksums,
training/selection dates and this limitation in the outputs.

Weather forecasts still obey their actual recorded/assumed publication eligibility;
the shared assumption is cycle time plus six hours. Only eligible vintages may
update future PV. The physical MPC remains price-blind: future realized imbalance
prices enter settlement only. The upper auction optimizer retains its existing
actual-auction-price oracle, so the complete trading benchmark is retrospective.

Use the same settings as the two-day smoke:

| Setting | Value |
|---|---|
| Farm / battery | Farm 1; 0.75 MWh, 0.5 MW |
| POI import/export proxy | 0.95328 MW each direction |
| Initial / operational SOC | 90%; 7–93% box |
| Directional efficiencies | sqrt(0.90) per direction |
| Self-discharge | Shared default zero |
| Degradation | Shared AC throughput convention and economics |
| MPC objective | 500 EUR/MWh expected absolute imbalance + wear/import/terminal costs |
| Ordinary midnight | Soft 50% SOC target at 50 EUR/MWh internal-energy deviation |
| Study end | Planned hard 50% SOC at midnight ending 31 January |
| Intermediate carryover | Actual state; no daily reset |
| Imbalance-price oracle | Disabled |
| Ensemble / scenario count | No ensemble in this comparison; one deterministic trajectory |
| Solver | Gurobi, two threads, relative gap 1e-4, no configured absolute gap |

The feedback controller's actual final inventory can differ from its planned
target. Report final stored energy and differences alongside unadjusted operating
profit. No terminal value becomes an actual settlement charge.

## Data preparation and acceptance

Use existing local farm weather/AC measurements and the read-only DuckDB catalog
at `data/api/market.duckdb`. No automatic acquisition or environment changes are
part of the launcher. Coordinate with any active downloader before preflight.

Preflight all 31 delivery days, the preceding-date forecasts needed for the first
submissions on 31 December 2025, native auction curves, NORD quarter-hour settlement,
and every forecast cutoff. Pin source snapshot IDs, checksums and model evidence.
The physical grid has **2,976 intervals per trajectory** and 3,100 distinct forecast
cutoffs under the supplied calendar.

Retain existing seasonal night policies and measurement preprocessing identically.
Daytime missing measurements may use the configured initial frozen TCN forecast,
up to the existing four-interval/day cap, with explicit fallback provenance.
Do not add episode exclusions, infer faults, silently drop dates or zero-fill
unexplained gaps. Missing enabled settlement, partial auction curves, unavailable
weather or incompatible artifacts stop preflight for review.

Accept a run only when physical balances, SOC carryover, power/POI bounds, command
exclusivity, forecast clocks, fixed commitments and signed settlement reconcile.
Check current-PV first-action predictions against execution, record every solver
failure/fallback, and resolve noon/midnight projection errors against observed
states. Missing plan coverage must be explicit. All 31 dates must appear in paired
metrics; failed or skipped dates cannot silently change the comparison sample.

Report absolute/signed imbalance, signed cash flow, settlement exposure, wear,
charging/discharging, curtailment, SOC, inventory, actual operating profit and
runtime. Preserve negative-price signs and zero-denominator nulls. Report paired
daily and complete-month differences without treating overlapping MPC objectives
as realized profit. Detailed per-day plots cover **1–3 January at most** with the
provided configuration; aggregate plots and CSV/JSON metrics cover the full month.

## Compute, outputs and commands

Execute comparisons sequentially with two CPU threads for Gurobi and TCN inference,
without GPUs. Fresh generation across both modes requires 5,952 MPC decisions and
372 auction solves: 124 for the frozen source and 248 across the two continuous
trajectories. Reusing a source ledger reduces that work.

MPC retains a ten-second total build/solve budget per decision, and auction solves
retain the shared 120-second limit. Their nominal limit sums are 16 h 32 min for
MPC and 12 h 24 min for auctions, plus preparation, reporting and any diagnostic
retries. These are configured allowances, not a prediction of runtime. The short
smoke is insufficient to guarantee full-month timing or performance.

Outputs use the distinct run ID
`farm1_january_2026_tcn_current_pv_all_auctions`, with separate
`frozen_commitments/` and `continuous/` directories under `outputs/realtime_mpc/farm_1/`.
Retain the resolved configuration, model/data evidence, immutable ledger, complete
plans, first actions, interval/daily/month metrics, projection diagnostics and plots.
Comparison orchestration restarts controller trajectories; individual continuous
replays retain midnight checkpoints. A failed month should be reviewed before an
explicit rerun, using `--mode` to select the affected comparison.

```bash
# Review the full plan; no dataset read, inference or optimization.
python scripts/run_realtime_mpc_comparison.py

# Explicit cached-data preflight; no optimization.
python scripts/run_realtime_mpc_comparison.py --check-only

# Only after approval of the resolved run plan.
python scripts/run_realtime_mpc_comparison.py --run

# Alternatively launch just one approved comparison.
python scripts/run_realtime_mpc_comparison.py --mode frozen_commitments --run
python scripts/run_realtime_mpc_comparison.py --mode continuous --run
```

The configuration directly imports the shared standard file:
`configs/full_experiments/realtime_farm1_january_2026_all_auctions.yaml`.
