# December 2025 auction-price participation comparison

Status: implementation audit complete; all eight 31-day input preflights passed
on 5 November 2026. Historical optimization has not started. Explicit experiment
approval is required before `--run`.
This supersedes the January plan for the requested study, while retaining the
existing December configurations prepared in `b2a614e`.

## Concrete experiment

Replay local delivery dates 1–31 December 2025 in Europe/Rome for farms 1,
345, 901 and 910. Each farm has a forecast-price arm and a perfect-auction-price
arm. Each arm contains these independent continuous trajectories:

1. DA
2. DA + IDA1
3. DA + IDA2
4. DA + IDA3
5. DA + IDA1 + IDA2
6. DA + IDA1 + IDA3
7. DA + IDA2 + IDA3
8. DA + IDA1 + IDA2 + IDA3
9. The existing all-auction sequential reference (`current_auction_only`).

The eight subsets use integrated planning across their remaining participating
auctions. The sequential reference plans only the current auction. This is 72
trajectories, 2,232 trajectory-days and 214,272 executed quarter-hours. With the
documented cancellation, there are 5,912 nominal auction solves across the
study (739 per farm/arm), before any interrupted-work repeats or solver retries.

Both arms settle submitted quantities at actual GME NORD clearing prices and
physical imbalance at the same actual Terna NORD quarter-hour settlement price.
Only auction-planning price information differs. Guaranteed full execution,
price-taking and signed incremental trades are assumed. Quantities are fixed
on submission; actual revenue is booked once on publication. Predicted
objective values are kept separate from realized cash.

## Inputs, timing and frozen models

Use the prepared `october_2025/ridge_price_only` artifacts under
`outputs/energy_price_forecasts/extended_2024_training_v2/`, with the existing
observation-history sidecars. All four models have a historical fit cutoff of
2025-09-30 09:00 UTC. Training labels end before that cutoff, with maximum
label-publication times on 29 September. Artifact creation/retrieval happened
retrospectively: this is a frozen historical-data fit, not evidence of a model
deployed before October.

Each model uses standardized ridge regression, alpha 20, and squared-error
loss plus ridge regularization. Features are its own auction's prices lagged
24, 48, 168 and 336 hours, plus seven deterministic calendar features (daily,
weekly and annual sine/cosine terms and a weekend indicator). There are no
regional weather or current-auction-outcome features. DA has 59,944 training
rows, IDA1 and IDA2 have 59,176 each, and IDA3 has 29,712. Some earlier hourly
labels were repeated over quarters; these are not independent observations.

Prediction selects only price observations whose availability clock is no
later than the decision. An unavailable nominal lag falls back to an older
published observation at the same local clock, within the configured 336-hour
lookback beyond the requested lag. Unresolved inputs fail rather than being
filled with zero. No fitting, model search, tuning, GPU work or model selection
using December profits is part of this run. The artifacts retain a historical
14-day residual-calibration procedure, but the runner requests one point
forecast, so no residual draws or stochastic price scenarios enter planning.

Submissions are five minutes before local gates: DA 11:55 D−1, IDA1 14:55 D−1,
IDA2 21:55 D−1, IDA3 09:55 D. Replay publication clocks are DA 13:30 D−1,
IDA1 15:30 D−1, IDA2 22:30 D−1 and IDA3 10:30 D. The DA history sidecar assumes
12:57 D−1; the remaining sidecars use the listed IDA publication clocks.
These are assumptions rather than observed historical publication timestamps.
There is no study decision between the two assumed DA publication times.
Current stored snapshots are pinned and checksummed, not reconstructed as-of
historical revisions.

IDA1 was cancelled for delivery on 10 December. The checked catalog contains
no prices and explicitly identifies that cancellation. Both arms omit its
submission, solve and trading variables. Other wholly absent or partial price
coverage fails. The loader treats the cancellation as known even at preceding
gates for that delivery day; this is a disclosed retrospective scheduling
assumption. It does not replace cancelled prices with zero. The IDA1 price lag
after cancellation can use an older published day.

PV planning uses deterministic latest-available weather, an assumed six-hour
cycle publication lag and unfitted pvlib explicit-efficiency conversion. The
preceding day's 00 UTC vintage is used at the initial DA decision; subsequent
eligible vintages update at later gates. No PV neural model or calibrated
conversion coefficient is used. The December local night-zero window is
18:00–06:45; direct irradiance exceeding GHI is clipped and flagged.

Physical execution uses the existing reviewed curated AC targets. These
include metered values, monthly night-zero policy values, reviewed synthetic
values and historical-GHI pvlib reconstruction for retained exclusions where
applicable. The original source labels, raw missingness and dataset checksums
are retained. Synthetic targets are counterfactual production, not meter
observations. This study makes no new cleaning decisions or zero-fill repairs.

The December preflight target counts (quarter-hours, common to both arms) are:

| Farm | Metered AC | Monthly night zero | Historical-GHI replacement | Retained-exclusion pvlib replacement |
| --- | ---: | ---: | ---: | ---: |
| 1 | 990 | 1,581 | 405 | 0 |
| 345 | 1,346 | 1,581 | 0 | 49 |
| 901 | 1,395 | 1,581 | 0 | 0 |
| 910 | 1,395 | 1,581 | 0 | 0 |

Each row totals 2,976 intervals. Night-policy counts need not imply the raw
metered value was nonzero or missing; the source category records the applied
policy.

## Matched physical and financial assumptions

| Farm | Battery MWh | Battery MW | Import/export limit MW |
| --- | ---: | ---: | ---: |
| 1 | 0.75 | 0.5 | 0.95328 |
| 345 | 1 | 0.5 | 1.5246 |
| 901 | 2 | 1 | 4.16 |
| 910 | 5 | 2.5 | 6.638 |

These battery and symmetric POI assumptions are illustrative, not confirmed
asset specifications. SOC bounds are 7–93%; initial SOC is 50%; combined charge
and discharge efficiencies are each approximately 0.948683. The final-study
planning target is 50% SOC, with no ordinary-midnight floor. Tracking execution
can miss that target, so final energy and SOC are reported without salvage
credit. Locations and other inherited PV geometry remain stated assumptions.

Every profile owns its state, outstanding commitments and financial ledger
throughout the month. The grid-tracking controller applies actual interval PV
to track committed net grid delivery subject to physical limits; that is an
idealized interval-average execution assumption, not a tested telemetry system.
There is no real-time economic MPC in this study. Aging is the existing generic
throughput charge, not a fitted cycle-life model or capacity-fade experiment.
Import fee and charged imbalance penalty are zero. Imbalance is not optimized
for profit; actual imbalance settlement is still included in realized profit.

GME transaction fees use the inherited standalone-operator volume tiers and
assumed unchanged tariffs. No external portfolio volume is assumed. Fee state
carries chronological submitted volume (including November submissions for
1 December) and calendar-month resets. No manual day-boundary resets occur.

## Metrics and outputs

Primary metric: monthly realized net profit, paired forecast minus oracle for
each identical farm/profile. Report daily and monthly auction revenue, signed
imbalance settlement, degradation, import fees, transaction fees, charged
imbalance penalty, absolute and signed imbalance, charge/discharge, curtailment,
fallbacks and final energy/SOC. Net profit is auction revenue plus settlement
less degradation and the listed charges. Forecast objectives and illustrative
penalties are not cash expenses. Participation rankings are descriptive; no
model is chosen by the December rankings.

The perfect-price arm is a price-information benchmark, not a global
perfect-control upper bound: PV, physical tracking and planning horizon remain
matched, execution can differ, and final inventory is not cash-valued.

Use one shared folder:
`outputs/continuous_optimization/december_2025_price_study/`.
It contains persistent paired preflight evidence, per-farm/arm configurations,
input provenance, per-profile accepted-trade and dispatch ledgers, forecast
provenance, solver diagnostics and checkpoints. The study-wide products are
`paired_monthly.csv`, `daily_all_farms.csv`, `report_sources.json` and a single
`summary.md` containing paired profits and component tables for all farms.
Monthly participation plots and common-bin daily-profit histograms remain
enabled. Per-date dispatch plots are disabled: zero dates, within the requested
maximum of three.

Report validation rejects wrong/missing/duplicate profiles or dates, nonfinite
accounting, inconsistent cash components, discontinuous daily SOC, mismatched
physical provenance or different run/preflight fingerprints. Runtime reporting
also validates interval continuity and reconciles trade revenue and fees.
Solver statuses, fallbacks and maximum reported relative gaps appear in the
shared summary rather than being hidden behind aggregate profit.

## Compute, checks and approval boundary

All four farm pairs passed physical-input matching. Every arm has 123 available
auction events and 123 eligible weather cutoffs; each forecast arm also passed
306 auction-price prediction checks. The only excluded event is 10 December
IDA1. Compact checksums, model settings, counts and cancellation snapshot
evidence are saved in [december_2025_price_preflight.json](december_2025_price_preflight.json);
full per-arm evidence is in the shared output folder's `preflight/` directory.
Verification passed 153 distinct unit tests across focused batches and eight
synthetic integration checks. Two initially failing real-time test fixtures
were made explicit about submitted-only commitments; production controller
code was unchanged. Report tests reject wrong dates/profiles, cash mismatch,
nonfinite values and SOC discontinuity and check bounded child-process threads.

Use the existing `bess` environment with at most two farms concurrently; each
farm runs its forecast and oracle arms serially, and each arm runs profiles
serially. Gurobi uses one thread, relative MIP gap 1e-4, no absolute gap and a
120-second limit per auction solve. Native BLAS/OpenMP threads are bounded at
one. No GPU use or environment/package changes are authorized or needed.

The nominal solve limits imply up to 197.1 aggregate solver-hours (about 98.5
hours at ideal two-worker utilization), plus preparation, model construction,
diagnostics and any solver retries; this is a cap calculation, not a runtime
prediction. CPU/disk conditions on the shared host must be checked immediately
before launching. The run writes separate logs for each farm/arm.

The sandbox reports a Gurobi HostID mismatch. Host-visible license initialization
has passed without creating or optimizing a model; use that execution context
for the approved study. Synthetic integration tests are routine software
verification and do not constitute a historical experiment.

Preflight command (no optimization):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python \
  scripts/run_december_price_comparison.py --check-only
```

After explicit approval only, run with the same environment and `--run`, then
validate actual output and review the report. `--report-only` can rebuild the
shared summary without optimization. Existing output directories are protected
from overwrite; inspect any interruption and use validated checkpoint identity
before a separately controlled resume. Commit only task-related changes with
a detailed message. No actual profit results exist at this preparation stage.
