# Parallel controller trajectories for the k25 studies

The controller queue is a preparation for a new run, not a resume mechanism for
the interrupted runs. `scripts/run_k25_controller_queue.py` accepts the same
three study names as `run_k25_farms.py`, prints a plan by default and requires
`--check-only` or `--run` for further work. Use a fresh run ID and obtain
explicit approval for the experiment before starting it.

For `january-august-deterministic`, the current study YAML covers 1 January to
29 August 2026, farms 1 and 345, and the unweighted and first-interval-weight-25
deterministic MPC cases. Each farm has one greedy-tracking trajectory and one
MPC trajectory for each objective. The queue can schedule these six trajectories
with up to three processes per farm and six overall. As soon as one finishes it
starts the next eligible trajectory. It does not split or parallelize time
within a trajectory.

The selected forecast models, cutoff rules, PV target dataset, auction prices,
initial SOC, planning and execution settings, solver tolerances and objective
come from the same study YAML as the existing runner. Forecast cycles have an
assumed six-hour availability lag; current-interval average PV is disclosed to
MPC retrospectively. The curated PV trajectory can include synthetic targets,
and actual auction-price curves are used in planning. Models were trained on
2025 data and selected using inspected 2026 validation; no training or
calibration is performed here. See `ensemble_k25_studies_plan.md` for the model,
objective, data and retrospective evaluation details.

The two deterministic MPC cases first minimize total absolute imbalance in
MWh, with K=25 applying only to the current interval in the weighted case.
They then minimize battery wear among plans within the configured primary
objective tolerance. The end of every daily MPC horizon is constrained to 50%
SOC. Every auction optimization also has the 50% equality at the end of its
delivery day, including the tracker baseline's auction plans. A failed MPC
solve uses the tracking fallback for that interval and records the reason; this
can cause actual SOC to miss the target. The tracker itself only follows grid
commitments, so its actual SOC is not guaranteed to equal its auction plan.
Earlier batches retain the settings loaded at launch, including the final-day
only auction target and soft MPC terminal target.

Identical tracker trajectories are run once per farm and compatible configuration;
both MPC cases compare against that same tracker result. The MPC trajectories
remain independent because their objectives differ. Compatible trajectories in
one batch share a preflight snapshot per farm and
input configuration. A lock prevents duplicate preparation, and each worker
loads an independent copy; the cache checksum and input fingerprint are checked.
The cache is local to the batch, so a separate check-only batch does not populate
the cache of a later run. Successful batches delete the temporary cache after
all workers and reports finish; failed or interrupted batches retain it for
diagnosis. Use `--keep-preflight-cache` to retain it on success, or
`--no-preflight-cache` to prepare each trajectory independently. In-memory model
and inference caches are released when their worker processes exit. Each
trajectory has its own state, market ledger, configuration,
checkpoints and log. Per-case reports are generated only
after both trajectories complete and their input fingerprints agree. Existing
reports continue to compare realized operating profit, signed settlement,
imbalance, curtailment, wear, SOC, auction revenue and solver diagnostics;
planning objectives are not realized cash flow. Prepared snapshots consume disk
space and each worker still holds a separate copy in memory; no live solver state
is shared.

For an older batch that did not have automatic cleanup, run
`python scripts/cleanup_k25_preflight.py <batch-directory>`. Add `--wait` to
wait for a running batch; the command checks that all trajectories and reports
completed before removing its cache. Failed or interrupted batches keep it.

The January–August deterministic study records detailed auction and MPC
validation residuals only for failed checks. Successful checks retain their
pass status and any warnings; validation itself is never skipped. The full
diagnostic mode remains available through
`continuous_optimization.validation_diagnostics: full`.

Each batch stores one input-provenance JSON per farm and input fingerprint in
`input_provenance/farm_<id>/`. The existing per-controller and per-case JSON
paths are hard links to that record, so existing readers still see the complete
JSON without consuming extra filesystem space. Treat the shared record and its
links as immutable. Copying individual paths to another filesystem may create
separate copies; archive the whole batch with hard-link preservation to keep
the space saving. Records with different fingerprints remain separate.

The plan covers 241 local delivery days, six trajectories and 92,528 nominal MPC
decisions. The original eight-trajectory auction submission count is an upper
bound after tracker reuse; cancellations and solver
retries affect actual counts. Each worker receives one configured native thread
and, by default, the study's five Gurobi threads. Four workers can therefore
request up to 20 solver threads on a ten-core Mac, in addition to other work;
set `--solver-threads 2` explicitly to limit the maximum to eight. CPU utilization
and runtime depend on input preparation, Python, inference, MILP size and system
load. Multiple in-memory copies of the prepared snapshot increase peak memory use.

After both complete batches are available, compare them without starting any
optimization. The tool requires matching input fingerprints and delivery days,
and writes a Markdown table, matched daily CSV and per-farm daily-SOC plots to a
new directory. The changes to both auction planning and MPC mean the comparison
is a full-policy comparison, not an isolated MPC effect.

```bash
python scripts/compare_k25_batches.py \
  --baseline outputs/energy_research/deterministic_six_workers_v2 \
  --candidate outputs/energy_research/deterministic_daily_hard_v1
```

The queue's scheduler, dispatch selection and report assembly have focused
software tests. Full data preflight, actual multi-process optimization, Gurobi
license concurrency, numerical equivalence with the sequential runner, memory
use and runtime remain unverified. The queue prints per-trajectory logs and
requires a new batch directory. Never run while another process is downloading
to the market DuckDB catalog. The preflight also runs independently for each
trajectory and can be expensive.
