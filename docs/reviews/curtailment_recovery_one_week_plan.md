# One-week deterministic MPC curtailment-recovery comparison

This plan was explicitly approved and the one-week comparison completed. See
`curtailment_recovery_one_week_results.md` and the associated audit reports.

Run 15–21 January 2026 in the Europe/Rome timezone for farms 1, 345, 901 and
910. Each farm has two independent continuous closed-loop trajectories:
original deterministic MPC and the same MPC with grid-neutral curtailment
recovery enabled. Each job also retains the existing grid-tracking reference.
All controllers use the same per-farm asset settings and 50% initial SOC. The
MPC includes the latest IDA2 plan for IDA3 until IDA3 becomes binding; actual
submitted orders remain fixed. Each controller can produce different later
auction orders and physical state; this is not a frozen-ledger comparison.

Recovery operates after physical and interconnection protection, holding the
protected grid exchange fixed while decreasing curtailed PV and either reducing
discharge or increasing charging. It is capped by battery power, energy
headroom, interval duration, charging efficiency and the SOC maximum. The
actual state, degradation and issued-versus-executed controls are carried
forward. The next MPC reoptimizes from that actual state. Noon and midnight
auction projections apply the same recovery rule using eligible forecast PV,
without a projection solve. Report final-study terminal-target outcomes;
recovery does not force executed final SOC to equal the planning target.

Both MPC cases minimize EUR 500/MWh absolute forecast imbalance plus modeled
battery degradation and import costs and a soft EUR 50/MWh ordinary-midnight
SOC deviation. A separate final-study 50% SOC target is binding in planning.
Neither physical MPC uses future imbalance settlement prices. The auction
optimizer retains the previous perfect auction-price information benchmark,
so actual-price availability for auction planning is retrospective. Auction
quantities are fixed at submission and settled at actual publication; realized
Terna NORD imbalance prices are used only for financial accounting.

PV forecasts use the same previously validation-selected reviewed-GHI neural
bundles as the three-day comparison: BiGRU farm 1, Transformer farm 345,
BiLSTM farm 901, TCN farm 910. They were trained previously, with historical
GHI, and selected using later 2026 validation; this is retrospective and no
training, selection, recalibration or GPU work occurs in this study. At each
cutoff the deterministic forecast uses the latest eligible weather cycle with
an assumed six-hour availability lag. Predictions obey the month-specific
nighttime-zero policy. Physical execution uses reviewed curated targets,
including source-marked synthetic values and historical-GHI fills where
applicable; those are counterfactual rather than metered production. The
interval-average realized PV is explicitly disclosed to the MPC for the
current interval, an idealized retrospective assumption; future actual PV is
not disclosed.

The matrix has eight jobs, two per farm; each job compares MPC with tracking.
For normal 96-interval days it has 5,376 MPC decisions and up to 448 auction
solves across the 16 controller trajectories. Two workers each use one CPU
thread and one Gurobi thread, no GPUs, 10 seconds per MPC and 120 seconds per
auction, relative MIP gap 1e-4 and no absolute gap. Their combined nominal
solver caps total about 29.9 aggregate CPU-hours; this is a limit, not a runtime
prediction, and preparation, reporting and retries add time. Resource use is
subject to the shared machine. No training loss, probability calibration or
model selection is part of this run. The week and models are not an untouched
sample and do not support conclusions about annual performance.

Metrics: recovered PV AC MWh and intervals; realized curtailment, charge,
discharge and wear; signed and absolute imbalance and signed settlement;
auction revenue, import costs, fees and realized profit; daily and final SOC,
terminal outcomes and noon/midnight projection errors; safety changes,
fallbacks, solver statuses and runtime. Paired differences are recovery minus
original; tracking is contextual. Realized profit is actual auction revenue
plus signed settlement minus executed degradation, import costs and fees; MPC
penalties and terminal values are not realized charges. Summaries show final
inventory because cash profit does not value stored energy. Retain interval
provenance and full plans. Output a single shared run directory with the paired
report and study plots; retain no more than three per-day plot dates.

Configuration: `configs/research/curtailment_recovery_one_week.yaml`.
Read-only preflight:

```bash
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/curtailment_recovery_one_week.yaml --check-only --workers 2
```

After approval, run the same command with `--run` instead of `--check-only`,
then run `scripts/audit_energy_research_results.py` and
`scripts/report_mpc_recovery.py` on its output directory. Licensed solver tests
and preflight do not count as a historical experiment.
