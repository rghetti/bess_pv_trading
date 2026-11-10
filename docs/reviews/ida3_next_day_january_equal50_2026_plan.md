# January 2026 extended IDA3 study with 50% midnight targets

This study uses the data, forecast assumptions, compute budget, January
delivery window and February 1 context described in
[the original January plan](ida3_next_day_january_2026_plan.md). It uses the
greedy grid tracker for farms 1, 345, 901 and 910; no MPC, model training,
tuning, calibration, new forecast scenarios or new data downloads are included.

Replace the 40% minimum with hard 50% SOC equalities at both future local
midnights in every two-day IDA3 auction plan. DA, IDA1 and IDA2 impose 50%
at the end of their delivery-day planning horizon. The projected state at the
start of a later-day auction plan is fixed by prior operation, so no equality
is imposed retroactively at that initial boundary. Actual greedy execution
does not enforce a daily SOC target; report planned and actual midnight SOC,
fallbacks and any discrepancies. Use the previous valid order after an
infeasible auction when available; otherwise stop and report the failure.

Compare January delivery outcomes against the existing same-day planning run
with its 50% daily equality, and against the completed extended-horizon run
with its 40% floor. The first comparison better isolates the change in
horizon, but the extended IDA3 planner also sees the next day's actual auction
prices at its earlier gate because the study uses a retrospective price oracle.
The second comparison changes SOC policy, including the starting-boundary
floor used by the older 40% run. Neither comparison is a causal real-time
price-forecast evaluation. Match realized PV, settlement prices and
auction-product prices and disclose the retrospective model selection and
synthetic PV labels.

The launcher defaults to four parallel farm workers and six Gurobi threads per
worker. Full input preflight is required before optimizing. Automatically
write per-farm and combined results, planned versus actual SOC, auction
fallbacks, solver timings, hourly tables and the January imbalance,
curtailment, net commitment, SOC and price plots for all three trajectories.
